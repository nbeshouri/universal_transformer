import argparse
import os
from collections import defaultdict
from itertools import chain
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.optim import RMSprop
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from universal_transformer import datasets, logger, models, tokenizers, vectors
from universal_transformer.wandb_utils import ConfigWrapper

TEMP_WEIGHTS_PATH = "state_dict.pickle"


class Trainer:

    def __init__(self, config, run):
        self.config = config

        self.tokenizer, self.output_tokenizer = tokenizers.get_tokenizers(config)
        self.data, self.tokenizer, self.output_tokenizer = datasets.get_dataset(
            config, self.tokenizer, self.output_tokenizer
        )
        config.train_size = len(self.data.train)
        config.val_size = len(self.data.val)

        embedding_matrix, output_embedding_matrix = vectors.get_vectors(config, self.tokenizer)
        self.run = run
        self.model = models.get_model(config, embedding_matrix, output_embedding_matrix)
        self.best_performance = None
        # self.step = 0
        self.run_history = defaultdict(lambda: [])

    def train(self):
        device = torch.device(self.config.device)
        self.model.to(device)

        if self.config.log is not None:
            wandb.watch(self.model, log=self.config.log)

        device = torch.device(self.config.device)
        self.model.to(device)

        best_performance = None
        step = 0
        run_history = defaultdict(lambda: [])

        for epoch in range(1, self.config.epochs + 1):
            if self.config.optimizer == "adam":
                optimizer = Adam(self.model.parameters(), lr=self.config.lr)
            elif self.config.optimizer == "rmsprop":
                optimizer = RMSprop(self.model.parameters(), lr=self.config.lr)
            else:
                raise ValueError(f'"{self.config.optimizer}" is an invalid optimizer name!')

            scheduler = None
            if self.config.get("learning_rate_decay_schedule", None) is not None:
                if self.config.learning_rate_decay_schedule == "linear":
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=len(self.data.train) * self.config.epochs,
                    )
                else:
                    raise ValueError(f'"{self.config.optimizer}" is an invalid optimizer name!')
            self.model.train()
            mini_batch_start_time = perf_counter()

            for preds, label_ids, task_ids, token_loss, loss in run_model_on_dataset(
                    self.model,
                    self.data.train,
                    self.tokenizer,
                    self.config,
                    yield_freq=self.config.get("log_freq"),
                    optimizer=optimizer,
                    scheduler=scheduler,
            ):
                step += 1
                train_metrics = compute_metrics(
                    token_loss=token_loss,
                    preds=preds,
                    label_ids=label_ids,
                    task_ids=task_ids,
                    loss=loss,
                    runtime=perf_counter() - mini_batch_start_time,
                    ignore_index=self.tokenizer.token_to_id[self.tokenizer.pad_token],
                )
                log_step(
                    "train", train_metrics, step=step, epoch=epoch, run_history=run_history
                )

                # Validate
                self.model.eval()
                with torch.no_grad():
                    start_time = perf_counter()
                    preds, label_ids, task_ids, token_loss, loss = iter(
                        next(
                            run_model_on_dataset(
                                self.model, self.data.val, self.tokenizer, self.config, yield_freq=None
                            )
                        )
                    )
                    val_metrics = compute_metrics(
                        token_loss=token_loss,
                        preds=preds,
                        label_ids=label_ids,
                        task_ids=task_ids,
                        loss=loss,
                        runtime=perf_counter() - start_time,
                        ignore_index=self.tokenizer.token_to_id[self.tokenizer.pad_token],
                    )
                    log_step(
                        "val", val_metrics, step=step, epoch=epoch, run_history=run_history
                    )
                    log_summary("val", run_history)

                    if self.config.checkpoint_metric is not None:
                        if (
                                best_performance is None
                                or val_metrics[self.config.checkpoint_metric] > best_performance
                        ):
                            best_performance = val_metrics[self.config.checkpoint_metric]
                            torch.save(self.model.state_dict(), TEMP_WEIGHTS_PATH)

        # Test
        if hasattr(self.data, "test"):
            if self.config.checkpoint_metric is not None:
                self.model.load_state_dict(torch.load(TEMP_WEIGHTS_PATH))
                logger.info(
                    f"Loaded checkpoint weights for metric: {self.config.checkpoint_metric}."
                )
            self.model.eval()
            with torch.no_grad():
                start_time = perf_counter()
                preds, label_ids, task_ids, token_loss, loss = iter(
                    next(
                        run_model_on_dataset(
                            self.model,
                            self.data.test,
                            self.tokenizer,
                            self.config,
                            teacher_forcing=False,
                            yield_freq=None,
                        )
                    )
                )
                test_metrics = compute_metrics(
                    token_loss=token_loss,
                    preds=preds,
                    label_ids=label_ids,
                    task_ids=task_ids,
                    loss=loss,
                    runtime=perf_counter() - start_time,
                    ignore_index=self.tokenizer.token_to_id[self.tokenizer.pad_token],
                )
                log_step(
                    "test", test_metrics, step=step, epoch=epoch, run_history=run_history
                )

        if (
                self.config.checkpoint_metric is not None
                and run.name is not None
                and self.config.save_weights
        ):
            # Save the best model weights.
            artifact = wandb.Artifact(
                f"{run.name.replace('-', '_')}_best_weights", type="weights"
            )
            artifact.add_file(TEMP_WEIGHTS_PATH)
            run.log_artifact(artifact)

        return run_history


def run_model_on_dataset(
    model,
    dataset,
    tokenizer,
    config,
    yield_freq=None,
    optimizer=None,
    scheduler=None,
    teacher_forcing=True,
):
    # dataloader = DataLoader(
    #     dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True
    # )
    # The custom sampler is required to get the datalaoder to pass
    # a list of indices to the dataset.
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(dataset),
        batch_size=config.batch_size,
        drop_last=False,
    )

    # If you don't override collate but do override sampler, it'll
    # insert an extra dimension at the front for each batch in the
    # training loop.
    def my_collate(x):
        return x[0]

    dataloader = DataLoader(
        dataset, pin_memory=True, sampler=sampler, collate_fn=my_collate
    )

    total_loss = 0
    total_token_loss = 0
    total_token_count = 0
    preds = []
    label_ids = []
    task_ids = []
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id[tokenizer.pad_token]
    )

    for i, batch in enumerate(dataloader):
        device = torch.device(config.device)
        batch = tuple(t.to(device) for t in batch)
        (
            batch_input_ids,
            batch_output_ids,
            batch_input_ids_padding_mask,
            batch_output_ids_padding_mask,
            batch_task_ids,
        ) = batch

        output_ids_inputs = batch_output_ids[:, :-1]
        output_ids_targets = batch_output_ids[:, 1:]

        if teacher_forcing:
            batch_logits, extra_output = model(
                source_ids=batch_input_ids,
                target_ids=output_ids_inputs,
                source_padding_mask=batch_input_ids_padding_mask,
                target_padding_mask=batch_output_ids_padding_mask[:, :-1],
            )
            if isinstance(batch_logits, tuple):
                batch_logits, extra_output = batch_logits
        else:
            preds_so_far = output_ids_inputs[:, :1]
            for _ in range(output_ids_inputs.size(1)):
                batch_logits, extra_output = model(
                    source_ids=batch_input_ids,
                    target_ids=preds_so_far,
                    source_padding_mask=batch_input_ids_padding_mask,
                    target_padding_mask=batch_output_ids_padding_mask[
                        :, : preds_so_far.size(1)
                    ],
                )
                preds_so_far = torch.cat(
                    [preds_so_far, batch_logits.argmax(-1)[:, -1:]], axis=1
                )

        # Need to flatten the inputs to the criterion so that logits
        # have shape (batch_size *sequences_length, vocab_size) and
        # the targets have shape (batch_size * sequences_length).
        token_loss = criterion(
            batch_logits.view(-1, batch_logits.size(-1)), output_ids_targets.reshape(-1)
        )

        loss = token_loss
        if config.dynamic_halting_loss_weight:
            if "input_n_updates" in extra_output:
                input_halting_loss = (
                    extra_output["input_n_updates"] + extra_output["input_remainders"]
                )
                input_halting_loss *= batch_input_ids_padding_mask
                input_halting_loss = (
                    input_halting_loss.sum() / batch_input_ids_padding_mask.sum()
                )
                # Can't be += of because pytorch does inplace replacement
                # and that would affect token_loss.
                loss = (
                    token_loss + config.dynamic_halting_loss_weight * input_halting_loss
                )

            if "output_n_updates" in extra_output:
                mask = batch_output_ids_padding_mask[:, 1:]
                output_halting_loss = (
                    extra_output["output_n_updates"] + extra_output["output_remainders"]
                )
                output_halting_loss *= mask
                output_halting_loss = output_halting_loss.sum() / mask.sum()
                loss = (
                    token_loss
                    + config.dynamic_halting_loss_weight * output_halting_loss
                )

        # Loss is averaged by the number of non-padding tokens.
        # Here I'm storing the raw value so that I average later
        # by the number of tokens between yields.
        token_count = batch_output_ids_padding_mask[:, 1:].sum().item()
        total_token_count += token_count
        total_loss += loss.item() * token_count
        total_token_loss += token_loss.item() * token_count

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_logits = batch_logits.detach().cpu().numpy()
        preds.extend(np.argmax(batch_logits, axis=-1))
        label_ids.extend(output_ids_targets.detach().cpu().numpy())
        task_ids.extend(batch_task_ids.detach().cpu().numpy())

        if (
            i == len(dataloader) - 1
            or yield_freq is not None
            and (i + 1) % yield_freq == 0
        ):
            yield preds, label_ids, task_ids, total_token_loss / total_token_count, total_loss / total_token_count
            total_loss = 0
            total_token_loss = 0
            total_token_count = 0
            preds = []
            label_ids = []
            task_ids = []


def log_step(
    run_type,
    metrics,
    epoch=None,
    run_history=None,
    **kwargs,
):
    log_dict = {f"{run_type}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    logger.info(log_dict)
    wandb.log(log_dict, **kwargs)
    if run_history is not None:
        run_history[run_type].append(metrics)


def compute_metrics(
    preds, label_ids, task_ids, token_loss, loss, runtime, ignore_index=None
):
    metrics = {
        # Think bAbI wants all or nothing accuracy. Can ignore padding
        # though because it's
        "loss": loss,
        "token_loss": token_loss,
        "perplexity": np.exp(token_loss),
        "examples_per_second": len(preds) / runtime,
        "sample_size": len(preds),
    }

    # Accuracy
    correct_sents = []
    correct_tokens = []
    for pred, target in zip(preds, label_ids):
        sent_correct = True
        for pred_id, target_id in zip(pred, target):
            if target_id == ignore_index:
                continue
            correct_tokens.append(pred_id == target_id)
            if pred_id != target_id:
                sent_correct = False
        correct_sents.append(sent_correct)
    metrics["sent_accuracy"] = np.mean(correct_sents)
    metrics["token_accuracy"] = np.mean(correct_tokens)

    # TODO: This is some hack crap, but need a way to not have this
    # run for non-babi stuff.
    if len(set(task_ids)) > 1:
        solved_tasks = 0
        for task in set(task_ids):
            correct_sents = np.array(correct_sents)
            metrics[f"accuracy_task_{task}"] = correct_sents[task_ids == task].mean()
            if metrics[f"accuracy_task_{task}"] >= 0.95:
                solved_tasks += 1
        metrics["solved_tasks"] = solved_tasks

    return metrics


def log_summary(run_type, run_history):
    metrics_df = pd.DataFrame(run_history[run_type])
    for agg_method in ["min", "max"]:
        for metric, value in metrics_df.agg(agg_method).items():
            wandb.run.summary[f"{run_type}_{metric}_{agg_method}"] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, default=None, metavar="N")
    parser.add_argument("--project", type=str, default="delete_me", metavar="N")

    args, unknown = parser.parse_known_args()

    if args.configs is not None:
        os.environ["WANDB_CONFIG_PATHS"] = args.configs

    run = wandb.init(project=args.project)

    config = ConfigWrapper(wandb.config)

    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    trainer = Trainer(config, run)
    trainer.train()
