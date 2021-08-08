import argparse
import os
from collections import defaultdict
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

TEMP_WEIGHTS_PATH = "state_dict.pickle"


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
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True
    )

    total_loss = 0
    preds = []
    logits = []
    label_ids = []
    task_ids = []
    batches_since_yield = 0
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
        loss = criterion(
            batch_logits.view(-1, batch_logits.size(-1)), output_ids_targets.reshape(-1)
        )

        if config.dynamic_halting_loss_weight:
            if "input_n_updates" in extra_output:
                input_halting_loss = extra_output["input_n_updates"] + extra_output["input_remainders"]
                input_halting_loss *= batch_input_ids_padding_mask
                input_halting_loss = input_halting_loss.sum() / batch_input_ids_padding_mask.sum()
                loss += config.dynamic_halting_loss_weight * input_halting_loss

            if "output_n_updates" in extra_output:
                mask = batch_output_ids_padding_mask[:, 1:]
                output_halting_loss = extra_output["output_n_updates"] + extra_output["output_remainders"]
                output_halting_loss *= mask
                output_halting_loss = input_halting_loss.sum() / mask.sum()
                loss += config.dynamic_halting_loss_weight * output_halting_loss

        total_loss += loss.item() * len(batch[0])  # Convert from mean to sum.

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_logits = batch_logits.detach().cpu().numpy()
        logits.append(batch_logits)
        preds.extend(np.argmax(batch_logits, axis=-1))
        label_ids.extend(output_ids_targets.detach().cpu().numpy())
        task_ids.extend(batch_task_ids.detach().cpu().numpy())
        batches_since_yield += 1

        if (
            i == len(dataloader) - 1
            or yield_freq is not None
            and (i + 1) % yield_freq == 0
        ):
            logits = np.concatenate(logits, axis=0)
            yield np.array(logits), np.array(preds), np.array(label_ids), np.array(
                task_ids
            ), total_loss / batches_since_yield
            total_loss = 0
            preds = []
            logits = []
            label_ids = []
            task_ids = []
            batches_since_yield = 0


def train(config, run):
    # Load stuff based on the config.
    tokenizer = tokenizers.get_tokenizer(config)
    data, tokenizer = datasets.get_dataset(config, tokenizer)
    config.train_size = len(data.train)
    config.val_size = len(data.val)

    embedding_matrix = vectors.get_vectors(config, tokenizer)

    model = models.get_model(config, embedding_matrix)

    if config.log is not None:
        wandb.watch(model, log=config.log)

    device = torch.device(config.device)
    model.to(device)

    best_performance = None
    step = 0
    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=config.lr)
        else:
            raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')

        scheduler = None
        if config.get("learning_rate_decay_schedule", None) is not None:
            if config.learning_rate_decay_schedule == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(data.train) * config.epochs,
                )
            else:
                raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')
        model.train()
        mini_batch_start_time = perf_counter()

        for logits, preds, label_ids, task_ids, loss in run_model_on_dataset(
            model,
            data.train,
            tokenizer,
            config,
            yield_freq=config.get("log_freq"),
            optimizer=optimizer,
            scheduler=scheduler,
        ):
            step += 1
            train_metrics = compute_metrics(
                logits=logits,
                preds=preds,
                label_ids=label_ids,
                task_ids=task_ids,
                loss=loss,
                runtime=perf_counter() - mini_batch_start_time,
                ignore_index=tokenizer.token_to_id[tokenizer.pad_token],
            )
            log_step("train", train_metrics, step=step, epoch=epoch)

            # Validate
            model.eval()
            with torch.no_grad():
                start_time = perf_counter()
                logits, preds, label_ids, task_ids, loss = iter(
                    next(
                        run_model_on_dataset(
                            model, data.val, tokenizer, config, yield_freq=None
                        )
                    )
                )
                val_metrics = compute_metrics(
                    logits=logits,
                    preds=preds,
                    label_ids=label_ids,
                    task_ids=task_ids,
                    loss=loss,
                    runtime=perf_counter() - start_time,
                    ignore_index=tokenizer.token_to_id[tokenizer.pad_token],
                )
                log_step("val", val_metrics, step=step, epoch=epoch)
                log_summary("val")

                if config.checkpoint_metric is not None:
                    if (
                        best_performance is None
                        or val_metrics[config.checkpoint_metric] > best_performance
                    ):
                        best_performance = val_metrics[config.checkpoint_metric]
                        torch.save(model.state_dict(), TEMP_WEIGHTS_PATH)

            model.train()  # Need to re-enter training model.

            mini_batch_start_time = perf_counter()

    # Test
    if hasattr(data, "test"):
        if config.checkpoint_metric is not None:
            model.load_state_dict(torch.load(TEMP_WEIGHTS_PATH))
            logger.info(
                f"Loaded checkpoint weights for metric: {config.checkpoint_metric}."
            )
        model.eval()
        with torch.no_grad():
            start_time = perf_counter()
            logits, preds, label_ids, task_ids, loss = iter(
                next(
                    run_model_on_dataset(
                        model,
                        data.test,
                        tokenizer,
                        config,
                        teacher_forcing=False,
                        yield_freq=None,
                    )
                )
            )
            test_metrics = compute_metrics(
                logits=logits,
                preds=preds,
                label_ids=label_ids,
                task_ids=task_ids,
                loss=loss,
                runtime=perf_counter() - start_time,
                ignore_index=tokenizer.token_to_id[tokenizer.pad_token],
            )
            log_step("test", test_metrics, step=step, epoch=epoch)

    if config.checkpoint_metric is not None and run.name is not None and config.save_weights:
        # Save the best model weights.
        artifact = wandb.Artifact(
            f"{run.name.replace('-', '_')}_best_weights", type="weights"
        )
        artifact.add_file(TEMP_WEIGHTS_PATH)
        run.log_artifact(artifact)


def log_step(
    run_type,
    metrics,
    epoch=None,
    **kwargs,
):
    log_dict = {f"{run_type}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    logger.info(log_dict)
    wandb.log(log_dict, **kwargs)

    _step_metrics[run_type].append(metrics)


_step_metrics = defaultdict(lambda: [])


def compute_metrics(
    logits, preds, label_ids, task_ids, loss, runtime, ignore_index=None
):
    metrics = {
        # Think bAbI wants all or nothing accuracy. Can ignore padding
        # though because it's
        "loss": loss,
        "examples_per_second": len(preds) / runtime,
        "sample_size": len(preds),
    }
    is_correct = ((preds == label_ids) | (label_ids == ignore_index)).all(axis=1)
    metrics["accuracy"] = is_correct.mean()
    solved_tasks = 0
    for task in set(task_ids):
        metrics[f"accuracy_task_{task}"] = is_correct[task_ids == task].mean()
        if metrics[f"accuracy_task_{task}"] >= 0.95:
            solved_tasks += 1
    metrics["solved_tasks"] = solved_tasks
    return metrics


def log_summary(run_type):
    metrics_df = pd.DataFrame(_step_metrics[run_type])
    for agg_method in ["min", "max"]:
        for metric, value in metrics_df.agg(agg_method).items():
            wandb.run.summary[f"{run_type}_{metric}_{agg_method}"] = value


class ConfigWrapper:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getattr__(self, key):
        try:
            return getattr(self.config, key)
        except:
            pass

    def __setattr__(self, key, value):
        if key == "config":
            self.__dict__["config"] = value
        else:
            setattr(self.config, key, value)


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

    train(config, run)
