import os
import re
import sys
import time
from glob import glob

import joblib
import torch
from torch.utils.data import TensorDataset

from universal_transformer import DATA_DIR_PATH, logger
from universal_transformer.class_registry import register_class, registry

memory = joblib.Memory(
    cachedir=os.path.join(DATA_DIR_PATH, "joblib_cache", "datasets", sys.platform)
)


@register_class(("dataset", "babi"))
class BabiDataset:
    def __init__(
        self,
        tokenizer,
        output_tokenizer,
        task="all",
        version="10k",
        group_story_sents=True,
        debug=False,
    ):
        self.debug = debug
        self.group_story_sents = group_story_sents

        task_prefix = "qa*" if task == "all" else f"qa{task}"
        version = "en-valid-10k" if version == "10k" else "en-valid"

        train_path = os.path.join(
            DATA_DIR_PATH, "babi", version, f"{task_prefix}_train.txt"
        )

        self.train = self.path_to_dataset(train_path, tokenizer, fit_tokenizer=True)

        val_path = os.path.join(
            DATA_DIR_PATH, "babi", version, f"{task_prefix}_valid.txt"
        )
        self.val = self.path_to_dataset(val_path, tokenizer)

        test_path = os.path.join(
            DATA_DIR_PATH, "babi", version, f"{task_prefix}_test.txt"
        )
        self.test = self.path_to_dataset(test_path, tokenizer)

    def path_to_dataset(self, glob_path, tokenizer, fit_tokenizer=False):
        examples = []
        for path in glob(glob_path):
            examples.extend(self.read_babi_lines(path))

        stories, answers, task_ids = zip(*examples)
        start_time = time.time()
        logger.info(f"Starting to compute for tensors for path: {glob_path}")
        tensors = self.stories_to_tensors(
            stories, answers, task_ids, tokenizer, fit_tokenizer
        )
        logger.info(
            f"Done to computing tensors. Took {time.time() - start_time} seconds."
        )
        return TensorDataset(*tensors)

    def stories_to_tensors(
        self, stories, answers, task_ids, tokenizer, fit_tokenizer=False
    ):
        assert len(stories) == len(answers)
        logger.info(f"Starting to fit tokenizer.")
        stories_flat = tuple(map(lambda x: " ".join(x), stories))
        if fit_tokenizer:
            tokenizer.fit(texts=stories_flat + answers)
        logger.info("Done fitting tokenizer.")

        logger.info(f"Starting to convert stories.")
        answers_ids, answers_attn_masks = texts_to_tensors(answers, tokenizer)
        if self.group_story_sents:
            stories_ids, stories_attn_masks = self.story_texts_to_tensors(
                stories, tokenizer
            )
        else:
            stories_ids, stories_attn_masks = texts_to_tensors(stories_flat, tokenizer)
        logger.info(f"Done converting stories.")

        task_numbers = torch.tensor(task_ids)
        return (
            stories_ids,
            answers_ids,
            stories_attn_masks,
            answers_attn_masks,
            task_numbers,
        )

    @staticmethod
    def story_texts_to_tensors(stories, tokenizer):
        max_story_length = max(list(map(len, stories)))
        max_sent_length = float("-inf")
        pad_id = tokenizer.token_to_id[tokenizer.pad_token]

        stories_ids = []
        for story in stories:
            story_ids = []
            for sent in story:
                sent_ids = tokenizer.encode(sent)
                max_sent_length = max(len(sent_ids), max_sent_length)
                story_ids.append(sent_ids)
            stories_ids.append(story_ids)

        for story_ids in stories_ids:
            for sent_ids in story_ids:
                for _ in range(max_sent_length - len(sent_ids)):
                    sent_ids.append(pad_id)
            for _ in range(max_story_length - len(story_ids)):
                story_ids.append([pad_id] * max_sent_length)

        stories_ids = torch.tensor(stories_ids)
        stories_attn_masks = (stories_ids != pad_id).any(axis=-1)
        return stories_ids, stories_attn_masks

    def read_babi_lines(self, path):
        task_id = int(re.search(r"\d+", os.path.basename(path)).group(0))
        story_lines = []
        stories_read = 0
        with open(path) as f:
            for line in f:
                if self.debug and stories_read >= 10:
                    break
                line_num = int(line.split(" ")[0])
                line = self.clean_line(line)
                if line_num == 1 and story_lines:
                    story_lines = []
                if "?" in line:
                    question, answer = re.split(r"(?<=\?)\s*", line)
                    stories_read += 1
                    yield story_lines + [question], answer, task_id
                else:
                    story_lines.append(line)

    @staticmethod
    def clean_line(line):
        line = re.sub(r"^[\d\s]*", "", line, flags=re.MULTILINE)
        line = re.sub(r"[\d\s]*$", "", line, flags=re.MULTILINE)
        return line


class WMTTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        hf_dataset,
        split,
        input_lang,
        output_lang,
        tokenizer,
        output_tokenizer,
        debug,
    ):
        self.hf_dataset = hf_dataset
        self.split = split
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.tokenizer = tokenizer
        self.output_tokenizer = output_tokenizer
        self.debug = debug

    def __len__(self):
        if self.debug:
            return 100
        return len(self.hf_dataset[self.split])

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        if isinstance(item, int):
            raise NotImplementedError()

        texts = self.hf_dataset[self.split][item]["translation"]
        if isinstance(item, int):
            texts = [texts]
        # TODO: These are coming in as ints, which is slower because
        # I can't batch them on the spacy side. See below for how to
        # fix: https://discuss.pytorch.org/t/force-dataloader-to-fetch-batched-index-from-custom-batch-sampler/86656/3

        input_texts = [text[self.input_lang] for text in texts]
        batch_input_ids, batch_input_ids_padding_mask = texts_to_tensors(
            input_texts, self.tokenizer
        )

        output_texts = [text[self.output_lang] for text in texts]
        batch_output_ids, batch_output_ids_padding_mask = texts_to_tensors(
            output_texts, self.output_tokenizer
        )

        batch_task_ids = torch.zeros(len(texts))

        return (
            batch_input_ids,
            batch_output_ids,
            batch_input_ids_padding_mask,
            batch_output_ids_padding_mask,
            batch_task_ids,
        )


@register_class(("dataset", "wmt"))
class WTMDataset:
    def __init__(self, tokenizer, output_tokenizer, debug=False):
        from datasets import load_dataset

        dataset = load_dataset("wmt14", "de-en")

        def get_train_iter(language):
            batch_size = 50000
            length = len(dataset["train"])
            if debug:
                batch_size = 100
                length = 100
            for i in range(0, length, batch_size):
                chunk = dataset["train"][i : i + batch_size]["translation"]
                yield [example[language] for example in chunk]

        logger.info("Fitting input tokenizer.")
        tokenizer.fit(text_batch_iter=get_train_iter("de"))
        logger.info("Done fitting input tokenizer.")
        logger.info("Fitting output tokenizer.")
        output_tokenizer.fit(text_batch_iter=get_train_iter("en"))
        logger.info("Done fitting output tokenizer.")

        self.train = WMTTorchDataset(
            hf_dataset=dataset,
            split="train",
            input_lang="de",
            output_lang="en",
            tokenizer=tokenizer,
            output_tokenizer=output_tokenizer,
            debug=debug,
        )
        self.val = WMTTorchDataset(
            hf_dataset=dataset,
            split="validation",
            input_lang="de",
            output_lang="en",
            tokenizer=tokenizer,
            output_tokenizer=output_tokenizer,
            debug=debug,
        )
        self.test = WMTTorchDataset(
            hf_dataset=dataset,
            split="test",
            input_lang="de",
            output_lang="en",
            tokenizer=tokenizer,
            output_tokenizer=output_tokenizer,
            debug=debug,
        )


def texts_to_tensors(texts, tokenizer):
    """Convert a sequence of texts and labels to a dataset."""
    token_ids_seqs = tokenizer.encode(texts)
    seq_length_max = len(token_ids_seqs[0])
    pad_token_id = tokenizer.token_to_id[tokenizer.pad_token]
    lengths = [
        ids.index(pad_token_id) if ids[-1] == pad_token_id else len(ids)
        for ids in token_ids_seqs
    ]
    att_masks = [[1] * length + [0] * (seq_length_max - length) for length in lengths]

    token_ids_seqs = torch.tensor(token_ids_seqs, dtype=torch.long)
    att_masks = torch.tensor(att_masks, dtype=torch.bool)

    return token_ids_seqs, att_masks


# TODO: Not doing anything with the output tokenizer and not fitting
# it anywhere.
def get_dataset(config, tokenizer=None, output_tokenizer=None):
    key = ("dataset", config.dataset)
    if key in registry:
        cls, dataset_kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        for key, value in config.items():
            if "dataset." not in key:
                continue
            key = key.replace("dataset.", "")
            if key in accepted_args:
                dataset_kwargs[key] = value
        dataset_kwargs_tuple = tuple(sorted(dataset_kwargs.items()))
        tokenizer_kwargs = {k: v for k, v in config.items() if "tokenizer" in k}
        tokenizer_kwargs_tuple = tuple(sorted(tokenizer_kwargs.items()))
        return _get_dataset(
            cls,
            tokenizer,
            output_tokenizer,
            dataset_kwargs_tuple,
            tokenizer_kwargs_tuple,
        )

    raise KeyError("Dataset not found!")


@memory.cache(ignore=["tokenizer", "output_tokenizer"])
def _get_dataset(
    cls, tokenizer, output_tokenizer, dataset_kwargs_tuple, tokenizer_kwargs_tuple
):
    # TODO: tokenizer_kwargs_tuple is here just for the caching. There's
    # really no reason why we couldn't be creating the tokenizer here directly. It's
    # worthless if it isn't fitted, so it's really dependent on the dataset.
    return (
        cls(
            tokenizer=tokenizer,
            output_tokenizer=output_tokenizer,
            **dict(dataset_kwargs_tuple),
        ),
        tokenizer,
        output_tokenizer,
    )
