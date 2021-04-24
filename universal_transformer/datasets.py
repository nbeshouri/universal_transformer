import os
import re
from itertools import chain

import torch
from torch.utils.data import TensorDataset

from universal_transformer import DATA_DIR_PATH
from universal_transformer.class_registry import registry, register_class


@register_class(("dataset", "babi"))
class BabiDataset:
    def __init__(self, tokenizer, group_story_sents=True, debug=False):
        self.debug = debug
        self.group_story_sents = group_story_sents
        train_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_train.txt")
        self.train = self.path_to_dataset(train_path, tokenizer, fit_tokenizer=True)

        val_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_valid.txt")
        self.val = self.path_to_dataset(val_path, tokenizer)

    def path_to_dataset(self, path, tokenizer, fit_tokenizer=False):
        stories, answers = list(zip(*self.read_babi_lines(path)))
        assert len(stories) == len(answers)
        if self.debug:
            stories = stories[:10]
            answers = answers[:10]
        stories_flat = tuple(map(lambda x: " ".join(x), stories))
        if fit_tokenizer:
            tokenizer.fit(stories_flat + answers)

        answers_ids, answers_attn_masks = texts_to_tensors(answers, tokenizer)
        if self.group_story_sents:
            stories_ids, stories_attn_masks = self.story_texts_to_tensors(
                stories, tokenizer
            )
        else:
            stories_ids, stories_attn_masks = texts_to_tensors(stories_flat, tokenizer)

        return TensorDataset(
            stories_ids, answers_ids, stories_attn_masks, answers_attn_masks
        )

    @staticmethod
    def story_texts_to_tensors(stories, tokenizer):
        max_story_length = max(list(map(len, stories)))
        max_sent_length = float("-inf")

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
                    sent_ids.append(0)
            for _ in range(max_story_length - len(story_ids)):
                story_ids.append([0] * max_sent_length)

        stories_ids = torch.tensor(stories_ids)
        stories_attn_masks = (stories_ids != 0).any(axis=-1)
        return stories_ids, stories_attn_masks

    @classmethod
    def read_babi_lines(cls, path):
        story_lines = []
        with open(path) as f:
            for line in f:
                line_num = int(line.split(" ")[0])
                line = cls.clean_line(line)
                if line_num == 1 and story_lines:
                    story_lines = []
                if "?" in line:
                    question, answer = re.split(r"(?<=\?)\s*", line)
                    yield story_lines + [question], answer
                else:
                    story_lines.append(line)

    @staticmethod
    def clean_line(line):
        line = re.sub(r"^[\d\s]*", "", line, flags=re.MULTILINE)
        line = re.sub(r"[\d\s]*$", "", line, flags=re.MULTILINE)
        return line


def texts_to_tensors(texts, tokenizer):
    """Convert a sequence of texts and labels to a dataset."""
    token_ids_seqs = tokenizer.encode_texts(texts)
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


def get_dataset(config, tokenizer=None):
    key = ("dataset", config.dataset)
    if key in registry:
        cls, kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {k.replace("dataset.", ""): v for k, v in config.items() if "dataset." in k}
        )
        kwargs["tokenizer"] = tokenizer
        return cls(**kwargs)

    raise KeyError("Dataset not found!")
