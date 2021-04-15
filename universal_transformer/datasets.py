import os
import re

import torch
from torch.utils.data import TensorDataset

from universal_transformer import DATA_DIR_PATH, utils


class DatasetBase:
    name = None


class BabiDataset(DatasetBase):
    name = "babi"

    def __init__(self, tokenizer, debug=False):
        self.debug = debug
        train_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_train.txt")
        self.train = self.path_to_dataset(train_path, tokenizer, fit_tokenizer=True)

        val_path = os.path.join(DATA_DIR_PATH, "babi", "en-valid", "qa1_valid.txt")
        self.val = self.path_to_dataset(val_path, tokenizer)

    def path_to_dataset(self, path, tokenizer, fit_tokenizer=False):
        stories, answers = list(zip(*self.read_babi_lines(path)))
        if self.debug:
            stories = stories[:10]
            answers = answers[:10]
        if fit_tokenizer:
            tokenizer.fit(stories + answers)
        stories, story_attn_masks = texts_to_tensors(stories, tokenizer)
        answers, answer_attn_masks = texts_to_tensors(answers, tokenizer)
        return TensorDataset(stories, answers, story_attn_masks, answer_attn_masks)

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
                    story = " ".join(story_lines + [question])
                    yield story, answer
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
    att_masks = torch.tensor(att_masks, dtype=torch.uint8)

    return token_ids_seqs, att_masks


def get_dataset(config, tokenizer=None):
    for sub in utils.get_subclasses(DatasetBase):
        if sub.name == config.dataset:
            accepted_args = set(sub.__init__.__code__.co_varnames)
            accepted_args.remove("self")
            kwargs = {
                k.replace("dataset.", ""): v for k, v in config.items() if "dataset." in k
            }
            kwargs["tokenizer"] = tokenizer
            return sub(**kwargs)
