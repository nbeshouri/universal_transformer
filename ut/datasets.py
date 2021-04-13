from ut import utils
import torch
import os
import re

from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from . import DATA_DIR_PATH, tokenizers


class DatasetBase:

    # TODO: Not actually using tensor_names.
    # Should do can match to requested inputs.
    def __init__(self, tensor_names):
        self.tensor_names = tensor_names


class BabiDataset(DatasetBase):

    name = 'babi'

    def __init__(self, tokenizer, config):
        train_path = os.path.join(
            DATA_DIR_PATH, 'en-valid/qa1_train.txt')
        self.train = self.path_to_dataset(train_path, tokenizer, fit_tokenizer=True)

        val_path = os.path.join(
            DATA_DIR_PATH, 'en-valid/qa1_valid.txt')
        self.val = self.path_to_dataset(val_path, tokenizer)

    @classmethod
    def path_to_dataset(cls, path, tokenizer, fit_tokenizer=False):
        stories, answers = list(zip(*cls.read_babi_lines(path)))
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
                line_num = int(line.split(' ')[0])
                line = cls.clean_line(line)
                if line_num == 1 and story_lines:
                    story_lines = []
                if '?' in line:
                    question, answer = re.split(r'(?<=\?)\s*', line)
                    story = ' '.join(story_lines + [question])
                    yield story, answer
                else:
                    story_lines.append(line)

    @staticmethod
    def clean_line(line):
        line = re.sub(r'^[\d\s]*', '', line, flags=re.MULTILINE)
        line = re.sub(r'[\d\s]*$', '', line, flags=re.MULTILINE)
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
    att_masks = [
        [1] * length + [0] * (seq_length_max - length) for length in lengths
    ]

    token_ids_seqs = torch.tensor(token_ids_seqs, dtype=torch.long)
    att_masks = torch.tensor(att_masks, dtype=torch.uint8)

    return token_ids_seqs, att_masks

# def prepare_tensor_dataset(texts, class_ids, seq_length_max, tokenizer):
#     """Convert a sequence of texts and labels to a dataset."""
#
#     class_ids = list(
#         class_ids
#     )  # HACK: Some how the fact that this one is a series breaks shit.
#
#     token_ids_seqs = [tokenizer.encode(t) for t in texts]
#     lengths = [
#         ids.index(tokenizer.pad_token) if ids[-1] == tokenizer.pad_token else len(ids)
#         for ids in token_ids_seqs
#     ]
#
#     att_masks = [[1] * len(ts) for ts in token_ids_seqs]
#     token_type_id_seqs = [
#         [1] * length + [0] * (seq_length_max - length) for length in lengths
#     ]
#
#     token_ids_seqs = torch.tensor(token_ids_seqs, dtype=torch.long)
#     att_masks = torch.tensor(att_masks, dtype=torch.long)
#     token_type_id_seqs = torch.tensor(token_type_id_seqs, dtype=torch.long)
#     class_ids = torch.tensor(class_ids, dtype=torch.long)
#
#     return TensorDataset(token_ids_seqs, att_masks, token_type_id_seqs, class_ids)
#
#
# class WikiCommentsDatasetBase(DatasetBase):
#
#     name = "wikipedia_comments"
#
#     def __init__(self, tokenizer, config):
#         self.tensor_names = ('input_ids', 'attention_mask', 'token_type_ids', 'labels')
#
#         train_df = pd.read_csv(
#             os.path.join(
#                 DATA_DIR_PATH, "datasets", "wikipedia_comments", "train.csv.zip"
#             )
#         )
#         test_df = pd.read_csv(
#             os.path.join(
#                 DATA_DIR_PATH, "datasets", "wikipedia_comments", "test.csv.zip"
#             )
#         )
#         test_labels_df = pd.read_csv(
#             os.path.join(
#                 DATA_DIR_PATH, "datasets", "wikipedia_comments", "test_labels.csv.zip"
#             )
#         )
#         test_df = pd.merge(test_df, test_labels_df)
#         test_df = test_df.query("toxic != -1")
#
#         if config.get("undersample", False):
#             train_df = under_sample(train_df, class_col="toxic")
#             test_df = under_sample(test_df, class_col="toxic")
#
#         self.train_texts = train_df.comment_text[: config.train_size_max]
#         self.val_texts = test_df.comment_text[: config.val_size_max]
#
#         tokenizer.fit(self.train_texts)
#
#         self.train = prepare_tensor_dataset(
#             texts=self.train_texts,
#             class_ids=train_df.toxic[: config.train_size_max],
#             seq_length_max=config.seq_length_max,
#             tokenizer=tokenizer,
#         )
#         self.val = prepare_tensor_dataset(
#             texts=test_df.comment_text[: config.val_size_max],
#             class_ids=test_df.toxic[: config.val_size_max],
#             seq_length_max=config.seq_length_max,
#             tokenizer=tokenizer,
#         )


def get_dataset(config, tokenizer):
    for sub in utils.get_subclasses(DatasetBase):
        if sub.name == config.dataset:
            return sub(tokenizer, config)
