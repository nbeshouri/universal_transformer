import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import math
from transformers import *

from ut import utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ModelBase:
    name = None

    # def __init__(self, **kwargs):
    #     pass

# class Simple(nn.Module, ModelBase):
#
#     name = "simple"
#
#     def __init__(self, embedding_matrix=None, hidden_size=128, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.FloatTensor(embedding_matrix)
#         )
#         self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
#         self.classifier = nn.Linear(hidden_size, 2)
#
#     def forward(self, input_ids=None, target_ids=None):
#         x = self.embbeding(input_ids)
#         _, hidden = self.encoder_rnn(x)
#
#         return output


class Vanilla(nn.Module, ModelBase):
    name = "vanilla"

    def __init__(self, embedding_matrix, **kwargs):
        super().__init__()
        self.embedding_size = embedding_matrix.shape[1]
        self.vocab_size = embedding_matrix.shape[0]
        self.transformer = nn.Transformer(d_model=self.embedding_size, nhead=5, **kwargs)
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.postional_embedding = PositionalEncoding(d_model=self.embedding_size, dropout=0)
        self.output_linear = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, *, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        tgt = self.postional_embedding(tgt)
        src = self.postional_embedding(src)
        x = self.transformer(src=src, tgt=tgt, src_key_padding_mask=1 - src_key_padding_mask, tgt_key_padding_mask=1 - tgt_key_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.output_linear(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_model(config, embedding_matrix=None):
    for sub in utils.get_subclasses(ModelBase):
        if sub.name == config.model:
            accepted_args = set(sub.__init__.__code__.co_varnames)
            accepted_args.remove("self")
            kwargs = {
                k.replace('model.', ''): v
                for k, v in config.items() if 'model.' in k
            }
            if embedding_matrix is not None:
                kwargs["embedding_matrix"] = embedding_matrix
            return sub(**kwargs)
