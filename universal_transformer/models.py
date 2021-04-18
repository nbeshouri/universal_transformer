import math
import os

import torch
import torch.nn as nn

from universal_transformer import utils
from universal_transformer.transformers import (
    UniversalTransformer,
    VanillaTransformer,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class ModelBase:
    name = None


class TransformerModelBase(nn.Module, ModelBase):
    transformer_class = None

    def __init__(self, embedding_matrix, **kwargs):
        super().__init__()
        self.embedding_size = embedding_matrix.shape[1]
        self.vocab_size = embedding_matrix.shape[0]
        self.transformer = self.transformer_class(
            d_model=self.embedding_size, nhead=5, **kwargs
        )
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        self.output_linear = nn.Linear(self.embedding_size, self.vocab_size)
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self._dummy_param.device

    def forward(
        self,
        *,
        source_ids,
        target_ids,
        source_padding_mask=None,
        target_padding_mask=None
    ):
        source_ids = self.embedding(source_ids)
        target_ids = self.embedding(target_ids)

        source_ids = source_ids.permute(1, 0, 2)
        target_ids = target_ids.permute(1, 0, 2)

        decoder_att_mask = self.transformer.generate_square_subsequent_mask(
            target_ids.size(0)
        )
        decoder_att_mask.to(self.device)

        output = self.transformer(
            src=source_ids,
            tgt=target_ids,
            tgt_mask=decoder_att_mask,
            src_key_padding_mask=1 - source_padding_mask,  # It things 1 means ignore.
            tgt_key_padding_mask=1 - target_padding_mask,
        )
        output = output.permute(1, 0, 2)
        output = self.output_linear(output)
        return output


# TODO: Do this with some kind of class decorator to register
# and name of the model and pass in arguments rather than with
# named subclasses (basically you named argument sets).
class VanillaTransformerModel(TransformerModelBase):
    name = "vanilla_transformer"
    transformer_class = VanillaTransformer


class UniversalTransformerModel(TransformerModelBase):
    name = "universal_transformer"
    transformer_class = UniversalTransformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_model(config, embedding_matrix=None):
    for sub in utils.get_subclasses(ModelBase):
        if sub.name == config.model:
            accepted_args = set(sub.__init__.__code__.co_varnames)
            accepted_args.remove("self")
            kwargs = {
                k.replace("model.", ""): v for k, v in config.items() if "model." in k
            }
            if embedding_matrix is not None:
                kwargs["embedding_matrix"] = embedding_matrix
            return sub(**kwargs)
