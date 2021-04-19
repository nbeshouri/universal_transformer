import math
import os

import torch
import torch.nn as nn

from universal_transformer.class_registry import registry, register_class
from universal_transformer.transformers import (
    UniversalTransformer,
    VanillaTransformer,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


@register_class(("model", "vanilla_transformer"), transformer_class=VanillaTransformer)
@register_class(
    ("model", "universal_transformer"), transformer_class=UniversalTransformer
)
class TransformerModelBase(nn.Module):
    transformer_class = None

    def __init__(self, embedding_matrix, transformer_class=None, **kwargs):
        super().__init__()
        self.transformer_class = transformer_class
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


def get_model(config, embedding_matrix=None):
    key = ("model", config.model)
    if key in registry:
        cls, kwargs = registry[key]
        accepted_args = set(cls.__init__.__code__.co_varnames)
        accepted_args.remove("self")
        kwargs.update(
            {k.replace("model.", ""): v for k, v in config.items() if "model." in k}
        )
        if embedding_matrix is not None:
            kwargs["embedding_matrix"] = embedding_matrix
        return cls(**kwargs)

    raise KeyError("Model not found!")
