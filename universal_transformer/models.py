import torch
import torch.nn as nn

from universal_transformer.class_registry import registry, register_class
from universal_transformer.transformers import (
    UniversalTransformer,
    VanillaTransformer,
)


@register_class(("model", "vanilla_transformer"), transformer_class=VanillaTransformer)
@register_class(
    ("model", "universal_transformer"), transformer_class=UniversalTransformer
)
class TransformerModelBase(nn.Module):
    def __init__(
        self, embedding_matrix, transformer_class=None, group_story_sents=True, **kwargs
    ):
        super().__init__()
        self.transformer_class = transformer_class
        self.group_story_sents = group_story_sents
        self.embedding_size = embedding_matrix.shape[1]
        self.vocab_size = embedding_matrix.shape[0]
        self.transformer = self.transformer_class(
            d_model=self.embedding_size, nhead=5, **kwargs
        )
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix)
        )
        if group_story_sents:
            self.sent_encoder = BagOfVectorsEncoder(self.embedding_size)
        self.output_linear = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(
        self,
        *,
        source_ids,
        target_ids,
        source_padding_mask=None,
        target_padding_mask=None
    ):
        source = self.embedding(source_ids)
        target = self.embedding(target_ids)

        if self.group_story_sents:
            source = self.sent_encoder(source)

        source = source.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        decoder_att_mask = self.transformer.generate_square_subsequent_mask(
            target.size(0)
        )
        decoder_att_mask = decoder_att_mask.to(source.device)

        output = self.transformer(
            src=source,
            tgt=target,
            tgt_mask=decoder_att_mask,
            src_key_padding_mask=~source_padding_mask,  # It thinks 1 means ignore.
            tgt_key_padding_mask=~target_padding_mask,
            memory_key_padding_mask=~source_padding_mask,
        )
        output = output.permute(1, 0, 2)
        output = self.output_linear(output)
        return output


class BagOfVectorsEncoder(nn.Module):
    def __init__(self, embedding_dim, max_sequence_length=100):
        super().__init__()
        self.mask = nn.Embedding(
            num_embeddings=max_sequence_length, embedding_dim=embedding_dim
        )

    def forward(self, x):
        # mask_indices has shape (1, num_words).
        mask_indices = torch.arange(x.size(-2)).reshape(1, -1)
        mask_indices = mask_indices.to(x.device)
        # mask has shape (1, num_words, embedding_size). This will
        # implicitly become (1, 1, num_words, embedding_size) and then
        # (batch_size, num_sents, num_words, embedding_size) when
        # multiplied by the embedded input.
        x = x * self.mask(mask_indices)
        # Sum out the "words" dimension.
        x = x.sum(axis=2)
        return x


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
