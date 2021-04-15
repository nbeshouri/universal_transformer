import torch.nn as nn


class UniversalTransformer(nn.Transformer):
    def __init__(self, d_model=512, nhead=8, dropout=0.1, max_steps=2):
        encoder = UniversalTransformerEncoder(
            d_model=d_model, nhead=nhead, dropout=dropout, max_steps=max_steps
        )
        decoder = UniversalTransformerDecoder(
            d_model=d_model, nhead=nhead, dropout=dropout, max_steps=max_steps
        )
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            custom_encoder=encoder,
            custom_decoder=decoder,
        )


class UniversalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_steps=2):
        super().__init__()
        self.max_steps = max_steps
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.

        """
        # TODO: Position embedding, Timing embedding, dynamic halting.
        for step in range(self.max_steps):
            scr2 = self.self_attn(
                src, src, src, attn_mask=mask, key_padding_mask=src_key_padding_mask
            )[0]
            scr = src + self.dropout1(scr2)

            scr = self.norm1(scr)
            scr2 = self.transition(scr)
            src = src + self.dropout2(scr2)

            src = self.norm2(src)

        return src


class UniversalTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, max_steps=2):
        super().__init__()
        self.max_steps = max_steps
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # TODO: Position embedding, Timing embedding, dynamic halting.
        for step in range(self.max_steps):
            tgt2 = self.self_attn(
                tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )[0]
            tgt = tgt + self.dropout1(tgt2)

            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)

            tgt = self.norm2(tgt)
            tgt2 = self.transition(tgt)
            tgt = tgt + self.dropout3(tgt2)

            tgt = self.norm3(tgt)
        return tgt
