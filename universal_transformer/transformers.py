import math

import torch
import torch.nn as nn


class VanillaTransformer(nn.Transformer):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(d_model=d_model, *args, **kwargs)
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=0)

    def forward(self, src, tgt, *args, **kwargs):
        src = self.positional_embedding(src)
        tgt = self.positional_embedding(tgt)
        return super().forward(src, tgt, *args, **kwargs)


class UniversalTransformer(nn.Transformer):
    def __init__(
        self, d_model=512, nhead=8, dropout=0.1, max_steps=2, halting_threshold=None
    ):
        encoder = UniversalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            max_steps=max_steps,
            halting_threshold=halting_threshold,
        )
        decoder = UniversalTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            max_steps=max_steps,
            halting_threshold=halting_threshold,
        )
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            custom_encoder=encoder,
            custom_decoder=decoder,
        )


class UniversalTransformerEncoder(nn.Module):
    def __init__(
        self, d_model, nhead, dropout=0.1, max_steps=2, halting_threshold=None
    ):
        super().__init__()
        self.max_steps = max_steps
        self.halting_threshold = halting_threshold
        if self.halting_threshold is not None:
            self.halting_prob_predictor = nn.Sequential(
                nn.Linear(d_model, 1), nn.Sigmoid()
            )
        self.positional_embedding = PositionalEncoding(d_model=d_model, dropout=0)
        self.temporal_embedding = TemporalEncoding(
            d_model=d_model, dropout=0, max_len=max_steps
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.transition = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.

        """

        def step_func(state, step):
            state = self.positional_embedding(state)
            state = self.temporal_embedding(state, step)
            state = self.self_attn(
                state,
                state,
                state,
                attn_mask=mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            state = state + self.dropout_1(state)

            state = self.norm_1(state)
            state = self.transition(state)
            state = state + self.dropout_2(state)

            state = self.norm_2(state)
            return state

        return self._run_steps(src, step_func)

    def _run_steps(self, state, step_func):
        if self.halting_threshold is None:
            state = self._run_fixed_loop(state, step_func)
        else:
            state = self._run_dynamic_halting_loop(state, step_func)
        return state

    def _run_fixed_loop(self, state, step_func):
        for step in range(self.max_steps):
            state = step_func(state, step)
        return state

    def _run_dynamic_halting_loop(self, state, step_func):
        # Note: To make it easier to compare, I've kept the structure
        # variable and variable names from page 14 of the paper.
        # I'm also not really doing anything with n_updates, but
        # presumably might use them for analysis in the feature.

        device = state.device
        # halting_probability is a per-position value that determines
        # if a position will halt. Once a position has halted,
        # it's halting probability is set to 1.
        halting_probability = torch.zeros(state.size(0), state.size(1)).to(device)
        # n_updates tracks the number of times that a position was updated
        # (a position that halts in the first iteration have a value
        # of 1 because it still be updated during the iteration it halted).
        n_updates = torch.zeros(state.size(0), state.size(1)).to(device)
        # remainders will contain the amount need top each positions
        # halting probability so that it equals 1 during the iteration
        # when it halted.
        remainders = torch.zeros(state.size(0), state.size(1)).to(device)
        # new_state is a weighted combination of the state values at
        # each iteration and is the the final state returned after
        # all positions have halted. The weight is either p or, on
        # a positions final iteration, the remainder. (Intuitively,
        # early small p iterations would have a relatively small effect
        # on the final output).
        new_state = torch.zeros_like(state).to(device)

        step = 0
        while (
            (halting_probability < self.halting_threshold)
            & (n_updates < self.max_steps)
        ).any():
            # p is the the amount that's going to be added to each
            # position's halting probability during this step.
            p = self.halting_prob_predictor(state).squeeze(-1)
            # still_running is a mask where 1 indicates that the position
            # is still running.
            still_running = (halting_probability < 1).float()
            # new_halted is a mask where 1 indicates that a position
            # has now halted and will stop updating after this loop
            # (though it will be updated in this loop below).
            new_halted = (halting_probability + p * still_running > self.halting_threshold).float() * still_running
            # Update still_running to drop out those that have halted.
            still_running = (halting_probability + p * still_running <= self.halting_threshold).float() * still_running
            # Add the extra halting probability p to the cumulative
            # halting_probability for positions that is still running.
            halting_probability += p * still_running
            # Add the amount need to top up each newly halted position.
            remainders += new_halted * (1 - halting_probability)
            # Top up newly halted positions to 1.
            halting_probability += new_halted * remainders
            # Update per-position update counts.
            n_updates += still_running + new_halted
            # Compute weights that control how much this iteration
            # will affect the final output for each position.
            update_weights = ((p * still_running) + (new_halted * remainders)).unsqueeze(-1)
            # Run the step function and update new_state.
            state = step_func(state, step)
            new_state = (state * update_weights) + (new_state * (1 - update_weights))
            step += 1

        return new_state


class UniversalTransformerDecoder(UniversalTransformerEncoder):
    def __init__(self, d_model, nhead, dropout=0.1, max_steps=2, halting_threshold=1):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            max_steps=max_steps,
            halting_threshold=halting_threshold,
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm_3 = nn.LayerNorm(d_model)

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
        def step_func(state, step):
            state = self.positional_embedding(state)
            state = self.temporal_embedding(state, step)
            state_2 = self.self_attn(
                state,
                state,
                state,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            state = state + self.dropout_1(state_2)

            state = self.norm_1(state)
            state_2 = self.multihead_attn(
                state,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            state = state + self.dropout_2(state_2)

            state = self.norm_2(state)
            state_2 = self.transition(state)
            state = state + self.dropout_3(state_2)

            state = self.norm_3(state)
            return state

        return self._run_steps(tgt, step_func)


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


class TemporalEncoding(PositionalEncoding):
    def forward(self, x, cur_step):
        x = x + self.pe[cur_step, :]
        return self.dropout(x)
