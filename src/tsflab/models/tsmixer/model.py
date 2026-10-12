"""Clean-room implementation of the basic TSMixer architecture."""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.mixer_block import MixerBlock
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Basic historical-target TSMixer from Appendix B.3.2."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.blocks = nn.ModuleList(
            MixerBlock(seq_len, enc_in, d_model, dropout) for _ in range(e_layers)
        )
        self.projection = ChannelWiseLinear(seq_len, pred_len, enc_in)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        hidden = x_enc
        for block in self.blocks:
            hidden = block(hidden)
        return self.projection(hidden.transpose(1, 2)).transpose(1, 2)
