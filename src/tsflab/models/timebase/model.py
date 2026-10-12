"""TimeBase model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


def cal_orthogonal_loss(matrix: torch.Tensor) -> torch.Tensor:
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    two_diag = torch.diag_embed(one_diag)
    off_diagonal = gram_matrix - two_diag
    loss = torch.norm(off_diagonal, dim=(-2, -1))
    return loss.mean()


def _check_lengths(seq_len: int, pred_len: int, period_len: int) -> None:
    if seq_len < 1 or pred_len < 1:
        raise ValueError("seq_len, pred_len, and enc_in must be positive")
    if period_len > seq_len:
        raise ValueError("period_len must be in [1, seq_len]")


class TimeBaseModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_len: int,
        basis_num: int,
        individual: bool,
        orthogonal_weight: float,
        use_period_norm: bool,
    ) -> None:
        super().__init__()
        _check_lengths(seq_len, pred_len, period_len)
        self.use_period_norm = use_period_norm
        self.orthogonal_weight = orthogonal_weight
        self.aux_loss: torch.Tensor | None = None
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.pad_seq_len = 0
        self.basis_num = basis_num

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1

        self.individual = individual
        self._build_projections()

    def _build_projections(self) -> None:
        """Segment-to-basis and basis-to-segment maps, one pair per channel when ``individual``."""
        if self.individual:
            self.ts2basis = nn.ModuleList()
            self.basis2ts = nn.ModuleList()
            for _ in range(self.enc_in):
                self.ts2basis.append(nn.Linear(self.seg_num_x, self.basis_num))
                self.basis2ts.append(nn.Linear(self.basis_num, self.seg_num_y))
        else:
            self.ts2basis = nn.Linear(self.seg_num_x, self.basis_num)
            self.basis2ts = nn.Linear(self.basis_num, self.seg_num_y)

    def _normalize_input(
        self, x: torch.Tensor, batch_size: int, channels: int
    ) -> tuple[torch.Tensor, dict]:
        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        x = x.reshape(batch_size, channels, -1)
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean
        x = x.reshape(-1, self.period_len, self.seg_num_x)
        return x, {"mean": mean}

    def _denormalize_output(
        self,
        x: torch.Tensor,
        norm_stats: dict,
        batch_size: int,
        channels: int,
    ) -> torch.Tensor:
        if self.use_period_norm:
            return x + norm_stats["period_mean"]
        x = x.reshape(batch_size, channels, -1)
        x = x + norm_stats["mean"]
        return x.reshape(-1, self.period_len, self.seg_num_y)

    def _project(
        self, x: torch.Tensor, batch_size: int, channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Basis coefficients and forecast segments of the normalized ``[B*C, period, seg_x]`` input."""
        if self.individual:
            x = x.reshape(batch_size, channels, self.period_len, self.seg_num_x)
            x_pred = torch.zeros(
                [batch_size, channels, self.period_len, self.seg_num_y],
                dtype=x.dtype,
                device=x.device,
            )
            x_basis = torch.zeros(
                [batch_size, channels, self.period_len, self.basis_num],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.enc_in):
                x_basis[:, i, :, :] = self.ts2basis[i](x[:, i, :, :])
                x_pred[:, i, :, :] = self.basis2ts[i](x_basis[:, i, :, :])
            x_basis = x_basis.reshape(-1, self.period_len, self.basis_num)
            x = x_pred.reshape(-1, self.period_len, self.seg_num_y)
        else:
            x_basis = self.ts2basis(x)
            x = self.basis2ts(x_basis)
        return x, x_basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"TimeBase expects input shaped (batch, {self.seq_len}, {self.enc_in})"
            )
        batch_size, _, channels = x.shape
        x = x.permute(0, 2, 1)

        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            x = torch.cat(
                [x, x[:, :, pad_start - self.pad_seq_len : pad_start]], dim=-1
            )

        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x = x.permute(0, 1, 3, 2).reshape(-1, self.period_len, self.seg_num_x)

        x, norm_stats = self._normalize_input(x, batch_size, channels)

        x, x_basis = self._project(x, batch_size, channels)

        x = self._denormalize_output(x, norm_stats, batch_size, channels)

        x = x.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y).permute(
            0, 1, 3, 2
        )
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        self.aux_loss = (
            self.orthogonal_weight * cal_orthogonal_loss(x_basis)
            if self.training and self.orthogonal_weight > 0
            else None
        )
        return x[:, : self.pred_len, :]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_len: int,
        basis_num: int,
        individual: bool,
        orthogonal_weight: float,
        use_period_norm: bool,
    ) -> None:
        super().__init__()
        self.aux_loss: torch.Tensor | None = None
        self.model = TimeBaseModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            period_len=period_len,
            basis_num=basis_num,
            individual=individual,
            orthogonal_weight=orthogonal_weight,
            use_period_norm=use_period_norm,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        output = self.model(x_enc)
        self.aux_loss = self.model.aux_loss
        return output
