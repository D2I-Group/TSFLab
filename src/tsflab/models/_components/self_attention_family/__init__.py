"""Attention layers used by transformer-style models."""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from tsflab.models._components.masking import ProbMask, TriangularCausalMask

try:
    from reformer_pytorch import LSHSelfAttention
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LSHSelfAttention = None


class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)

        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))

        normalizer_row_refine = torch.einsum(
            "nhld,nhd->nhl",
            queries + 1e-6,
            (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6,
        )
        normalizer_col_refine = torch.einsum(
            "nhsd,nhd->nhs",
            keys + 1e-6,
            (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6,
        )

        normalizer_row_refine = torch.sigmoid(normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]

        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1, 2)
        return x.contiguous(), None


class FlashAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, queries, keys, values, mask=None):
        block_size = 32
        neg_inf = -1e10
        epsilon = 1e-10

        # Device-agnostic: follow the input tensor's device (was hardcoded "cuda").
        output = torch.zeros_like(queries, requires_grad=True)
        l_block = torch.zeros(queries.shape[:-1], device=queries.device)[..., None]
        m_block = torch.ones(queries.shape[:-1], device=queries.device)[..., None] * neg_inf

        q_block_size = min(block_size, queries.shape[-1])
        kv_block_size = block_size

        q_blocks = torch.split(queries, q_block_size, dim=2)
        k_blocks = torch.split(keys, kv_block_size, dim=2)
        v_blocks = torch.split(values, kv_block_size, dim=2)
        if mask is not None:
            mask_blocks = list(torch.split(mask, kv_block_size, dim=1))

        tr = len(q_blocks)
        tc = len(k_blocks)

        o_blocks = list(torch.split(output, q_block_size, dim=2))
        l_blocks = list(torch.split(l_block, q_block_size, dim=2))
        m_blocks = list(torch.split(m_block, q_block_size, dim=2))

        for j in range(tc):
            k_j = k_blocks[j]
            v_j = v_blocks[j]
            if mask is not None:
                mask_j = mask_blocks[j]

            for i in range(tr):
                q_i = q_blocks[i]
                o_i = o_blocks[i]
                l_i = l_blocks[i]
                m_i = m_blocks[i]

                scale = 1 / np.sqrt(queries.shape[-1])
                q_scaled = q_i * scale

                scores = torch.einsum("... i d, ... j d -> ... i j", q_scaled, k_j)
                if mask is not None:
                    mask_j_temp = rearrange(mask_j, "b j -> b 1 1 j")
                    scores = torch.where(mask_j_temp > 0, scores, neg_inf)

                m_block_ij, _ = torch.max(scores, dim=-1, keepdims=True)
                p_ij = torch.exp(scores - m_block_ij)
                if mask is not None:
                    p_ij = torch.where(mask_j_temp > 0, p_ij, 0.0)

                l_block_ij = torch.sum(p_ij, dim=-1, keepdims=True) + epsilon

                p_ij_vj = torch.einsum("... i j, ... j d -> ... i d", p_ij, v_j)

                m_i_new = torch.maximum(m_block_ij, m_i)
                l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_block_ij - m_i_new) * l_block_ij

                o_blocks[i] = (l_i / l_i_new) * torch.exp(m_i - m_i_new) * o_i + (
                    torch.exp(m_block_ij - m_i_new) / l_i_new
                ) * p_ij_vj
                l_blocks[i] = l_i_new
                m_blocks[i] = m_i_new

        output = torch.cat(o_blocks, dim=2)
        l_block = torch.cat(l_blocks, dim=2)
        m_block = torch.cat(m_blocks, dim=2)
        return output, l_block, m_block

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = self.flash_attention_forward(
            queries.permute(0, 2, 1, 3),
            keys.permute(0, 2, 1, 3),
            values.permute(0, 2, 1, 3),
            attn_mask,
        )[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, length, num_heads, embed_dim = queries.shape
        _, series_length, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(embed_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, length, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        values = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return values.contiguous(), attn
        return values.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(self, queries, keys, sample_k, n_top):
        batch_size, num_heads, length_k, embed_dim = keys.shape
        _, _, length_q, _ = queries.shape

        keys_expand = keys.unsqueeze(-3).expand(batch_size, num_heads, length_q, length_k, embed_dim)
        index_sample = torch.randint(length_k, (length_q, sample_k))
        keys_sample = keys_expand[:, :, torch.arange(length_q).unsqueeze(1), index_sample, :]
        qk_sample = torch.matmul(queries.unsqueeze(-2), keys_sample.transpose(-2, -1)).squeeze(-2)

        measurement = qk_sample.max(-1)[0] - torch.div(qk_sample.sum(-1), length_k)
        m_top = measurement.topk(n_top, sorted=False)[1]

        q_reduce = queries[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            m_top,
            :,
        ]
        qk = torch.matmul(q_reduce, keys.transpose(-2, -1))

        return qk, m_top

    def _get_initial_context(self, values, length_q):
        batch_size, num_heads, length_v, embed_dim = values.shape
        if not self.mask_flag:
            values_sum = values.mean(dim=-2)
            context = (
                values_sum.unsqueeze(-2).expand(batch_size, num_heads, length_q, values_sum.shape[-1]).contiguous()
            )
        else:
            assert length_q == length_v
            context = values.cumsum(dim=-2)
        return context

    def _update_context(self, context, values, scores, index, length_q, attn_mask):
        batch_size, num_heads, length_v, embed_dim = values.shape

        if self.mask_flag:
            attn_mask = ProbMask(batch_size, num_heads, length_q, index, scores, device=values.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            index,
            :,
        ] = torch.matmul(attn, values).type_as(context)
        if self.output_attention:
            attns = torch.ones([batch_size, num_heads, length_v, length_v]) / length_v
            attns = attns.type_as(attn).to(attn.device)
            attns[
                torch.arange(batch_size)[:, None, None],
                torch.arange(num_heads)[None, :, None],
                index,
                :,
            ] = attn
            return context, attns
        return context, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, length_q, num_heads, embed_dim = queries.shape
        _, length_k, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        u_part = self.factor * np.ceil(np.log(length_k)).astype("int").item()
        u = self.factor * np.ceil(np.log(length_q)).astype("int").item()

        u_part = u_part if u_part < length_k else length_k
        u = u if u < length_q else length_q

        scores_top, index = self._prob_qk(queries, keys, sample_k=u_part, n_top=u)

        scale = self.scale or 1.0 / sqrt(embed_dim)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, length_q)
        context, attn = self._update_context(context, values, scores_top, index, length_q, attn_mask)

        # [B, H, L_q, D] -> [B, L_q, H, D], as in the original Informer code, so
        # AttentionLayer's view(B, L, -1) concatenates heads per time step.
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        batch_size, length_q, _ = queries.shape
        _, length_k, _ = keys.shape
        num_heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, length_q, num_heads, -1)
        keys = self.key_projection(keys).view(batch_size, length_k, num_heads, -1)
        values = self.value_projection(values).view(batch_size, length_k, num_heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        out = out.view(batch_size, length_q, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        causal=False,
        bucket_size=4,
        n_hashes=4,
    ):
        super().__init__()
        if LSHSelfAttention is None:
            raise ImportError("reformer_pytorch is required for ReformerLayer. Install it to use this attention type.")
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries):
        batch_size, length, channels = queries.shape
        if length % (self.bucket_size * 2) == 0:
            return queries
        fill_len = (self.bucket_size * 2) - (length % (self.bucket_size * 2))
        return torch.cat(
            [queries, torch.zeros([batch_size, fill_len, channels]).to(queries.device)],
            dim=1,
        )

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        batch_size, length, channels = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :length, :]
        return queries, None
