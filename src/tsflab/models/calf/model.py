"""CALF: cross-modal fine-tuning of a GPT-2 backbone for forecasting.

Clean-room TSFLab implementation of Liu et al., "CALF: Aligning LLMs for Time
Series Forecasting via Cross-modal Fine-Tuning" (AAAI 2025). Each channel's whole
look-back window is one token (channel tokens, not patches). Two branches share
one pretrained GPT-2 initialisation:

* temporal target branch -- projected time tokens ``X_time`` through GPT-2 blocks
  fine-tuned with LoRA on the fused QKV projection plus trainable layer norms and
  positional table (paper "Parameter Efficient Training");
* textual source branch -- aligned text tokens ``X_text`` (Eq. 3) through a frozen
  copy of the same blocks with only the positional table trainable.

Cross-modal match (Eqs. 2-3): ``D_hat = PCA(D)`` reduces the GPT-2 word-token
table to ``n_principal`` principal word embeddings (computed once, kept as a
buffer); multi-head cross-attention with ``X_time`` as query and ``D_hat`` as
key/value gives ``X_text``. Training adds the feature regularisation loss (Eq. 4,
``sum_l gamma^(L-l) sim(phi_text^l(F_text^l), phi_time^l(F_time^l))``) and the
output consistency loss (Eq. 5, ``sim(Y_text, Y_time)``) to the supervised loss
(Eq. 6). Only ``Y_time`` is the forecast.

The GPT-2 trunk is the shared ``gpt2_backbone`` component (pre-LN, causal
attention over the token axis, tanh-GELU MLP) with its LoRA-on-``c_attn`` and
fused-attention options, so that the pinned ``openai-community/gpt2`` weights
can be loaded from a checksum-verified local ``model.safetensors`` without the
``transformers`` package. Without that artifact the backbone is randomly
initialised; see the model card.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.gpt2_backbone import GPT2Backbone, GPT2Config
from tsflab.models._components.revin import RevIN

# GPT-2 base (124M) geometry; fixed because the pretrained table defines it.
GPT2_WIDTH = 768
GPT2_HEADS = 12
GPT2_MAX_POSITIONS = 1024
GPT2_LAYERS = 12
GPT2_LN_EPS = 1e-5
# Cross-modal match module: one post-norm encoder layer and the cross-attention
# use 12 heads, a 2048-wide ReLU feed-forward, and dropout 0.1 (official code
# relies on the torch defaults for these).
MATCH_FF = 2048
MATCH_DROPOUT = 0.1

_SIMILARITY = {
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "mse": F.mse_loss,
}


class CALFGPT2Backbone(GPT2Backbone):
    """First ``num_layers`` GPT-2 blocks (``gpt2_backbone``) fed with input embeddings.

    One ``dropout`` for embeddings, attention weights and residuals, the fused
    causal attention kernel, and, for the temporal branch, a LoRA adapter on
    every fused QKV projection. ``forward_with_hidden_states`` returns the final
    hidden state (after ``ln_f``) and ``num_layers + 1`` hidden features: the
    position-embedded input, the outputs of blocks ``1..L-1``, and the final
    normalised state -- the features compared by the feature regularisation loss.
    """

    def __init__(
        self, num_layers: int, dropout: float, lora: tuple[int, float, float] | None = None
    ) -> None:
        rank, alpha, lora_dropout = lora if lora else (0, 1.0, 0.0)
        super().__init__(GPT2Config(
            n_layer=num_layers, n_embd=GPT2_WIDTH, n_head=GPT2_HEADS,
            n_positions=GPT2_MAX_POSITIONS, embd_pdrop=dropout, attn_pdrop=dropout,
            resid_pdrop=dropout, layer_norm_epsilon=GPT2_LN_EPS,
            attn_implementation="sdpa", lora_rank=rank, lora_alpha=alpha,
            lora_dropout=lora_dropout,
        ))

    def reset_parameters(self) -> None:
        """GPT-2 initialisation in parameter order (normal std 0.02, zero biases,
        residual output projections scaled by ``1/sqrt(2 * 12)``, the released
        GPT-2 depth, whatever ``num_layers``); layer norms and LoRA keep their
        own start. Overwritten by the pretrained tensors when
        the weight artifact is loaded."""
        for name, param in self.named_parameters():
            if "lora_" in name or "ln_" in name:
                continue
            if name.endswith("bias"):
                nn.init.zeros_(param)
            elif name.endswith("c_proj.weight"):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * GPT2_LAYERS))
            else:
                nn.init.normal_(param, std=0.02)


class CrossModalMatch(nn.Module):
    """Time-token embedding and principal-word cross-attention (Eqs. 1-3).

    ``[B, C, L]`` series -> ``X_time`` (``Linear(L, 768)`` then one Transformer
    encoder layer over channel tokens) and ``X_text`` (multi-head cross-attention,
    query ``X_time``, key/value the principal word embeddings ``D_hat``).
    """

    def __init__(self, seq_len: int, n_principal: int) -> None:
        super().__init__()
        self.embed = nn.Linear(seq_len, GPT2_WIDTH)
        self.encoder = nn.TransformerEncoderLayer(
            GPT2_WIDTH, GPT2_HEADS, dim_feedforward=MATCH_FF,
            dropout=MATCH_DROPOUT, batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(GPT2_WIDTH, GPT2_HEADS, batch_first=True)
        # D_hat = PCA(D): fixed principal word embeddings [n_principal, 768];
        # a random stand-in until pretrained GPT-2 weights are loaded.
        self.register_buffer(
            "principal_embeddings", torch.randn(n_principal, GPT2_WIDTH) * 0.02
        )

    def forward(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_time = self.encoder(self.embed(series))
        words = self.principal_embeddings.unsqueeze(0).expand(series.shape[0], -1, -1)
        x_text, _ = self.cross_attention(x_time, words, words, need_weights=False)
        return x_time, x_text


def _loads_gpt2_tensor(name: str, gpt_layers: int) -> bool:
    """Whether a Hugging Face GPT-2 tensor is copied into a ``gpt_layers``-deep backbone."""
    if name.endswith((".attn.bias", ".attn.masked_bias")):
        return False  # causal-mask buffers
    if name.startswith("h.") and int(name.split(".")[1]) >= gpt_layers:
        return False
    return name.startswith(("h.", "wpe.", "ln_f."))


class Model(nn.Module):
    """CALF temporal/textual twin-branch forecaster; ``forward`` returns ``Y_time``."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        gpt_layers: int = 6,
        n_principal: int = 500,
        lora_rank: int = 8,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        dropout: float = 0.1,
        feature_weight: float = 0.01,
        output_weight: float = 1.0,
        feature_decay: float = 0.8,
        alignment_loss: str = "l1",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.gpt_layers = gpt_layers
        self.feature_weight = float(feature_weight)
        self.output_weight = float(output_weight)
        self.feature_decay = float(feature_decay)
        self.alignment_loss = alignment_loss
        self.pretrained = False

        self.revin = RevIN(enc_in, eps=1e-5, affine=False)
        self.match = CrossModalMatch(seq_len, n_principal)
        self.gpt2_time = CALFGPT2Backbone(gpt_layers, dropout, (lora_rank, lora_alpha, lora_dropout))
        self.gpt2_text = CALFGPT2Backbone(gpt_layers, dropout)
        # Both branches start from identical backbone weights.
        self.gpt2_text.load_state_dict(self.gpt2_time.state_dict(), strict=False)
        # phi_time^l and phi_text^l of Eq. (4), one per compared feature.
        self.time_proj = nn.ModuleList(
            nn.Linear(GPT2_WIDTH, GPT2_WIDTH, bias=False) for _ in range(gpt_layers + 1)
        )
        self.text_proj = nn.ModuleList(
            nn.Linear(GPT2_WIDTH, GPT2_WIDTH, bias=False) for _ in range(gpt_layers + 1)
        )
        # Forecast head shared by both branches.
        self.head = nn.Linear(GPT2_WIDTH, pred_len)
        self._freeze_backbones()

    def _freeze_backbones(self) -> None:
        """Temporal: LoRA, layer norms, positions trainable. Textual: positions only."""
        for name, param in self.gpt2_time.named_parameters():
            param.requires_grad = "lora_" in name or "ln_" in name or name.startswith("wpe")
        for name, param in self.gpt2_text.named_parameters():
            param.requires_grad = name.startswith("wpe")

    def _branch(self, backbone: CALFGPT2Backbone, tokens: torch.Tensor):
        hidden, features = backbone.forward_with_hidden_states(tokens)
        # Residual from the branch input around the GPT-2 stack, then the head.
        forecast = self.head(hidden + tokens).transpose(1, 2)
        return self.revin(forecast, "denorm"), features

    def cross_modal_outputs(self, x_enc: torch.Tensor) -> dict[str, object]:
        """Run both branches; return both forecasts and projected hidden features."""
        series = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        x_time, x_text = self.match(series)
        y_time, f_time = self._branch(self.gpt2_time, x_time)
        y_text, f_text = self._branch(self.gpt2_text, x_text)
        return {
            "time": y_time,
            "text": y_text,
            "features_time": [proj(f) for proj, f in zip(self.time_proj, f_time)],
            "features_text": [proj(f) for proj, f in zip(self.text_proj, f_text)],
        }

    def cross_modal_loss(self, outputs: dict[str, object]) -> torch.Tensor:
        """``lambda_feature * L_feature + lambda_output * L_output`` (Eqs. 4-6)."""
        sim = _SIMILARITY[self.alignment_loss]
        pairs = zip(reversed(outputs["features_time"]), reversed(outputs["features_text"]))
        # gamma^(L-l): the deepest feature has weight 1, shallower ones decay.
        feature = sum(self.feature_decay**depth * sim(t, s) for depth, (t, s) in enumerate(pairs))
        consistency = sim(outputs["time"], outputs["text"])
        return self.feature_weight * feature + self.output_weight * consistency

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """Temporal-branch forecast ``[B, pred_len, C]``; the textual branch is training-only."""
        series = self.revin(x_enc, "norm").transpose(1, 2)
        x_time, _ = self.match(series)
        return self._branch(self.gpt2_time, x_time)[0]

    def _load_backbones(self, state: dict[str, torch.Tensor]) -> None:
        for backbone in (self.gpt2_time, self.gpt2_text):
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            missing = [key for key in missing if "lora_" not in key]
            if missing or unexpected:
                raise ValueError(
                    f"GPT-2 weights mismatch: missing={missing}, unexpected={unexpected}"
                )

    @torch.no_grad()
    def load_gpt2(self, weights: dict[str, torch.Tensor]) -> None:
        """Copy pretrained GPT-2 tensors into both branches and refit ``D_hat``.

        ``weights`` uses the Hugging Face ``GPT2Model`` names (``wte.weight``,
        ``wpe.weight``, ``h.<i>.*``, ``ln_f.*``); GPT-2's ``Conv1D`` weights are
        stored ``[in, out]`` and are transposed into ``nn.Linear`` layout.
        """
        weights = {name.removeprefix("transformer."): t for name, t in weights.items()}
        state = {}
        for name, tensor in weights.items():
            if not _loads_gpt2_tensor(name, self.gpt_layers):
                continue
            if name.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
                tensor = tensor.t()
            state[name] = tensor.float().contiguous()
        self._load_backbones(state)
        buffer = self.match.principal_embeddings
        buffer.copy_(principal_word_embeddings(weights.get("wte.weight"), buffer.shape[0]))
        self.pretrained = True


def principal_word_embeddings(wte: torch.Tensor | None, n_components: int) -> torch.Tensor:
    """``D_hat = PCA(D)`` following the official offline recipe (Eq. 2).

    The vocabulary table ``D`` is ``[vocab, 768]``. The official recipe fits PCA
    on its transpose, treating each of the 768 embedding coordinates as a sample
    over the vocabulary axis; the ``[768, n_components]`` scores, transposed,
    are ``n_components`` principal word vectors of width 768. Exact SVD is used
    here (the reference relies on scikit-learn's randomized solver); component
    signs are arbitrary in both.
    """
    if wte is None:
        raise ValueError("GPT-2 weights do not contain the word-token table 'wte.weight'")
    samples = wte.float().t()  # [768, vocab]
    centred = samples - samples.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centred, full_matrices=False)
    scores = u[:, :n_components] * s[:n_components]  # [768, n_components]
    return scores.t().contiguous()
