"""Runtime specification for GraphLassoMTSF."""

import torch
from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.graphlassomtsf.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    rnn_units: int = Field(default=32, ge=1)
    num_rnn_layers: int = Field(default=1, ge=1)
    max_diffusion_step: int = Field(default=1, ge=0)
    lasso_lambda: float = Field(default=0.02, gt=0.0)
    cov_eps: float = Field(default=1e-3, ge=0.0)
    use_curriculum_learning: bool = True
    cl_decay_steps: int = Field(default=2000, ge=1)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


def _full_training_series(train_loader, enc_in: int):
    """Return the full scaled ``(T, N)`` training series behind the loader, or ``None``."""
    data = getattr(getattr(train_loader, "dataset", None), "data", None)
    if data is None:
        return None
    series = torch.as_tensor(data).float()
    if series.ndim != 2 or series.shape[1] != enc_in or series.shape[0] < 2:
        return None
    return series


def training_setup(model, train_loader, *, pred_len, features):
    """Phase 1 (Eq. 1): graphical-lasso precision matrix of the training split.

    The official script fits it on the whole standardized training series, so the
    loader's dataset series is used when available; otherwise the newest history
    step of every training window.
    """
    del pred_len, features
    series = _full_training_series(train_loader, model.enc_in)
    if series is None:
        series = torch.cat([batch_x[:, -1, :].float() for batch_x, *_ in train_loader])
    model.fit_precision(series)


def training_objective(model, batch: TrainingBatch, criterion):
    """Configured loss with the official curriculum (scheduled teacher forcing) in the decoder.

    Teacher forcing needs a label for every node, so it is skipped when the target
    holds fewer channels than the model (``MS``).
    """
    targets = batch.target
    probability = 0.0
    if model.use_curriculum_learning and targets.shape[-1] == model.enc_in:
        probability = model.teacher_forcing_probability()
    forecast = model.forecast(batch.x, targets=targets, teacher_probability=probability)
    model.batches_seen += 1
    return forecast, criterion(batch.align(forecast), targets)


SPEC = ModelSpec(
    name="GraphLassoMTSF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_graphlassomtsf.toml",
    capabilities=frozenset({"time-series"}),
    components=("graph_conv_gru",),
    contract_task={"seq_len": 14, "pred_len": 7, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
