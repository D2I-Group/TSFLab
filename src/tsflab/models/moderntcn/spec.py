"""Runtime specification for ModernTCN."""
from pydantic import BaseModel, Field, model_validator
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.moderntcn.model import Model

def _check_stages(arrays):
    if len({len(x) for x in arrays}) != 1: raise ValueError("stage lists must have equal length")
    if any(x <= 0 for values in arrays for x in values): raise ValueError("stage values must be positive")

class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0); ffn_ratio: int = Field(default=1, gt=0)
    num_blocks: list[int] = Field(default_factory=lambda:[1], min_length=1)
    large_size: list[int] = Field(default_factory=lambda:[13], min_length=1)
    small_size: list[int] = Field(default_factory=lambda:[5], min_length=1)
    dims: list[int] = Field(default_factory=lambda:[32], min_length=1)
    patch_size: int = Field(default=16, gt=0); patch_stride: int = Field(default=16, gt=0)
    downsample_ratio: int = Field(default=2, gt=1)
    dropout: float = Field(default=0.1, ge=0, lt=1); head_dropout: float = Field(default=0.1, ge=0, lt=1)
    use_multi_scale: bool = True; revin: bool = True; affine: bool = True
    subtract_last: bool = False; decomposition: bool = False
    kernel_size: int = Field(default=25, gt=0)
    @model_validator(mode="after")
    def architecture(self):
        _check_stages((self.num_blocks,self.large_size,self.small_size,self.dims))
        if any(x % 2 == 0 for x in self.large_size+self.small_size): raise ValueError("kernels must be odd")
        if self.decomposition and self.kernel_size % 2 == 0: raise ValueError("kernel_size must be odd")
        return self

def build_model(cfg, params): return Model(cfg.task.seq_len, cfg.task.pred_len, **params)
SPEC = ModelSpec(name="ModernTCN", module="tsflab.models.moderntcn", model_class=Model,
    factory=build_model, params_schema=ModelParameterConfig, config_path="configs/models/ModernTCN.toml",
    model_card="src/tsflab/models/moderntcn/README.md", smoke_config=None,
    capabilities=frozenset(["time-series"]), components=("revin","series_decomposition"),
    contract_task={"seq_len":96,"pred_len":96,"label_len":0})
