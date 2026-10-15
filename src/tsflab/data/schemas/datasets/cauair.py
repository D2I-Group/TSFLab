"""Parameter schema for the CauAir datasets."""

from tsflab.data.schemas.base import DatasetParameters


class DatasetParameterConfig(DatasetParameters):
    """Validated parameters for ``cauair_st`` / ``cauair_ts`` datasets.

    Parameters
    ----------
    input_dim : int
        Number of channels kept from ``data`` (value + covariates).
    npz_name : str
        Bundle file name inside ``root_path``.
    scale : bool
        Whether to z-score with the bundle's per-channel ``mean``/``std``.
    """

    input_dim: int = 8
    npz_name: str = "his.npz"
    scale: bool = True
    max_windows: int | None = None
