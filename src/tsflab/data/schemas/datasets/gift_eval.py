"""Parameters for GIFT-EVAL dataset windows and scaling."""

from tsflab.data.schemas.base import DatasetParameters


class GiftEvalParameterConfig(DatasetParameters):
    """Parameters for the GIFT-EVAL dataset wrapper.

    Attributes
    ----------
    scale : bool
        Whether to apply StandardScaler (fitted on training data).
    windows : int or None
        Number of rolling test windows.  ``None`` means auto-calculate
        following the GIFT-EVAL protocol.
    """

    scale: bool = True
    windows: int | None = None
