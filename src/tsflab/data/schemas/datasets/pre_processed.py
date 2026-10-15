"""Parameter contract for pre-windowed NumPy forecasting datasets."""

from tsflab.data.schemas.base import DatasetParameters


class PreProcessedParameterConfig(DatasetParameters):
    """No dataset-level params; preprocessing is done by `tsf data prepare`."""
