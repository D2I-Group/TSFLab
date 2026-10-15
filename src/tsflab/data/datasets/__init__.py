"""Built-in forecasting dataset implementations."""

from tsflab.data.datasets.base import ForecastingDataset
from tsflab.data.datasets.custom import Dataset_Custom
from tsflab.data.datasets.ett import Dataset_ETT_hour, Dataset_ETT_minute
from tsflab.data.datasets.periodic_data import Dataset_periodic
from tsflab.data.datasets.solar import Dataset_Solar
from tsflab.data.datasets.trend_data import Dataset_trend

__all__ = [
    "ForecastingDataset",
    "Dataset_Custom",
    "Dataset_ETT_hour",
    "Dataset_ETT_minute",
    "Dataset_periodic",
    "Dataset_Solar",
    "Dataset_trend",
]
