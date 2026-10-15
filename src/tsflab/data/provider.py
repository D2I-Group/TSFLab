"""Dataset construction and DataLoader wiring."""

from __future__ import annotations

from typing import cast

from torch.utils.data import DataLoader

from tsflab.catalog.registry.datasets import DATASET_REGISTRY


def build_data_loader(
    dataset_name: str,
    root_path: str,
    data_path: str,
    size: tuple[int, int, int],
    flag: str,
    features: str,
    dataset_params: dict,
    batch_size: int,
    num_workers: int,
    *,
    drop_last: bool | None = None,
    window_context: str = "carry",
) -> tuple[object, DataLoader]:
    """Build a dataset instance and its DataLoader.

    Parameters
    ----------
    dataset_name : str
        Registered dataset name.
    root_path : str
        Dataset root directory.
    data_path : str
        Data file name.
    size : tuple[int, int, int]
        Sequence length, label length, prediction length.
    flag : str
        Split flag: "train", "val", or "test".
    features : str
        Feature mode ("M", "S", "MS").
    dataset_params : dict
        Dataset parameters for the selected dataset.
    batch_size : int
        Batch size for the loader.
    num_workers : int
        DataLoader worker count.
    drop_last : bool | None
        Training-only final-batch policy. None keeps a short batch when the
        entire training split is smaller than batch_size. Evaluation never drops.

    Returns
    -------
    tuple[object, DataLoader]
        Dataset instance and DataLoader.
    """
    dataset_spec = DATASET_REGISTRY.get(dataset_name)
    if window_context not in {"carry", "within-split"}:
        raise ValueError("window_context must be carry or within-split")
    dataset_cls = cast(type, dataset_spec.dataset_class)

    dataset_kwargs = dict(dataset_params)

    data_set = dataset_cls(
        root_path=root_path,
        data_path=data_path,
        size=size,
        flag=flag,
        features=features,
        **dataset_kwargs,
    )
    if window_context == "within-split":
        from tsflab.data.datasets.base import ForecastingDataset

        if not isinstance(data_set, ForecastingDataset) or not hasattr(data_set, "_history_context_rows"):
            raise ValueError(
                "within-split windows require a chronological ForecastingDataset with declared context rows"
            )
        context_rows = data_set._history_context_rows
        data_set.data = data_set.data[context_rows:]
        if data_set.time_stamp is not None:
            data_set.time_stamp = data_set.time_stamp[context_rows:]
        if len(data_set.data) < size[0] + size[2]:
            raise ValueError("within-split partition is too short for the requested lookback and horizon")
    shuffle_flag = flag == "train"
    # Training drops the incomplete final batch (the Informer/Autoformer
    # convention, kept by many official repositories). A trailing batch of one
    # sample breaks BatchNorm in train mode and gives one noisy update; the
    # shuffle changes the dropped samples every epoch. A training split smaller
    # than one batch keeps its single short batch. Validation and test never
    # drop samples, so every evaluation window is scored.
    drop_last = flag == "train" and (len(data_set) >= batch_size if drop_last is None else drop_last)
    if drop_last and len(data_set) < batch_size:
        raise ValueError("training.drop_last=true yields no batches; reduce batch_size or retain the partial batch")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
