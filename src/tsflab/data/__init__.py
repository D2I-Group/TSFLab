"""Dataset construction and data-loader entry points.

``build_data_loader`` is imported lazily so torch-free modules in this package
(calendar features, store readers, converters) stay importable without torch.
"""

__all__ = ["build_data_loader"]


def __getattr__(name: str):
    if name == "build_data_loader":
        from tsflab.data.provider import build_data_loader

        return build_data_loader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
