"""Opt-in sample forecast figures in model-space units, outside metric computation."""

import warnings


def log_predictions(model, loader, config, device, tracker):
    count = tracker.options.prediction_samples
    if not count:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from tsflab.experiments.runner.evaluator import _point_reduce, _resolve_output_kind
    from tsflab.experiments.runner.model_io import (
        call_forecaster,
        make_decoder_input,
        slice_prediction_target,
    )

    try:
        values, targets, marks, future_marks = next(iter(loader))
        values, targets = (
            values[:count].float().to(device),
            targets[:count].float().to(device),
        )
        marks = marks[:count].float().to(device) if marks is not None else None
        future_marks = future_marks[:count].float().to(device) if future_marks is not None else None
        model.eval()
        with torch.no_grad():
            decoder = make_decoder_input(targets, config.task.label_len, config.task.pred_len, device)
            outputs = call_forecaster(model, values, marks, decoder, future_marks)
            outputs, targets = slice_prediction_target(outputs, targets, config.task.pred_len, config.task.features)
        pred, actual = outputs.cpu().numpy(), targets.cpu().numpy()
        kind, _ = _resolve_output_kind(model)
        levels = list(config.evaluation.quantile_levels)
        point = _point_reduce(pred, kind, levels) if pred.ndim == 4 else pred
        for sample in range(len(actual)):
            fig, axis = plt.subplots(figsize=(7, 3))
            try:
                axis.plot(actual[sample, :, 0], label="Observed")
                axis.plot(point[sample, :, 0], label="Forecast")
                if kind == "quantile":
                    low = int(np.argmin(np.abs(np.asarray(levels) - 0.1)))
                    high = int(np.argmin(np.abs(np.asarray(levels) - 0.9)))
                    axis.fill_between(
                        np.arange(pred.shape[1]),
                        pred[sample, :, 0, low],
                        pred[sample, :, 0, high],
                        alpha=0.2,
                        label=f"Quantiles {levels[low]:g}–{levels[high]:g}",
                    )
                elif kind == "distribution":
                    location, scale = pred[sample, :, 0, 0], pred[sample, :, 0, 1]
                    axis.fill_between(
                        np.arange(pred.shape[1]),
                        location - 1.281552 * scale,
                        location + 1.281552 * scale,
                        alpha=0.2,
                        label="Gaussian 80% interval",
                    )
                axis.set(
                    xlabel="Forecast step",
                    ylabel="Model-space value",
                    title=f"Sample {sample + 1}, target channel 1",
                )
                axis.legend()
                tracker.figure(fig, f"prediction-{sample + 1}", config.training.epochs + 1)
            finally:
                plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Optional prediction plot unavailable: {exc}", stacklevel=2)
