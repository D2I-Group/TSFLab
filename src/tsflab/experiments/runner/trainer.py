"""Training utilities for time-series forecasting models."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from tsflab.catalog.registry.losses import get_loss
from tsflab.experiments.runner.callbacks import Callback, CallbackContext
from tsflab.experiments.runner.evaluator import _CANONICAL_LEVELS, _resolve_output_kind
from tsflab.experiments.runner.model_io import (
    call_forecaster,
    make_decoder_input,
    slice_prediction_target,
    unwrap_model,
)
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.experiments.utils.training import (
    CheckpointManager,
    EarlyStopping,
    adjust_learning_rate,
)


@dataclass
class TrainResult:
    """Summary of training outputs.

    Parameters
    ----------
    best_model_path : str
        Path to the best checkpoint saved during training.
    train_time_sec : float
        Total training time in seconds.
    epochs_run : int
        Epochs trained (fewer than configured after early stopping).
    """

    best_model_path: str
    train_time_sec: float
    epochs_run: int = 0


def _collect_aux_loss(model: nn.Module) -> torch.Tensor | None:
    """Read an optional auxiliary loss exposed by the model, if any.

    Non-invasive convention: a model may stash an auxiliary regularization term
    (e.g. a regularization term) on itself as ``aux_loss``. When present and a
    finite scalar/tensor, the trainer adds it to the main training loss. Models
    that do not set any of these attributes are completely unaffected -- the
    return value is ``None`` and the train loop behaves as before.

    Parameters
    ----------
    model : nn.Module
        The model being trained (the underlying module if DataParallel-wrapped).

    Returns
    -------
    torch.Tensor | None
        A finite auxiliary loss tensor, or ``None`` when not applicable.
    """
    target = model.module if isinstance(model, nn.DataParallel) else model
    aux = getattr(target, "aux_loss", None)
    if not torch.is_tensor(aux) or aux.numel() != 1 or not torch.isfinite(aux):
        return None
    return aux


def _assert_custom_objective_supported(model: nn.Module, training_objective=None) -> None:
    """Fail fast when a registered objective would bypass DataParallel replicas."""
    if isinstance(model, nn.DataParallel) and training_objective is not None:
        raise RuntimeError(
            "ModelSpec.training_objective is not supported with "
            "torch.nn.DataParallel; disable use_multi_gpu for this run."
        )


def _training_loss(model, outputs, batch_y, pred_len, features, criterion):
    """Resolve the training loss for one forward pass.

    Uses the configured observation criterion plus the one canonical optional
    ``aux_loss`` regularizer. Full paper objectives use ``ModelSpec`` instead.
    """
    sliced_out, sliced_tgt = slice_prediction_target(outputs, batch_y, pred_len, features)
    loss = criterion(sliced_out, sliced_tgt)
    aux = _collect_aux_loss(model)
    if aux is not None:
        loss = loss + aux
    return loss


def _forward_training(
    model,
    training_objective,
    batch_x,
    batch_x_mark,
    dec_inp,
    batch_y_mark,
    batch_y,
    pred_len,
    features,
    criterion,
):
    """Run one forward pass and resolve its explicitly declared objective.

    A special paper objective is registered on ``ModelSpec`` and returns both
    the forecast (or ``None``) and a scalar loss; see
    ``tsflab.experiments.runner.objective``. This avoids a second stochastic
    forward and keeps model-specific objective discovery out of the trainer.
    """
    if training_objective is None:
        outputs = call_forecaster(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs, _training_loss(model, outputs, batch_y, pred_len, features, criterion)

    batch = TrainingBatch(
        x=batch_x,
        x_mark=batch_x_mark,
        dec_inp=dec_inp,
        y_mark=batch_y_mark,
        y=batch_y,
        pred_len=pred_len,
        features=features,
    )
    outputs, loss = training_objective(unwrap_model(model), batch, criterion)
    if outputs is not None:
        sliced_outputs = batch.align(outputs)
        if sliced_outputs.shape != batch.target.shape:
            raise ValueError(
                "training_objective forecast shape "
                f"{tuple(sliced_outputs.shape)} does not match target "
                f"{tuple(batch.target.shape)}"
            )
    if not torch.is_tensor(loss) or loss.numel() != 1 or not torch.isfinite(loss):
        raise ValueError("training_objective must return a finite scalar tensor loss")
    return outputs, loss


def train(
    model: nn.Module,
    train_loader,
    vali_loader,
    device: torch.device,
    epochs: int,
    patience: int,
    loss_name: str,
    loss_params: dict,
    optimizer: torch.optim.Optimizer,
    lradj: str,
    base_lr: float,
    total_epochs: int,
    label_len: int,
    pred_len: int,
    features: str,
    use_amp: bool,
    checkpoint_dir: str,
    checkpoint_cfg,
    callbacks: list[Callback] | None = None,
    training_objective=None,
    training_setup=None,
    resume: bool = False,
    tracker=None,
    checkpoint_every_batches: int = 0,
    min_lr: float = 0.0,
) -> TrainResult:
    """Train a model with early stopping and checkpointing.

    Pluggable callbacks
    -------------------
    An optional ``callbacks`` list (default empty) lets training tricks hook
    into the loop without changing default behavior. When ``callbacks`` is empty
    every hook is skipped and the loop is byte-identical to the pre-callback
    trainer. See ``tsflab.experiments.runner.callbacks`` for the hook contract.

    Auxiliary-loss convention
    -------------------------
    After each forward pass the trainer reads an optional auxiliary loss off the
    model via ``getattr(model, 'aux_loss', None)``. When a finite scalar tensor
    is found it is added to the main loss; this is a strict no-op
    for models that never set those attributes. See ``_collect_aux_loss``.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    train_loader : DataLoader
        Training data loader.
    vali_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    epochs : int
        Number of epochs to train.
    patience : int
        Early stopping patience.
    loss_name : str
        Loss name.
    loss_params : dict
        Keyword arguments for loss construction.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    lradj : str
        Learning rate schedule type.
    base_lr : float
        Base learning rate for scheduling.
    total_epochs : int
        Total epochs used for scheduling.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").
    use_amp : bool
        Whether to enable mixed precision.
    checkpoint_dir : str
        Directory to save checkpoints.
    checkpoint_cfg : TrainCheckpointConfig
        Checkpointing settings.

    Returns
    -------
    TrainResult
        Best checkpoint path and training time.
    """
    if checkpoint_every_batches and getattr(train_loader, "num_workers", 0):
        raise ValueError("batch recovery currently requires num_workers=0 for deterministic replay")
    model.train()
    _assert_custom_objective_supported(model, training_objective)
    criterion = get_loss(loss_name, **loss_params)
    early_stopping = EarlyStopping(patience=patience)
    checkpoint_manager = CheckpointManager(
        strategy=checkpoint_cfg.strategy,
        save_k=checkpoint_cfg.save_k,
        path=checkpoint_dir,
    )
    use_amp = use_amp and device.type != "cpu"
    scaler = torch.amp.GradScaler() if use_amp else None

    callbacks = callbacks or []
    # Fast path flag: when there are no callbacks, the loop must remain
    # byte-identical to the pre-callback trainer (zero_grad + step every batch,
    # no hook invocations, no extra tensor reslicing).
    has_callbacks = bool(callbacks)
    ctx = CallbackContext(
        model=model,
        optimizer=optimizer,
        total_epochs=epochs,
        pred_len=pred_len,
    )
    for cb in callbacks:
        cb.on_train_start(ctx)

    from tsflab.experiments.infra.checkpoint import save_checkpoint

    latest = os.path.join(checkpoint_dir, "latest.pth")
    start_epoch, prior_elapsed, completed, progress = _start_or_resume(
        model=model,
        train_loader=train_loader,
        vali_loader=vali_loader,
        optimizer=optimizer,
        scaler=scaler,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        callbacks=callbacks,
        training_setup=training_setup,
        resume=resume,
        tracker=tracker,
        latest=latest,
        pred_len=pred_len,
        features=features,
    )
    start_time = time.perf_counter()
    epochs_run = start_epoch
    for epoch in range(start_epoch, epochs if not completed else start_epoch):
        epochs_run = epoch + 1
        epoch_losses, skip, saved_rng, epoch_rng, loader_rng = _start_epoch(progress, train_loader)

        def batch_checkpoint(next_batch):
            if checkpoint_every_batches and next_batch % checkpoint_every_batches == 0:
                save_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    early_stopping=early_stopping,
                    manager=checkpoint_manager,
                    callbacks=callbacks,
                    epoch=epoch,
                    elapsed=prior_elapsed + time.perf_counter() - start_time,
                    loaders=(train_loader, vali_loader),
                    progress={
                        "next_batch": next_batch,
                        "epoch_rng": epoch_rng,
                        "loader_rng": loader_rng,
                        "losses": epoch_losses,
                    },
                )

        ctx.epoch = epoch
        _train_batches(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            label_len=label_len,
            pred_len=pred_len,
            features=features,
            callbacks=callbacks,
            has_callbacks=has_callbacks,
            ctx=ctx,
            training_objective=training_objective,
            skip=skip,
            saved_rng=saved_rng,
            epoch_losses=epoch_losses,
            batch_checkpoint=batch_checkpoint,
        )

        progress = None

        completed = _end_epoch(
            model=model,
            train_loader=train_loader,
            vali_loader=vali_loader,
            device=device,
            criterion=criterion,
            label_len=label_len,
            pred_len=pred_len,
            features=features,
            optimizer=optimizer,
            scaler=scaler,
            base_lr=base_lr,
            lradj=lradj,
            total_epochs=total_epochs,
            min_lr=min_lr,
            early_stopping=early_stopping,
            checkpoint_manager=checkpoint_manager,
            checkpoint_dir=checkpoint_dir,
            latest=latest,
            callbacks=callbacks,
            ctx=ctx,
            tracker=tracker,
            epoch=epoch,
            epochs=epochs,
            epoch_losses=epoch_losses,
            prior_elapsed=prior_elapsed,
            start_time=start_time,
        )
        epochs_run = epoch + 1
        if early_stopping.early_stop:
            break

    train_time = prior_elapsed + time.perf_counter() - start_time
    best_model_path = f"{checkpoint_dir}/best_checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    return TrainResult(best_model_path=best_model_path, train_time_sec=train_time, epochs_run=epochs_run)


def _start_tracker(tracker, resume, latest) -> None:
    """Start the tracker at the epoch training resumes from."""
    if not tracker:
        return
    # Start after constructors but preserve the restored RNG below.
    if resume and os.path.isfile(latest):
        metadata = torch.load(latest, map_location="cpu", weights_only=False)
        tracker.start(metadata["epoch"] + 1)
    else:
        tracker.start(1)


def _start_or_resume(
    *,
    model,
    train_loader,
    vali_loader,
    optimizer,
    scaler,
    early_stopping,
    checkpoint_manager,
    callbacks,
    training_setup,
    resume,
    tracker,
    latest,
    pred_len,
    features,
):
    """Run the fresh-start setup or restore ``latest``; return (epoch, elapsed, completed, progress)."""
    from tsflab.experiments.infra.checkpoint import restore_checkpoint

    _start_tracker(tracker, resume, latest)
    if training_setup is not None and not (resume and os.path.isfile(latest)):
        training_setup(unwrap_model(model), train_loader, pred_len=pred_len, features=features)
    if not (resume and os.path.isfile(latest)):
        return 0, 0.0, False, None
    state = restore_checkpoint(
        latest,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        early_stopping=early_stopping,
        manager=checkpoint_manager,
        callbacks=callbacks,
        loaders=(train_loader, vali_loader),
    )
    return state["epoch"], state["elapsed"], state["completed"], state.get("progress")


def _start_epoch(progress, train_loader):
    """Restore mid-epoch progress, if any; return (losses, skip, saved_rng, epoch_rng, loader_rng)."""
    from tsflab.experiments.infra.checkpoint import capture_rng, set_rng

    epoch_losses = list(progress["losses"]) if progress else []
    skip = progress["next_batch"] if progress else 0
    saved_rng = capture_rng()
    if progress:
        set_rng(progress["epoch_rng"])
        if progress["loader_rng"] is not None:
            train_loader.generator.set_state(progress["loader_rng"])
    epoch_rng = capture_rng()
    loader_rng = train_loader.generator.get_state() if getattr(train_loader, "generator", None) is not None else None
    return epoch_losses, skip, saved_rng, epoch_rng, loader_rng


def _batch_to_device(batch_x, batch_y, batch_x_mark, batch_y_mark, device):
    """Move one batch to ``device`` as float tensors; absent marks stay ``None``."""
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    if batch_x_mark is not None:
        batch_x_mark = batch_x_mark.float().to(device)
    if batch_y_mark is not None:
        batch_y_mark = batch_y_mark.float().to(device)
    return batch_x, batch_y, batch_x_mark, batch_y_mark


def _train_step(
    *,
    model,
    optimizer,
    criterion,
    scaler,
    use_amp,
    device,
    batch_x,
    batch_y,
    batch_x_mark,
    batch_y_mark,
    dec_inp,
    pred_len,
    features,
    training_objective,
):
    """Run one training micro-batch without callbacks and return its loss."""
    optimizer.zero_grad()
    if use_amp:
        with torch.amp.autocast(device_type=device.type):
            outputs, loss = _forward_training(
                model,
                training_objective,
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                batch_y,
                pred_len,
                features,
                criterion,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs, loss = _forward_training(
            model,
            training_objective,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            batch_y,
            pred_len,
            features,
            criterion,
        )
        loss.backward()
        optimizer.step()
    return loss


def _train_batches(
    *,
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    use_amp,
    device,
    label_len,
    pred_len,
    features,
    callbacks,
    has_callbacks,
    ctx,
    training_objective,
    skip,
    saved_rng,
    epoch_losses,
    batch_checkpoint,
) -> None:
    """Train one epoch's batches, skipping the ``skip`` already replayed ones."""
    from tsflab.experiments.infra.checkpoint import set_rng

    n_batches = len(train_loader) if hasattr(train_loader, "__len__") else -1
    for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        if batch_idx < skip:
            if batch_idx + 1 == skip:
                set_rng(saved_rng)
            continue
        batch_x, batch_y, batch_x_mark, batch_y_mark = _batch_to_device(
            batch_x, batch_y, batch_x_mark, batch_y_mark, device
        )

        dec_inp = make_decoder_input(batch_y, label_len, pred_len, device)

        ctx.batch_idx = batch_idx
        ctx.is_last_batch = n_batches >= 0 and batch_idx == n_batches - 1

        step = dict(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            use_amp=use_amp,
            device=device,
            batch_x=batch_x,
            batch_y=batch_y,
            batch_x_mark=batch_x_mark,
            batch_y_mark=batch_y_mark,
            dec_inp=dec_inp,
            pred_len=pred_len,
            features=features,
            training_objective=training_objective,
        )
        if not has_callbacks:
            loss = _train_step(**step)
            epoch_losses.append(loss.item())
            batch_checkpoint(batch_idx + 1)
            continue

        # --- Callback path ---
        do_step = _train_step_with_callbacks(callbacks=callbacks, ctx=ctx, **step)
        epoch_losses.append(ctx.extra["last_loss_value"])
        if do_step:
            batch_checkpoint(batch_idx + 1)


def _end_epoch(
    *,
    model,
    train_loader,
    vali_loader,
    device,
    criterion,
    label_len,
    pred_len,
    features,
    optimizer,
    scaler,
    base_lr,
    lradj,
    total_epochs,
    min_lr,
    early_stopping,
    checkpoint_manager,
    checkpoint_dir,
    latest,
    callbacks,
    ctx,
    tracker,
    epoch,
    epochs,
    epoch_losses,
    prior_elapsed,
    start_time,
) -> bool:
    """Validate, checkpoint, schedule, and log one finished epoch; return whether training is complete."""
    from tsflab.experiments.infra.checkpoint import save_checkpoint, save_weights

    validation_metrics = {} if tracker else None
    vali_loss = validate(
        model,
        vali_loader,
        device,
        criterion,
        label_len,
        pred_len,
        features,
        metrics=validation_metrics,
    )
    train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    current_lr = optimizer.param_groups[0].get("lr", base_lr)
    print(
        f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {vali_loss:.6f} | lr: {current_lr:.6g}"
    )
    is_best = early_stopping.step(vali_loss)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
        save_weights(best_path, model)
    checkpoint_manager.save(model, epoch + 1, vali_loss, is_best)
    for cb in callbacks:
        cb.on_epoch_end(ctx)
    completed = early_stopping.early_stop or epoch + 1 >= epochs
    if not early_stopping.early_stop:
        adjust_learning_rate(optimizer, epoch + 1, lradj, base_lr, total_epochs, min_lr=min_lr)
    save_checkpoint(
        latest,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        early_stopping=early_stopping,
        manager=checkpoint_manager,
        callbacks=callbacks,
        epoch=epoch + 1,
        elapsed=prior_elapsed + time.perf_counter() - start_time,
        completed=completed,
        loaders=(train_loader, vali_loader),
    )
    if tracker:
        tracker.log(
            {
                "train/loss": train_loss,
                "validation/loss": vali_loss,
                **{f"validation/{key}": value for key, value in validation_metrics.items()},
                "train/lr": current_lr,
                "train/elapsed_sec": prior_elapsed + time.perf_counter() - start_time,
            },
            epoch + 1,
        )
    return completed


def _run_compute_loss_hooks(callbacks, ctx, criterion):
    """Invoke on_compute_loss hooks, recomputing the loss if a hook resliced.

    A hook may either return a replacement loss (short-circuit) or mutate
    ``ctx.outputs`` / ``ctx.targets`` in place and return None. In the latter
    case the loss is recomputed from the (possibly resliced) tensors after all
    hooks run. ``ctx.loss`` is updated in place.
    """
    resliced = False
    initial_outputs = ctx.outputs
    initial_targets = ctx.targets
    for cb in callbacks:
        returned = cb.on_compute_loss(ctx)
        if returned is not None:
            ctx.loss = returned
        if ctx.outputs is not initial_outputs or ctx.targets is not initial_targets:
            resliced = True
            initial_outputs = ctx.outputs
            initial_targets = ctx.targets
    if resliced:
        ctx.loss = criterion(ctx.outputs, ctx.targets)


def _train_step_with_callbacks(
    *,
    model,
    optimizer,
    criterion,
    scaler,
    use_amp,
    device,
    batch_x,
    batch_y,
    batch_x_mark,
    batch_y_mark,
    dec_inp,
    pred_len,
    features,
    callbacks,
    ctx: CallbackContext,
    training_objective=None,
) -> bool:
    """Run a single training micro-batch with callback hooks.

    Returns whether an optimizer step was performed. Stores the (unscaled)
    base-loss value for logging in ``ctx.extra['last_loss_value']``.

    The optimizer step / zero_grad coordination follows the gradient-accumulation
    contract: when a callback's ``on_optimizer_step`` returns False the step and
    the subsequent ``zero_grad`` are both skipped so gradients accumulate.
    """
    if use_amp:
        with torch.amp.autocast(device_type=device.type):
            outputs, base_loss = _forward_training(
                model,
                training_objective,
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                batch_y,
                pred_len,
                features,
                criterion,
            )
            outputs = _callback_outputs(outputs, model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = slice_prediction_target(outputs, batch_y, pred_len, features)
            ctx.outputs = outputs
            ctx.targets = batch_y_sliced
            ctx.loss = base_loss
            ctx.extra["last_loss_value"] = base_loss.item()
            _run_compute_loss_hooks(callbacks, ctx, criterion)
        scaler.scale(ctx.loss).backward()
        scaler.unscale_(optimizer)
        for cb in callbacks:
            cb.on_backward(ctx)
        do_step = _resolve_step(callbacks, ctx)
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # Keep the scaler's internal state consistent without stepping.
            scaler.update()
    else:
        outputs, base_loss = _forward_training(
            model,
            training_objective,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            batch_y,
            pred_len,
            features,
            criterion,
        )
        outputs = _callback_outputs(outputs, model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs, batch_y_sliced = slice_prediction_target(outputs, batch_y, pred_len, features)
        ctx.outputs = outputs
        ctx.targets = batch_y_sliced
        ctx.loss = base_loss
        ctx.extra["last_loss_value"] = base_loss.item()
        _run_compute_loss_hooks(callbacks, ctx, criterion)
        ctx.loss.backward()
        for cb in callbacks:
            cb.on_backward(ctx)
        do_step = _resolve_step(callbacks, ctx)
        if do_step:
            optimizer.step()
            optimizer.zero_grad()
    return do_step


def _callback_outputs(outputs, model, batch_x, batch_x_mark, dec_inp, batch_y_mark):
    """Give callbacks a forecast when the objective returned none (no gradient)."""
    if outputs is not None:
        return outputs
    with torch.no_grad():
        return call_forecaster(unwrap_model(model), batch_x, batch_x_mark, dec_inp, batch_y_mark)


def _resolve_step(callbacks, ctx: CallbackContext) -> bool:
    """Resolve whether to step the optimizer this micro-batch.

    A step is performed unless any callback's ``on_optimizer_step`` returns
    False (gradient accumulation). Hooks returning None/True do not veto.
    """
    do_step = True
    for cb in callbacks:
        decision = cb.on_optimizer_step(ctx)
        if decision is False:
            do_step = False
    return do_step


def _point_forecast(model: nn.Module, criterion: nn.Module, outputs: torch.Tensor) -> torch.Tensor:
    """Point forecast for validation metrics: the median quantile or the distribution mean, as in evaluation."""
    if outputs.ndim != 4:
        return outputs
    output_type, _ = _resolve_output_kind(model)
    if output_type != "quantile":
        return outputs[..., 0]
    levels = getattr(criterion, "q", None)
    if levels is None:
        levels = getattr(unwrap_model(model), "quantile_levels", None)
    levels = torch.as_tensor(_CANONICAL_LEVELS if levels is None else levels, dtype=torch.float64).cpu()
    if levels.numel() != outputs.shape[-1]:
        return outputs[..., outputs.shape[-1] // 2]
    return outputs[..., int(torch.argmin((levels - 0.5).abs()))]


def validate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    criterion: nn.Module,
    label_len: int,
    pred_len: int,
    features: str,
    *,
    metrics: dict[str, float] | None = None,
) -> float:
    """Evaluate selection loss and optional forecast metrics in one forward pass.

    When supplied, ``metrics`` receives global-element MSE and MAE. These are
    diagnostic records; the configured criterion still controls early stopping.


    Parameters
    ----------
    model : nn.Module
        Forecasting model.
    data_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    criterion : nn.Module
        Loss function.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    features : str
        Feature mode ("M", "S", "MS").

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    weighted_loss = 0.0
    samples = 0
    squared_error = absolute_error = 0.0
    elements = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            if batch_y_mark is not None:
                batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = make_decoder_input(batch_y, label_len, pred_len, device)
            outputs = call_forecaster(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs, batch_y_sliced = slice_prediction_target(outputs, batch_y, pred_len, features)
            loss = criterion(outputs, batch_y_sliced)
            count = batch_x.shape[0]
            weighted_loss += loss.item() * count
            samples += count
            if metrics is not None:
                error = _point_forecast(model, criterion, outputs) - batch_y_sliced
                squared_error += error.square().sum().item()
                absolute_error += error.abs().sum().item()
                elements += error.numel()
    if metrics is not None:
        metrics.update(
            mse=squared_error / elements if elements else float("nan"),
            mae=absolute_error / elements if elements else float("nan"),
        )
    model.train()
    return weighted_loss / samples if samples else float("nan")
