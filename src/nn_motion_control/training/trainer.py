import logging
import os
import time

import torch
from torch import autocast

from nn_motion_control.core.checkpoints import write_json_sidecar
from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.loaders import BatchLoader

logger = logging.getLogger(os.path.basename(__file__))

PERCENT = 100.0
REDUCED_DTYPES = (torch.bfloat16, torch.float16)


def config_overrides(source: dict, casts: dict) -> dict:
    """
    Trainer keyword overrides the config supplies, each cast to its type.
    """

    return {
        key: cast(source[key]) for key, cast in casts.items() if key in source
    }


class Trainer:
    """
    Train and validate a model with mixed precision,
    gradient accumulation and early stopping.
    Best checkpoint is saved to disk.
    """

    def __init__(
        self,
        model,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        device: str,
        scaler_class,
        optimizer_class,
        criterion_class,
        node_info: DatasetMetadata,
        max_epochs: int,
        learning_rate: float,
        min_delta,
        patience,
        model_name,
        save_path: str,
        logging: bool,
        accumulation_steps,
        training_dtype: torch.dtype,
        window_size: int = 1,
        seed: int = 42,
        grad_clip_norm: float = 1.0,
        validate_first: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss scaling is only needed for float16 (bf16 has fp32 range),
        # disabling it for other dtypes keeps the step a no-op instead of
        # rescaling needlessly.
        self.scaler = scaler_class(
            device=self.device, enabled=(training_dtype == torch.float16)
        )
        self.num_epochs = max_epochs
        self.learning_rate = learning_rate
        self.min_delta = min_delta
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=self.learning_rate
        )
        self.node_info = node_info
        self.patience = patience
        self.model_name = model_name
        self.save_path = save_path
        self.logging = logging
        self.accumulation_steps = accumulation_steps
        self.training_dtype = training_dtype
        self.window_size = window_size
        self.seed = seed
        self.grad_clip_norm = grad_clip_norm
        self.validate_first = validate_first

        self.means = node_info.target_denorm_params["mean"]
        self.stds = node_info.target_denorm_params["std"]
        self.weights = node_info.loss_weights

        # Construct the loss with the per-target weight vector when
        # supported, else fall back to a plain built-in loss. Move it to
        # the device so any registered weight buffer lives alongside the
        # model.
        try:
            self.criterion = criterion_class(weights=self.weights)
        except TypeError:
            self.criterion = criterion_class()
        self.criterion = self.criterion.to(self.device)

        self.train_losses = []
        self.val_losses = []
        # Per-epoch profiling: train/val wall time and peak CUDA memory,
        # to see how cost scales (e.g. with the rollout horizon curriculum)
        # across a run.
        self.epoch_train_times: list[float] = []
        self.epoch_val_times: list[float] = []
        self.epoch_peak_mem_gb: list[float] = []
        self.best_epoch = 0
        self.test_results = {}

    def _forward_loss(self, batch):
        """
        Compute the loss for one batch (w/ one-step prediction).
        Overwritable.
        """

        data, labels = batch[0], batch[1]
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        outputs = self.model(data)
        return self.criterion(outputs, labels)

    def _on_epoch_start(self, epoch: int) -> None:
        """
        Per-epoch setup hook: a no-op for one-step,
        used by rollout for its schedules.
        """

    def _is_improvement(self, val_loss: float, best_val_loss: float) -> bool:
        """
        Whether val_loss beats the best by a relative fraction of
        the current best, so the check is robust across loss magnitudes.

        An absolute margin mis-scaled to the loss (e.g. 1e-6 on a ~1e-5 loss)
        can otherwise never trigger, freezing the best on epoch 1.
        """

        return val_loss < best_val_loss * (1.0 - self.min_delta)

    def _train_epoch(self):
        """
        Run one training epoch with gradient accumulation and mixed precision.
        """

        self.model.train()
        # Loss sums and batch counts stay on the device and are read once
        # per epoch, so a batch costs no host sync; the only one left is
        # the finiteness check at each optimiser step.
        total_loss = torch.zeros((), device=self.device)
        counted_batches = torch.zeros((), device=self.device)
        n_batches = 0
        self._skipped_steps = 0
        self._steps = 0
        # Gradient norms before clipping over the stepped batches, so a
        # growing or exploding gradient shows in the log.
        self._norm_sum = torch.zeros((), device=self.device)
        self._norm_max = torch.zeros((), device=self.device)
        # Gradients not yet stepped in the current accumulation cycle.
        has_pending = False
        accumulating = self.accumulation_steps > 1

        for batch_idx, batch in enumerate(self.train_loader):
            n_batches += 1
            first = (batch_idx % self.accumulation_steps) == 0
            if first:
                self.optimizer.zero_grad(set_to_none=True)
                # Finite batches stepped into this cycle, on the device.
                self._cycle_finite = torch.zeros((), device=self.device)

            # Mixed precision context for later quantisation support.
            # The per-batch forward and loss live in _forward_loss so
            # subclasses can override them.
            with self._autocast():
                loss = self._forward_loss(batch) / self.accumulation_steps

            if self.training_dtype == torch.float16:
                # float16 loss scaling already syncs to skip overflowing
                # steps, so the batch is skipped on the host as before.
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at batch {batch_idx}")
                    if first:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue
                finite = None
            else:
                finite = torch.isfinite(loss)
                loss = torch.where(finite, loss, torch.zeros_like(loss))
                self._cycle_finite = self._cycle_finite + finite.float()

            # When accumulating, a non-finite batch must leave the cycle's
            # other gradients as they were, so its contribution is masked
            # out against a copy taken before it (zero for the first).
            keep = (
                self._grads() if finite is not None and accumulating else None
            )
            self.scaler.scale(loss).backward()
            if keep is not None:
                assert finite is not None

                for p, g in zip(self._params(), keep, strict=True):
                    if p.grad is not None:
                        prev = torch.zeros_like(p.grad) if g is None else g
                        p.grad.copy_(torch.where(finite, p.grad, prev))
            has_pending = True

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self._optimizer_step()
                has_pending = False

            total_loss += loss.detach().float() * self.accumulation_steps
            counted_batches += 1.0 if finite is None else finite.float()

        # Flush gradients from an incomplete final accumulation cycle so
        # the last 1..accumulation_steps-1 batches are not silently
        # discarded.
        if has_pending:
            self._optimizer_step()
        stepped = self._steps - self._skipped_steps
        if stepped:
            logger.info(
                "Gradient norm before clipping: mean %.3g, max %.3g",
                float(self._norm_sum) / stepped,
                float(self._norm_max),
            )
        if self._skipped_steps:
            logger.warning(
                "%d of %d optimiser steps (%.0f%%) skipped: no finite batch "
                "or a non-finite gradient",
                self._skipped_steps,
                self._steps,
                PERCENT * self._skipped_steps / max(self._steps, 1),
            )
        if n_batches == 0:
            logger.warning("The training loader yielded no batches")

        return self._mean_loss(total_loss, counted_batches, n_batches)

    def _autocast(self) -> autocast:
        # Autocast only applies to a reduced precision; float32 runs as is.
        return autocast(
            device_type=self.device,
            dtype=self.training_dtype,
            enabled=self.training_dtype in REDUCED_DTYPES,
        )

    def _params(self) -> list[torch.nn.Parameter]:
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def _grads(self) -> list[torch.Tensor | None]:
        return [
            None if p.grad is None else p.grad.detach().clone()
            for p in self._params()
        ]

    def _optimizer_step(self):
        """
        Unscale, clip and apply one optimiser step, then update the AMP scaler.

        Outside float16 (whose loss scaler skips overflowing steps itself)
        the step is skipped entirely, neither the weights nor the optimiser
        state moving, unless the cycle had a finite batch and the gradient
        norm is finite: one host sync per step. A non-finite loss can still
        backpropagate finite zero gradients, so the norm alone is not enough.
        """

        self._steps = getattr(self, "_steps", 0) + 1
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_clip_norm
        )
        if self.training_dtype != torch.float16:
            cycle = getattr(self, "_cycle_finite", None)
            ok = torch.isfinite(norm)
            if cycle is not None:
                ok = ok & (cycle > 0)
            if not ok:
                self._skipped_steps = getattr(self, "_skipped_steps", 0) + 1
                self.optimizer.zero_grad(set_to_none=True)
                return
        if hasattr(self, "_norm_sum") and torch.isfinite(norm):
            self._norm_sum += norm.detach()
            self._norm_max = torch.maximum(self._norm_max, norm.detach())
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _validate_epoch(self):
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        counted_batches = torch.zeros((), device=self.device)
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Same dtype as training so val loss is comparable for
                # early stopping.
                with self._autocast():
                    loss = self._forward_loss(batch)
                finite = torch.isfinite(loss)
                total_loss += torch.where(finite, loss, 0.0).float()
                counted_batches += finite.float()
                n_batches += 1

        # One divergent validation batch fails the epoch: averaging only the
        # finite ones would rank a partly unstable model as the best.
        if int(counted_batches.item()) < n_batches:
            logger.warning(
                "Validation had non-finite batches, scoring the epoch inf"
            )
            return float("inf")

        return self._mean_loss(total_loss, counted_batches, n_batches)

    @staticmethod
    def _mean_loss(
        total: torch.Tensor, counted: torch.Tensor, n_batches: int
    ) -> float:
        """
        Mean loss over the finite batches, read back with one host sync.
        """

        counted_n = int(counted.item())
        if counted_n < n_batches:
            logger.warning(
                "%d of %d batches had a non-finite loss and were excluded",
                n_batches - counted_n,
                n_batches,
            )

        if counted_n == 0:
            # Nothing finite to average: never let it read as the best loss.
            return float("inf")

        return float(total.item()) / counted_n

    def get_training_info(self):
        """
        Return the train/val loss history plus the best epoch number.
        """

        return self.train_losses, self.val_losses, self.best_epoch

    def _save_checkpoint(self):
        """
        Save the model plus everything needed to run/denormalize it later.

        Normalization is fit at train time (not stored in the dataset),
        so the fitted stats must travel with the weights or
        inference cannot recover physical units.
        """

        ni = self.node_info
        # Unwrap a torch.compile wrapper so saved keys have no
        # '_orig_mod.' prefix and the checkpoint loads into a plain
        # module.
        core_model = getattr(self.model, "_orig_mod", self.model)
        checkpoint = {
            "model_state_dict": core_model.state_dict(),
            "model_name": self.model_name,
            "window_size": self.window_size,
            "seed": self.seed,
            "schema_version": 2,
            "input_labels": list(ni.input_denorm_params["mean"].keys()),
            "target_labels": list(ni.target_denorm_params["mean"].keys()),
            "input_denorm_params": ni.input_denorm_params,
            "target_denorm_params": ni.target_denorm_params,
            "input_stats": {
                "mean": ni.input_stats.mean.cpu(),
                "std": ni.input_stats.std.cpu(),
            },
            "target_stats": {
                "mean": ni.target_stats.mean.cpu(),
                "std": ni.target_stats.std.cpu(),
            },
            "loss_weights": ni.loss_weights.cpu(),
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(checkpoint, self.save_path)

        # Readable sidecar with the denorm params for inspection.
        write_json_sidecar(
            self.save_path,
            {
                "model_name": self.model_name,
                "window_size": self.window_size,
                "seed": self.seed,
                "input_labels": checkpoint["input_labels"],
                "target_labels": checkpoint["target_labels"],
                "input_denorm_params": ni.input_denorm_params,
                "target_denorm_params": ni.target_denorm_params,
            },
        )

    def train(self):
        """
        Run the training loop with validation and early stopping.
        """

        best_val_loss = float("inf")
        epochs_no_improve = 0
        if self.validate_first:
            # A warm-started model sets the bar: an epoch must beat it to
            # replace it as the best checkpoint.
            best_val_loss = self._validate_epoch()
            self._save_checkpoint()
            logger.info("Epoch 0 (warm start) Val Loss: %.3e", best_val_loss)
        logger.debug(
            f"Training using {self.criterion} loss and "
            f"{self.optimizer} optimizer"
        )

        for epoch in range(self.num_epochs):
            try:
                self._on_epoch_start(epoch)
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.perf_counter()
                train_loss = self._train_epoch()
                t_train = time.perf_counter() - t0

                t0 = time.perf_counter()
                val_loss = self._validate_epoch()
                t_val = time.perf_counter() - t0

                peak_gb = (
                    # Reports in bytes, return GB
                    torch.cuda.max_memory_allocated() / 1e9
                    if self.device == "cuda"
                    else 0.0
                )

                # Store losses plus per-epoch profiling
                # for plotting and cost analysis.
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.epoch_train_times.append(t_train)
                self.epoch_val_times.append(t_val)
                self.epoch_peak_mem_gb.append(peak_gb)

                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} "
                    f"({t_train:.1f}s train, {t_val:.1f}s val, "
                    f"{peak_gb:.1f} GB peak), "
                    f"Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}"
                )

                if self._is_improvement(val_loss, best_val_loss):
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self._save_checkpoint()
                    self.best_epoch = epoch + 1
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs."
                    )
                    logger.info(
                        f"The best model was at epoch {self.best_epoch}."
                    )
                    break

            # If we get fed up waiting for training we can Ctrl+C
            # and this will attempt to finish the current epoch
            # so you can see the result.
            except KeyboardInterrupt:
                try:
                    logger.info(
                        f"User ended learning process after {epoch} epochs."
                    )
                    logger.info(f"Finishing epoch {epoch + 1}...")

                    train_loss = self._train_epoch()
                    val_loss = self._validate_epoch()
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)

                    if self._is_improvement(val_loss, best_val_loss):
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        self._save_checkpoint()
                        self.best_epoch = epoch + 1
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} "
                            f"epochs."
                        )
                        logger.info(
                            f"The best model was at epoch {self.best_epoch}."
                        )
                        break

                    break

                # Ctrl+C again to exit completely.
                except RuntimeError:
                    break
