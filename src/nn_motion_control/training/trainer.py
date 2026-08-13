import logging
import os
import time

import torch
from torch import autocast

from nn_motion_control.core.checkpoints import write_json_sidecar
from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.loaders import BatchLoader

logger = logging.getLogger(os.path.basename(__file__))

# Divisor for reporting peak CUDA memory in gigabytes rather than bytes.
BYTES_PER_GB = 1e9


def config_overrides(source: dict, casts: dict) -> dict:
    """
    Trainer keyword overrides the config supplies, each cast to its type.

    An absent key is omitted so the trainer signature's own default
    applies, keeping default values in one place rather than duplicated
    as fallbacks at the call site. Each key names both the config field
    and the trainer keyword.
    """

    return {
        key: cast(source[key]) for key, cast in casts.items() if key in source
    }


class Trainer:
    """
    Train and validate a model with mixed precision, gradient accumulation
    and early stopping; the best checkpoint (weights + norm stats +
    provenance) is saved to disk.
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
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # Loss scaling is only needed for float16 (bf16 has fp32 range);
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
        # Per-epoch profiling: train/val wall time and peak CUDA memory, to
        # see how cost scales (e.g. with the rollout horizon curriculum)
        # across a run.
        self.epoch_train_times: list[float] = []
        self.epoch_val_times: list[float] = []
        self.epoch_peak_mem_gb: list[float] = []
        self.best_epoch = 0
        self.test_results = {}

    def _forward_loss(self, batch):
        """
        Compute the loss for one batch (one-step prediction).

        Subclasses override this to change the per-batch objective; the
        surrounding AMP, gradient accumulation, clipping and early
        stopping are shared.
        """

        data, labels = batch[0], batch[1]
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        outputs = self.model(data)
        return self.criterion(outputs, labels)

    def _on_epoch_start(self, epoch: int) -> None:
        """
        Per-epoch setup hook: a no-op for one-step, used by rollout for
        its schedules.
        """

    def _is_improvement(self, val_loss: float, best_val_loss: float) -> bool:
        """
        Whether val_loss beats the best by the min_delta margin.

        min_delta is a relative fraction of the current best, so the
        check is robust across loss magnitudes. An absolute margin
        mis-scaled to the loss (e.g. 1e-6 against a ~1e-5 loss) can
        otherwise never trigger, freezing the best on epoch 1.
        """

        return val_loss < best_val_loss * (1.0 - self.min_delta)

    def _train_epoch(self):
        """
        Run one training epoch with gradient accumulation and mixed precision.

        Accumulation simulates a larger batch than fits in memory: the loss
        is scaled down per batch and the optimiser step is deferred until
        accumulation_steps batches have contributed their gradients.
        """

        self.model.train()
        total_loss = 0
        # Batches that contributed to the mean; NaN/Inf batches are skipped.
        counted_batches = 0
        # Gradients not yet stepped in the current accumulation cycle.
        has_pending = False

        for batch_idx, batch in enumerate(self.train_loader):
            if (batch_idx % self.accumulation_steps) == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context for later quantisation support. The
            # per-batch forward and loss live in _forward_loss so
            # subclasses can override them.
            with autocast(device_type=self.device, dtype=self.training_dtype):
                loss = self._forward_loss(batch) / self.accumulation_steps

            # Guard against NaN/Inf loss, which can destabilize training.
            if torch.isnan(loss) or torch.isinf(loss):
                if torch.isnan(loss):
                    logger.warning(f"NaN at batch {batch_idx}, skipping")
                else:
                    logger.warning(f"Inf at batch {batch_idx}, skipping")

                if (batch_idx % self.accumulation_steps) == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            has_pending = True

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self._optimizer_step()
                has_pending = False

            total_loss += loss.item() * self.accumulation_steps
            counted_batches += 1

        # Flush gradients from an incomplete final accumulation cycle so
        # the last 1..accumulation_steps-1 batches are not silently
        # discarded.
        if has_pending:
            self._optimizer_step()

        return total_loss / max(counted_batches, 1)

    def _optimizer_step(self):
        """
        Unscale, clip and apply one optimiser step, then update the AMP scaler.
        """

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        # Batches counted, mirroring _train_epoch's averaging.
        counted_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Same dtype as training so val loss is comparable for
                # early stopping.
                with autocast(
                    device_type=self.device, dtype=self.training_dtype
                ):
                    loss = self._forward_loss(batch)

                if torch.isnan(loss):
                    logger.warning("NaN in validation, skipping")
                elif torch.isinf(loss):
                    logger.warning("Inf in validation, skipping")
                else:
                    total_loss += loss.item()
                    counted_batches += 1

        return total_loss / max(counted_batches, 1)

    def get_training_info(self):
        """
        Return the train/val loss history plus the best epoch number.
        """

        return self.train_losses, self.val_losses, self.best_epoch

    def _save_checkpoint(self):
        """
        Save the model plus everything needed to run/denormalize it later.

        Normalization is fit at train time (not stored in the
        dataset), so the fitted stats must travel with the weights or
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

        # Human-readable sidecar with the denorm params (no tensors)
        # for inspection.
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
                    torch.cuda.max_memory_allocated() / BYTES_PER_GB
                    if self.device == "cuda"
                    else 0.0
                )

                # Store losses plus per-epoch profiling for plotting and
                # cost analysis.
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
                            f"Early stopping triggered after "
                            f"{epoch + 1} epochs."
                        )
                        logger.info(
                            f"The best model was at epoch {self.best_epoch}."
                        )
                        break

                    break
                except RuntimeError:
                    break
