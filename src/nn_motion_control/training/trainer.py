import json
import logging
import os
import time
from pathlib import Path

import torch
from torch import autocast

from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.loaders import BatchLoader

logger = logging.getLogger(os.path.basename(__file__))


class Trainer:
    """
    Train and validate a model with mixed precision, gradient accumulation and early
    stopping; the best checkpoint (weights + norm stats + provenance) is saved to disk.
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
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # Loss scaling is only needed for float16 (bf16 has fp32 range); disabling it
        # for other dtypes keeps the step a no-op instead of rescaling needlessly.
        self.scaler = scaler_class(
            device=self.device, enabled=(training_dtype == torch.float16)
        )
        self.num_epochs = max_epochs
        self.learning_rate = learning_rate
        self.min_delta = min_delta
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.node_info = node_info
        self.patience = patience
        self.model_name = model_name
        self.save_path = save_path
        self.logging = logging
        self.accumulation_steps = accumulation_steps
        self.training_dtype = training_dtype
        self.window_size = window_size
        self.seed = seed

        self.means = node_info.target_denorm_params["mean"]
        self.stds = node_info.target_denorm_params["std"]
        self.weights = node_info.loss_weights

        # Construct the loss with the per-target weight vector when supported, else
        # fall back to a plain built-in loss. Move it to the device so any registered
        # weight buffer lives alongside the model.
        try:
            self.criterion = criterion_class(weights=self.weights)
        except TypeError:
            self.criterion = criterion_class()
        self.criterion = self.criterion.to(self.device)

        self.train_losses = []
        self.val_losses = []
        # Per-epoch profiling: train/val wall time and peak CUDA memory, so we can see
        # how cost scales (e.g. with the rollout horizon curriculum) across a run.
        self.epoch_train_times: list[float] = []
        self.epoch_val_times: list[float] = []
        self.epoch_peak_mem_gb: list[float] = []
        self.stopped_early = 0
        self.test_results = {}

    def _forward_loss(self, batch):
        """
        Compute the loss for one batch (one-step prediction).

        Subclasses override this to change the per-batch objective; the surrounding
        AMP, gradient accumulation, clipping and early stopping are shared.
        """

        data, labels = batch[0], batch[1]
        data = data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        outputs = self.model(data)
        return self.criterion(outputs, labels)

    def _on_epoch_start(self, epoch: int) -> None:
        """
        Per-epoch setup hook. No-op for one-step; rollout uses it for its schedules.
        """

    def _is_improvement(self, val_loss: float, best_val_loss: float) -> bool:
        """
        Whether val_loss beats the best by the min_delta margin.

        min_delta is a relative fraction of the current best, so the check is robust
        across loss magnitudes. An absolute margin mis-scaled to the loss (e.g. 1e-6
        against a ~1e-5 loss) can otherwise never trigger, freezing the best on epoch 1.
        """

        return val_loss < best_val_loss * (1.0 - self.min_delta)

    def _train_epoch(self):
        """
        Runs one epoch of training with gradient accumulation and mixed precision.

        Use gradient accumulation to simulate larger batch sizes if hardware limited.
        Scales loss for accumulation and unscales before optimizer step.

        Returns average training loss for the epoch.
        """

        self.model.train()
        total_loss = 0
        counted_batches = 0  # non-skipped batches actually contributing to the mean

        # Iterate over all training batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Zero gradients at the start of each accumulation cycle
            if (batch_idx % self.accumulation_steps) == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context for later quantisation support. The per-batch
            # forward and loss live in _forward_loss so subclasses can override them.
            with autocast(device_type=self.device, dtype=self.training_dtype):
                loss = self._forward_loss(batch) / self.accumulation_steps

            # Guard against NaN/Inf loss which can destabilize training
            if torch.isnan(loss) or torch.isinf(loss):
                if torch.isnan(loss):
                    logger.warning(f"NaN at batch {batch_idx}, skipping")
                else:
                    logger.warning(f"Inf at batch {batch_idx}, skipping")

                if (batch_idx % self.accumulation_steps) == 0:  # Only if we zeroed
                    self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()  # Compute gradients

            # Accumulate gradients and step optimizer on completion of each mini-batch
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Step optimizer and update scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item() * self.accumulation_steps
            counted_batches += 1

        # Average over batches that actually contributed (skipped NaN/Inf excluded)
        return total_loss / max(counted_batches, 1)

    def _validate_epoch(self):
        """
        Performs validation after each training epoch to monitor for overfitting.
        """
        self.model.eval()
        total_loss = 0

        # No gradient computation in validation
        with torch.no_grad():
            for batch in self.val_loader:
                # Same dtype as training so val loss is comparable for early stopping.
                with autocast(device_type=self.device, dtype=self.training_dtype):
                    loss = self._forward_loss(batch)

                if torch.isnan(loss):
                    logger.warning("NaN in validation, skipping")
                elif torch.isinf(loss):
                    logger.warning("Inf in validation, skipping")
                else:
                    total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def get_training_info(self):
        """
        Returns all the training and corresponding validation
        losses, as well as the epoch with the last best model.
        """

        return self.train_losses, self.val_losses, self.stopped_early

    def _save_checkpoint(self):
        """
        Save the model plus everything needed to run/denormalize it later.

        Normalization is fit at train time (not stored in the dataset), so the fitted
        stats must travel with the weights or inference cannot recover physical units.
        """

        ni = self.node_info
        # Unwrap a torch.compile wrapper so saved keys have no `_orig_mod.` prefix and
        # the checkpoint loads into a plain module.
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
        torch.save(checkpoint, self.save_path)

        # Human-readable sidecar with the denorm params (no tensors) for inspection.
        sidecar = Path(self.save_path).with_suffix(".norm.json")
        sidecar.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "window_size": self.window_size,
                    "seed": self.seed,
                    "input_labels": checkpoint["input_labels"],
                    "target_labels": checkpoint["target_labels"],
                    "input_denorm_params": ni.input_denorm_params,
                    "target_denorm_params": ni.target_denorm_params,
                },
                indent=2,
            )
        )

    def train(self):
        """
        Executes the training loop for the model,
        including validation checks.

        Implements early stopping.
        """

        best_val_loss = float("inf")
        epochs_no_improve = 0
        logger.debug(
            f"Training using {self.criterion} loss and {self.optimizer} optimizer"
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
                    torch.cuda.max_memory_allocated() / 1e9
                    if self.device == "cuda"
                    else 0.0
                )

                # Store losses + per-epoch profiling for plotting / cost analysis.
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.epoch_train_times.append(t_train)
                self.epoch_val_times.append(t_val)
                self.epoch_peak_mem_gb.append(peak_gb)

                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} "
                    f"({t_train:.1f}s train, {t_val:.1f}s val, {peak_gb:.1f} GB peak), "
                    f"Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}"
                )

                # Early stopping check
                if self._is_improvement(val_loss, best_val_loss):
                    best_val_loss = val_loss
                    epochs_no_improve = 0

                    # Save best model state
                    self._save_checkpoint()
                    self.stopped_early = epoch + 1
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    logger.info(f"The best model was at epoch {self.stopped_early}.")
                    break

            except KeyboardInterrupt:
                try:
                    logger.info(f"User ended learning process after {epoch} epochs.")
                    logger.info(f"Finishing epoch {epoch + 1}...")

                    train_loss = self._train_epoch()
                    val_loss = self._validate_epoch()

                    # Store losses for future plotting
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)

                    # Early stopping check
                    if self._is_improvement(val_loss, best_val_loss):
                        best_val_loss = val_loss
                        epochs_no_improve = 0

                        # Save best model state
                        self._save_checkpoint()
                        self.stopped_early = epoch + 1
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs."
                        )
                        logger.info(
                            f"The best model was at epoch {self.stopped_early}."
                        )
                        break

                    break
                except RuntimeError:
                    break
