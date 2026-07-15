import json
import logging
import os
from pathlib import Path

import torch
from torch import autocast
from torch.utils.data import DataLoader

from nn_motion_control.data.dataset import DatasetMetadata

logger = logging.getLogger(os.path.basename(__file__))


class Trainer:
    """
    Train and validate a model with mixed precision, gradient accumulation and early
    stopping; the best checkpoint (weights + norm stats + provenance) is saved to disk.
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        self.scaler = scaler_class(device=self.device)
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
        self.stopped_early = 0
        self.test_results = {}

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
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            # Zero gradients at the start of each accumulation cycle
            if (batch_idx % self.accumulation_steps) == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Async CPU->GPU data transfer
            data, labels = (
                data.to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )

            # Mixed precision context for later quantisation support
            with autocast(device_type=self.device, dtype=self.training_dtype):
                outputs = self.model(data)
                # The loss carries the per-target weight vector internally.
                loss = self.criterion(outputs, labels) / self.accumulation_steps

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
            for data, labels in self.val_loader:
                data, labels = (
                    data.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                )
                # Check model against unseen data to monitor overfitting. Use the same
                # dtype as training so val loss is comparable for early stopping.
                with autocast(device_type=self.device, dtype=self.training_dtype):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

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
        """Save the model plus everything needed to run/denormalize it later.

        Normalization is fit at train time (not stored in the dataset), so the fitted
        stats must travel with the weights or inference cannot recover physical units.
        """
        ni = self.node_info
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
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
                train_loss = self._train_epoch()
                val_loss = self._validate_epoch()

                # Store losses for future plotting
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.num_epochs}, "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                    )

                # Early stopping check
                if val_loss < best_val_loss - self.min_delta:
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
                    if val_loss < best_val_loss - self.min_delta:
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
