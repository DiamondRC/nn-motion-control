import logging
import os

import torch
from torch import autocast
from torch.utils.data import DataLoader

from deltabot_nn_controller.dataset_utils.dataloader import DatasetMetadata

logger = logging.getLogger(os.path.basename(os.path.basename(__file__)))


class Trainer:
    """
    Trainer class to handle training, validation, and testing loops for the model.
    Implements early stopping based on validation loss and saves the best model.

    Includes a diagnostic method to profile a single batch for performance bottlenecks.

    Args:
        TODO
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
        self.criterion = criterion_class()
        self.patience = patience
        self.model_name = model_name
        self.save_path = save_path
        self.logging = logging
        self.accumulation_steps = accumulation_steps
        self.training_dtype = training_dtype

        self.means = node_info.target_denorm_params["mean"]
        self.stds = node_info.target_denorm_params["std"]
        self.weights = node_info.loss_weights

        self.train_losses = []
        self.val_losses = []
        self.stopped_early = 0
        self.test_results = {}

    def profile_one_batch(self):
        """
        Verbose diagnostic function to profile the model.

        Passes a single batch through the model with autocast,
        checks for performance bottlenecks and log timing and memory usage.
        """

        logger.debug("Profiling one batch...")
        data, labels = next(iter(self.val_loader))
        data, labels = data.to(self.device), labels.to(self.device)

        logger.debug("Checking CUDA status before profiling")
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        logger.debug("Running forward pass with autocast for profiling...")
        start_event.record()

        with autocast(device_type=self.device, dtype=self.training_dtype):
            outputs = self.model(data)

        end_event.record()
        end_event.synchronize()

        logger.debug(f"Batch shape: {data.shape}")
        logger.debug(f"Forward time: {start_event.elapsed_time(end_event):.2f}ms")
        logger.debug(f"Peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        logger.debug(f"Output range: [{outputs.min()}, {outputs.max()}]")

    def _train_epoch(self):
        """
        Runs one epoch of training with gradient accumulation and mixed precision.

        Use gradient accumulation to simulate larger batch sizes if hardware limited.
        Scales loss for accumulation and unscales before optimizer step.

        Returns average training loss for the epoch.
        """

        self.model.train()
        total_loss = 0
        num_batches_epoch = len(self.train_loader)  # len accounts for batch size

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

                # Scale for accumulation
                try:
                    loss = (
                        self.criterion(outputs, labels, self.weights.to(self.device))
                        / self.accumulation_steps
                    )
                except TypeError:
                    # Fallback for built-in losses that don't accept weights
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

        return total_loss / num_batches_epoch

    def _validate_epoch(self):
        """
        Perfroms validation after each training epoch to monitor for overfitting.
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
                # Check model against unseen data to monitor overfitting
                with autocast(device_type=self.device, dtype=torch.bfloat16):
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
                    torch.save(self.model.state_dict(), self.save_path)
                    self.stopped_early = epoch + 1
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    logger.info(f"The best model was at epoch {self.stopped_early}.")
                    break

            except KeyboardInterrupt:
                try:
                    logger.info(f"User termined learning process after {epoch} epochs.")
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
                        torch.save(self.model.state_dict(), self.save_path)
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
