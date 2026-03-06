import matplotlib.pyplot as plt
import torch
from torch import autocast


class Trainer:
    """
    Trainer class to handle training, validation, and testing loops for the model.
    Implements early stopping based on validation loss and saves the best model.

    Includes a diagnostic method to profile a single batch for performance bottlenecks.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        device: Device to run training on (e.g., 'cuda' or 'cpu')
        scaler_class: Class for mixed precision scaling (e.g., GradScaler)
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        criterion_class: Loss function class (e.g., nn.MSELoss)
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for the optimizer
        min_delta: Minimum change in validation loss to qualify as an improvement
        patience: Number of epochs with no improvement after which we stop training
        save_path: Path to save the best model state dict
        logging: Whether to print detailed logs during training
        accumulation_steps: Number of batches to accumulate gradients over
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        scaler_class,
        optimizer_class,
        criterion_class,
        max_epochs,
        learning_rate,
        min_delta,
        patience,
        save_path,
        logging,
        accumulation_steps,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.scaler = scaler_class(device=self.device)
        self.criterion = criterion_class()
        self.num_epochs = max_epochs
        self.learning_rate = learning_rate
        self.min_delta = min_delta
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.patience = patience
        self.save_path = save_path
        self.logging = logging
        self.accumulation_steps = accumulation_steps

        self.train_losses = []
        self.val_losses = []
        self.stopped_early = 0
        self.test_results = {}

        # Return model shape and parameter count
        if self.logging:
            print("\nParameter devices:")
            for name, param in self.model.named_parameters():
                print(f"  {name}: {param.device}, shape={param.shape}")
            print(
                f"GPU param count: \
{sum(1 for p in self.model.parameters() if p.device.type == 'cuda')}"
            )
            print(
                f"Model size: \
{sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9:.2f}GB"
            )
            print(f"Baseline GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    def profile_one_batch(self):
        """
        Verbose diagnostic function to profile the model.

        Passes a single batch through the model with autocast,
        checks for performance bottlenecks and log timing and memory usage.
        """

        print("\nProfiling one batch...")
        data, labels = next(iter(self.test_loader))
        data, labels = data.to(self.device), labels.to(self.device)

        print("Checking CUDA status before profiling")
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        print("Running forward pass with autocast for profiling...")
        start_event.record()

        # TODO - unify dtypes for profiling, training and future quant
        with autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(data)
            # loss = self.criterion(outputs, labels)

        end_event.record()
        end_event.synchronize()

        print(f"Batch shape: {data.shape}")
        print(f"Forward time: {start_event.elapsed_time(end_event):.2f}ms")
        print(f"Peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
        print(f"Output range: [{outputs.min()}, {outputs.max()}]")

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
            with autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(data)
                loss = (
                    self.criterion(outputs, labels) / self.accumulation_steps
                )  # Scale for accumulation

            # Guard against NaN/Inf loss which can destabilize training
            if torch.isnan(loss) or torch.isinf(loss):
                if torch.isnan(loss):
                    print(f"NaN at batch {batch_idx}, skipping")
                else:
                    print(f"Inf at batch {batch_idx}, skipping")

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
                with autocast(device_type=self.device):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)

                if torch.isnan(loss):
                    print("NaN in validation, skipping")
                elif torch.isinf(loss):
                    print("Inf in validation, skipping")
                else:
                    total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _test(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # Check best model on unseen test data to estimate real-world performance
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = (
                    data.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                )
                with autocast(device_type=self.device):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                    # Collect predictions and targets for metrics
                    all_predictions.extend(outputs.cpu().flatten().numpy())
                    all_targets.extend(labels.cpu().flatten().numpy())

        avg_loss = total_loss / len(self.test_loader)

        # Convert to tensors for metric computation
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)

        # Common regression metrics
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
        mape = (
            torch.mean(
                torch.abs(
                    (all_targets - all_predictions) / (torch.abs(all_targets) + 1e-8)
                )
            )
            * 100
        )

        print(
            f"Test Results - Loss: {avg_loss:.4f}, \
MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%"
        )

        return {"loss": avg_loss, "mae": mae, "rmse": rmse, "mape": mape}

    def _plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.legend()
        plt.show()

    def train(self):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            # Store losses for future plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, \
Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
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
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Load best model and test
        print("\nLoading best model for testing...")
        self.model.load_state_dict(torch.load(self.save_path))
        test_results = self._test()
        self.test_results = test_results
        self._plot_losses()
