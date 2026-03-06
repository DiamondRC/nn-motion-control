import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast


class TestModel:
    def __init__(
        self,
        model,
        test_loader,
        training_losses,
        validation_losses,
        criterion_class,
        early_stop_epoch,
        normalisation_consts,
        save_path,
        device,
        logging,
    ):
        # Instantiate user args
        self.model = model
        self.test_loader = test_loader
        self.train_losses = training_losses
        self.val_losses = validation_losses
        self.criterion = criterion_class()
        self.early_stop_epoch = early_stop_epoch
        self.norm_consts = normalisation_consts
        self.save_path = save_path
        self.device = device

    def _plot_losses(self):
        """ """

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.legend()
        plt.show()

    def _testing_loop(self):
        """ """

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

        self.avg_loss = total_loss / len(self.test_loader)

        # Convert to tensors for metric computation
        self.all_predictions = torch.tensor(all_predictions)
        self.all_targets = torch.tensor(all_targets)

    def _run_metrics(self):
        """ """

        # Common regression metrics
        self.mae = torch.mean(torch.abs(self.all_predictions - self.all_targets)).item()
        self.rmse = torch.sqrt(
            torch.mean((self.all_predictions - self.all_targets) ** 2)
        ).item()
        self.mape = (
            torch.mean(
                torch.abs(
                    (self.all_targets - self.all_predictions)
                    / (torch.abs(self.all_targets) + 1e-8)
                )
            )
            * 100
        )

        print(
            f"Test Results - Loss: {self.avg_loss:.4f}, "
            f"MAE: {self.mae:.4f}, RMSE: {self.rmse:.4f}, MAPE: {self.mape:.2f}%"
        )

    def _example_data(self):
        """ """

        # TODO - Rewrite for readability
        self.model.eval()

        print(f"self.norm_consts: {self.norm_consts}")

        (
            t_mean,
            t_std,
            x_pos_mean,
            x_pos_std,
            x_vel_mean,
            x_vel_std,
            y_pos_mean,
            y_pos_std,
            y_vel_mean,
            y_vel_std,
            z_pos_mean,
            z_pos_std,
            z_vel_mean,
            z_vel_std,
            x_dac_mean,
            x_dac_std,
            y_dac_mean,
            y_dac_std,
            z_dac_mean,
            z_dac_std,
        ) = self.norm_consts

        # Denormalise them and display to user
        def _denormalise_column(normed_values):
            """
            Denormalize 1D normalized values using z-score standardization inverse.
            """

            (
                t_mean,
                t_std,
                x_pos_mean,
                x_pos_std,
                x_vel_mean,
                x_vel_std,
                y_pos_mean,
                y_pos_std,
                y_vel_mean,
                y_vel_std,
                z_pos_mean,
                z_pos_std,
                z_vel_mean,
                z_vel_std,
                x_dac_mean,
                x_dac_std,
                y_dac_mean,
                y_dac_std,
                z_dac_mean,
                z_dac_std,
            ) = self.norm_consts

            normed_values[0] = (
                normed_values[0].astype(np.float64) * x_dac_std
            ) + x_dac_mean
            normed_values[1] = (
                normed_values[1].astype(np.float64) * y_dac_std
            ) + y_dac_mean
            normed_values[2] = (
                normed_values[2].astype(np.float64) * z_dac_std
            ) + z_dac_mean
            normed_values[3] = (
                normed_values[3].astype(np.float64) * x_dac_std
            ) + x_dac_mean
            normed_values[4] = (
                normed_values[4].astype(np.float64) * y_dac_std
            ) + y_dac_mean
            normed_values[5] = (
                normed_values[5].astype(np.float64) * z_dac_std
            ) + z_dac_mean
            normed_values[6] = (
                normed_values[6].astype(np.float64) * x_dac_std
            ) + x_dac_mean
            normed_values[7] = (
                normed_values[7].astype(np.float64) * y_dac_std
            ) + y_dac_mean
            normed_values[8] = (
                normed_values[8].astype(np.float64) * z_dac_std
            ) + z_dac_mean

            return normed_values

        def _denormalise_windowed(normed_windows):
            """Denormalize windows: (N, window_size, 7) -> (N, window_size, 7)"""

            (
                t_mean,
                t_std,
                x_pos_mean,
                x_pos_std,
                x_vel_mean,
                x_vel_std,
                y_pos_mean,
                y_pos_std,
                y_vel_mean,
                y_vel_std,
                z_pos_mean,
                z_pos_std,
                z_vel_mean,
                z_vel_std,
                x_dac_mean,
                x_dac_std,
                y_dac_mean,
                y_dac_std,
                z_dac_mean,
                z_dac_std,
            ) = self.norm_consts

            normed_windows[:, :, 0] = (
                normed_windows[:, :, 0].astype(np.float64) * t_std
            ) + t_mean
            normed_windows[:, :, 1] = (
                normed_windows[:, :, 1].astype(np.float64) * x_pos_std
            ) + x_pos_mean
            normed_windows[:, :, 2] = (
                normed_windows[:, :, 2].astype(np.float64) * x_vel_std
            ) + x_vel_mean
            normed_windows[:, :, 3] = (
                normed_windows[:, :, 3].astype(np.float64) * y_pos_std
            ) + y_pos_mean
            normed_windows[:, :, 4] = (
                normed_windows[:, :, 4].astype(np.float64) * y_vel_std
            ) + y_vel_mean
            normed_windows[:, :, 5] = (
                normed_windows[:, :, 5].astype(np.float64) * z_pos_std
            ) + z_pos_mean
            normed_windows[:, :, 6] = (
                normed_windows[:, :, 6].astype(np.float64) * z_vel_std
            ) + z_vel_mean

            return normed_windows

        # Access first N windows via dataset (respects __len__ and __getitem__)
        dataset = self.test_loader.dataset
        n = 3  # Number of windows to sample
        sample_data_windows = []
        sample_labels = []
        sample_preds = []

        with torch.no_grad():
            for i in range(n):
                window_data, label = dataset[
                    i
                ]  # Returns (window_size, 16 features), scalar label

                window_data = window_data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                # Model inference on window
                with autocast(device_type=self.device):
                    output = self.model(
                        window_data.unsqueeze(0)
                    )  # Add batch dim: (1, 1)

                # Store CPU numpy for denormalization
                sample_data_windows.append(window_data.cpu().numpy())
                sample_labels.append(label.cpu().numpy())
                sample_preds.append(output.cpu().numpy().flatten())

        # Stack windows: shape (N, window_size, 16)
        all_windows = np.stack(sample_data_windows)  # (N, window_size, 16)
        all_labels = np.concatenate(sample_labels)  # (N,)
        sample_preds = np.concatenate(sample_preds)  # TODO

        denorm_windows = _denormalise_windowed(all_windows)
        denorm_labels = _denormalise_column(all_labels)
        sample_preds = _denormalise_column(sample_preds)

        print("\nDisplaying denormalised values...")
        print(f"First window, first 3 timesteps:\n{denorm_windows[0, :3, :3]}")
        print(f"Real values:      {denorm_labels}")
        print(f"Model prediction: {sample_preds}")

    def test(self):
        """ """

        # Load best model and test
        print("\nLoading best model for testing...")
        self.model.load_state_dict(torch.load(self.save_path))

        # # Display training losses
        # self._plot_losses()

        # # Test the loaded model
        # self._testing_loop()

        # # Display metrics about the trained model
        # self._run_metrics()

        # Undo normalisation to display real results to user
        self._example_data()
