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
        test_display_num,
    ):
        # Instantiate user args
        self.model = model
        self.test_loader = test_loader
        self.train_losses = training_losses
        self.val_losses = validation_losses
        self.criterion = criterion_class()
        self.early_stop_epoch = early_stop_epoch
        self.save_path = save_path
        self.device = device
        self.test_display_num = test_display_num

        # Unpack normalisation arguements
        self.t_mean = (normalisation_consts[0],)
        self.t_std = (normalisation_consts[1],)
        self.x_pos_mean = (normalisation_consts[2],)
        self.x_pos_std = (normalisation_consts[3],)
        self.x_vel_mean = (normalisation_consts[4],)
        self.x_vel_std = (normalisation_consts[5],)
        self.x_acc_mean = (normalisation_consts[6],)
        self.x_acc_std = (normalisation_consts[7],)
        self.x_jer_mean = (normalisation_consts[8],)
        self.x_jer_std = (normalisation_consts[9],)
        self.y_pos_mean = (normalisation_consts[10],)
        self.y_pos_std = (normalisation_consts[11],)
        self.y_vel_mean = (normalisation_consts[12],)
        self.y_vel_std = (normalisation_consts[13],)
        self.y_acc_mean = (normalisation_consts[14],)
        self.y_acc_std = (normalisation_consts[15],)
        self.y_jer_mean = (normalisation_consts[16],)
        self.y_jer_std = (normalisation_consts[17],)
        self.z_pos_mean = (normalisation_consts[18],)
        self.z_pos_std = (normalisation_consts[19],)
        self.z_vel_mean = (normalisation_consts[20],)
        self.z_vel_std = (normalisation_consts[21],)
        self.z_acc_mean = (normalisation_consts[22],)
        self.z_acc_std = (normalisation_consts[23],)
        self.z_jer_mean = (normalisation_consts[24],)
        self.z_jer_std = (normalisation_consts[25],)
        self.x_dac_mean = (normalisation_consts[26],)
        self.x_dac_std = (normalisation_consts[27],)
        self.y_dac_mean = (normalisation_consts[28],)
        self.y_dac_std = (normalisation_consts[29],)
        self.z_dac_mean = (normalisation_consts[30],)
        self.z_dac_std = (normalisation_consts[31],)

    def _testing_loop(self):
        """
        Test the model and collect the results for alter analysis.
        """

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

        def _denormalise_inputs(inputs):
            inputs = np.array(inputs)

            stds = np.array(
                [
                    self.t_std,
                    self.x_pos_std,
                    self.x_vel_std,
                    self.x_acc_std,
                    self.y_pos_std,
                    self.y_vel_std,
                    self.y_acc_std,
                    self.z_pos_std,
                    self.z_vel_std,
                    self.z_acc_std,
                ]
            )
            means = np.array(
                [
                    self.t_mean,
                    self.x_pos_mean,
                    self.x_vel_mean,
                    self.x_acc_mean,
                    self.y_pos_mean,
                    self.y_vel_mean,
                    self.y_acc_mean,
                    self.z_pos_mean,
                    self.z_vel_mean,
                    self.z_acc_mean,
                ]
            )

            denormalize = np.add(np.multiply(inputs.astype(np.float64), stds), means)

            return denormalize

        def _denormalise_outputs(outputs):
            outputs = np.array(outputs)

            # Model outputs predictions in triplets,
            # create read-only view in threes.
            stds = np.array([self.x_dac_std, self.y_dac_std, self.z_dac_std])
            means = np.array([self.x_dac_mean, self.y_dac_mean, self.z_dac_mean])

            # Split into triplets
            n_points = len(outputs) // 3
            outputs_reshaped = outputs.reshape(3, n_points)

            # Broadcasting applies stds[0]/means[0] to col0, etc.
            denormalized_reshaped = (outputs_reshaped.astype(np.float64) * stds) + means
            denormalize = denormalized_reshaped.ravel()  # flatten back to 1D

            return denormalize

        # Denormalise Data
        # self.all_inputs_denorm = _denormalise_inputs()
        self.all_predictions_denorm = _denormalise_outputs(all_predictions)
        self.all_targets_denorm = _denormalise_outputs(all_targets)

    def _run_metrics(self):
        """
        Calculates some common metrics to benchmark the performance of the model.
        """

        print("\nDisplaying some model predictions against truth values:")
        print(
            f"\n{'-' * 47}"
            f"\n{'Axis':>4} | {'Predicted':>12} | {'Actual':>12} | {'RMSE':>10}"
            f"\n{'-' * 47}"
        )
        for i in range(0, self.test_display_num, 3):
            # Display individually to see results easily
            preds = self.all_predictions_denorm[i : i + 3]
            targets = self.all_targets_denorm[i : i + 3]
            # diffs = np.abs(preds - targets) / (np.abs(targets) + epsilon) * 100
            diffs = np.sqrt((preds - targets) ** 2)

            axes = ["x", "y", "z"]
            for j, (p, t, d) in enumerate(zip(preds, targets, diffs, strict=False)):
                print(f"{axes[j]:>4} | {p:12.3f} | {t:12.3f} | {d:9.3f}%")
            print(f"{'-' * 47}")

            # # Couple related data for clarity
            # print(
            #     f"\nModel predicts ({preds[0]:.3f}, {preds[1]:.3f}, {preds[2]:.3f}) "
            #     f"versus ({targets[0]:.3f}, {targets[1]:.3f}, {targets[2]:.3f}) "
            #     f"(DAC Values)\n"
            #     f"That's a difference of "
            #     f"({diffs[0]:.3f}%, {diffs[1]:.3f}%, {diffs[2]:.3f}%)!\n"
            # )

        # Common regression metrics
        self.mae = np.mean(
            np.abs(self.all_predictions_denorm - self.all_targets_denorm)
        ).item()
        self.rmse = np.sqrt(
            np.mean((self.all_predictions_denorm - self.all_targets_denorm) ** 2)
        ).item()

        print(
            f"\nCommon metrics:"
            f"\nAvg Loss: {self.avg_loss:.4f}"
            f"\nAvg MAE: {self.mae:.4f}"
            f"\nAvg RMSE: {self.rmse:.4f}"
        )

    def test(self):
        """
        Execute the testing sequence for the saved model.
        """

        # Load best model and test
        print("\nLoading best model for testing...")
        self.model.load_state_dict(torch.load(self.save_path))

        # Test the loaded model
        self._testing_loop()

        # Display metrics about the trained model
        self._run_metrics()

    def plot_losses(self):
        """
        Plots the training losses against the validation losses for visual inspection.
        """

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid()
        plt.legend()
        plt.show()
