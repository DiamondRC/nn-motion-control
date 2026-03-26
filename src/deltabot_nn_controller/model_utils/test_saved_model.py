import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast

logger = logging.getLogger(os.path.basename(__file__))


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
        plot_path,
        plot_name,
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
        self.plot_path = plot_path
        self.plot_name = plot_name
        self.device = device
        self.test_display_num = test_display_num

        # Unpack normalisation arguements
        self.normalisation_consts = normalisation_consts
        # input state
        self.t_mean = normalisation_consts[0]
        self.t_std = normalisation_consts[1]
        self.x_pos_mean = normalisation_consts[2]
        self.x_pos_std = normalisation_consts[3]
        self.x_vel_mean = normalisation_consts[4]
        self.x_vel_std = normalisation_consts[5]
        self.x_acc_mean = normalisation_consts[6]
        self.x_acc_std = normalisation_consts[7]
        self.x_jer_mean = normalisation_consts[8]
        self.x_jer_std = normalisation_consts[9]
        self.y_pos_mean = normalisation_consts[10]
        self.y_pos_std = normalisation_consts[11]
        self.y_vel_mean = normalisation_consts[12]
        self.y_vel_std = normalisation_consts[13]
        self.y_acc_mean = normalisation_consts[14]
        self.y_acc_std = normalisation_consts[15]
        self.y_jer_mean = normalisation_consts[16]
        self.y_jer_std = normalisation_consts[17]
        self.z_pos_mean = normalisation_consts[18]
        self.z_pos_std = normalisation_consts[19]
        self.z_vel_mean = normalisation_consts[20]
        self.z_vel_std = normalisation_consts[21]
        self.z_acc_mean = normalisation_consts[22]
        self.z_acc_std = normalisation_consts[23]
        self.z_jer_mean = normalisation_consts[24]
        self.z_jer_std = normalisation_consts[25]
        # dac
        self.x_dac_mean = normalisation_consts[26]
        self.x_dac_std = normalisation_consts[27]
        self.y_dac_mean = normalisation_consts[28]
        self.y_dac_std = normalisation_consts[29]
        self.z_dac_mean = normalisation_consts[30]
        self.z_dac_std = normalisation_consts[31]
        # next state
        self.x_pos_nxt_mean = normalisation_consts[32]
        self.x_pos_nxt_std = normalisation_consts[33]
        self.x_vel_nxt_mean = normalisation_consts[34]
        self.x_vel_nxt_std = normalisation_consts[35]
        self.x_acc_nxt_mean = normalisation_consts[36]
        self.x_acc_nxt_std = normalisation_consts[37]
        self.x_jer_nxt_mean = normalisation_consts[38]
        self.x_jer_nxt_std = normalisation_consts[39]
        self.y_pos_nxt_mean = normalisation_consts[40]
        self.y_pos_nxt_std = normalisation_consts[41]
        self.y_vel_nxt_mean = normalisation_consts[42]
        self.y_vel_nxt_std = normalisation_consts[43]
        self.y_acc_nxt_mean = normalisation_consts[44]
        self.y_acc_nxt_std = normalisation_consts[45]
        self.y_jer_nxt_mean = normalisation_consts[46]
        self.y_jer_nxt_std = normalisation_consts[47]
        self.z_pos_nxt_mean = normalisation_consts[48]
        self.z_pos_nxt_std = normalisation_consts[49]
        self.z_vel_nxt_mean = normalisation_consts[50]
        self.z_vel_nxt_std = normalisation_consts[51]
        self.z_acc_nxt_mean = normalisation_consts[52]
        self.z_acc_nxt_std = normalisation_consts[53]
        self.z_jer_nxt_mean = normalisation_consts[54]
        self.z_jer_nxt_std = normalisation_consts[55]

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

        def _denorm_outputs(outputs):
            """
            Used for plant state outputs
            """
            outputs = np.array(outputs)

            # Split into states
            n_points = len(outputs) // 13
            outputs_reshaped = outputs.reshape(13, n_points)

            stds = np.array(
                [
                    self.t_std,
                    self.x_pos_nxt_std,
                    self.x_vel_nxt_std,
                    self.x_acc_nxt_std,
                    self.x_jer_nxt_std,
                    self.y_pos_nxt_std,
                    self.y_vel_nxt_std,
                    self.y_acc_nxt_std,
                    self.y_jer_nxt_std,
                    self.z_pos_nxt_std,
                    self.z_vel_nxt_std,
                    self.z_acc_nxt_std,
                    self.z_jer_nxt_std,
                ]
            )
            means = np.array(
                [
                    self.t_mean,
                    self.x_pos_nxt_mean,
                    self.x_vel_nxt_mean,
                    self.x_acc_nxt_mean,
                    self.x_jer_nxt_mean,
                    self.y_pos_nxt_mean,
                    self.y_vel_nxt_mean,
                    self.y_acc_nxt_mean,
                    self.y_jer_nxt_mean,
                    self.z_pos_nxt_mean,
                    self.z_vel_nxt_mean,
                    self.z_acc_nxt_mean,
                    self.z_jer_nxt_mean,
                ]
            )

            # Add new axis for broadcasting (13,) -> (13,1)
            stds = stds[:, np.newaxis]  # Now shape: (13, 1)
            means = means[:, np.newaxis]

            denormalized = (outputs_reshaped.astype(np.float64) * stds) + means
            # denormalize = denormalized.ravel()  # flatten back to 1D

            return denormalized

        # Denormalise Data
        # self.all_inputs_denorm = _denormalise_inputs()
        self.all_predictions_denorm = _denorm_outputs(all_predictions)
        self.all_targets_denorm = _denorm_outputs(all_targets)

    def _run_metrics(self):
        """
        Calculates some common metrics to benchmark the performance of the model.
        """

        logger.info("Displaying some model predictions against truth values:")
        logger.info(f"{'-' * 62}")
        logger.info(f"{'State':>8} | {'Predicted':>16} | {'Actual':>16} | {'Diff':>12}")
        logger.info(f"{'-' * 62}")

        # Take a few predictions and targets to benchmark the model
        preds = self.all_predictions_denorm[:, : self.test_display_num]
        targets = self.all_targets_denorm[:, : self.test_display_num]
        diffs = np.sqrt((preds - targets) ** 2)

        axes = [
            "times",
            "x_pos",
            "x_vel",
            "x_acc",
            "x_jer",
            "y_pos",
            "y_vel",
            "y_acc",
            "y_jer",
            "z_pos",
            "z_vel",
            "z_acc",
            "z_jer",
        ]

        # Display model prediction against target
        no_states = len(preds[:, 0])
        for i in range(0, self.test_display_num):
            for j in range(0, no_states):
                p = preds[j, i]
                t = targets[j, i]
                d = diffs[j, i]
                logger.info(f"{axes[j]:>8} | {p:>16.4f} | {t:>16.4f} | {d:>16.4f}")
            logger.info(f"{'-' * 62}")

        def _calculate_mae(pred, tar):
            return np.mean(np.abs(pred - tar)).item()

        def _calculate_rmse(pred, tar):
            return np.sqrt(np.mean((pred - tar) ** 2)).item()

        def _calculate_percentiles(pred, tar, ps=(95, 99)):
            set = np.sqrt((pred - tar) ** 2)
            return {p: np.percentile(set, p).item() for p in ps}

        def _calculate_metrics(pred, tar, name):
            # Calc MAE, RMSE
            mae = _calculate_mae(pred, tar)
            rmse = _calculate_rmse(pred, tar)

            abs_p = _calculate_percentiles(pred, tar)

            # Return values for logging
            logger.info(f"{name} 95% Abs Err: {abs_p[95]:.2f}%")
            logger.info(f"{name} 99% Abs Err: {abs_p[99]:.2f}%")
            logger.info(f"{name} MAE: {mae:.4f}")
            logger.info(f"{name} RMSE: {rmse:.4f}")
            logger.info(f"{'-' * 62}")

        # Common regression metrics
        self.avg_mae = _calculate_mae(
            self.all_predictions_denorm, self.all_targets_denorm
        )
        self.avg_rmse = _calculate_rmse(
            self.all_predictions_denorm, self.all_targets_denorm
        )

        # Return average metrics for whole run
        logger.info("Common metrics:")
        logger.info(f"{'-' * 62}")
        logger.info(f"Avg Loss: {self.avg_loss:.4f}")
        logger.info(f"Avg MAE: {self.avg_mae:.4f}")
        logger.info(f"Avg RMSE: {self.avg_rmse:.4f}")

        # Return metric for each item
        logger.info(f"{'-' * 62}")
        logger.info("Variable specific metrics:")
        logger.info(f"{'-' * 62}")
        for idx, axis in enumerate(axes):
            _calculate_metrics(
                self.all_predictions_denorm[idx],
                self.all_targets_denorm[idx],
                axis,
            )

    def test(self):
        """
        Execute the testing sequence for the saved model.
        """

        # Load best model and test
        logger.info("Loading best model for testing...")
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

        plot_name = os.path.join(self.plot_path, f"{self.plot_name}.svg")
        plt.savefig(fname=plot_name, format="svg", bbox_inches="tight")
        plt.close()
