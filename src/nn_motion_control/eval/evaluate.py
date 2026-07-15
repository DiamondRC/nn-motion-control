"""
Evaluate a trained model on the held-out test split and report physical-unit metrics.
"""

import logging
import os
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast
from torch.utils.data import DataLoader

from nn_motion_control.data.dataset import DatasetMetadata

logger = logging.getLogger(os.path.basename(__file__))


class Evaluator:
    """
    Load the best checkpoint, run the test split, and report denormalised metrics.
    """

    def __init__(
        self,
        model,
        test_loader: DataLoader,
        training_losses,
        validation_losses,
        criterion_class,
        early_stop_epoch,
        node_info: DatasetMetadata,
        allowed_targets: Sequence[Mapping[str, float]],
        save_path: str,
        plot_path: str,
        plot_name: str,
        device: str,
        test_display_num: int,
        training_dtype: torch.dtype,
    ):
        self.model = model
        self.test_loader = test_loader
        self.train_losses = training_losses
        self.val_losses = validation_losses
        self.early_stop_epoch = early_stop_epoch
        self.save_path = save_path
        self.plot_path = plot_path
        self.plot_name = plot_name
        self.device = device
        self.test_display_num = test_display_num
        self.training_dtype = training_dtype

        self.means = node_info.target_denorm_params["mean"]
        self.stds = node_info.target_denorm_params["std"]
        self.weights = node_info.loss_weights
        self.data_labels = [next(iter(d)) for d in allowed_targets]

        # Same per-target-weighted loss as training (falls back for built-in losses).
        try:
            self.criterion = criterion_class(weights=self.weights)
        except TypeError:
            self.criterion = criterion_class()
        self.criterion = self.criterion.to(self.device)

    def _testing_loop(self):
        """
        Run the test split and collect (denormalised) predictions and targets.
        """

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # Check best model on unseen test data to estimate real-world performance.
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = (
                    data.to(self.device, non_blocking=True),
                    labels.to(self.device, non_blocking=True),
                )
                with autocast(device_type=self.device, dtype=self.training_dtype):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                all_predictions.extend(outputs.float().cpu().flatten().numpy())
                all_targets.extend(labels.cpu().flatten().numpy())

        self.avg_loss = total_loss / len(self.test_loader)

        self.all_predictions = torch.tensor(all_predictions)
        self.all_targets = torch.tensor(all_targets)

        self.all_predictions_denorm = self._denorm_outputs(all_predictions)
        self.all_targets_denorm = self._denorm_outputs(all_targets)

    def _denorm_outputs(self, outputs) -> np.ndarray:
        """
        Reshape flattened, sample-major outputs to [state, point] and denormalise.
        """

        outputs = np.array(outputs)

        # Predictions were flattened sample-major ([s0_t0..t11, s1_t0..t11, ...]),
        # so reshape to [n_points, states] then transpose to get one row per state.
        lengths = len(self.stds)
        n_points = len(outputs) // lengths
        outputs_reshaped = outputs.reshape(n_points, lengths).T

        means = np.array(list(self.means.values()))
        stds = np.array(list(self.stds.values()))
        return (outputs_reshaped.astype(np.float64) * stds[:, np.newaxis]) + means[
            :, np.newaxis
        ]

    def _run_metrics(self):
        """
        Compute and log common regression metrics against the truth values.
        """

        separator = "-" * 65
        logger.info("Displaying some model predictions against truth values:")
        logger.info(f"{separator}")
        logger.info(f"{'State':>8} | {'Predicted':>16} | {'Actual':>16} | {'Diff':>12}")
        logger.info(f"{separator}")

        # Take a few predictions and targets to benchmark the model.
        preds = self.all_predictions_denorm[:, : self.test_display_num]
        targets = self.all_targets_denorm[:, : self.test_display_num]
        diffs = np.sqrt((preds - targets) ** 2)

        no_states = len(preds[:, 0])
        for i in range(self.test_display_num):
            for j in range(no_states):
                p = preds[j, i]
                t = targets[j, i]
                d = diffs[j, i]
                logger.info(
                    f"{self.data_labels[j]:>8} | {p:>16.4f} | {t:>16.4f} | {d:>15.4f}"
                )
            logger.info(f"{separator}")

        def _calculate_mae(pred, tar):
            return np.mean(np.abs(pred - tar)).item()

        def _calculate_rmse(pred, tar):
            return np.sqrt(np.mean((pred - tar) ** 2)).item()

        def _calculate_percentiles(pred, tar, ps=(95, 99)):
            errs = np.sqrt((pred - tar) ** 2)
            return {p: np.percentile(errs, p).item() for p in ps}

        def _calculate_metrics(pred, tar, name):
            mae = _calculate_mae(pred, tar)
            rmse = _calculate_rmse(pred, tar)
            abs_p = _calculate_percentiles(pred, tar)

            logger.info(f"{name} 95% Abs Err: {abs_p[95]:.2f}%")
            logger.info(f"{name} 99% Abs Err: {abs_p[99]:.2f}%")
            logger.info(f"{name} MAE: {mae:.4f}")
            logger.info(f"{name} RMSE: {rmse:.4f}")
            logger.info(f"{separator}")

        self.avg_mae = _calculate_mae(
            self.all_predictions_denorm, self.all_targets_denorm
        )
        self.avg_rmse = _calculate_rmse(
            self.all_predictions_denorm, self.all_targets_denorm
        )

        logger.info("Common metrics:")
        logger.info(f"{separator}")
        logger.info(f"Avg Loss: {self.avg_loss:.4f}")
        logger.info(f"Avg MAE: {self.avg_mae:.4f}")
        logger.info(f"Avg RMSE: {self.avg_rmse:.4f}")

        logger.info(f"{separator}")
        logger.info("Variable specific metrics:")
        logger.info(f"{separator}")
        for idx, axis in enumerate(self.data_labels):
            _calculate_metrics(
                self.all_predictions_denorm[idx],
                self.all_targets_denorm[idx],
                axis,
            )

    def test(self):
        """
        Load the best checkpoint, run the test split, and log metrics.
        """

        # Checkpoints are bundles (weights + norm stats + provenance); tolerate a bare
        # state_dict for backward compatibility.
        logger.info("Loading best model for testing...")
        checkpoint = torch.load(self.save_path, map_location=self.device)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)

        self._testing_loop()
        self._run_metrics()

    def plot_losses(self):
        """
        Plot the training losses against the validation losses for visual inspection.
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
