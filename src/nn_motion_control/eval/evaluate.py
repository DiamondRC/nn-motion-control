"""
Evaluate a trained model on the held-out test split and report
physical-unit metrics.
"""

import logging
import os
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import autocast

from nn_motion_control.core.checkpoints import (
    load_checkpoint_bundle,
    state_dict_from,
)
from nn_motion_control.data.dataset import DatasetMetadata
from nn_motion_control.data.loaders import BatchLoader
from nn_motion_control.eval.metrics import DEFAULT_GATE, channel_metrics

logger = logging.getLogger(os.path.basename(__file__))

TEST_METRICS_RULE_WIDTH = 65  # console separator width for test tables


def to_flat_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Flatten a tensor to a 1-D float32 numpy array.

    The .float() is required: numpy has no bf16 dtype, so bf16/half
    tensors (produced by autocast or bf16 configs) must be upcast
    before conversion.
    """

    return tensor.detach().float().cpu().flatten().numpy()


class Evaluator:
    """
    Load the best checkpoint, run the test split and report denormalised
    metrics.
    """

    def __init__(
        self,
        model,
        test_loader: BatchLoader,
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

        # Same per-target-weighted loss as training (falls back for
        # built-in losses).
        try:
            self.criterion = criterion_class(weights=self.weights)
        except TypeError:
            self.criterion = criterion_class()
        self.criterion = self.criterion.to(self.device)

    def _testing_loop(self):
        """
        Run the test split and collect (denormalised) predictions and
        targets.
        """

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # Check best model on unseen test data to estimate real-world
        # performance.
        with torch.no_grad():
            for batch in self.test_loader:
                data, labels = (
                    batch[0].to(self.device, non_blocking=True),
                    batch[1].to(self.device, non_blocking=True),
                )
                with autocast(
                    device_type=self.device, dtype=self.training_dtype
                ):
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()

                all_predictions.extend(to_flat_numpy(outputs))
                all_targets.extend(to_flat_numpy(labels))

        self.avg_loss = total_loss / len(self.test_loader)

        self.all_predictions_denorm = self._denorm_outputs(all_predictions)
        self.all_targets_denorm = self._denorm_outputs(all_targets)

    def _denorm_outputs(self, outputs) -> np.ndarray:
        """
        Reshape flattened, sample-major outputs to [state, point] and
        denormalise.
        """

        outputs = np.array(outputs)

        # Predictions were flattened sample-major ([s0_t0..t11,
        # s1_t0..t11, ...]), so reshape to [n_points, states] then
        # transpose to get one row per state.
        lengths = len(self.stds)
        n_points = len(outputs) // lengths
        outputs_reshaped = outputs.reshape(n_points, lengths).T

        means = np.array(list(self.means.values()))
        stds = np.array(list(self.stds.values()))

        return (
            outputs_reshaped.astype(np.float64) * stds[:, np.newaxis]
        ) + means[:, np.newaxis]

    def _run_metrics(self):
        """
        Compute and log common regression metrics against the truth
        values.
        """

        separator = "-" * TEST_METRICS_RULE_WIDTH
        logger.info("Displaying some model predictions against truth values:")
        logger.info(f"{separator}")
        logger.info(
            f"{'State':>8} | {'Predicted':>16} | {'Actual':>16} | {'Diff':>12}"
        )
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
                    f"{self.data_labels[j]:>8} | {p:>16.4f} | "
                    f"{t:>16.4f} | {d:>15.4f}"
                )
            logger.info(f"{separator}")

        preds_all = self.all_predictions_denorm
        targets_all = self.all_targets_denorm
        self.avg_mae = float(np.mean(np.abs(preds_all - targets_all)))
        self.avg_rmse = float(np.sqrt(np.mean((preds_all - targets_all) ** 2)))

        logger.info("Common metrics:")
        logger.info(f"{separator}")
        logger.info(f"Avg Loss: {self.avg_loss:.4f}")
        logger.info(f"Avg MAE: {self.avg_mae:.4f}")
        logger.info(f"Avg RMSE: {self.avg_rmse:.4f}")

        # Per-channel metrics: absolute (physical units) plus
        # std-normalised errors, with the acceptance gate (P95 |err|
        # within 5% of the target's std; P99 tracks the tail).
        self.channel_metrics = {
            name: channel_metrics(name, preds_all[idx], targets_all[idx])
            for idx, name in enumerate(self.data_labels)
        }

        logger.info(f"{separator}")
        gate_pct = DEFAULT_GATE * 100
        logger.info(
            f"Per-channel metrics (gate: P95 <= {gate_pct:.0f}% of "
            f"std; P99 = tail):"
        )
        logger.info(f"{separator}")

        for m in self.channel_metrics.values():
            verdict = "PASS" if m.passes else "FAIL"
            logger.info(
                f"{m.name:>13} | MAE {m.mae:9.3f} RMSE {m.rmse:9.3f} | "
                f"P95|e| {m.p95_abs:9.3f} P99|e| {m.p99_abs:9.3f} | "
                f"P95 {m.p95_frac * 100:6.2f}% P99 {m.p99_frac * 100:6.2f}% | "
                f"FIT {m.fit:6.2f}% | {verdict}"
            )
        n_pass = sum(m.passes for m in self.channel_metrics.values())
        logger.info(f"{separator}")
        logger.info(
            f"Acceptance gate (P95): {n_pass}/"
            f"{len(self.channel_metrics)} channels "
            f"within {gate_pct:.0f}%"
        )

    def test(self):
        """
        Load the best checkpoint, run the test split and log metrics.
        """

        # Checkpoints are bundles (weights + norm stats + provenance);
        # tolerate a bare state_dict for backward compatibility.
        logger.info("Loading best model for testing...")
        bundle = load_checkpoint_bundle(self.save_path, self.device)
        self.model.load_state_dict(state_dict_from(bundle))

        self._testing_loop()
        self._run_metrics()

    def plot_losses(self):
        """
        Plot the training losses against the validation losses for
        visual inspection.
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
