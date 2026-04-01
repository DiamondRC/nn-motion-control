import gc
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(os.path.basename(__file__))


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    train_ratio: float
    val_ratio: float
    seed: int = 42
    cpu_core_util: int = 50
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory: bool = True
    auto_tune_dataloader: bool = True
    do_logging: bool = False


class PVT2DACDataset(Dataset):
    """
    Dataset for loading deltabot data from an HDF5 file.

    This version loads data into RAM if memory is sufficient, selects only
    requested labels, supports optional sliding windows, and exposes helper
    methods to create train/val/test DataLoaders.
    """

    # TODO - Hard coded for now
    DATA_KEY = "inputs"
    TARGET_KEY = "targets"
    INPUT_LABEL_KEY = "input_labels"
    TARGET_LABEL_KEY = "target_labels"
    INPUT_NORM_KEY = "input_norm_params"
    TARGET_NORM_KEY = "target_norm_params"

    def __init__(
        self,
        h5_path: str,
        batch_size: int,
        train_ratio: float,
        val_ratio: float,
        do_auto_tune_dataloader: bool,
        cpu_core_util: int,
        num_workers: int,
        prefetch_factor: int | None,
        logging: bool,
        window_size: int,
        seed: int,
        allowed_inputs: Sequence[str],
        allowed_targets: Sequence[str],
        do_persistent_workers: bool = False,
    ):
        super().__init__()

        if window_size < 1:
            raise ValueError(f"{window_size=} must be >= 1")
        if not 0 < train_ratio < 1:
            raise ValueError(f"{train_ratio=} must be in (0, 1)")
        if not 0 <= val_ratio < 1:
            raise ValueError(f"{val_ratio=} must be in [0, 1)")
        if train_ratio + val_ratio >= 1:
            raise ValueError(f"{train_ratio=} + {val_ratio=} must be < 1")
        if not allowed_inputs:
            raise ValueError("allowed_inputs must not be empty")
        if not allowed_targets:
            raise ValueError("allowed_targets must not be empty")

        self.config = DataLoaderConfig(
            batch_size=batch_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            cpu_core_util=cpu_core_util,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=do_persistent_workers,
            auto_tune_dataloader=do_auto_tune_dataloader,
            do_logging=logging,
        )

        self.window_size = window_size
        self.loaded_ram = False
        self.allowed_inputs = list(allowed_inputs)
        self.allowed_targets = list(allowed_targets)

        if self.config.do_logging:
            logger.debug("Loading dataset to from HDF5...")

        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # File operations
        with h5py.File(h5_path, "r") as f:
            # Grab labels
            self.input_labels = np.array(
                [i.decode("utf-8") for i in f[self.INPUT_LABEL_KEY]]
            )
            self.target_labels = np.array(
                [i.decode("utf-8") for i in f[self.TARGET_LABEL_KEY]]
            )

            # Collect norm params
            self.input_norm_params = np.array(f[self.INPUT_NORM_KEY])
            self.output_norm_params = np.array(f[self.TARGET_NORM_KEY])

            # Check size of dataset in GB
            dataset_size_gb = (f[self.DATA_KEY].nbytes + f[self.TARGET_KEY].nbytes) / (
                1024**3
            )

            # Compare sizes with a little headroom (+20% max system)
            estimated_needed_gb = dataset_size_gb * 1.2
            if available_memory_gb < estimated_needed_gb:
                raise MemoryError(
                    f"Not enough available RAM to load dataset. "
                    f"Dataset size: {dataset_size_gb:.2f} GB, "
                    f"estimated needed: {estimated_needed_gb:.2f} GB, "
                    f"available: {available_memory_gb:.2f} GB"
                )
            else:
                # Load entire dataset into RAM for faster access during training
                self.data = f[self.DATA_KEY][:]
                self.targets = f[self.TARGET_KEY][:]

                # Since we've loaded everything to RAM,
                # we don't need (persistent) workers or prefetching
                self.loaded_ram = True
                if self.config.do_logging:
                    logger.debug(f"Loaded {len(self.data)} samples to RAM")

        # Select inputs and targets
        self._select_data()

        if self.window_size > len(self.data):
            raise ValueError(
                f"{self.window_size=} cannot exceed no. samples ({len(self.data)})"
            )

        # Convert to PyTorch tensors
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).float()

        # Instantiate dataloader params
        self._auto_tune_dataloader()

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = (
            self._create_dataloaders()
        )

    def __len__(self):
        if self.window_size > 1:
            # Handle sliding windows
            return len(self.data) - self.window_size + 1
        else:
            return len(self.data)

    def __getitem__(self, idx: int):
        if self.window_size > 1:
            # Create data window of shape (window_size, features)
            window_data = self.data[idx : idx + self.window_size]

            # Use target at the end of the window
            label = self.targets[idx + self.window_size - 1]
            # print(window_data.shape, label.shape)
            return window_data, label
        else:
            # If no windowing, return single sample
            return self.data[idx], self.targets[idx]

    def _auto_tune_dataloader(self):
        """
        Auto-tune dataloader parameters based on system resources.

        If the dataset is loaded into RAM, we can disable (persistent) workers
        as well as prefetching since we have no I/O bottlenecks.
        Otherwise, we need to use workers and prefetching to keep the GPU fed with data.
        """

        # Profile system resources
        cpu_count = os.cpu_count() or 1
        ram_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = torch.cuda.device_count()

        # First check if we need I/O resources.
        if self.loaded_ram and self.config.auto_tune_dataloader:
            workers = max(int(cpu_count * (self.config.cpu_core_util / 100)), 0)
            self.config = DataLoaderConfig(
                **{
                    **self.config.__dict__,
                    "num_workers": max(0, min(workers, cpu_count)),
                    "prefetch_factor": max(2, workers) if workers > 0 else None,
                }
            )
            if self.config.do_logging:
                logger.debug(
                    f"Auto-detected: {cpu_count} cores, "
                    f"{ram_gb:.1f}GB RAM, {gpu_count} GPU(s)"
                )
            logger.debug(
                f"Using: num_workers={self.config.num_workers}, "
                f"prefetch_factor={self.config.prefetch_factor}"
            )

        # If autoconfiguration is disabled, use user-provided values.
        else:
            if self.config.cpu_core_util > 90 and self.config.do_logging:
                logger.debug(
                    f"WARNING: Using a high percentage of"
                    f"system CPU cores ({self.config.cpu_core_util})%"
                )
            logger.debug(
                f"Using manually set dataloader config: "
                f"{self.config.num_workers=}, {self.config.prefetch_factor=}"
            )

    def _select_data(self):
        """
        Select only data specified in the model json.
        TODO - Don't need to load entire dataset if we're
        using a subset for the dataloaders
        """

        input_map = {label: idx for idx, label in enumerate(self.input_labels)}
        target_map = {label: idx for idx, label in enumerate(self.target_labels)}

        try:
            in_idxs = [input_map[item] for item in self.allowed_inputs]
        except KeyError as e:
            raise ValueError(
                f"Invalid input label: {e.args[0]}."
                f"Available inputs are: {list(self.input_labels)}"
            ) from e

        try:
            out_idxs = [target_map[item] for item in self.allowed_targets]
        except KeyError as e:
            raise ValueError(
                f"Invalid target label: {e.args[0]}."
                f"Available targets are: {list(self.target_labels)}"
            ) from e

        self.data = self.data[:, in_idxs]
        self.targets = self.targets[:, out_idxs]

    def _split_lengths(self) -> tuple[int, int, int]:
        n = len(self)
        train_size = int(self.config.train_ratio * n)
        val_size = int(self.config.val_ratio * n)
        test_size = n - train_size - val_size

        if train_size <= 0 or val_size < 0 or test_size <= 0:
            raise ValueError(
                f"Invalid split sizes: {train_size=}, {val_size=}, {test_size=}."
                f"total={n}"
            )
        return train_size, val_size, test_size

    def _create_dataloaders(self):
        """
        Split loaded dataset into train/val/test and create DataLoaders for each.
        """

        # Set seed for reproducible splits
        torch.manual_seed(self.config.seed)
        _ = torch.Generator().manual_seed(self.config.seed)

        # Partition dataset into train/val/test
        train_size, val_size, test_size = self._split_lengths()
        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

        # Force multiprocessing to use temporary files for inter-process communication.
        # Avoids descriptor exhaustion and cleanup deadlocks on Linux.
        if self.config.num_workers > 0:
            mp.set_sharing_strategy("file_system")

        common_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory and torch.cuda.is_available(),
            "persistent_workers": self.config.persistent_workers
            and self.config.num_workers > 0,
        }

        if self.config.num_workers > 0 and self.config.prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = self.config.prefetch_factor

        # Instantiate DataLoaders
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **common_kwargs,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **common_kwargs,
        )

        return train_loader, val_loader, test_loader

    def cleanup_dataloaders(self):
        """
        Clean up DataLoader workers to exit gracefully.

        Usage: Call this method after training/testing is complete to ensure all
        worker processes are terminated and resources are freed.
        """

        # Dump the CUDA cache
        torch.cuda.empty_cache()
        # Run garbage collection
        gc.collect()

    def get_data_labels(self):
        """
        Returns the labels of the input and target data.
        """

        return self.input_labels, self.target_labels

    def get_normalisation_params(self):
        """
        Returns the normalisation params.
        """

        return self.input_norm_params, self.output_norm_params
