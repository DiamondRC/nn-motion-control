import gc
import logging
import os

import h5py
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class PVT2DACDataset(Dataset):
    """
    Custom Dataset for loading deltabot data from an HDF5 file.

    Tries to load the entire dataset into RAM for faster access during training.
    Provides a method to create DataLoaders with train/val/test splits.
    """

    def __init__(
        self,
        h5_path,
        batch_size,
        train_ratio,
        val_ratio,
        do_auto_tune_dataloader,
        cpu_core_util,
        num_workers,
        prefetch_factor,
        logging,
        window_size,
        seed,
        # Persistant_workers seems to cause all sorts of I/O issues,
        # so disabled for now.
        do_persistent_workers=False,
        data_key="inputs",
        label_key="outputs",
    ):
        # Instantiate user-configurable options
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.do_auto_tune_dataloader = do_auto_tune_dataloader
        self.cpu_core_util = cpu_core_util
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.do_persistent_workers = do_persistent_workers
        self.logging = logging
        self.window_size = window_size
        self.seed = seed
        self.loaded_ram = False

        # Log dataset loading
        if self.logging:
            logger.debug("Loading dataset to RAM...")

        # Check if we can send to RAM, otherwise fallback to on-demand loading
        available_memory = psutil.virtual_memory().total / (1024**3)
        with h5py.File(h5_path, "r") as f:
            # Check size of dataset in GB
            dataset_size_gb = (f[data_key].nbytes + f[label_key].nbytes) / (1024**3)

            # Collect normalisation params
            self.norm_params = np.array(f["norm_params"])
            # .values.astype(
            #     np.float32
            # )

            # Compare sizes with a little headroom (+20% max system)
            if available_memory < dataset_size_gb + (available_memory * 0.2):
                raise MemoryError(
                    f"Not enough RAM to load dataset ({dataset_size_gb:.2f}GB), "
                    f"available: {available_memory:.2f}GB"
                )
            else:
                # Load entire dataset into RAM for faster access during training
                self.data = f[data_key][:]
                self.labels = f[label_key][:]

                # Since we've loaded everything to RAM,
                # we don't need (persistent) workers or prefetching
                self.loaded_ram = True
                if self.logging:
                    logger.debug(f"Loaded {len(self.data)} samples to RAM")

        # Instantiate dataloader params
        self._auto_tune_dataloader()

        # Convert to PyTorch tensors
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

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

    def __getitem__(self, idx):
        if self.window_size > 1:
            # Create data window of shape (window_size, features)
            window_data = self.data[idx : idx + self.window_size]

            # Use target at the end of the window
            label = self.labels[idx + self.window_size - 1]
            # print(window_data.shape, label.shape)
            return window_data, label
        else:
            # If no windowing, return single sample
            return self.data[idx], self.labels[idx]

    def _auto_tune_dataloader(self):
        """
        Auto-tune dataloader parameters based on system resources.

        If the dataset is loaded into RAM, we can disable (persistent) workers
        as well as prefetching since we have no I/O bottlenecks.
        Otherwise, we need to use workers and prefetching to keep the GPU fed with data.
        """

        # Profile system resources
        cpu_count = os.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024**3)
        gpu_count = torch.cuda.device_count()

        # First check if we need I/O resources.
        if self.loaded_ram:
            self.num_workers = max(int(cpu_count * (self.cpu_core_util / 100)), 8)
            self.prefetch_factor = max(self.num_workers, 4)
            if self.logging:
                logger.debug(
                    f"Auto-detected: {cpu_count} cores, "
                    f"{ram_gb:.1f}GB RAM, {gpu_count} GPU(s)"
                )
            logger.debug(
                f"Using: num_workers={self.num_workers}, "
                f"prefetch_factor={self.prefetch_factor}"
            )

        # If autoconfiguration is disabled, use user-provided values.
        else:
            if self.cpu_core_util > 90:
                logger.debug(
                    f"WARNING: Using a high percentage of"
                    f"system CPU cores ({self.cpu_core_util})%"
                )
            logger.debug(
                f"Using manually set: self.num_workers={self.num_workers}, "
                f"prefetch_factor={self.prefetch_factor}"
            )

    def _create_dataloaders(self):
        """
        Split loaded dataset into train/val/test and create DataLoaders for each.

        Args:
            train_ratio (float): Proportion of data for training dataset
            val_ratio (float): Proportion of data for validation dataset
            batch_size (int): Batching size for the DataLoaders
            self.num_workers (int): Number of worker processes for data loading
            prefetch_factor (int): Number of batches to prefetch per worker
            seed (int): Fix seed for reproducible data splits
        """

        # Set seed for reproducible splits
        torch.manual_seed(self.seed)

        # Calculate split sizes, test gets the remainder
        self.train_size = int(self.train_ratio * len(self))
        self.val_size = int(self.val_ratio * len(self))
        self.test_size = len(self) - self.train_size - self.val_size

        # Partition dataset into train/val/test
        train_dataset, val_dataset, test_dataset = random_split(
            self, [self.train_size, self.val_size, self.test_size]
        )

        # Force multiprocessing to use temporary files for inter-process communication.
        # Avoids descriptor exhaustion and cleanup deadlocks on Linux.
        if self.num_workers > 0:
            mp.set_sharing_strategy("file_system")

        # Instanticate DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=self.do_persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=self.do_persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=self.do_persistent_workers,
            prefetch_factor=self.prefetch_factor,
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

    def get_normalisation_params(self):
        """ """

        return self.norm_params
