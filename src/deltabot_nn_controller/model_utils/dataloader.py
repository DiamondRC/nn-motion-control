import gc

import h5py
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, random_split


class PVT2DACDataset(Dataset):
    """
    Custom Dataset for loading deltabot data from an HDF5 file.

    Tries to load the entire dataset into RAM for faster access during training.
    Provides a method to create DataLoaders with train/val/test splits.
    """

    def __init__(
        self, h5_path, logging, window_size, data_key="inputs", label_key="outputs"
    ):
        # Instantiate user-configurable options
        self.logging = logging
        self.window_size = window_size

        # Log dataset loading
        if self.logging:
            print("\nLoading dataset to RAM...")

        # Try to load entire dataset into system RAM
        # try:
        #     with h5py.File(h5_path, "r") as f:
        #         self.data = f[data_key][:]
        #         self.labels = f[label_key][:]
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load dataset into RAM: {e}") from e

        with h5py.File(h5_path, "r") as f:
            self.data = f[data_key][:]
            self.labels = f[label_key][:]

        # Convert to PyTorch tensors
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

        if self.logging:
            print(f"Loaded {len(self.data)} samples to RAM")

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

    def get_dataloaders(
        self, train_ratio, val_ratio, batch_size, num_workers, prefetch_factor, seed
    ):
        """
        Split loaded dataset into train/val/test and create DataLoaders for each.

        Args:
            train_ratio (float): Proportion of data for training dataset
            val_ratio (float): Proportion of data for validation dataset
            batch_size (int): Batching size for the DataLoaders
            num_workers (int): Number of worker processes for data loading
            prefetch_factor (int): Number of batches to prefetch per worker
            seed (int): Fix seed for reproducible data splits
        """

        # Set seed for reproducible splits
        torch.manual_seed(seed)

        # Calculate split sizes, test gets the remainder
        train_size = int(train_ratio * len(self))
        val_size = int(val_ratio * len(self))
        test_size = len(self) - train_size - val_size

        # Partition dataset into train/val/test
        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_size, val_size, test_size]
        )

        # Force multiprocessing to use temporary files for inter-process communication.
        # Avoids descriptor exhaustion and cleanup deadlocks on Linux.
        if num_workers > 0:
            mp.set_sharing_strategy("file_system")

        # Instanticate DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            # pin_memory_device="cuda",
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor,
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
