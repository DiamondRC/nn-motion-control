import gc
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(os.path.basename(__file__))


@dataclass(frozen=True)
class DatasetMetadata:
    input_labels: np.ndarray
    target_labels: np.ndarray
    input_denorm_params: dict[str, dict[str, float]]
    target_denorm_params: dict[str, dict[str, float]]
    loss_weights: torch.Tensor


@dataclass(frozen=True)
class AllDataInfo:
    trn_loader: DataLoader
    val_loader: DataLoader
    tst_loader: DataLoader
    node_info: DatasetMetadata


def build_time_series_splits(
    h5_path: str,
    allowed_inputs: Sequence[str],
    allowed_targets: Sequence[dict[str, float]],
    window_size: int,
    train_ratio: float,
    val_ratio: float,
    training_dtype: torch.dtype,
    seed: int = 42,
    load_into_ram: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    cpu_core_util: int = 50,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    auto_tune_workers: bool = True,
    enable_logging: bool = False,
) -> AllDataInfo:
    """
    Create contiguous or random splits in loaded data.

    Contiguous helps reduce leakage from overlapping windows.
    """

    split_mode: str = "contiguous" if window_size > 1 else "random"

    if load_into_ram:
        ensure_enough_ram(h5_path)

    dataset = H5TimeSeriesDataset(
        h5_path=h5_path,
        allowed_inputs=allowed_inputs,
        allowed_targets=allowed_targets,
        window_size=window_size,
        load_into_ram=load_into_ram,
        dtype=training_dtype,
    )

    if auto_tune_workers:
        num_workers, prefetch_factor, persistent_workers = setup_workers(
            num_workers=num_workers,
            cpu_core_util=cpu_core_util,
            load_into_ram=load_into_ram,
            enable_logging=enable_logging,
        )

    n = len(dataset)
    if split_mode == "contiguous":
        train_idx, val_idx, test_idx = split_indices_contiguous(
            n, train_ratio, val_ratio
        )
    elif split_mode == "random":
        train_idx, val_idx, test_idx = split_indices_random(
            n, train_ratio, val_ratio, seed
        )
    else:
        # Leaving incase of future split method
        raise ValueError("split_mode must be 'contiguous' or 'random'")

    return AllDataInfo(
        *make_dataloaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            shuffle_train=(split_mode == "random"),
            seed=seed,
            enable_logging=enable_logging,
        ),
        node_info=dataset.meta,
    )


def validate_labels(
    requested: Sequence[str], available: Sequence[str], kind: str
) -> list[int]:
    """
    Generic handler for matching two sequences of strings.
    """

    mapping = {label: idx for idx, label in enumerate(available)}
    missing = [x for x in requested if x not in mapping]

    try:
        if missing:
            raise ValueError(
                f"Invalid {kind} labels: {missing}."
                f"Available {kind} labels: {list(available)}"
            )
    except ValueError:
        logger.exception(
            f"Invalid {kind} labels: {missing}."
            f"Available {kind} labels: {list(available)}"
        )

    return [mapping[x] for x in requested]


def split_indices_contiguous(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Contiguous data spliting for sliding window data spliting.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(f"{train_ratio=} must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"{val_ratio=} must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError(f"{train_ratio=} + {val_ratio=} must be < 1")

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    try:
        if train_end <= 0 or val_end <= train_end or val_end >= n:
            raise ValueError(f"Invalid split sizes for {n=}")
    except ValueError:
        logger.exception(f"Invalid split sizes for {n=}")

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    return train_idx, val_idx, test_idx


def split_indices_random(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomised spliting when not using sliding windows
    over temporal data.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(f"{train_ratio=} must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"{val_ratio=} must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError(f"{train_ratio=} + {val_ratio=} must be < 1")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    return train_idx, val_idx, test_idx


class H5TimeSeriesDataset(Dataset):
    # TODO - hardcoded
    DATA_KEY = "inputs"
    TARGET_KEY = "targets"

    def __init__(
        self,
        h5_path: str,
        allowed_inputs: Sequence[str],
        allowed_targets: Sequence[dict[str, float]],
        window_size: int = 1,
        load_into_ram: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self._file = None
        self._inputs = None
        self._targets = None
        self._input_idx = None
        self._target_idx = None

        self.window_size = window_size
        self.h5_path = h5_path
        self.dtype = dtype

        self.allowed_inputs = list(allowed_inputs)
        self.allowed_targets = [list(d.keys())[0] for d in allowed_targets]
        loss_weighting = [list(d.values())[0] for d in allowed_targets]

        self.meta = self._load_metadata(loss_weighting)

        try:
            if self.window_size > self.num_samples:
                raise ValueError(
                    f"{self.window_size=} cannot exceed {self.num_samples=}"
                )
        except ValueError:
            logger.exception(f"{self.window_size=} cannot exceed {self.num_samples=}")

        if load_into_ram:
            self._load_all_to_ram()

    def _load_metadata(self, loss_weighting: list[float]) -> DatasetMetadata:
        logger.debug("Gathering dataset metadata...")

        with h5py.File(self.h5_path, "r") as f:
            input_labels = np.array([x.decode("utf-8") for x in f["input_labels"]])
            target_labels = np.array([x.decode("utf-8") for x in f["target_labels"]])
            input_norm_params = np.array(f["input_norm_params"])
            target_norm_params = np.array(f["target_norm_params"])

            self.num_samples = int(f["inputs"].shape[0])

        self._input_idx = validate_labels(self.allowed_inputs, input_labels, "input")
        self._target_idx = validate_labels(
            self.allowed_targets, target_labels, "target"
        )

        # Pack denorm params according to user selection
        input_denorm_params = {
            "mean": dict(zip(self.allowed_inputs, input_norm_params[0], strict=False)),
            "std": dict(zip(self.allowed_inputs, input_norm_params[1], strict=False)),
        }

        target_denorm_params = {
            "mean": dict(
                zip(self.allowed_targets, target_norm_params[0], strict=False)
            ),
            "std": dict(zip(self.allowed_targets, target_norm_params[1], strict=False)),
        }

        # Create loss function weighting for each param
        weighting_tensor = torch.tensor(loss_weighting)
        loss_weights = torch.stack(
            [
                (1.0 / (target_denorm_params["std"][label] ** 2 + 1e-8))
                * weighting_tensor
                for label in self.allowed_targets
            ]
        )

        return DatasetMetadata(
            input_labels=input_labels,
            target_labels=target_labels,
            input_denorm_params=input_denorm_params,
            target_denorm_params=target_denorm_params,
            loss_weights=loss_weights,
        )

    def _open_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _load_all_to_ram(self):
        with h5py.File(self.h5_path, "r") as f:
            x = f[self.DATA_KEY][:, self._input_idx]
            y = f[self.TARGET_KEY][:, self._target_idx]

        self._inputs = torch.from_numpy(np.asarray(x)).to(self.dtype)
        self._targets = torch.from_numpy(np.asarray(y)).to(self.dtype)

    def __len__(self) -> int:
        """
        Accounts for windowing.
        """
        return self.num_samples - self.window_size + 1

    def __getitem__(self, idx: int):
        """
        Initially load from file, then pull from loaded data.
        """

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Load from RAM if available
        if self._inputs is not None and self._targets is not None:
            if self.window_size == 1:
                return self._inputs[idx], self._targets[idx]
            x = self._inputs[idx : idx + self.window_size].T
            y = self._targets[idx + self.window_size - 1]
            return x, y

        # Otherwise fetch from file
        f = self._open_file()
        if self.window_size == 1:
            x = f[self.DATA_KEY][idx, self._input_idx]
            y = f[self.TARGET_KEY][idx, self._target_idx]
            return torch.as_tensor(x, dtype=self.dtype), torch.as_tensor(
                y, dtype=self.dtype
            )

        x = f[self.DATA_KEY][idx : idx + self.window_size, self._input_idx].T
        y = f[self.TARGET_KEY][idx + self.window_size - 1, self._target_idx]
        return torch.as_tensor(x, dtype=self.dtype), torch.as_tensor(
            y, dtype=self.dtype
        )

    def close_file(self):
        """
        Closes the opened datafile.
        """

        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def cleanup(self):
        """
        Manually clear file handling to try to prevent I/O issues.
        """

        self.close_file()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[int(self.indices[idx])]


def setup_workers(
    num_workers: int,
    cpu_core_util: int,
    load_into_ram: bool,
    enable_logging: bool = False,
) -> tuple[int, int | None, bool]:
    """
    Attempts to auto-tune workers to the systems resources,
    otherwise uses user specification.
    """

    cpu_count = os.cpu_count() or 1
    if not load_into_ram:
        return num_workers, None, False

    target_workers = max(int(cpu_count * (cpu_core_util / 100.0)), 0)
    target_workers = min(target_workers, cpu_count)

    prefetch_factor = max(2, target_workers) if target_workers > 0 else None
    persistent_workers = target_workers > 0

    if enable_logging:
        logger.debug(
            f"Auto-tuned workers: {cpu_count=}, {target_workers=}, "
            f"{prefetch_factor=}, {persistent_workers=}"
        )

    return target_workers, prefetch_factor, persistent_workers


def make_dataloaders(
    dataset: Dataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    seed: int = 42,
    enable_logging: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the training, validation and testing dataloaders.
    """

    generator = torch.Generator().manual_seed(seed)

    train_ds = SubsetDataset(dataset, train_idx)
    val_ds = SubsetDataset(dataset, val_idx)
    test_ds = SubsetDataset(dataset, test_idx)

    # Force multiprocessing to use temporary files for inter-process communication.
    # Avoids descriptor exhaustion and cleanup deadlocks on Linux.
    if num_workers > 0:
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": persistent_workers and num_workers > 0,
    }

    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle_train,
        generator=generator if shuffle_train else None,
        **common_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    if enable_logging:
        logger.debug(
            f"Created dataloaders with {batch_size=}, {num_workers=}, "
            f"pin_memory={common_kwargs['pin_memory']}"
        )

    return train_loader, val_loader, test_loader


def ensure_enough_ram(h5_path: str):
    """
    Estimates the size of the data and compares to system resources.
    """

    with h5py.File(h5_path, "r") as f:
        data_size = (f["inputs"].nbytes + f["targets"].nbytes) / (1024**3)

    # Compare sizes with a little headroom (+20% max system)
    required = data_size * 1.2
    available = psutil.virtual_memory().available / (1024**3)

    try:
        if available < required:
            raise MemoryError(
                f"Not enough available RAM. {required:.2f=} GB, {available:.2f=} GB"
            )
    except MemoryError:
        logger.exception(
            f"Not enough available RAM. {required:.2f=} GB, {available:.2f=} GB"
        )
