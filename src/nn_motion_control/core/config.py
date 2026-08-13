from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from nn_motion_control.core.system import SystemSpec

_LOSS_MODULE = "nn_motion_control.training.losses"


def resolve_class(name: str, module_paths: tuple[str, ...] = ()) -> type:
    """
    Resolve a class by name from custom modules, then torch nn/optim/amp.
    """

    for module_path in module_paths:
        module = import_module(module_path)
        if hasattr(module, name):
            return getattr(module, name)
    for namespace in (nn, optim, torch.amp):
        if hasattr(namespace, name):
            return getattr(namespace, name)
    raise ValueError(f"Unknown class name: {name}")


class RunConfiguration:
    """
    Parse an artifact config + its SystemSpec into flat run hyperparameters.
    """

    def __init__(self, json_path: str):
        self.json_path = json_path
        self._dir = Path(json_path).resolve().parent
        self.model_config = self._load()
        self.system = SystemSpec.from_toml(
            self._resolve(self.model_config["system"])
        )
        self._store_hyperparams()

    def _load(self) -> dict[str, Any]:
        path = Path(self.json_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file {self.json_path} not found")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _resolve(self, rel: str) -> str:
        """
        Resolve a config-relative path against the config file's directory.
        """

        p = Path(rel)

        return str(p if p.is_absolute() else (self._dir / p).resolve())

    def _expand_targets(
        self, targets: dict[str, dict]
    ) -> list[dict[str, float]]:
        """
        Expand target channels (axis-major) into a list of dicts.

        Each target selects a prediction form, resolved to a dataset column
        suffix: 'delta' -> '_delta' (next - current) or 'absolute' -> '_nxt'.
        'form' defaults to 'delta'; the legacy 'predict: "next"' key means
        'absolute'.
        """

        suffixes = {"delta": "_delta", "absolute": "_nxt"}
        out: list[dict[str, float]] = []

        for axis in self.system.axes:
            for name, spec in targets.items():
                ch = self.system.channel(name)
                label = ch.label(axis if ch.per_axis else None)
                form = self._target_form(name, spec)
                out.append({f"{label}{suffixes[form]}": spec.get("weight", 1)})

        return out

    @staticmethod
    def _target_form(name: str, spec: dict) -> str:
        """
        Resolve a target's prediction form ('delta' or 'absolute').
        """

        if "form" in spec:
            form = spec["form"]
            if form not in ("delta", "absolute"):
                raise ValueError(
                    f"Target '{name}': form must be 'delta' or 'absolute', "
                    f"got {form!r}"
                )

            return form
        # Legacy: 'predict: "next"' selected the absolute next-state column.
        if spec.get("predict") == "next":
            return "absolute"

        return "delta"

    def _store_hyperparams(self) -> None:
        cfg = self.model_config
        run = cfg.get("run", {})
        arch = cfg["architecture"]
        train = cfg["training"]
        loader = train.get("dataloader", {})

        # Scaffolding / run
        self.model_name: str = cfg["model_name"]
        out_dir = self._resolve(run.get("out_dir", "runs"))
        # Each artifact owns a directory so its checkpoint, run history and
        # eval plots do not collide with other models under a shared out_dir.
        model_dir = f"{out_dir}/{self.model_name}"
        self.m_save_dir = f"{model_dir}/{self.model_name}.pth"
        self.logging_dir = f"{model_dir}/history"
        self.eval_dir = f"{model_dir}/eval"
        self.seed = run.get("seed", 42)
        self.do_verb_log = run.get("verbose_logging", True)
        # torch.compile the model for training (applied only on CUDA).
        self.do_compile = run.get("compile", True)
        # torch.compile mode: "default" fuses kernels; "reduce-overhead"
        # captures a CUDA graph, collapsing per-kernel launch overhead (the
        # win when host-launch bound on a fast GPU). The rollout path reads
        # its own from the rollout block.
        self.compile_mode = run.get("compile_mode", "default")
        # Run the temporal-conv stack as channels-last (NHWC) conv2d so cuDNN
        # skips the NCHW<->NHWC transpose around every conv. Numerically
        # equivalent; the rollout path reads its own flag from the rollout
        # block.
        self.channels_last = run.get("channels_last", True)
        # Optional rollout-training block; absent means one-step training.
        self.rollout: dict[str, Any] | None = cfg.get("rollout")
        # Optional checkpoint to initialise weights from (warm start /
        # fine-tuning).
        init = run.get("init_checkpoint")
        self.init_checkpoint: str | None = self._resolve(init) if init else None

        # Data
        self.datafile_dir = self._resolve(cfg["data"])
        self.train_ratio = train.get("train_ratio", 0.8)
        self.val_ratio = train.get("validation_ratio", 0.1)

        # Dataloader performance knobs
        self.do_dataloader_auto_tune = loader.get("auto_tune", False)
        self.p_cpu_util = loader.get("cpu_util", 80)
        self.num_workers = loader.get("num_workers", 0)
        self.prefetch_factor = loader.get("prefetch_factor", 2)

        # Architecture + I/O (channel names -> concrete dataset labels via
        # the system)
        self.hidden_layers = arch["hidden_layers"]
        self.window_size = arch.get("window_size", 1)
        if self.window_size < 1:
            raise ValueError(f"{self.window_size=} must be >= 1")
        # Channel names (roles) kept alongside the expanded labels: the
        # rollout uses them to derive which columns are state/command from
        # the SystemSpec.
        self.input_channels = list(cfg["inputs"])
        self.target_channels = list(cfg["targets"].keys())
        self.input_params = self.system.labels(cfg["inputs"])
        self.target_params = self._expand_targets(cfg["targets"])

        self.input_size = list(self.hidden_layers[0].values())[0][0]
        self.target_size = list(self.hidden_layers[-1].values())[0][-1]
        n_in, n_tgt = len(self.input_params), len(self.target_params)
        if n_in == 0 or n_tgt == 0:
            raise ValueError(
                "Inputs and targets must each expand to >= 1 channel"
            )
        if self.input_size != n_in:
            raise ValueError(
                f"Model input size ({self.input_size}) must match the expanded "
                f"inputs ({n_in})"
            )
        if self.target_size != n_tgt:
            raise ValueError(
                f"Model output size ({self.target_size}) must match the "
                f"expanded targets ({n_tgt})"
            )

        # Training
        self.batch_size = train["batch_size"]
        self.max_epochs = train.get("max_epochs", 1000)
        self.patience = train.get("patience", 100)
        # Relative early-stopping threshold: an epoch must beat the best by
        # this fraction to count (robust across loss magnitudes; 0 saves any
        # improvement).
        self.min_delta = train.get("min_delta", 1e-4)
        self.lr_rate = train["learning_rate"]
        self.accum_steps = train.get("accumulation_steps", 1)
        self.dtype = getattr(torch, train.get("training_dtype", "float32"))
        self.loss_function = resolve_class(
            train["loss_function"], module_paths=(_LOSS_MODULE, "torch.nn")
        )
        self.optimiser = getattr(torch.optim, train["optimiser"])
        self.grad_scaler = getattr(torch.amp, train["grad_scaler"])

        # Testing
        self.display_no = train.get("test_display_num", 3)
