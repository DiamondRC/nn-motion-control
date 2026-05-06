import json
import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim


def resolve_class(name: str, module_paths: tuple[(str, ...)] = ()) -> type[nn.Module]:
    """
    Used to match custom loss classes.
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
    Handles incoming model definitions from json files and
    extracts hyperparams from the provided model config.
    """

    def __init__(self, json_path):
        self.json_path = json_path
        self.model_config = self.get_config()
        self.run_config = self._store_hyperparams()

    def get_config(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Config file {self.json_path} not found")
        with open(self.json_path, encoding="utf-8") as f:
            self.model_config = json.load(f)
            return self.model_config

    def _store_hyperparams(self):
        """
        Grabs all hyperparams from the json dict.
        """
        # Scaffolding
        self.model_name = self.model_config["model_name"]
        self.m_save_dir = f"{self.model_config['model_save_dir']}/{self.model_name}.pth"
        self.logging_dir = self.model_config["logging_dir"]
        self.seed = self.model_config["seed"]
        self.do_verb_log = self.model_config["verbose_logging"]

        # Performance
        self.do_dataloader_auto_tune = self.model_config["auto_tune_dataloader"]
        self.p_cpu_util = self.model_config["percent_cpu_core_util"]
        self.num_workers = self.model_config["num_workers"]
        self.prefetch_factor = self.model_config["prefetch_factor"]
        self.accum_steps = self.model_config["accumulation_steps"]

        # Data
        self.datafile_dir = self.model_config["data_dir"]
        self.train_ratio = self.model_config["train_ratio"]
        self.val_ratio = self.model_config["validation_ratio"]

        # Training
        self.hidden_layers = self.model_config["hidden_layers"]
        self.window_size = self.model_config.get("window_size", 1)
        if self.window_size < 1:
            raise ValueError(f"{self.window_size=} must be >= 1")
        self.input_params = self.model_config["input_params"]
        self.target_params = self.model_config["target_params"]
        self.input_size = list(self.hidden_layers[0].values())[0][0]
        self.target_size = list(self.hidden_layers[-1].values())[0][-1]
        input_size = len(self.input_params)
        target_size = len(self.target_params)
        if input_size == 0:
            raise ValueError(f"{input_size=} must be >= 1")
        elif target_size == 0:
            raise ValueError(f"{target_size=} must be >= 1")
        elif self.input_size != input_size:
            raise ValueError(
                f"The specified model input size ({self.input_size}) "
                f"must match the size of the model inputs ({input_size})"
            )
        elif self.target_size != target_size:
            raise ValueError(
                f"The specified model out size ({self.target_size}) "
                f"must match the size of the model outputs ({target_size})"
            )

        self.batch_size = self.model_config["batch_size"]
        self.max_epochs = self.model_config["max_epochs"]
        self.patience = self.model_config["patience"]
        self.min_delta = self.model_config["min_delta"]
        self.lr_rate = self.model_config["learning_rate"]
        self.dtype = getattr(torch, self.model_config["training_dtype"])
        self.loss_function = resolve_class(
            self.model_config["loss_function"],
            module_paths=(
                "deltabot_nn_controller.model_zoo.losses.weighted_mse",
                "torch.nn",
            ),
        )
        self.optimiser = getattr(torch.optim, self.model_config["optimiser"])
        self.grad_scaler = getattr(torch.amp, self.model_config["grad_scaler"])

        # Testing
        self.display_no = self.model_config["test_display_num"]
