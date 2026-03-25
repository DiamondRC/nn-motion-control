import json
import os


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
        self.input_size = self.model_config["input_size"]
        self.window_size = self.model_config["window_size"]
        self.batch_size = self.model_config["batch_size"]
        self.max_epochs = self.model_config["max_epochs"]
        self.patience = self.model_config["patience"]
        self.min_delta = self.model_config["min_delta"]
        self.lr_rate = self.model_config["learning_rate"]

        # Testing
        self.display_no = self.model_config["test_display_num"]
