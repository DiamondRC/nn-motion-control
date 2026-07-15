import logging
import os
import random
import shutil
import time
from collections.abc import Sized
from datetime import datetime
from typing import cast

import numpy as np
import torch

from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.data import build_time_series_splits
from nn_motion_control.eval.evaluate import Evaluator
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.training.logging_setup import ModelLogger
from nn_motion_control.training.trainer import Trainer


class CompleteRun:
    def __init__(self, model_cfg_pth):
        # Load run config
        self.m = RunConfiguration(model_cfg_pth)

        # Begin logging process
        self._setup_run()
        self._set_seed()
        self._cuda_profiling()
        self.logger.info("Starting Run...")

        # Instantiate model architecture
        self._create_model()

        # Create dataloaders
        self._create_run_dataloaders()

        # Train model
        self._run_trainer()

        # Train the model
        self._train_model()

        # Test the trained model
        self._test_model()

        # Finish
        self._end_run()

    def _setup_run(self):
        # Start timing model runtime
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.start_time = time.perf_counter()

        # Begin logging
        ModelLogger(self.m.logging_dir, self.timestamp)
        self.logger = logging.getLogger(os.path.basename(__file__))

        # Snapshot the exact run config alongside the log so past runs are reproducible
        # (the source JSON is mutable and git-tracked).
        try:
            shutil.copy(
                self.m.json_path,
                os.path.join(self.m.logging_dir, f"{self.timestamp}.config.json"),
            )
        except OSError as exc:
            self.logger.warning(f"Could not snapshot run config: {exc}")

    def _set_seed(self):
        """Seed every RNG so weight init, dropout and shuffling are reproducible."""
        seed = self.m.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"Global RNG seeded with {seed}")

    def _cuda_profiling(self):
        if self.m.do_verb_log:
            self.logger.debug(f"Torch version: {torch.__version__}")

        if torch.cuda.is_available():
            if self.m.do_verb_log:
                self.logger.debug(f"CUDA version: {torch.version.cuda}")
                self.logger.debug(f"Using CUDA device {torch.cuda.get_device_name(0)}")
            self.device = "cuda"
        else:
            # raise Exception("CUDA not available")
            self.logger.warning("CUDA not available, using CPU (not recommended)")
            self.device = "cpu"

    def _create_run_dataloaders(self):
        self.dloader = build_time_series_splits(
            h5_path=self.m.datafile_dir,
            allowed_inputs=self.m.input_params,
            allowed_targets=self.m.target_params,
            window_size=self.m.window_size,
            train_ratio=self.m.train_ratio,
            val_ratio=self.m.val_ratio,
            seed=self.m.seed,
            batch_size=self.m.batch_size,
            num_workers=self.m.num_workers,
            cpu_core_util=self.m.p_cpu_util,
            prefetch_factor=self.m.prefetch_factor,
            persistent_workers=False,
            pin_memory=True,
            auto_tune_workers=self.m.do_dataloader_auto_tune,
            enable_logging=self.m.do_verb_log,
            training_dtype=self.m.dtype,
            load_into_ram=True,
        )
        self.logger.info(
            "Splits -> train:%d val:%d test:%d",
            len(cast(Sized, self.dloader.trn_loader.dataset)),
            len(cast(Sized, self.dloader.val_loader.dataset)),
            len(cast(Sized, self.dloader.tst_loader.dataset)),
        )

    def _create_model(self):
        # TODO - model selection
        self.model = JsonModel(config=self.m)
        self.model.to(self.device)

    def _run_trainer(self):
        # Instantiate trainer with model, dataloaders, and training hyperparams
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.dloader.trn_loader,
            val_loader=self.dloader.val_loader,
            device=self.device,
            scaler_class=self.m.grad_scaler,
            optimizer_class=self.m.optimiser,
            criterion_class=self.m.loss_function,
            node_info=self.dloader.node_info,
            max_epochs=self.m.max_epochs,
            learning_rate=self.m.lr_rate,
            patience=self.m.patience,
            min_delta=self.m.min_delta,
            model_name=self.m.model_name,
            save_path=self.m.m_save_dir,
            logging=self.m.do_verb_log,
            accumulation_steps=self.m.accum_steps,
            training_dtype=self.m.dtype,
            window_size=self.m.window_size,
            seed=self.m.seed,
        )

    def _train_model(self):
        self.logger.info("Starting training loop...")
        self.trainer.train()
        self.logger.info("Training loop complete.")
        # self.dloader.cleanup() TODO - re-enable

    def _test_model(self):
        # Grab training info
        train_losses, val_losses, early_stop_epoch = self.trainer.get_training_info()

        self.tester = Evaluator(
            model=self.model,
            test_loader=self.dloader.tst_loader,
            training_losses=train_losses,
            validation_losses=val_losses,
            criterion_class=self.m.loss_function,
            early_stop_epoch=early_stop_epoch,
            node_info=self.dloader.node_info,
            allowed_targets=self.m.target_params,
            device=self.device,
            save_path=self.m.m_save_dir,
            plot_path=self.m.logging_dir,
            plot_name=self.timestamp,
            test_display_num=self.m.display_no,
            training_dtype=self.m.dtype,
        )

        self.tester.test()

    def _end_run(self):
        # Complete timing measurement
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time

        # Format nicely for long runs
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        self.hms = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".rstrip("0").rstrip(":")

        self.logger.debug(
            f"Model training and testing took {self.hms} (Hours/Mins/Secs)."
        )

        # Display the losses
        self.tester.plot_losses()

        self.logger.info("Finished run.")
