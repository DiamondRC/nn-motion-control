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
from nn_motion_control.data import build_rollout_splits, build_time_series_splits
from nn_motion_control.eval.evaluate import Evaluator
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.channels_last import is_channels_last_able
from nn_motion_control.plant.plant import Plant, RolloutLayout
from nn_motion_control.training.logging_setup import ModelLogger
from nn_motion_control.training.rollout import RolloutTrainer
from nn_motion_control.training.trainer import Trainer


class CompleteRun:
    def __init__(self, model_cfg_pth):
        # Load run config
        self.m = RunConfiguration(model_cfg_pth)
        self.is_rollout = self.m.rollout is not None

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
        """
        Seed every RNG so weight init, dropout and shuffling are reproducible.
        """

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

            # Fixed input shapes -> let cuDNN autotune; allow TF32 on matmuls.
            torch.backends.cudnn.benchmark = True
            # High for correctness.
            # Already have bf16 autocast,
            # this is a tiny edge case so we might as well refuse a precision loss.
            torch.set_float32_matmul_precision("high")
        else:
            # raise Exception("CUDA not available")
            self.logger.warning("CUDA not available, using CPU (not recommended)")
            self.device = "cpu"

    def _create_run_dataloaders(self):
        if self.is_rollout:
            return self._create_rollout_dataloaders()
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
            device=self.device,
        )
        self.logger.info(
            "Splits -> train:%d val:%d test:%d",
            len(cast(Sized, self.dloader.trn_loader.dataset)),
            len(cast(Sized, self.dloader.val_loader.dataset)),
            len(cast(Sized, self.dloader.tst_loader.dataset)),
        )

    def _create_rollout_dataloaders(self):
        ro = self.m.rollout
        assert ro is not None
        self.rollout_layout = RolloutLayout.from_config(self.m)
        max_h = int(ro["horizon"])
        # Subsample redundant overlapping window starts; validation strides harder since
        # it now free-runs the full horizon every epoch. Compiling the step needs a
        # constant batch shape, so drop the ragged last batch when compile is on.
        compiling = self.m.do_compile and self.device == "cuda"
        self.dloader = build_rollout_splits(
            h5_path=self.m.datafile_dir,
            allowed_inputs=self.m.input_params,
            allowed_targets=self.m.target_params,
            window_size=self.m.window_size,
            max_horizon=max_h,
            pos_cols=self.rollout_layout.pos_cols,
            dac_cols=self.rollout_layout.dac_cols,
            train_ratio=self.m.train_ratio,
            val_ratio=self.m.val_ratio,
            training_dtype=self.m.dtype,
            batch_size=self.m.batch_size,
            seed=self.m.seed,
            device=self.device,
            train_start_stride=int(ro.get("start_stride", 1)),
            val_start_stride=int(ro.get("val_start_stride", 1)),
            drop_last=compiling,
        )
        self.logger.info(
            "Rollout splits (H_max=%d) -> train:%d val:%d test:%d",
            max_h,
            len(cast(Sized, self.dloader.trn_loader.dataset)),
            len(cast(Sized, self.dloader.val_loader.dataset)),
            len(cast(Sized, self.dloader.tst_loader.dataset)),
        )

    def _create_model(self):
        # TODO - model selection
        self.model = JsonModel(config=self.m)
        self.model.to(self.device)

        # Optional warm start: load weights from a prior checkpoint before training.
        if self.m.init_checkpoint:
            bundle = torch.load(self.m.init_checkpoint, map_location=self.device)
            state = (
                bundle["model_state_dict"]
                if isinstance(bundle, dict) and "model_state_dict" in bundle
                else bundle
            )
            self.model.load_state_dict(state)
            self.logger.info("Initialised weights from %s", self.m.init_checkpoint)

        # Train through a compiled view when enabled.
        # The rollout path compiles its own one-step view in _run_rollout_trainer,
        # so skip this compile there.
        if self.m.do_compile and self.device == "cuda" and not self.is_rollout:
            self.logger.info("Compiling model with torch.compile...")
            self.train_model = torch.compile(self.model)
        else:
            self.train_model = self.model

    def _run_trainer(self):
        if self.is_rollout:
            return self._run_rollout_trainer()
        # Instantiate trainer with model, dataloaders and training hyperparams
        self.trainer = Trainer(
            model=self.train_model,
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

    def _run_rollout_trainer(self):
        ro = self.m.rollout
        assert ro is not None
        ni = self.dloader.node_info
        # Channels-last forward: run the convs as NHWC conv2d so cuDNN's tensor-core
        # kernels skip the NCHW<->NHWC transposes that otherwise dominate the
        # memory-bound rollout (~40% of runtime). Same maths; falls back for non-TCN.
        if (
            self.device == "cuda"
            and ro.get("channels_last", True)
            and is_channels_last_able(self.model.network)
        ):
            self.model.enable_channels_last()
            self.logger.info("Channels-last conv2d rollout forward enabled")
        # Compile the one-step model, not the unroll: each call sees a constant
        # [B, F, W] window regardless of the curriculum horizon, so it is compile-safe
        # and cuts per-step launch overhead.
        # Autograd still composes the H differentiable calls. mode is configurable
        # ("default"; try "reduce-overhead" for CUDA graphs once validated on hardware).
        step_model = self.model
        is_streaming = ro.get("rollout_kind", "windowed") == "streaming"
        if self.m.do_compile and self.device == "cuda" and not is_streaming:
            mode = ro.get("compile_mode", "default")
            self.logger.info("Compiling rollout step (torch.compile mode=%s)", mode)
            # torch.compile returns an OptimizedModule (an nn.Module) but is typed as a
            # bare callable; cast so Plant still sees a Module.
            step_model = cast(torch.nn.Module, torch.compile(self.model, mode=mode))
        plant = Plant(
            step_model,
            ni.input_stats,
            ni.target_stats,
            self.rollout_layout,
            self.device,
            rollout_kind=ro.get("rollout_kind", "windowed"),
        )
        self.trainer = RolloutTrainer(
            plant,
            max_horizon=int(ro["horizon"]),
            curriculum_start=int(ro.get("curriculum_start", 4)),
            curriculum_ramp=int(ro.get("curriculum_ramp", 20)),
            ss_start=float(ro.get("ss_start", 1.0)),
            ss_end=float(ro.get("ss_end", 0.0)),
            ss_ramp=int(ro.get("ss_ramp", 30)),
            step_weight=float(ro.get("step_weight", 0.1)),
            hw_mode=ro.get("horizon_weighting", "uniform"),
            auto_balance=bool(ro.get("auto_balance", False)),
            train_loader=self.dloader.trn_loader,
            val_loader=self.dloader.val_loader,
            device=self.device,
            scaler_class=self.m.grad_scaler,
            optimizer_class=self.m.optimiser,
            criterion_class=self.m.loss_function,
            node_info=ni,
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
        if self.is_rollout:
            # The one-step evaluator cannot consume rollout batches.
            self.logger.info(
                "Rollout training complete; run the error-vs-horizon eval separately."
            )
            return

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

        # Plot the losses (one-step runs only; rollout eval is a separate step).
        if not self.is_rollout:
            self.tester.plot_losses()

        self.logger.info("Finished run.")
