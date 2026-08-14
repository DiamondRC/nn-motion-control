import logging
import os
import shutil
import time
from collections.abc import Sized
from datetime import datetime
from typing import cast

import torch

from nn_motion_control.core.checkpoints import (
    load_checkpoint_bundle,
    state_dict_from,
)
from nn_motion_control.core.config import RunConfiguration
from nn_motion_control.core.seeding import seed_everything
from nn_motion_control.data import build_time_series_splits
from nn_motion_control.eval.evaluate import Evaluator
from nn_motion_control.models.builder import JsonModel
from nn_motion_control.models.channels_last import is_channels_last_able
from nn_motion_control.plant.plant import (
    Plant,
    RolloutLayout,
    rollout_splits_from_config,
)
from nn_motion_control.training.logging_setup import ModelLogger
from nn_motion_control.training.rollout import RolloutTrainer
from nn_motion_control.training.trainer import Trainer, config_overrides


class CompleteRun:
    def __init__(self, model_cfg_pth):
        self.m = RunConfiguration(model_cfg_pth)
        self.is_rollout = self.m.rollout is not None

        self._setup_run()
        self._set_seed()
        self._cuda_profiling()
        self.logger.info("Starting Run...")

        self._create_model()
        self._create_run_dataloaders()
        self._run_trainer()
        self._train_model()
        self._test_model()
        self._end_run()

    def _setup_run(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.start_time = time.perf_counter()

        ModelLogger(self.m.logging_dir, self.timestamp)
        self.logger = logging.getLogger(os.path.basename(__file__))

        # Snapshot the exact run config alongside the log so past runs
        # are reproducible.
        try:
            shutil.copy(
                self.m.json_path,
                os.path.join(
                    self.m.logging_dir, f"{self.timestamp}.config.json"
                ),
            )
        except OSError as exc:
            self.logger.warning(f"Could not snapshot run config: {exc}")

    def _set_seed(self):
        """
        Seed every RNG so weight init, dropout and shuffling are reproducible.
        """

        seed_everything(self.m.seed)
        self.logger.info(f"Global RNG seeded with {self.m.seed}")

    def _cuda_profiling(self):
        if self.m.do_verb_log:
            self.logger.debug(f"Torch version: {torch.__version__}")

        if torch.cuda.is_available():
            if self.m.do_verb_log:
                self.logger.debug(f"CUDA version: {torch.version.cuda}")
                self.logger.debug(
                    f"Using CUDA device {torch.cuda.get_device_name(0)}"
                )
            self.device = "cuda"

            # Fixed input shapes -> let cuDNN autotune
            # allow TF32 on matmuls.
            torch.backends.cudnn.benchmark = True
            # 'high' (TF32) over 'medium': bf16 autocast already covers
            # the bulk of compute, so the fp32 matmuls this setting
            # affects are a tiny edge case not worth trading away.
            torch.set_float32_matmul_precision("high")
        else:
            self.logger.warning(
                "CUDA not available, using CPU (not recommended)"
            )
            self.device = "cpu"

            # float16 autocast is unsupported/unreliable on CPU - bf16
            # has fp32 range and works - so fall back to it
            # rather than silently misbehaving.
            if self.m.dtype == torch.float16:
                self.logger.warning(
                    "training_dtype float16 is unreliable on CPU, "
                    "using bfloat16"
                )
                self.m.dtype = torch.bfloat16

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

        # Subsample redundant overlapping window starts
        compiling = self.m.do_compile and self.device == "cuda"
        self.dloader = rollout_splits_from_config(
            self.m,
            self.rollout_layout,
            max_horizon=max_h,
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
        self.model = JsonModel(config=self.m)
        self.model.to(self.device)

        # Optional warm start: load weights from a prior checkpoint
        # before training.
        if self.m.init_checkpoint:
            bundle = load_checkpoint_bundle(self.m.init_checkpoint, self.device)
            self.model.load_state_dict(state_dict_from(bundle))
            self.logger.info(
                "Initialised weights from %s", self.m.init_checkpoint
            )

        # Channels-last forward: run convs as NHWC conv2d so cuDNN's
        # tensor-core kernels skip the NCHW<->NHWC transpose
        # that otherwise dominates each conv on the DGXSpark.
        if (
            not self.is_rollout
            and self.device == "cuda"
            and self.m.channels_last
            and is_channels_last_able(self.model.network)
        ):
            self.model.enable_channels_last()
            self.logger.info("Channels-last conv2d single-step forward enabled")

        # The rollout path compiles its own one-step view in
        # _run_rollout_trainer - skip compile there.
        if self.m.do_compile and self.device == "cuda" and not self.is_rollout:
            mode = self.m.compile_mode
            self.logger.info(
                "Compiling model with torch.compile (mode=%s)...", mode
            )
            self.train_model = torch.compile(self.model, mode=mode)
        else:
            self.train_model = self.model

    def _run_trainer(self):
        if self.is_rollout:
            return self._run_rollout_trainer()

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

        # Channels-last forward: run the convs as NHWC conv2d so
        # cuDNN's tensor-core kernels skip the NCHW<->NHWC transposes
        # that otherwise dominate the memory-bound rollout.
        if (
            self.device == "cuda"
            and ro.get("channels_last", True)
            and is_channels_last_able(self.model.network)
        ):
            self.model.enable_channels_last()
            self.logger.info("Channels-last conv2d rollout forward enabled")

        # Compile the one-step model, but not the unroll
        step_model = self.model
        is_streaming = ro.get("rollout_kind", "windowed") == "streaming"

        if self.m.do_compile and self.device == "cuda" and not is_streaming:
            mode = ro.get("compile_mode", "default")
            self.logger.info(
                "Compiling rollout step (torch.compile mode=%s)", mode
            )
            # torch.compile returns an OptimizedModule (an nn.Module)
            # but is typed as a bare callable, cast so Plant still
            # sees a Module.
            step_model = cast(
                torch.nn.Module, torch.compile(self.model, mode=mode)
            )

        plant = Plant(
            step_model,
            ni.input_stats,
            ni.target_stats,
            self.rollout_layout,
            self.device,
            rollout_kind=ro.get("rollout_kind", "windowed"),
        )

        # Curriculum and loss knobs the config may override
        overrides = config_overrides(
            ro,
            {
                "curriculum_start": int,
                "curriculum_ramp": int,
                "ss_start": float,
                "ss_end": float,
                "ss_ramp": int,
                "step_weight": float,
                "auto_balance": bool,
                "axis_weights": list,
            },
        )

        # The config names horizon weighting 'horizon_weighting',
        # the trainer 'hw_mode'.
        if "horizon_weighting" in ro:
            overrides["hw_mode"] = str(ro["horizon_weighting"])

        self.trainer = RolloutTrainer(
            plant,
            max_horizon=int(ro["horizon"]),
            **overrides,
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

    def _test_model(self):
        if self.is_rollout:
            # The one-step evaluator cannot consume rollout batches.
            self.logger.info(
                "Rollout training complete. Run the error-vs-horizon "
                "eval separately."
            )
            return  # awesome python

        train_losses, val_losses, early_stop_epoch = (
            self.trainer.get_training_info()
        )

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
        end_time = time.perf_counter()
        elapsed = end_time - self.start_time

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        self.hms = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".rstrip(
            "0"
        ).rstrip(":")

        self.logger.debug(
            f"Model training and testing took {self.hms} (Hours/Mins/Secs)."
        )

        # Rollout eval (error-vs-horizon) is a separate step,
        # so only one-step runs have loss curves to plot here.
        if not self.is_rollout:
            self.tester.plot_losses()

        self.logger.info("Finished run.")
