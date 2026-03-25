import logging
import time
from datetime import datetime

import torch
from torch.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import Adam

from deltabot_nn_controller.dataset_utils.dataloader import PVT2DACDataset
from deltabot_nn_controller.model_utils.instantiate_model import RunConfiguration
from deltabot_nn_controller.model_utils.setup_logging import ModelLogger
from deltabot_nn_controller.model_utils.test_saved_model import TestModel
from deltabot_nn_controller.model_utils.training_loop import Trainer
from deltabot_nn_controller.model_zoo.models.mlp import MLP


class CompleteRun:
    def __init__(self, model_cfg_pth):
        # Load run config
        self.m = RunConfiguration(model_cfg_pth)

        # Begin logging process
        self._setup_run()
        self._cuda_profiling()

        # Create dataloaders
        self._create_run_dataloaders()

        # Instantiate model architecture
        self._create_model()

        # Train model
        self._run_trainer()

        # Pass dummy data through the model
        if self.m.do_verb_log:
            self._dummy_test()

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
        self.logger = logging.getLogger(__name__)

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
        self.dataset = PVT2DACDataset(
            h5_path=self.m.datafile_dir,
            batch_size=self.m.batch_size,
            train_ratio=self.m.train_ratio,
            val_ratio=self.m.val_ratio,
            do_auto_tune_dataloader=self.m.do_dataloader_auto_tune,
            cpu_core_util=self.m.p_cpu_util,
            num_workers=self.m.num_workers,
            prefetch_factor=self.m.prefetch_factor,
            seed=self.m.seed,
            logging=self.m.do_verb_log,
            window_size=self.m.window_size,
        )
        self.tra_loader, self.val_loader, self.tst_loader = (
            self.dataset.train_loader,
            self.dataset.val_loader,
            self.dataset.test_loader,
        )

    def _create_model(self):
        # TODO - model selection
        self.model = MLP(logging=self.m.do_verb_log, config=self.m.model_config)
        self.model.to(self.device)

    def _run_trainer(self):
        # Instantiate trainer with model, dataloaders, and training hyperparams
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.tra_loader,
            val_loader=self.val_loader,
            device=self.device,
            scaler_class=GradScaler,  # TODO
            optimizer_class=Adam,  # allow configing
            criterion_class=MSELoss,  # of these
            max_epochs=self.m.max_epochs,
            learning_rate=self.m.lr_rate,
            patience=self.m.patience,
            min_delta=self.m.min_delta,
            model_name=self.m.model_name,
            save_path=self.m.m_save_dir,
            logging=self.m.do_verb_log,
            accumulation_steps=self.m.accum_steps,
        )

    def _dummy_test(self):
        # Pass dummy data through model to verify forward pass and log initial stats
        self.logger.debug("Profiling model with dummy input...")
        dummy_input = torch.randn(1, self.m.input_size * self.m.window_size).to(
            self.device
        )
        self.logger.debug(f"Dummy input shape: {dummy_input.shape}")

        with torch.no_grad():
            model_dummy_output = self.model(dummy_input)
        self.logger.debug("Profiling model with dummy input... DONE")

        self.logger.debug(
            f"Model sent to device: {next(self.model.parameters()).device}"
        )
        self.logger.debug(
            f"First layer sample weights: "
            f"{self.model.network[0].weight.flatten()[:5].cpu().detach().numpy()}"
        )
        self.logger.debug("First layer weight range: ")
        self.logger.debug(
            f"[{self.model.network[0].weight.min():.3f}, {self.model.network[0].weight.max():.3f}]"  # noqa: E501
        )
        self.logger.debug(
            f"Final layer sample weights: "
            f"{self.model.network[-1].weight.flatten()[:5].cpu().detach().numpy()}"
        )
        self.logger.debug("Final layer weight range: ")
        self.logger.debug(
            f"[{self.model.network[-1].weight.min():.3f}, {self.model.network[-1].weight.max():.3f}]"  # noqa: E501
        )

        self.logger.debug(
            f"Model dummy output range: "
            f"[{model_dummy_output.min():.3f}, {model_dummy_output.max():.3f}]"
        )

        # Log input/output ranges for user data to verify normalisation,
        # then profile a single batch of data.
        self.logger.debug("Profiling user data...")
        data_sample, label_sample = next(iter(self.tra_loader.dataset))
        self.logger.debug(
            f"Data Inputs range: [{data_sample.min():.3f}, {data_sample.max():.3f}]"
        )
        self.logger.debug(
            f"Data Targets range: [{label_sample.min():.3f}, {label_sample.max():.3f}]"
        )
        self.logger.debug("Profiling user data... DONE")
        self.trainer.profile_one_batch()

    def _train_model(self):
        self.logger.info("Starting training loop...")
        self.trainer.train()
        self.logger.info("Training loop complete.")
        self.dataset.cleanup_dataloaders()

    def _test_model(self):
        # Grab training info
        train_losses, val_losses, early_stop_epoch = self.trainer.get_training_info()
        norm_consts = self.dataset.get_normalisation_params()

        self.tester = TestModel(
            model=self.model,
            test_loader=self.tst_loader,
            training_losses=train_losses,
            validation_losses=val_losses,
            criterion_class=MSELoss,  # TODO - make this configurable
            early_stop_epoch=early_stop_epoch,
            normalisation_consts=norm_consts,
            device=self.device,
            save_path=self.m.m_save_dir,
            plot_path=self.m.logging_dir,
            plot_name=self.timestamp,
            logging=self.m.do_verb_log,
            test_display_num=self.m.display_no,
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

        self.logger.info("Finished model execution.")
