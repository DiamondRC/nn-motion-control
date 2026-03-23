import logging
import logging.config
import os
import time
from datetime import datetime

import torch
from model_utils.dataloader import PVT2DACDataset
from model_utils.setup_logging import setup_logging
from model_utils.test_saved_model import TestModel
from model_utils.training_loop import Trainer
from model_zoo.json_manager import load_config
from model_zoo.models.mlp import MLP
from torch.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import Adam

os.system("clear")

# -------------------------------
# Hyperparams
# -------------------------------

DO_VERBOSE_LOGGING = True
DATAFILE = "./data/pvt_to_dac_training.h5"

# BATCH_SIZE = 131072
BATCH_SIZE = 1028 * 8
PERCENTAGE_CPU_CORE_UTIL = 80
AUTO_TUNE_DATALOADER = True
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MAX_EPOCHS = 5000
PATIENCE = 200
MIN_DELTA = 1e-4
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1  # Set >1 to simulate larger batch sizes on limited hardware
MODEL_SAVE_PATH = (
    "src/deltabot_nn_controller/model_zoo/models/model_states/best_model.pth"
)
SEED = 42
INPUT_SIZE = 16  # controls dummy test size
WINDOW_SIZE = 1  # controls dummy window size
DISPLAY_TEST_NUM = 3  # Displays this many test points
LOGGING_PATH = "logs/"


# -------------------------------
# Begin logging
# -------------------------------

# Start timing model runtime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
start_time = time.perf_counter()

# Begin logging
setup_logging(LOGGING_PATH, timestamp)
logger = logging.getLogger(__name__)


# -------------------------------
# CUDA Logging
# -------------------------------

if DO_VERBOSE_LOGGING:
    logger.debug(f"Torch version: {torch.__version__}")

if torch.cuda.is_available():
    if DO_VERBOSE_LOGGING:
        logger.debug(f"CUDA version: {torch.version.cuda}")
        logger.debug(f"Using CUDA device {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    # raise Exception("CUDA not available")
    logger.warning("CUDA not available, using CPU (not recommended)")
    DEVICE = "cpu"


# -------------------------------
# Create dataloaders
# -------------------------------

dataset = PVT2DACDataset(
    h5_path=DATAFILE,
    batch_size=BATCH_SIZE,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    do_auto_tune_dataloader=AUTO_TUNE_DATALOADER,
    cpu_core_util=PERCENTAGE_CPU_CORE_UTIL,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    seed=SEED,
    logging=DO_VERBOSE_LOGGING,
    window_size=WINDOW_SIZE,
)
train_loader, val_loader, test_loader = (
    dataset.train_loader,
    dataset.val_loader,
    dataset.test_loader,
)

# -------------------------------
# Instantiate model architecture
# -------------------------------

# TODO - model selection
model = MLP(logging=DO_VERBOSE_LOGGING, config=load_config())
model.to(DEVICE)


# -------------------------------
# Train loop
# -------------------------------

# Instantiate trainer with model, dataloaders, and training hyperparams
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    scaler_class=GradScaler,
    optimizer_class=Adam,
    criterion_class=MSELoss,
    max_epochs=MAX_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    save_path=MODEL_SAVE_PATH,
    logging=DO_VERBOSE_LOGGING,
    accumulation_steps=ACCUMULATION_STEPS,
)

# Pass dummy data through model to verify forward pass and log initial stats
if DO_VERBOSE_LOGGING:
    logger.debug("Profiling model with dummy input...")
    dummy_input = torch.randn(1, INPUT_SIZE * WINDOW_SIZE).to(DEVICE)
    logger.debug(f"Dummy input shape: {dummy_input.shape}")
    with torch.no_grad():
        model_dummy_output = model(dummy_input)
    logger.debug("Profiling model with dummy input... DONE")

    logger.debug(f"Model sent to device: {next(model.parameters()).device}")

    logger.debug(
        f"First layer sample weights: "
        f"{model.network[0].weight.flatten()[:5].cpu().detach().numpy()}"
    )
    logger.debug("First layer weight range: ")
    logger.debug(
        f"[{model.network[0].weight.min():.3f}, {model.network[0].weight.max():.3f}]"
    )
    logger.debug(
        f"Final layer sample weights: "
        f"{model.network[-1].weight.flatten()[:5].cpu().detach().numpy()}"
    )
    logger.debug("Final layer weight range: ")
    logger.debug(
        f"[{model.network[-1].weight.min():.3f}, {model.network[-1].weight.max():.3f}]"
    )

    logger.debug(
        f"Model dummy output range: "
        f"[{model_dummy_output.min():.3f}, {model_dummy_output.max():.3f}]"
    )

# Log input/output ranges for user data to verify normalisation,
# then profile a single batch of data.
if DO_VERBOSE_LOGGING:
    logger.debug("Profiling user data...")
    data_sample, label_sample = next(iter(train_loader.dataset))
    logger.debug(
        f"Data Inputs range: [{data_sample.min():.3f}, {data_sample.max():.3f}]"
    )
    logger.debug(
        f"Data Targets range: [{label_sample.min():.3f}, {label_sample.max():.3f}]"
    )
    logger.debug("Profiling user data... DONE")
    trainer.profile_one_batch()

logger.info("Starting training loop...")
trainer.train()
logger.info("Training loop complete.")
dataset.cleanup_dataloaders()


# -------------------------------
# Test Model
# -------------------------------

# Grab training info
train_losses, val_losses, early_stop_epoch = trainer.get_training_info()
norm_consts = dataset.get_normalisation_params()

tester = TestModel(
    model=model,
    test_loader=test_loader,
    training_losses=train_losses,
    validation_losses=val_losses,
    criterion_class=MSELoss,
    early_stop_epoch=early_stop_epoch,
    normalisation_consts=norm_consts,
    device=DEVICE,
    save_path=MODEL_SAVE_PATH,
    plot_path=LOGGING_PATH,
    plot_name=timestamp,
    logging=DO_VERBOSE_LOGGING,
    test_display_num=DISPLAY_TEST_NUM,
)

tester.test()

# -------------------------------
# Complete Timing
# -------------------------------

# Complete timing measurement
end_time = time.perf_counter()
elapsed = end_time - start_time

# Format nicely for long runs
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60
hms = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".rstrip("0").rstrip(":")

logger.debug(f"Model training and testing took {hms} (Hours/Mins/Secs).")


# -------------------------------
# Finish Execution
# -------------------------------

# Display the losses
tester.plot_losses()

logger.info("Finished model execution.")
