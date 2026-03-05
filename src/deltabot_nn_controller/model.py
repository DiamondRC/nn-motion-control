import os

import psutil
import torch
from model_utils.dataloader import PVT2DACDataset
from model_utils.training_loop import Trainer
from model_zoo.json_manager import load_config
from model_zoo.models.mlp import MLP
from torch.amp import GradScaler

# from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from torch.optim import Adam

os.system("clear")

# -------------------------------
# Hyperparams
# -------------------------------

DO_VERBOSE_LOGGING = True
DATAFILE = "./data/pvt_to_dac_training.h5"

# BATCH_SIZE = 131072
BATCH_SIZE = 1028
PERCENTAGE_CPU_CORE_UTIL = 80
AUTO_TUNE_DATALOADER = True
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
MAX_EPOCHS = 20
PATIENCE = 500
MIN_DELTA = 1e-4
LEARNING_RATE = 3e-4
ACCUMULATION_STEPS = 1  # Set >1 to simulate larger batch sizes on limited hardware
MODEL_SAVE_PATH = (
    "src/deltabot_nn_controller/model_zoo/models/model_states/best_model.pth"
)
SEED = 42

# -------------------------------
# Configure system resources
# -------------------------------

# Profile system resources
cpu_count = os.cpu_count()
ram_gb = psutil.virtual_memory().total / (1024**3)
gpu_count = torch.cuda.device_count()

# Heuristic for dataloader workers and prefetching based on system resources
if AUTO_TUNE_DATALOADER:
    NUM_WORKERS = max(int(cpu_count * (PERCENTAGE_CPU_CORE_UTIL / 100)), 8)
    PREFETCH_FACTOR = max(NUM_WORKERS, 4)

    if DO_VERBOSE_LOGGING:
        print(
            f"Auto-detected: {cpu_count} cores, {ram_gb:.1f}GB RAM, {gpu_count} GPU(s)"
        )
    print(f"Using: num_workers={NUM_WORKERS}, prefetch_factor={PREFETCH_FACTOR}")
else:
    if PERCENTAGE_CPU_CORE_UTIL > 90:
        print(
            f"WARNING: Using a high percentage of \
system CPU cores ({PERCENTAGE_CPU_CORE_UTIL})%"
        )
    print(
        f"Using manually set: num_workers={NUM_WORKERS}, \
prefetch_factor={PREFETCH_FACTOR}"
    )


# -------------------------------
# CUDA Logging
# -------------------------------

if DO_VERBOSE_LOGGING:
    print("Torch version: ", torch.__version__)

if torch.cuda.is_available():
    if DO_VERBOSE_LOGGING:
        print("CUDA version: ", torch.version.cuda)
        print(f"Using CUDA device {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    # raise Exception("CUDA not available")
    print("CUDA not available, using CPU (not recommended)")
    DEVICE = "cpu"


# -------------------------------
# Create dataloaders
# -------------------------------

dataset = PVT2DACDataset(h5_path=DATAFILE, logging=DO_VERBOSE_LOGGING)
train_loader, val_loader, test_loader = dataset.get_dataloaders(
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    seed=SEED,
)


# -------------------------------
# Instantiate model architecture
# -------------------------------

# TODO - model selection
model = MLP(config=load_config())
model.to(DEVICE)


# -------------------------------
# Train loop (placeholder)
# -------------------------------

# Instantiate trainer with model, dataloaders, and training hyperparams
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
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
    print("\nProfiling model with dummy input...")
    dummy_input = torch.randn(1, 7).to(DEVICE)
    with torch.no_grad():
        model_dummy_output = model(dummy_input)
        print("Profiling model with dummy input... DONE")

    print(f"Model sent to device: {next(model.parameters()).device}")

    print(
        f"First layer sample weights: \
{model.network[0].weight.flatten()[:5]}"
    )
    print(
        f"First layer weight range: \
[{model.network[0].weight.min():.3f}, {model.network[0].weight.max():.3f}]",
    )
    print(
        f"Final layer sample weights: \
{model.network[-1].weight.flatten()[:5]}"
    )
    print(
        f"Final layer weight range: \
[{model.network[-1].weight.min():.3f}, {model.network[-1].weight.max():.3f}]",
    )

    print(
        f"Model dummy output range: \
[{model_dummy_output.min():.3f}, {model_dummy_output.max():.3f}]"
    )

# Log input/output ranges for user data to verify normalisation
if DO_VERBOSE_LOGGING:
    print("\nProfiling user data...")
    data_sample, label_sample = next(iter(train_loader.dataset))
    print(f"Data Inputs range: [{data_sample.min():.3f}, {data_sample.max():.3f}]")
    print(f"Data Targets range: [{label_sample.min():.3f}, {label_sample.max():.3f}]")
    print("Profiling user data... DONE")

# Begin training loop
trainer.profile_one_batch()

print("\nStarting training loop...")
trainer.train()
