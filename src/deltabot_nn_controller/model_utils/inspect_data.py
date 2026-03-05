import h5py

DATA_DIR = "./data/"
OUTPUT_FILE = "./data/pvt_to_dac_training.h5"

with h5py.File(OUTPUT_FILE, "r") as f:
    input_data = f["inputs"][:]
    output_data = f["outputs"][:]
    if "norm_params" in f:
        norm_params = f["norm_params"][:]
        print(
            f"Loaded from {OUTPUT_FILE}: inputs {input_data.shape}, \
outputs {output_data.shape}, norm_params {len(norm_params)}"
        )
        print()

        print(
            f"max/min input_data: \
{input_data.max(axis=0)}, {input_data.min(axis=0)}"
        )
        print()
        print(
            f"max/min output_data: \
{output_data.max(axis=0)}, {output_data.min(axis=0)}"
        )
        print()
        print(f"norm_params: {norm_params}")
    else:
        print(
            f"Loaded from {OUTPUT_FILE}: \
inputs {input_data.shape}, outputs {output_data.shape}"
        )
