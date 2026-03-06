import os

import h5py
import numpy as np
import pandas as pd

os.system("clear")

DATA_DIR = "./data/"
OUTPUT_FILE = "./data/pvt_to_dac_training.h5"

DO_PVT = True
DO_SHIFTING = True
DO_NORMALISE = True
DO_DATA_ANALYSIS = False


# -------------------------------
# Helper Functions
# -------------------------------


def parse_all_files(data_dir):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing {filename}...")
            # Read line-by-line, skip header if any, parse floats
            df = pd.read_csv(
                filepath,
                sep=r"\s+",
                header=None,
                names=[
                    "timestep",
                    "x_input",
                    "x_input_real",
                    "y_input",
                    "y_input_real",
                    "z_input",
                    "z_input_real",
                    "x_pos",
                    "y_pos",
                    "z_pos",
                ],
            )

            # Discard verification data
            df = df[
                [
                    "timestep",
                    "x_input_real",
                    "y_input_real",
                    "z_input_real",
                    "x_pos",
                    "y_pos",
                    "z_pos",
                ]
            ]

            if DO_SHIFTING:
                # Shift positions to start at (0,0,0)
                df["x_pos"] -= np.mean(df["x_pos"])
                df["y_pos"] -= np.mean(df["y_pos"])
                df["z_pos"] -= np.mean(df["z_pos"])

            if DO_DATA_ANALYSIS:
                print(df["x_pos"][:10])
                print(df["x_pos"].iloc[0])

                print(f"Raw:     {df['x_pos'][0]}")
                print(f"Float32: {np.float32(df['x_pos'][0])}")
                recovered = np.float64(np.float32(df["x_pos"][0]))
                print(
                    f"Error:   {abs(recovered - df['x_pos'][0]):.2e} mm = \
{abs(recovered - df['x_pos'][0]) * 1e9:.1f} nm"
                )

                print(np.max(df["x_pos"]), np.min(df["x_pos"]))

            if DO_PVT:
                # Catch div by zero
                dt = np.diff(df["timestep"])
                dt_safe = np.clip(dt, 1e-9, None)

                # Calculate velocities
                df["x_vel"] = np.concatenate(([0.0], np.diff(df["x_pos"]) / dt_safe))
                df["y_vel"] = np.concatenate(([0.0], np.diff(df["y_pos"]) / dt_safe))
                df["z_vel"] = np.concatenate(([0.0], np.diff(df["z_pos"]) / dt_safe))

                # Reorder outputs as inputs
                df = df[
                    [
                        "timestep",
                        "x_pos",
                        "x_vel",
                        "y_pos",
                        "y_vel",
                        "z_pos",
                        "z_vel",
                        "x_input_real",
                        "y_input_real",
                        "z_input_real",
                    ]
                ]

            all_data.append(df)

    return all_data


def collect_files(files):
    # Concatenate all
    files = pd.concat(files, ignore_index=True)
    return files


# -------------------------------
# Main Logic
# -------------------------------


# Handle individual file processing
all_data = parse_all_files(DATA_DIR)

# Create one dataset
all_data = collect_files(all_data)

# Split into input and output data and write to HDF5
with h5py.File(OUTPUT_FILE, "w") as f:
    # Cast everything as float32 to match training format
    if DO_NORMALISE:
        # Normalise interferometer data
        def normalise_column(col):
            # Keep mean/std as float64 for accurate denormalisation later
            col_f64 = col.astype(np.float64)
            input_mean = col_f64.mean()
            input_std = col_f64.std() + 1e-10

            # Normalise to float32 for training
            col_norm = (
                col.astype(np.float32) - input_mean.astype(np.float32)
            ) / input_std.astype(np.float32)

            print(f"Max/min norm column: {col_norm.max()}, {col_norm.min()}")

            return col_norm, input_mean, input_std

        # Normalise each PVT axis and store the mean/std for later denormalisation
        all_data["timestep"], t_mean, t_std = normalise_column(all_data["timestep"])
        all_data["x_pos"], x_pos_mean, x_pos_std = normalise_column(all_data["x_pos"])
        all_data["x_vel"], x_vel_mean, x_vel_std = normalise_column(all_data["x_vel"])
        all_data["y_pos"], y_pos_mean, y_pos_std = normalise_column(all_data["y_pos"])
        all_data["y_vel"], y_vel_mean, y_vel_std = normalise_column(all_data["y_vel"])
        all_data["z_pos"], z_pos_mean, z_pos_std = normalise_column(all_data["z_pos"])
        all_data["z_vel"], z_vel_mean, z_vel_std = normalise_column(all_data["z_vel"])

        # Normalise each DAC axis and store the mean/std for later denormalisation
        all_data["x_input_real"], x_dac_mean, x_dac_std = normalise_column(
            all_data["x_input_real"]
        )
        all_data["y_input_real"], y_dac_mean, y_dac_std = normalise_column(
            all_data["y_input_real"]
        )
        all_data["z_input_real"], z_dac_mean, z_dac_std = normalise_column(
            all_data["z_input_real"]
        )

        # Gather denormalisation parameters into one array for later use in inference
        norm_params = np.array(
            [
                t_mean,
                t_std,
                x_pos_mean,
                x_pos_std,
                x_vel_mean,
                x_vel_std,
                y_pos_mean,
                y_pos_std,
                z_pos_mean,
                z_pos_std,
                x_dac_mean,
                x_dac_std,
                y_dac_mean,
                y_dac_std,
                z_dac_mean,
                z_dac_std,
            ]
        )

        if DO_PVT:
            print("Writing PVT dataset with normalisation params...")
            input_data = all_data[
                ["timestep", "x_pos", "x_vel", "y_pos", "y_vel", "z_pos", "z_vel"]
            ].values
            output_data = all_data[
                ["x_input_real", "y_input_real", "z_input_real"]
            ].values
        else:
            input_data = all_data[
                ["timestep", "x_input_real", "y_input_real", "z_input_real"]
            ].values
            output_data = all_data[["x_pos", "y_pos", "z_pos"]].values

    else:
        if DO_PVT:
            input_data = all_data[
                ["timestep", "x_pos", "x_vel", "y_pos", "y_vel", "z_pos", "z_vel"]
            ].values.astype(np.float32)

            output_data = all_data[
                ["x_input_real", "y_input_real", "z_input_real"]
            ].values.astype(np.float32)

        else:
            input_data = all_data[
                ["timestep", "x_input_real", "y_input_real", "z_input_real"]
            ].values.astype(np.float32)

            # Outputs dataset (x_pos, y_pos, z_pos)
            output_data = all_data[["x_pos", "y_pos", "z_pos"]].values.astype(
                np.float32
            )

    # Create input and output datasets
    f.create_dataset(
        "inputs",
        data=input_data,
        compression="gzip",
        compression_opts=4,
        chunks=True,
    )

    f.create_dataset(
        "outputs",
        data=output_data,
        compression="gzip",
        compression_opts=4,
        chunks=True,
    )

    if DO_NORMALISE:
        f.create_dataset(
            "norm_params",
            data=norm_params,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

if DO_NORMALISE:
    print(
        f"Converted to {OUTPUT_FILE}: inputs {input_data.shape}, \
outputs {output_data.shape}, norm_params {len(norm_params)}"
    )

else:
    print(
        f"Converted to {OUTPUT_FILE}: \
inputs {input_data.shape}, outputs {output_data.shape}"
    )
