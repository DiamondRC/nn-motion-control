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
                    f"Error:   {abs(recovered - df['x_pos'][0]):.2e} mm = "
                    f"{abs(recovered - df['x_pos'][0]) * 1e9:.1f} nm"
                )

                print(np.max(df["x_pos"]), np.min(df["x_pos"]))

            if DO_PVT:
                # Catch div by zero
                dt = np.diff(df["timestep"])
                dt_safe = np.clip(dt, 1e-9, None)

                # Velocities (correct)
                positions = np.stack([df["x_pos"], df["y_pos"], df["z_pos"]])
                velocities_unpadded = np.diff(positions, axis=1) / dt_safe  # (3, n-1)

                # Accelerations from unpadded velocities
                accelerations_unpadded = (
                    np.diff(velocities_unpadded, axis=1) / dt_safe[:-1]
                )  # (3, n-2)

                # Jerk from acceleration

                jerks_unpadded = np.diff(accelerations_unpadded, axis=1) / dt_safe[:-2]

                # Pad all
                velocities_padded = np.pad(
                    velocities_unpadded,
                    ((0, 0), (1, 0)),
                    "constant",
                    constant_values=np.nan,
                )
                accelerations_padded = np.pad(
                    accelerations_unpadded,
                    ((0, 0), (2, 0)),
                    "constant",
                    constant_values=np.nan,
                )
                jerks_padded = np.pad(
                    jerks_unpadded, ((0, 0), (3, 0)), "constant", constant_values=np.nan
                )

                # Unpack
                df["x_vel"] = velocities_padded[0]
                df["y_vel"] = velocities_padded[1]
                df["z_vel"] = velocities_padded[2]
                df["x_acc"] = accelerations_padded[0]
                df["y_acc"] = accelerations_padded[1]
                df["z_acc"] = accelerations_padded[2]
                df["x_jer"] = jerks_padded[0]
                df["y_jer"] = jerks_padded[1]
                df["z_jer"] = jerks_padded[2]

                # Shift the DAC values
                df["x_input_real"] = df["x_input_real"].shift(1)
                df["y_input_real"] = df["y_input_real"].shift(1)
                df["z_input_real"] = df["z_input_real"].shift(1)

                df = df.dropna()

                # Reorder outputs as inputs
                df = df[
                    [
                        "timestep",
                        "x_pos",
                        "x_vel",
                        "x_acc",
                        "x_jer",
                        "y_pos",
                        "y_vel",
                        "y_acc",
                        "y_jer",
                        "z_pos",
                        "z_vel",
                        "z_acc",
                        "z_jer",
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

# plt.figure(figsize=(10, 5))
# plt.plot(all_data["x_pos"], label="pos")
# plt.plot(all_data["x_vel"], label="vel")
# plt.plot(all_data["x_acc"], label="acc")
# plt.plot(all_data["x_jer"], label="jer")
# plt.xlabel("time")
# plt.ylabel("Loss")
# plt.title("Data Analysis")
# plt.grid()
# plt.legend()
# plt.show()

# Split into input and output data and write to HDF5
with h5py.File(OUTPUT_FILE, "w") as f:
    # Cast everything as float32 to match training format
    if DO_NORMALISE:
        # Normalise interferometer data
        def normalise_column(col, label):
            # Keep mean/std as float64 for accurate denormalisation later
            col_f64 = col.astype(np.float64)
            input_mean = col_f64.mean()
            input_std = col_f64.std() + 1e-10

            # Normalise to float32 for training
            col_norm = (
                col.astype(np.float32) - input_mean.astype(np.float32)
            ) / input_std.astype(np.float32)

            print(
                f"\nNormalising {label}..."
                f"\nMax/min norm column: {col_norm.max()}, {col_norm.min()}"
            )

            return col_norm, input_mean, input_std

        # Normalise each PVT axis and store the mean/std for later denormalisation
        all_data["timestep"], t_mean, t_std = normalise_column(
            all_data["timestep"], "timestep"
        )
        all_data["x_pos"], x_pos_mean, x_pos_std = normalise_column(
            all_data["x_pos"], "x_pos"
        )
        all_data["y_pos"], y_pos_mean, y_pos_std = normalise_column(
            all_data["y_pos"], "y_pos"
        )
        all_data["z_pos"], z_pos_mean, z_pos_std = normalise_column(
            all_data["z_pos"], "z_pos"
        )

        # Normalise each DAC axis and store the mean/std for later denormalisation
        all_data["x_input_real"], x_dac_mean, x_dac_std = normalise_column(
            all_data["x_input_real"], "x_input_real"
        )
        all_data["y_input_real"], y_dac_mean, y_dac_std = normalise_column(
            all_data["y_input_real"], "y_input_real"
        )
        all_data["z_input_real"], z_dac_mean, z_dac_std = normalise_column(
            all_data["z_input_real"], "z_input_real"
        )

        if DO_PVT:
            # Normalise velocity and acceleration values, store results
            all_data["x_vel"], x_vel_mean, x_vel_std = normalise_column(
                all_data["x_vel"], "x_vel"
            )
            all_data["x_acc"], x_acc_mean, x_acc_std = normalise_column(
                all_data["x_acc"], "x_acc"
            )
            all_data["x_jer"], x_jer_mean, x_jer_std = normalise_column(
                all_data["x_jer"], "x_jer"
            )
            all_data["y_vel"], y_vel_mean, y_vel_std = normalise_column(
                all_data["y_vel"], "y_vel"
            )
            all_data["y_acc"], y_acc_mean, y_acc_std = normalise_column(
                all_data["y_acc"], "y_acc"
            )
            all_data["y_jer"], y_jer_mean, y_jer_std = normalise_column(
                all_data["y_jer"], "y_jer"
            )
            all_data["z_vel"], z_vel_mean, z_vel_std = normalise_column(
                all_data["z_vel"], "z_vel"
            )
            all_data["z_acc"], z_acc_mean, z_acc_std = normalise_column(
                all_data["z_acc"], "z_acc"
            )
            all_data["z_jer"], z_jer_mean, z_jer_std = normalise_column(
                all_data["z_jer"], "z_jer"
            )

        # Gather denormalisation parameters into one array for later use in inference
        if DO_PVT:
            norm_params = np.array(
                [
                    t_mean,
                    t_std,
                    x_pos_mean,
                    x_pos_std,
                    x_vel_mean,
                    x_vel_std,
                    x_acc_mean,
                    x_acc_std,
                    x_jer_mean,
                    x_jer_std,
                    y_pos_mean,
                    y_pos_std,
                    y_vel_mean,
                    y_vel_std,
                    y_acc_mean,
                    y_acc_std,
                    y_jer_mean,
                    y_jer_std,
                    z_pos_mean,
                    z_pos_std,
                    z_vel_mean,
                    z_vel_std,
                    z_acc_mean,
                    z_acc_std,
                    z_jer_mean,
                    z_jer_std,
                    x_dac_mean,
                    x_dac_std,
                    y_dac_mean,
                    y_dac_std,
                    z_dac_mean,
                    z_dac_std,
                ]
            )
        else:
            norm_params = np.array(
                [
                    t_mean,
                    t_std,
                    x_pos_mean,
                    x_pos_std,
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
            print("\nWriting PVT dataset with normalisation params...")
            input_data = all_data[
                [
                    "timestep",
                    "x_pos",
                    "x_vel",
                    "x_acc",
                    "x_jer",
                    "y_pos",
                    "y_vel",
                    "y_acc",
                    "y_jer",
                    "z_pos",
                    "z_vel",
                    "z_acc",
                    "z_jer",
                ]
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
                "timestep",
                "x_pos",
                "x_vel",
                "x_acc",
                "x_jer",
                "y_pos",
                "y_vel",
                "y_acc",
                "y_jer",
                "z_pos",
                "z_vel",
                "z_acc",
                "z_jer",
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
    )

    f.create_dataset(
        "outputs",
        data=output_data,
    )

    if DO_NORMALISE:
        f.create_dataset(
            "norm_params",
            data=norm_params,
        )

if DO_NORMALISE:
    print(
        f"Converted to {OUTPUT_FILE}: inputs {input_data.shape}, "
        f"outputs {output_data.shape}, norm_params {len(norm_params)}"
    )

else:
    print(
        f"Converted to {OUTPUT_FILE}: "
        f"inputs {input_data.shape}, outputs {output_data.shape}"
    )
