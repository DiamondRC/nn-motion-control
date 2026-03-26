import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

os.system("clear")

DATA_DIR = "./data/"
OUTPUT_FILE = "./data/pvt_to_dac_training.h5"


class CreateTrainingData:
    def __init__(self, data_dir, out_dir):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.all_data = []
        self.in_norm_mean = []
        self.in_norm_std = []
        self.out_norm_mean = []
        self.out_norm_std = []

        self.input_labels = [
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
            "x_DAC_real",
            "y_DAC_real",
            "z_DAC_real",
        ]

        self.output_labels = [
            "timestep_nxt",
            "x_pos_nxt",
            "x_vel_nxt",
            "x_acc_nxt",
            "x_jer_nxt",
            "y_pos_nxt",
            "y_vel_nxt",
            "y_acc_nxt",
            "y_jer_nxt",
            "z_pos_nxt",
            "z_vel_nxt",
            "z_acc_nxt",
            "z_jer_nxt",
        ]

        self.parse_all_files()
        self.concat_files()
        self.do_normalisation()
        self.save_all()

    def parse_all_files(self):
        """
        Transforms data in txt files into PVTAJ data
        for later neural network training.

        Want x_DAC_i+1, x_pos_i -> x_pos_i+1
        """

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.data_dir, filename)
                print(f"Processing {filename}...")

                # Read line-by-line, skip header if any, parse floats
                df = pd.read_csv(
                    filepath,
                    sep=r"\s+",
                    header=None,
                    names=[
                        "timestep",
                        "x_input",
                        "x_DAC_real",
                        "y_input",
                        "y_DAC_real",
                        "z_input",
                        "z_DAC_real",
                        "x_pos",
                        "y_pos",
                        "z_pos",
                    ],
                )

                # Discard verification data
                df = df[
                    [
                        "timestep",
                        "x_DAC_real",
                        "y_DAC_real",
                        "z_DAC_real",
                        "x_pos",
                        "y_pos",
                        "z_pos",
                    ]
                ]

                # Adjust positions to centre at (0,0,0)
                df["x_pos"] -= np.mean(df["x_pos"])
                df["y_pos"] -= np.mean(df["y_pos"])
                df["z_pos"] -= np.mean(df["z_pos"])

                # Shift the DAC values down to get the corresponding
                # demand to position. We record DAC demand and current
                # position simultaneously, so we need to shift the DACs
                # to match the position outcome.

                # Shift the DACs instead of position to keep extra position
                # data which we can discard later when we do our deriviates.
                df["x_DAC_real"] = df["x_DAC_real"].shift(1)
                df["y_DAC_real"] = df["y_DAC_real"].shift(1)
                df["z_DAC_real"] = df["z_DAC_real"].shift(1)

                # Clean up after shifting
                df = df.dropna()

                # Create dt and catch 0 divisions
                dt = np.diff(df["timestep"])
                dt_safe = np.clip(dt, 1e-9, None)

                # Create relative timesteps, acting as effectively indices
                df["timestep"] = (
                    df["timestep"].astype(np.int64)
                    - df["timestep"].astype(np.int64).iloc[0]
                )

                # Velocities
                positions = np.stack([df["x_pos"], df["y_pos"], df["z_pos"]])
                velocities_unpadded = np.diff(positions, axis=1) / dt_safe

                # Accelerations from unpadded velocities
                accelerations_unpadded = (
                    np.diff(velocities_unpadded, axis=1) / dt_safe[:-1]
                )

                # Jerk from acceleration
                jerks_unpadded = np.diff(accelerations_unpadded, axis=1) / dt_safe[:-2]

                # Pad to reallign data
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

                # Drop all rows with nan values
                df = df.dropna()

                # Now shift states to create the i+1 next states
                # Should have x_DAC_i+1, x_pos_i -> x_pos_i+1, which is
                # what we want and is less lossy than x_DAC_i, x_pos_i+1 -> x_pos_i+2
                for out_label in self.output_labels:
                    base_label = out_label.replace("_nxt", "")
                    if base_label in df.columns:
                        df[out_label] = df[base_label].shift(-1)

                # Final shifting cleanup
                df = df.dropna()

                # Return data
                self.all_data.append(df)

        print("\nFinished processing individual files.")

    def concat_files(self):
        self.all_data = pd.concat(self.all_data, ignore_index=True)
        print("\nFinished combining files.")

    def do_normalisation(self):
        """
        Normalises all data except timestep information.
        Stores the normalisation params for later data recovery.
        """

        def _normalise_column(col, label):
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

        # Normalise all the input data
        print(f"np.shape(self.all_data): {np.shape(self.all_data)}")
        for in_label in self.input_labels:
            if not in_label == "timestep":
                col_norm, mean, std = _normalise_column(
                    self.all_data[in_label], in_label
                )
                self.all_data[in_label] = col_norm
                self.in_norm_mean.append(mean)
                self.in_norm_std.append(std)
            else:
                self.in_norm_mean.append(1)
                self.in_norm_std.append(1)

        # Normalise all the output data
        for out_label in self.output_labels:
            if not out_label == "timestep_nxt":
                col_norm, mean, std = _normalise_column(
                    self.all_data[out_label], out_label
                )
                self.all_data[out_label] = col_norm
                self.out_norm_mean.append(mean)
                self.out_norm_std.append(std)
            else:
                self.out_norm_mean.append(1)
                self.out_norm_std.append(1)

    def save_all(self):
        """
        Saves all the data, including any normalisation information.
        """
        print("\nDisplaying data for inspection.")
        self.input_data = self.all_data[self.input_labels].to_numpy()
        self.output_data = self.all_data[self.output_labels].to_numpy()

        print(f"\nInput Data:\n{self.input_data[:][:4]}")
        print(f"\nOutput Data\n{self.output_data[:][:4]}")

        print("\nSaving all data.")

        # Delete file to prevent blockingIO
        try:
            Path(self.out_dir).unlink()
        except FileNotFoundError:
            pass

        # Do writing
        with h5py.File(self.out_dir, "w") as f:
            f.create_dataset("inputs", data=self.input_data)
            f.create_dataset("targets", data=self.output_data)
            f.create_dataset(
                "input_norm_params", data=[self.in_norm_mean, self.in_norm_std]
            )
            f.create_dataset(
                "target_norm_params", data=[self.out_norm_mean, self.out_norm_std]
            )
            f.create_dataset("input_labels", data=self.input_labels)
            f.create_dataset("target_labels", data=self.output_labels)

            print("Data saved successfully.")


CreateTrainingData(DATA_DIR, OUTPUT_FILE)
