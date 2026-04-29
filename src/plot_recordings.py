"""
plot_recordings.py

Loads Sensor Logger CSV files (accelerometer + gyroscope) and plots them
so you can visually compare multiple recordings of the same gesture.

Usage:
    python plot_recordings.py path/to/folder
    python plot_recordings.py path/to/folder --pattern "digit_3_*.csv"

Expected CSV format (Sensor Logger default):
    time,seconds_elapsed,z,y,x   (for Accelerometer.csv)
    time,seconds_elapsed,z,y,x   (for Gyroscope.csv)

Sensor Logger exports each session as a folder containing several CSVs
(Accelerometer.csv, Gyroscope.csv, etc.). For this script we expect each
recording to be saved as its own folder, OR as a single combined CSV per
recording — adjust load_recording() if your export format differs.
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_recording(path: str) -> pd.DataFrame:
    """
    Load one recording into a single DataFrame.

    Two cases are handled:
    1. `path` is a folder containing Accelerometer.csv and Gyroscope.csv
       (Sensor Logger's default export).
    2. `path` is a single CSV file with columns time, ax, ay, az, gx, gy, gz.
    """
    p = Path(path)

    if p.is_dir():
        accel_path = p / "Accelerometer.csv"
        gyro_path = p / "Gyroscope.csv"
        if not accel_path.exists():
            raise FileNotFoundError(f"No Accelerometer.csv in {p}")

        accel = pd.read_csv(accel_path)
        accel = accel.rename(columns={"x": "ax", "y": "ay", "z": "az"})

        if gyro_path.exists():
            gyro = pd.read_csv(gyro_path)
            gyro = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})
            df = pd.merge_asof(
                accel.sort_values("seconds_elapsed"),
                gyro[["seconds_elapsed", "gx", "gy", "gz"]].sort_values("seconds_elapsed"),
                on="seconds_elapsed",
                direction="nearest",
            )
        else:
            df = accel
        df = df.rename(columns={"seconds_elapsed": "t"})
    else:
        df = pd.read_csv(p)
        if "t" not in df.columns and "seconds_elapsed" in df.columns:
            df = df.rename(columns={"seconds_elapsed": "t"})

    return df


def plot_recording(df: pd.DataFrame, ax_accel, ax_gyro, label: str = ""):
    """Plot one recording on the given axes."""
    ax_accel.plot(df["t"], df["ax"], label="ax", linewidth=0.8)
    ax_accel.plot(df["t"], df["ay"], label="ay", linewidth=0.8)
    ax_accel.plot(df["t"], df["az"], label="az", linewidth=0.8)
    ax_accel.set_title(f"{label} — accelerometer")
    ax_accel.set_xlabel("time [s]")
    ax_accel.set_ylabel("a [m/s²]")
    ax_accel.legend(fontsize=7, loc="upper right")
    ax_accel.grid(alpha=0.3)

    if "gx" in df.columns:
        ax_gyro.plot(df["t"], df["gx"], label="gx", linewidth=0.8)
        ax_gyro.plot(df["t"], df["gy"], label="gy", linewidth=0.8)
        ax_gyro.plot(df["t"], df["gz"], label="gz", linewidth=0.8)
        ax_gyro.set_title(f"{label} — gyroscope")
        ax_gyro.set_xlabel("time [s]")
        ax_gyro.set_ylabel("ω [rad/s]")
        ax_gyro.legend(fontsize=7, loc="upper right")
        ax_gyro.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot Sensor Logger recordings.")
    parser.add_argument("folder", help="Folder containing recordings")
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to filter recordings (e.g. 'digit_3_*')",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not paths:
        print(f"No recordings found in {args.folder} matching {args.pattern}")
        return

    print(f"Found {len(paths)} recordings:")
    for p in paths:
        print(f"  {p}")

    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.5 * n), squeeze=False)

    for i, path in enumerate(paths):
        try:
            df = load_recording(path)
            label = Path(path).stem
            plot_recording(df, axes[i, 0], axes[i, 1], label=label)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    plt.tight_layout()
    plt.savefig("recordings_overview.png", dpi=120)
    plt.show()
    print("Saved overview to recordings_overview.png")


if __name__ == "__main__":
    main()
