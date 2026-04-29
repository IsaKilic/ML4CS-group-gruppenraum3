"""
make_dummy_data.py

Generates fake Sensor Logger recordings so you can test the pipeline before
the Apple Watch arrives. Each "recording" is a folder containing
Accelerometer.csv and Gyroscope.csv, mimicking the real Sensor Logger export.

Usage:
    python make_dummy_data.py
    -> creates ./dummy_data/digit_3_run01/ ... digit_3_run10/

The signals are not real gestures — they are just a quiet stretch + a
noisy "movement" stretch + another quiet stretch. Enough to verify that
plotting and loading work end-to-end.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def synth_recording(seed: int = 0, fs: int = 100, total_s: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a fake recording: rest -> movement -> rest."""
    rng = np.random.default_rng(seed)
    n = int(total_s * fs)
    t = np.arange(n) / fs

    # gravity baseline (watch upright on wrist => ~9.8 in one axis)
    ax = np.full(n, 0.1) + rng.normal(0, 0.05, n)
    ay = np.full(n, 9.8) + rng.normal(0, 0.05, n)
    az = np.full(n, 0.0) + rng.normal(0, 0.05, n)

    gx = rng.normal(0, 0.02, n)
    gy = rng.normal(0, 0.02, n)
    gz = rng.normal(0, 0.02, n)

    # inject a "gesture" between t=1.5s and t=3.0s
    start, end = int(1.5 * fs), int(3.0 * fs)
    movement_n = end - start
    movement_t = np.linspace(0, 2 * np.pi, movement_n)

    ax[start:end] += 3.0 * np.sin(2 * movement_t) + rng.normal(0, 0.5, movement_n)
    ay[start:end] += 2.0 * np.cos(movement_t) + rng.normal(0, 0.5, movement_n)
    az[start:end] += 1.5 * np.sin(3 * movement_t) + rng.normal(0, 0.5, movement_n)

    gx[start:end] += 1.5 * np.cos(2 * movement_t) + rng.normal(0, 0.2, movement_n)
    gy[start:end] += 1.0 * np.sin(movement_t) + rng.normal(0, 0.2, movement_n)
    gz[start:end] += 0.8 * np.cos(3 * movement_t) + rng.normal(0, 0.2, movement_n)

    accel = pd.DataFrame({"seconds_elapsed": t, "x": ax, "y": ay, "z": az})
    gyro = pd.DataFrame({"seconds_elapsed": t, "x": gx, "y": gy, "z": gz})
    return accel, gyro


def main():
    out_root = Path("dummy_data")
    out_root.mkdir(exist_ok=True)

    for i in range(1, 11):
        folder = out_root / f"digit_3_run{i:02d}"
        folder.mkdir(exist_ok=True)
        accel, gyro = synth_recording(seed=i)
        accel.to_csv(folder / "Accelerometer.csv", index=False)
        gyro.to_csv(folder / "Gyroscope.csv", index=False)
        print(f"created {folder}")

    print("\nDone. Test with:")
    print("    python plot_recordings.py dummy_data --pattern 'digit_3_*'")


if __name__ == "__main__":
    main()
