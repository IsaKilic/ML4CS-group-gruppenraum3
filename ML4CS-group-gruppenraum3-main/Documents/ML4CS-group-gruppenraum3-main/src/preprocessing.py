from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

def load_csv(path):
    return pd.read_csv(path)

def load_recording(run_folder: Path):
    acc = pd.read_csv(run_folder / "Accelerometer.csv")
    gyr = pd.read_csv(run_folder / "Gyroscope.csv")
    return acc, gyr

def extract_features(acc: pd.DataFrame, gyr: pd.DataFrame) -> np.ndarray:
    features = []
    acc_cols = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]
    gyr_cols = ["Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"]

    for df, cols in [(acc, acc_cols), (gyr, gyr_cols)]:
        for col in cols:
            values = df[col].values
            features.extend([
                np.mean(values),
                np.std(values),
                np.max(values),
                np.min(values),
            ])

    return np.array(features)

def load_all_data(data_folder: Path, digits: list):
    X = []
    y = []

    for run_folder in sorted(data_folder.iterdir()):
        if not run_folder.is_dir():
            continue

        name = run_folder.name
        for digit in digits:
            if name.startswith(f"digit_{digit}_"):
                try:
                    acc, gyr = load_recording(run_folder)
                    features = extract_features(acc, gyr)
                    X.append(features)
                    y.append(digit)
                    print(f"Geladen: {name} → Label {digit}")
                except Exception as e:
                    print(f"Fehler bei {name}: {e}")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    data_folder = Path("/Users/isakilic/Downloads/real_data")
    X, y = load_all_data(data_folder, digits=[3])
    print(f"\nFeature Matrix: {X.shape}")
    print(f"Labels: {y}")