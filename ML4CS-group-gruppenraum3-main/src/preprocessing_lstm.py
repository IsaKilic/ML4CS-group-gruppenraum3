# preprocessing_lstm.py
# Laedt die rohen Zeitreihen (statt 24 statistische Features)
# und bringt alle Aufnahmen per Interpolation auf eine feste Laenge.
# Literatur: Al Abir et al. (2021) — "Choice of Proper Interpolation Technique"

import pandas as pd
import numpy as np
from pathlib import Path

SEQUENCE_LENGTH = 100  # Jede Aufnahme wird auf 100 Zeitschritte gebracht


def find_columns(df, sensor):
    """Erkennt die Spaltennamen automatisch (verschiedene Phyphox-Formate)."""
    cols = {}
    for axis in ["x", "y", "z"]:
        for candidate in df.columns:
            c = candidate.lower()
            # Beispiele: "Acceleration x (m/s^2)", "X (m/s^2)", "Linear Acceleration x"
            if c.strip().startswith(axis + " ") or f" {axis} " in c or c == axis:
                cols[axis] = candidate
                break
    if len(cols) != 3:
        raise ValueError(f"Konnte {sensor}-Achsen nicht erkennen. Spalten: {list(df.columns)}")
    return cols


def resample_to_fixed_length(signal, target_len=SEQUENCE_LENGTH):
    """Interpoliert ein Signal auf eine feste Laenge (linear interpolation)."""
    if len(signal) == target_len:
        return signal
    x_old = np.linspace(0.0, 1.0, num=len(signal))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, signal)


def load_recording_sequence(folder):
    """Laedt eine Aufnahme als Zeitreihe der Form (SEQUENCE_LENGTH, 6)."""
    folder = Path(folder)
    acc_file = folder / "Accelerometer.csv"
    gyr_file = folder / "Gyroscope.csv"
    if not acc_file.exists():
        # Lineare Beschleunigung heisst evtl. anders
        candidates = list(folder.glob("*.csv"))
        acc_candidates = [c for c in candidates if "acc" in c.name.lower() or "linear" in c.name.lower()]
        if acc_candidates:
            acc_file = acc_candidates[0]
    if not acc_file.exists() or not gyr_file.exists():
        raise FileNotFoundError(f"CSV-Dateien fehlen in {folder}")

    acc = pd.read_csv(acc_file)
    gyr = pd.read_csv(gyr_file)

    acc_cols = find_columns(acc, "Accelerometer")
    gyr_cols = find_columns(gyr, "Gyroscope")

    channels = []
    for axis in ["x", "y", "z"]:
        channels.append(resample_to_fixed_length(acc[acc_cols[axis]].to_numpy()))
    for axis in ["x", "y", "z"]:
        channels.append(resample_to_fixed_length(gyr[gyr_cols[axis]].to_numpy()))

    # Form: (SEQUENCE_LENGTH, 6)
    return np.stack(channels, axis=1)


def load_all_sequences(data_folder, digits):
    """Laedt alle Aufnahmen als Sequenzen.
    Rueckgabe: X mit Form (N, SEQUENCE_LENGTH, 6), y mit Form (N,)"""
    data_folder = Path(data_folder)
    X, y = [], []
    for digit in digits:
        for folder in sorted(data_folder.glob(f"digit_{digit}_run*")):
            try:
                seq = load_recording_sequence(folder)
                X.append(seq)
                y.append(digit)
                print(f"Geladen: {folder.name} -> Label {digit}")
            except Exception as e:
                print(f"FEHLER bei {folder.name}: {e}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Normalisierung pro Kanal (wichtig fuer LSTM!)
    # z-score: (x - mean) / std, berechnet ueber alle Trainingsdaten
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Mean und Std speichern fuer die Live-Demo
    np.save(data_folder / "norm_mean.npy", mean)
    np.save(data_folder / "norm_std.npy", std)

    print(f"\n{len(X)} Aufnahmen geladen, Form: {X.shape}")
    return X, y
