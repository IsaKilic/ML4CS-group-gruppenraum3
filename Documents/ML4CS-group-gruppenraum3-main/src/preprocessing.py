from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)

def load_recording(run_folder: Path):
    """Lädt Accelerometer und Gyroscope aus einem Run-Ordner."""
    acc = pd.read_csv(run_folder / "Accelerometer.csv")
    gyr = pd.read_csv(run_folder / "Gyroscope.csv")
    return acc, gyr

def extract_features(acc: pd.DataFrame, gyr: pd.DataFrame) -> np.ndarray:
    """Berechnet Features aus Accelerometer und Gyroscope Daten."""
    features = []
    
    for df in [acc, gyr]:
        x = df.iloc[:, 1].values
        y = df.iloc[:, 2].values
        z = df.iloc[:, 3].values
        
        for axis in [x, y, z]:
            features.extend([
                np.mean(axis),
                np.std(axis),
                np.max(axis),
                np.min(axis),
            ])
    
    return np.array(features)

def load_all_data(data_folder: Path, digits: list):
    """Lädt alle Aufnahmen und gibt Features + Labels zurück."""
    X = []
    y = []
    
    # Suche alle Ordner die mit digit_ZAHL anfangen
    for run_folder in sorted(data_folder.iterdir()):
        if not run_folder.is_dir():
            continue
        
        # Extrahiere die Zahl aus dem Ordnernamen z.B. digit_3_run01 → 3
        name = run_folder.name  # z.B. "digit_3_run01"
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
    data_folder = Path("dummy_data")
    X, y = load_all_data(data_folder, digits=[3])
    print(f"\nFeature Matrix: {X.shape}")
    print(f"Labels: {y}")