import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── 1. DATEN EINLESEN ───────────────────────────────────────
def load_data(data_folder):
    X = []  # Sensordaten
    y = []  # Labels (0, 1, 2, 3)
    
    for label in range(4):  # 0 bis 3
        folder = os.path.join(data_folder, f"zahl_{label}")
        
        if not os.path.exists(folder):
            print(f"Ordner nicht gefunden: {folder}")
            continue
            
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                filepath = os.path.join(folder, filename)
                df = pd.read_csv(filepath)
                
                # Nur Accelerometer Spalten X, Y, Z
                data = df.iloc[:, 1:4].values
                X.append(data)
                y.append(label)
                print(f"Geladen: {filename} → Label {label}")
    
    return X, y

# ─── 2. FEATURES BERECHNEN ───────────────────────────────────
def extract_features(X):
    features = []
    
    for sample in X:
        f = [
            np.mean(sample[:, 0]),   # Mittelwert X
            np.mean(sample[:, 1]),   # Mittelwert Y
            np.mean(sample[:, 2]),   # Mittelwert Z
            np.std(sample[:, 0]),    # Standardabweichung X
            np.std(sample[:, 1]),    # Standardabweichung Y
            np.std(sample[:, 2]),    # Standardabweichung Z
            np.max(sample[:, 0]),    # Maximum X
            np.max(sample[:, 1]),    # Maximum Y
            np.max(sample[:, 2]),    # Maximum Z
            np.min(sample[:, 0]),    # Minimum X
            np.min(sample[:, 1]),    # Minimum Y
            np.min(sample[:, 2]),    # Minimum Z
        ]
        features.append(f)
    
    return np.array(features)

# ─── 3. DATEN VISUALISIEREN ──────────────────────────────────
def plot_samples(X, y, n_samples=2):
    fig, axes = plt.subplots(4, n_samples, figsize=(12, 10))
    
    for label in range(4):
        indices = [i for i, l in enumerate(y) if l == label]
        
        for j in range(min(n_samples, len(indices))):
            idx = indices[j]
            ax = axes[label, j]
            ax.plot(X[idx][:, 0], label="X")
            ax.plot(X[idx][:, 1], label="Y")
            ax.plot(X[idx][:, 2], label="Z")
            ax.set_title(f"Zahl {label} - Aufnahme {j+1}")
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# ─── MAIN ────────────────────────────────────────────────────
data_folder = r"C:\Users\ikili\Documents\ML4CS-group-gruppenraum3-main\data"

print("Lade Daten...")
X, y = load_data(data_folder)

if len(X) == 0:
    print("\nNoch keine Daten vorhanden!")
    print("Lege CSV Dateien in die entsprechenden Ordner:")
    print("  data/zahl_0/  → CSVs für die 0")
    print("  data/zahl_1/  → CSVs für die 1")
    print("  data/zahl_2/  → CSVs für die 2")
    print("  data/zahl_3/  → CSVs für die 3")
else:
    print(f"\n{len(X)} Aufnahmen geladen!")
    print(f"Labels: {y}")
    
    # Features berechnen
    features = extract_features(X)
    print(f"Feature Matrix: {features.shape}")
    
    # Visualisieren
    plot_samples(X, y)