from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent))
from preprocessing import load_all_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y_encoded, cv=3)
    print(f"Cross-Validation Genauigkeit: {scores.mean():.2f} (+/- {scores.std():.2f})")
    model.fit(X, y_encoded)
    print(f"Modell trainiert auf {len(X)} Aufnahmen!")
    return model, le

if __name__ == "__main__":
    data_folder = Path("/Users/isakilic/Downloads/real_data")
    print("Lade Daten...")
    X, y = load_all_data(data_folder, digits=[3])
    print(f"{len(X)} Aufnahmen geladen, Feature Matrix: {X.shape}")
    print("\nTrainiere Modell...")
    model, le = train_model(X, y)