from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append(str(Path(__file__).parent))
from preprocessing import load_all_data

def train_model(X: np.ndarray, y: np.ndarray):
    """Trainiert einen Random Forest Klassifikator."""
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-Validation (testet wie gut das Modell generalisiert)
    scores = cross_val_score(model, X, y_encoded, cv=3)
    print(f"Cross-Validation Genauigkeit: {scores.mean():.2f} (+/- {scores.std():.2f})")
    
    # Modell auf allen Daten trainieren
    model.fit(X, y_encoded)
    print(f"Modell trainiert auf {len(X)} Aufnahmen!")
    
    return model, le

if __name__ == "__main__":
    data_folder = Path("dummy_data")
    
    print("Lade Daten...")
    X, y = load_all_data(data_folder, digits=[3])
    
    print(f"\n{len(X)} Aufnahmen geladen!")
    print(f"Feature Matrix: {X.shape}")
    
    print("\nTrainiere Modell...")
    model, le = train_model(X, y)