from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent))
from preprocessing import load_recording, extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from preprocessing import load_all_data
import joblib

def train_and_save(data_folder, digits):
    """Trainiert das Modell und speichert es."""
    X, y = load_all_data(data_folder, digits)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # Modell speichern
    joblib.dump(model, "/Users/isakilic/Downloads/real_data/model.pkl")
    joblib.dump(le, "/Users/isakilic/Downloads/real_data/label_encoder.pkl")
    print("Modell gespeichert!")
    return model, le

def predict_digit(run_folder, model, le):
    """Vorhersage für eine neue Aufnahme."""
    run_folder = Path(run_folder)
    
    # Daten einlesen
    acc, gyr = load_recording(run_folder)
    
    # Features berechnen
    features = extract_features(acc, gyr)
    
    # Vorhersage
    prediction_encoded = model.predict([features])
    prediction = le.inverse_transform(prediction_encoded)
    
    # Wahrscheinlichkeiten
    probabilities = model.predict_proba([features])[0]
    
    print(f"\n🎯 Erkannte Zahl: {prediction[0]}")
    print("\nWahrscheinlichkeiten:")
    for digit, prob in zip(le.classes_, probabilities):
        bar = "█" * int(prob * 20)
        print(f"  Zahl {digit}: {bar} {prob:.1%}")
    
    return prediction[0]

if __name__ == "__main__":
    data_folder = Path("/Users/isakilic/Downloads/real_data")
    
    # Schritt 1: Modell trainieren auf allen Zahlen
    print("Trainiere Modell...")
    model, le = train_and_save(data_folder, digits=[0, 1, 2, 3])
    
    # Schritt 2: Neue Aufnahme vorhersagen
    # Ändere diesen Pfad zu deiner neuen Aufnahme!
    test_folder = "/Users/isakilic/Downloads/real_data/digit_3_run01"
    print(f"\nVorhersage für: {test_folder}")
    predict_digit(test_folder, model, le)