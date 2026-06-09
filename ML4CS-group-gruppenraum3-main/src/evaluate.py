from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent))
from preprocessing import load_all_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # kein Fenster nötig

def train_and_evaluate(data_folder, digits):
    """Trainiert und evaluiert das Modell."""
    
    # ─── 1. DATEN LADEN ──────────────────────────────────────
    print("Lade Daten...")
    X, y = load_all_data(data_folder, digits)
    print(f"{len(X)} Aufnahmen geladen, {len(digits)} Klassen")
    
    if len(X) == 0:
        print("Keine Daten gefunden!")
        return
    
    # ─── 2. LABELS ENCODIEREN ────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # ─── 3. MODELL TRAINIEREN ────────────────────────────────
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # ─── 4. CROSS VALIDATION ─────────────────────────────────
    cv = min(3, len(X) // len(digits))  # max 3 folds
    scores = cross_val_score(model, X, y_encoded, cv=cv)
    print(f"\nCross-Validation ({cv} folds):")
    print(f"  Genauigkeit: {scores.mean():.2%} (+/- {scores.std():.2%})")
    
    # ─── 5. MODELL AUF ALLEN DATEN TRAINIEREN ────────────────
    model.fit(X, y_encoded)
    y_pred = model.predict(X)
    
    # ─── 6. CLASSIFICATION REPORT ────────────────────────────
    print("\nClassification Report:")
    print(classification_report(
        y_encoded, y_pred,
        target_names=[f"Digit {d}" for d in le.classes_]
    ))
    
    # ─── 7. CONFUSION MATRIX ─────────────────────────────────
    cm = confusion_matrix(y_encoded, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # ─── 8. CONFUSION MATRIX PLOTTEN ─────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    classes = [f"Digit {d}" for d in le.classes_]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max()/2 else "black")
    
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    
    output_path = data_folder / "confusion_matrix.png"
    plt.savefig(output_path)
    print(f"\nConfusion Matrix gespeichert: {output_path}")
    
    # ─── 9. FEATURE IMPORTANCE ───────────────────────────────
    feature_names = []
    sensors = ["Acc_x", "Acc_y", "Acc_z", "Gyr_x", "Gyr_y", "Gyr_z"]
    stats = ["mean", "std", "max", "min"]
    for sensor in sensors:
        for stat in stats:
            feature_names.append(f"{sensor}_{stat}")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    print("\nTop 10 wichtigste Features:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    return model, le

if __name__ == "__main__":
    data_folder = Path("/Users/isakilic/Downloads/real_data")
    
    # Ändere digits wenn du mehr Daten hast!
    digits = [0, 1, 2, 3]  # → [0, 1, 2, 3] wenn alle Daten da sind
    
    train_and_evaluate(data_folder, digits)