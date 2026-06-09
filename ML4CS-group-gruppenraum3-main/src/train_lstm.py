# train_lstm.py
# Trainiert ein LSTM auf den rohen Zeitreihen und evaluiert mit Kreuzvalidierung.
# Architektur angelehnt an: Bastas et al. (2020), Guillaume Chevalier (LSTM-HAR, GitHub)

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing_lstm import load_all_sequences, SEQUENCE_LENGTH

DATA_FOLDER = Path("U:\Airwriting\real_data")
DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # bei weniger Daten: [0,1,2,3]
MODEL_PATH = DATA_FOLDER / "lstm_model.keras"


def build_model(n_classes):
    """LSTM-Modell: bewusst klein gehalten gegen Overfitting bei kleinen Datensaetzen."""
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, 6)),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Lade Daten...")
    X, y = load_all_sequences(DATA_FOLDER, DIGITS)
    n_classes = len(set(y.tolist()))

    # Labels auf 0..n-1 mappen (falls Ziffern fehlen)
    unique = sorted(set(y.tolist()))
    label_map = {d: i for i, d in enumerate(unique)}
    y_mapped = np.array([label_map[v] for v in y])

    # ── 3-fache Kreuzvalidierung (stratifiziert = balanciert pro Fold) ──
    print("\n=== 3-fache Kreuzvalidierung ===")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_mapped), 1):
        model = build_model(n_classes)
        early = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(
            X[train_idx], y_mapped[train_idx],
            validation_split=0.2,
            epochs=150, batch_size=16,
            callbacks=[early], verbose=0,
        )
        _, acc = model.evaluate(X[test_idx], y_mapped[test_idx], verbose=0)
        accs.append(acc)
        preds = model.predict(X[test_idx], verbose=0).argmax(axis=1)
        all_true.extend(y_mapped[test_idx])
        all_pred.extend(preds)
        print(f"Fold {fold}: {acc*100:.1f}%")

    print(f"\nKreuzvalidierung: {np.mean(accs)*100:.1f}% (+/- {np.std(accs)*100:.1f}%)")
    print("\nClassification Report:")
    names = [f"Digit {d}" for d in unique]
    print(classification_report(all_true, all_pred, target_names=names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_pred))

    # ── Finales Modell auf ALLEN Daten trainieren und speichern ──
    print("\nTrainiere finales Modell auf allen Daten...")
    final = build_model(n_classes)
    early = keras.callbacks.EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True)
    final.fit(X, y_mapped, epochs=150, batch_size=16, callbacks=[early], verbose=0)
    final.save(MODEL_PATH)
    np.save(DATA_FOLDER / "label_map.npy", np.array(unique))
    print(f"Modell gespeichert: {MODEL_PATH}")


if __name__ == "__main__":
    main()
