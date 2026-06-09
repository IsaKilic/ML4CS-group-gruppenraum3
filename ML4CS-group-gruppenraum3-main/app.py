# app.py
# Live-Demo Backend:
#   Phyphox (Remote-Zugriff) --> Flask --> Webseite
#
# Ablauf:
#   1. Phyphox auf dem iPhone: Experiment starten, "Remote-Zugriff erlauben" aktivieren
#   2. PHYPHOX_URL unten anpassen (steht in der Phyphox-App, z.B. http://192.168.2.108:8080)
#   3. python3 app.py
#   4. Browser oeffnen: http://localhost:5001

import threading
import time
import requests
import numpy as np
from collections import deque
from pathlib import Path
from flask import Flask, jsonify, render_template

# ════════════════ EINSTELLUNGEN ════════════════
PHYPHOX_URL = "http://192.168.2.108:8080"   # <-- IP aus der Phyphox App eintragen!
DATA_FOLDER = Path("U:\Airwriting\real_data")
SEQUENCE_LENGTH = 100

# Buffer-Namen in Phyphox. Diese haengen vom Experiment ab!
# Pruefen unter: http://<PHYPHOX_URL>/config
# Standard "Beschleunigung (ohne g)" + Gyroskop im selben Experiment:
ACC_BUFFERS = ["accX", "accY", "accZ"]
GYR_BUFFERS = ["gyrX", "gyrY", "gyrZ"]

# Segmentierung: wann beginnt/endet eine Geste?
START_THRESHOLD = 1.5     # m/s^2 — Bewegungsstaerke ab der die Geste beginnt
END_SILENCE_SEC = 0.6     # Sekunden Ruhe = Geste ist zu Ende
MIN_GESTURE_SEC = 0.5     # kuerzere "Gesten" werden ignoriert (Wackler)
POLL_INTERVAL = 0.1       # alle 100ms neue Daten von Phyphox holen
# ═══════════════════════════════════════════════

app = Flask(__name__)

# Geteilter Zustand zwischen Polling-Thread und Webserver
state = {
    "connected": False,
    "recording": False,        # laeuft gerade eine Geste?
    "prediction": None,        # letzte erkannte Ziffer
    "confidence": 0.0,
    "probabilities": {},       # {"0": 0.05, "1": 0.87, ...}
    "live_signal": [],         # letzte ~3s Beschleunigungs-Magnitude fuer den Graph
    "status_text": "Warte auf Phyphox...",
}

# Modell laden
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model(DATA_FOLDER / "lstm_model.keras")
label_map = np.load(DATA_FOLDER / "label_map.npy")
norm_mean = np.load(DATA_FOLDER / "norm_mean.npy")
norm_std = np.load(DATA_FOLDER / "norm_std.npy")
print(f"Modell geladen. Klassen: {label_map.tolist()}")


def resample(signal, target_len=SEQUENCE_LENGTH):
    x_old = np.linspace(0, 1, num=len(signal))
    x_new = np.linspace(0, 1, num=target_len)
    return np.interp(x_new, x_old, signal)


def fetch_phyphox():
    """Holt die aktuellen Sensor-Buffer von Phyphox."""
    buffers = ACC_BUFFERS + GYR_BUFFERS
    query = "&".join(f"{b}=full" for b in buffers)
    r = requests.get(f"{PHYPHOX_URL}/get?{query}", timeout=2)
    data = r.json()["buffer"]
    result = {}
    for b in buffers:
        result[b] = np.array(data[b]["buffer"], dtype=np.float64)
    return result


def clear_phyphox():
    """Setzt die Phyphox-Buffer zurueck (nach jeder Geste)."""
    try:
        requests.get(f"{PHYPHOX_URL}/control?cmd=clear", timeout=2)
    except Exception:
        pass


def predict_gesture(gesture_data):
    """gesture_data: dict mit 6 Kanaelen -> Vorhersage."""
    channels = []
    for b in ACC_BUFFERS + GYR_BUFFERS:
        sig = gesture_data[b]
        if len(sig) < 10:
            return  # zu kurz
        channels.append(resample(sig))
    X = np.stack(channels, axis=1)[np.newaxis, ...]  # (1, 100, 6)
    X = (X - norm_mean) / norm_std

    probs = model.predict(X, verbose=0)[0]
    pred_idx = int(probs.argmax())

    state["prediction"] = int(label_map[pred_idx])
    state["confidence"] = float(probs[pred_idx])
    state["probabilities"] = {str(int(label_map[i])): float(p) for i, p in enumerate(probs)}
    state["status_text"] = f"Erkannt: {label_map[pred_idx]} ({probs[pred_idx]*100:.0f}%)"
    print(state["status_text"])


def polling_loop():
    """Hintergrund-Thread: holt Daten, segmentiert, sagt vorher."""
    gesture = {b: [] for b in ACC_BUFFERS + GYR_BUFFERS}
    in_gesture = False
    silence_start = None
    gesture_start = None
    last_len = 0

    while True:
        try:
            data = fetch_phyphox()
            state["connected"] = True
        except Exception:
            state["connected"] = False
            state["status_text"] = "Phyphox nicht erreichbar — Remote-Zugriff aktiv?"
            time.sleep(1)
            continue

        n = len(data[ACC_BUFFERS[0]])
        if n == 0 or n == last_len:
            time.sleep(POLL_INTERVAL)
            continue

        # Nur die NEUEN Samples seit letztem Poll
        new_slice = slice(last_len, n)
        last_len = n

        acc_mag = np.sqrt(
            data[ACC_BUFFERS[0]][new_slice] ** 2
            + data[ACC_BUFFERS[1]][new_slice] ** 2
            + data[ACC_BUFFERS[2]][new_slice] ** 2
        )

        # Live-Signal fuer den Graph (letzte ~300 Werte)
        state["live_signal"] = (state["live_signal"] + acc_mag.tolist())[-300:]

        moving = acc_mag.max() > START_THRESHOLD if len(acc_mag) else False
        now = time.time()

        if not in_gesture:
            if moving:
                in_gesture = True
                gesture_start = now
                silence_start = None
                gesture = {b: [] for b in gesture}
                state["recording"] = True
                state["status_text"] = "Geste laeuft..."
        if in_gesture:
            for b in gesture:
                gesture[b].extend(data[b][new_slice].tolist())
            if moving:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = now
                elif now - silence_start > END_SILENCE_SEC:
                    # Geste zu Ende
                    in_gesture = False
                    state["recording"] = False
                    duration = now - gesture_start
                    if duration >= MIN_GESTURE_SEC:
                        state["status_text"] = "Berechne Vorhersage..."
                        predict_gesture({b: np.array(v) for b, v in gesture.items()})
                    else:
                        state["status_text"] = "Zu kurz — ignoriert"
                    clear_phyphox()
                    last_len = 0
                    state["live_signal"] = []

        time.sleep(POLL_INTERVAL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify(state)


if __name__ == "__main__":
    t = threading.Thread(target=polling_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5001, debug=False)
