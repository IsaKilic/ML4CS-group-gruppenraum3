# Air-Writing Live Demo — Anleitung (Gruppenraum 3)

LSTM-basierte Ziffernerkennung (0–9) mit Live-Web-Demo.
Pipeline: Phyphox (iPhone) → Flask Backend → Browser.

---

## Schritt 0 — Installation (einmalig)

```bash
pip3 install tensorflow flask requests scikit-learn pandas numpy
```

Falls TensorFlow auf dem Mac Probleme macht:
```bash
pip3 install tensorflow-macos
```

---

## Schritt 1 — Daten sammeln (WICHTIG: neues Phyphox-Experiment!)

1. Phyphox öffnen → **"Beschleunigung (ohne g)"** Experiment suchen
   - NICHT das normale "Beschleunigung mit g"! Das "ohne g" subtrahiert
     die Gravitation → Handhaltung wird egal (Prof-Feedback!)
2. Ein eigenes Experiment erstellen das BEIDE Sensoren enthält:
   - Phyphox → "+" → "Einfaches Experiment hinzufügen"
   - Sensoren: Beschleunigung (ohne g) + Gyroskop
3. Pro Ziffer (0–9) mindestens 30 Aufnahmen PRO PERSON:
   - Aufnahme starten → Ziffer in die Luft schreiben (2–3 Sek) → stoppen
   - Als ZIP exportieren, benennen: `Digit5-01.zip`, `Digit5-02.zip`, ...
4. Entpacken wie gewohnt:
   ```bash
   cd ~/Downloads
   for i in 01 02 03 ... 30; do
       unzip -o "Digit5-$i.zip" -d "real_data/digit_5_run$i"
   done
   ```

Ziel: 10 Ziffern × 30 Aufnahmen × 3 Personen = 900 Aufnahmen.
(Notfalls reichen auch 10 Ziffern × 30 gesamt für einen ersten Test.)

---

## Schritt 2 — LSTM trainieren

```bash
cd src
python3 train_lstm.py
```

Das Skript:
- Lädt alle Aufnahmen als Zeitreihen (100 Zeitschritte × 6 Achsen)
- Macht 3-fache Kreuzvalidierung (für die Präsentation!)
- Speichert das finale Modell als `lstm_model.keras`

Dauer: ca. 5–15 Minuten je nach Datenmenge.

---

## Schritt 3 — Live-Demo starten

1. **iPhone:** Phyphox öffnen → euer Experiment → Play drücken
   → Menü (⋮) → **"Remote-Zugriff erlauben"**
   → Die angezeigte URL merken (z.B. `http://192.168.2.108:8080`)

2. **Mac:** In `app.py` oben die `PHYPHOX_URL` anpassen.

3. **Buffer-Namen prüfen** (nur beim ersten Mal):
   Browser öffnen: `http://<phyphox-ip>:8080/config`
   → Dort stehen die Buffer-Namen (z.B. `accX` oder `lin_accX`)
   → Falls anders, in `app.py` bei `ACC_BUFFERS` / `GYR_BUFFERS` anpassen.

4. **Backend starten:**
   ```bash
   python3 app.py
   ```

5. **Browser öffnen:** http://localhost:5001
   → Für die Präsentation: Laptop an Beamer, Browser im Vollbild (Cmd+Ctrl+F)

6. **Ziffer in die Luft schreiben** → Vorhersage erscheint automatisch!

WICHTIG: iPhone und Mac müssen im SELBEN WLAN sein.
Eduroam blockiert das oft → persönlichen Hotspot vom zweiten Handy nutzen!

---

## Troubleshooting

| Problem | Lösung |
|---|---|
| "Phyphox nicht erreichbar" | Gleiche WLAN? Remote-Zugriff aktiv? IP richtig? |
| Erkennt Geste nicht | `START_THRESHOLD` in app.py senken (z.B. 1.0) |
| Geste endet zu früh | `END_SILENCE_SEC` erhöhen (z.B. 0.8) |
| Vorhersage immer falsch | Buffer-Namen prüfen! `/config` aufrufen |
| TensorFlow Fehler beim Laden | Gleiche TF-Version wie beim Training nutzen |

---

## Architektur (für die Präsentation)

```
📱 iPhone (Phyphox Remote)
      │ WLAN, JSON, alle 100ms
      ▼
💻 Flask Backend (app.py)
   ├─ Automatische Segmentierung (Schwellwert-basiert)
   ├─ Resampling auf 100 Zeitschritte (Interpolation, Al Abir et al. 2021)
   └─ LSTM-Vorhersage (Architektur nach Bastas et al. 2020)
      │ HTTP-Polling, alle 200ms
      ▼
🌐 Webseite (Live-Graph + Vorhersage + Wahrscheinlichkeiten)
```

## Quellen (im Bericht zitieren!)

- Bastas et al. (2020): Air-Writing Recognition using Deep CNN and RNN Architectures, ICFHR
- Al Abir et al. (2021): Deep Learning Based Air-Writing Recognition with the
  Choice of Proper Interpolation Technique, Sensors (MDPI)
- LSTM-Architektur angelehnt an: github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
