# Week 03 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Set up preprocessing and training pipeline, and test it with real sensor data from a smartphone.

## Work Done This Week

### 1. Data Work
Switched from Apple Watch to smartphone (via Phyphox app) as the primary data source.
Recorded first real sensor data using Phyphox — both Accelerometer and Gyroscope simultaneously exported as CSV.
Data structure: `seconds_elapsed, x, y, z` for both sensors.
Currently only digit "3" recorded (10 dummy runs already in place from last week).

### 2. Analysis / Modeling Work
Implemented and tested full preprocessing pipeline in `preprocessing.py`:
- Loads Accelerometer and Gyroscope CSVs from each run folder
- Extracts 24 statistical features per recording (mean, std, max, min for each axis of both sensors)
- Returns feature matrix `(n_samples, 24)` and label array

Implemented `train.py` with a Random Forest Classifier:
- Cross-validation (cv=3) on current data
- Achieved 100% accuracy on single-class data (digit 3 only — expected)
- Pipeline ready for multi-class training once more digits are recorded

Visualised real Gyroscope data in `accelerometer.py` — confirmed clean signal with clear motion pattern visible between seconds 1–5.

### 3. Repository / Documentation Work
Scripts pushed to GitHub:
- `preprocessing.py` — feature extraction pipeline
- `train.py` — model training with Random Forest
- `accelerometer.py` — sensor data visualisation

## Experiments Conducted

| Experiment | Change Made | Result | Interpretation |
|-----------|-------------|--------|----------------|
| Real data recording | Used Phyphox instead of Apple Watch | Clean CSV export with acc + gyro | Smartphone is a viable alternative |
| Preprocessing pipeline | Ran on 10 digit_3 recordings | Feature matrix (10, 24) generated | Pipeline works correctly on real data |
| Random Forest training | Trained on digit_3 only | 100% cross-val accuracy | Expected for single class — needs more digits |

## Results
Full pipeline from raw sensor data to trained model is working end-to-end on real smartphone data.

## Challenges
- Apple Watch still not available — continuing with smartphone via Phyphox
- Only digit "3" recorded so far — multi-class training not yet possible
- Data collection for digits 0, 1, 2 still pending

## Key Insights
- Phyphox exports clean, well-structured CSVs — easy to work with
- Feature extraction with statistical measures (mean, std, max, min) works well as a baseline
- Pipeline is modular and ready to scale to more digits

## Plan for Next Week
- Record 10 runs each for digits 0, 1, 2 using Phyphox
- Retrain model on all 4 classes (0–3)
- Evaluate real multi-class accuracy with cross-validation
- Start thinking about feature engineering improvements

## Contributions
- Member 1 : Isa Kilic
- Member 2: Efe Özüm
- Member 3: Tony Duong
