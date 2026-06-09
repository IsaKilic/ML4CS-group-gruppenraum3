# Week 02 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal
Set up the data collection pipeline and test it with synthetic data while waiting for the Apple Watch.

## Work Done This Week

### 0. Project setup
Project idea confirmed: air-writing digit recognition (0–9) using Apple Watch accelerometer and gyroscope data.

### 1. Data Work
No real data collected yet — Apple Watch still not available. Synthetic recordings generated with `make_dummy_data.py` to test the pipeline.

### 2. Analysis / Modeling Work
Implemented `plot_recordings.py` to load and visualise sensor recordings. Pipeline tested end-to-end on 10 dummy recordings of "digit 3" — plots rendered correctly.

### 3. Repository / Documentation Work
Pipeline scripts (`make_dummy_data.py`, `plot_recordings.py`) pushed to GitHub. Folder structure in place.

## Experiments Conducted

| Experiment | Change Made | Result | Interpretation |
|-----------|-------------|--------|----------------|
| Dummy pipeline test | Ran plot_recordings.py on 10 synthetic recordings | All 10 plots rendered correctly | Pipeline is ready for real data |

## Results
Pipeline verified on synthetic data. Ready to switch to real Apple Watch recordings as soon as the device is available.

## Challenges
Apple Watch still not available — real data collection delayed again.

## Key Insights
Having a working pipeline before real data arrives means no time is lost once the watch is here.

## Plan for Next Week
- Receive Apple Watch and install Sensor Logger
- Record pilot dataset: digit "3", 10 recordings
- Visualise real data and check signal quality
- Start literature scan on gesture / air-writing recognition

## Contributions
- Member 1 (Team Captain): Isa Kilic 
- Member 2: Efe Özüm 
- Member 3: Tony Duong 