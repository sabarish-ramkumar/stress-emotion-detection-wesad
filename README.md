# Stress and Emotion Detection using Physiological Signals (WESAD)

A complete end-to-end machine learning mini-project that classifies **Baseline / Stress / Amusement** states from chest-worn physiological sensor data using classical ML models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

## Overview

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset, focusing on chest-worn (RespiBAN) signals: ECG, EDA, EMG, Respiration, Temperature, and Accelerometer data sampled at 700 Hz.

The pipeline:

1. Loads raw physiological signal data
2. Cleans and windows the signal (60s windows, 50% overlap)
3. Extracts statistical features (HRV, EDA slope, breathing rate, etc.)
4. Trains and tunes 4 classifiers: Logistic Regression, SVM, Random Forest, XGBoost
5. Evaluates with subject-wise train/test split to avoid data leakage
6. Selects the best model based on macro F1-score

## Dataset Description

**Dataset:** Wearable Stress and Affect Detection (**WESAD**)

**Source:** Schmidt et al., 2018 — UCI Machine Learning Repository

**Citation:** Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018, Boulder, USA, 2018.

**Download:** https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

**Device:** The Raw sensor data was recorded with two devices: a chest-worn device (RespiBAN) and a wrist-worn device (Empatica E4).
The RespiBAN device provides the following sensor data: electrocardiogram (ECG), electrodermal activity (EDA), electromyogram (EMG), respiration, body temperature, and three-axis acceleration. All signals are sampled at 700 Hz.
The Empatica E4 device provides the following sensor data: blood volume pulse (BVP, 64 Hz), electrodermal activity (EDA, 4 Hz), body temperature (4 Hz), and three-axis acceleration (32 Hz).

Thanks to the creators of this dataset **Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, Kristof Van Laerhoven**

**RespiBAN (Chest-worn)** - used in this project. Sampling rate: 700 Hz for all signals and it was feeded by (RespiBAN Professional)

**Signals used:**

| Signal | Description                                                |
| ------ | ---------------------------------------------------------- |
| ECG    | Electrocardiogram — heart electrical activity              |
| EDA    | Electrodermal Activity — skin conductance (sweat response) |
| EMG    | Electromyogram — muscle activity                           |
| RESP   | Respiration — chest expansion/contraction (breathing)      |
| TEMP   | Body temperature                                           |
| ACC    | 3-axis accelerometer (movement)                            |

**File format:** Each subject has an `SX.pkl` file — a Python dictionary containing `signal['chest']` (dict of arrays), `label` (array of condition codes), and `subject` (ID string).

**Labels used in this project (3-class):**

- `1` → Baseline
- `2` → Stress
- `3` → Amusement

(Labels `0, 4, 5, 6, 7` — transient/meditation/recovery — are dropped.)

- **Source:** [WESAD on UCI ML Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
- **Subjects:** 15 (S2–S17, excluding S1, S12)
- **Classes used:** Baseline, Stress, Amusement (3-class)

### Setup

1. Download the WESAD ZIP from the link above.
2. Extract and copy each subject's `SX.pkl` file into `data/raw/`.
3. In the notebook, set `USE_SYNTHETIC_DATA = False` and confirm `DATA_DIR` points to `data/raw/`.

> The notebook also includes a **synthetic data generator** so you can run and test the entire pipeline immediately, even before downloading the real dataset.

## Project Structure

```
📁 stress-emotion-detection-wesad/
├── README.md
├── requirements.txt
├── 📁 data/
│   ├── raw/                 # place SX.pkl files here (gitignored)
├── 📁 notebooks/
│   └── Stress_Emotion_Detection.ipynb
├── 📁models/
|    ├── feature_scaler.pkl       # Fitted StandardScaler instance
     ├── selected_features.pkl    # Serialized list of columns passing VarianceThreshold
     └── final_model.pkl          # Exported serialization of the optimized classifier
```

## How to Run

```bash
# Initialize and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Installing Requirements
pip install -r requirements.txt

#Launching Notebook File
jupyter notebook notebooks/ Stress_Emotion_Detection.ipynb
```

Run all cells top to bottom. With synthetic data, the full pipeline runs in under a minute.

## Models Compared

| Model               | Notes                                                  |
| ------------------- | ------------------------------------------------------ |
| Logistic Regression | Linear baseline                                        |
| SVM (RBF kernel)    | Non-linear, margin-based                               |
| Random Forest       | Tree ensemble, gives feature importance                |
| XGBoost             | Gradient-boosted trees, typically best on tabular data |

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score (macro-averaged), Confusion Matrix, ROC Curve (one-vs-rest).

## Key Design Decisions

- **Subject-wise train/test split** — prevents data leakage between windows from the same person.
- **60-second windows, 50% overlap** — balances feature stability with sample count.
- **Feature engineering over raw signals** — HRV (SDNN, RMSSD), EDA slope, breathing rate are well-established physiological stress markers.

## References

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). _Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection._ ICMI 2018.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
