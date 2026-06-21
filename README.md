# Stress and Emotion Detection using Physiological Signals (WESAD)

A complete end-to-end machine learning mini-project that classifies **Baseline / Stress / Amusement** states from chest-worn physiological sensor data using classical ML models.

## Overview

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset, focusing on chest-worn (RespiBAN) signals: ECG, EDA, EMG, Respiration, Temperature, and Accelerometer data sampled at 700 Hz.

The pipeline:
1. Loads raw physiological signal data
2. Cleans and windows the signal (60s windows, 50% overlap)
3. Extracts statistical features (HRV, EDA slope, breathing rate, etc.)
4. Trains and tunes 4 classifiers: Logistic Regression, SVM, Random Forest, XGBoost
5. Evaluates with subject-wise train/test split to avoid data leakage
6. Selects the best model based on macro F1-score

## Dataset

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
stress-emotion-detection-wesad/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # place SX.pkl files here (gitignored)
│   └── processed/
├── notebooks/
│   └── WESAD_Stress_Detection.ipynb
├── reports/
└── presentation/
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/WESAD_Stress_Detection.ipynb
```

Run all cells top to bottom. With synthetic data, the full pipeline runs in under a minute.

## Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| SVM (RBF kernel) | Non-linear, margin-based |
| Random Forest | Tree ensemble, gives feature importance |
| XGBoost | Gradient-boosted trees, typically best on tabular data |

## Evaluation Metrics

Accuracy, Precision, Recall, F1-Score (macro-averaged), Confusion Matrix, ROC Curve (one-vs-rest).

## Key Design Decisions

- **Subject-wise train/test split** — prevents data leakage between windows from the same person.
- **60-second windows, 50% overlap** — balances feature stability with sample count.
- **Feature engineering over raw signals** — HRV (SDNN, RMSSD), EDA slope, breathing rate are well-established physiological stress markers.

## References

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* ICMI 2018.

## License

MIT
