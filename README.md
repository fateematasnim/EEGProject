# EEG Signal Classification using Machine Learning (DEAP Dataset)

## Overview

Electroencephalography (EEG) signals are complex, noisy, and difficult to interpret visually. This project builds an end-to-end machine learning pipeline to extract meaningful features from EEG signals and classify brain states.

Using data inspired by the DEAP dataset, this project demonstrates how signal processing and machine learning can be combined to transform raw brain signals into structured, predictive insights.

---

## Problem Statement

It is difficult to analyze raw EEG signals and directly identify underlying mental or cognitive states. The goal of this project is to extract meaningful features from EEG signals and use machine learning models to classify brain activity patterns.

---

## Dataset

This project uses EEG signal data derived from the DEAP dataset.

The DEAP dataset contains:
- EEG recordings from 32 participants
- Signals captured while participants watched emotional stimuli (music videos)
- Self-reported labels such as arousal, valence, and dominance

In this implementation:
- EEG signals are structured channel-wise
- Time-series data is processed into windows
- Synthetic labels are generated for classification purposes (for demonstration)

---

## Methodology

### 1. Data Preprocessing
- Removed unnecessary columns
- Handled missing values using forward/backward fill
- Structured multichannel EEG data

### 2. Feature Engineering

Extracted both statistical and frequency-domain features:

#### Statistical Features:
- Mean
- Standard Deviation
- Min / Max
- Skewness
- Kurtosis

#### Frequency Features (Band Power):
- Delta (0.5–4 Hz)
- Theta (4–8 Hz)
- Alpha (8–13 Hz)
- Beta (13–30 Hz)
- Gamma (30–45 Hz)

Band power was computed using Welch’s method.

---

### 3. Window-Based Dataset Construction
- EEG signals were split into fixed time windows
- Features extracted per window
- Labels generated based on signal variability (proxy classification)

---

### 4. Exploratory Data Analysis (EDA)
- Feature distributions visualized
- Correlation analysis performed
- Class balance evaluated

---

### 5. Machine Learning Models

Two models were trained:

- Logistic Regression (baseline, interpretable)
- Random Forest (nonlinear, robust)

---

### 6. Model Evaluation

Models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score

---

### 7. Feature Importance

Random Forest feature importance was used to identify the most influential EEG features contributing to predictions.

---

## Results

- Successfully built a full ML pipeline for EEG classification
- Random Forest achieved higher performance due to its ability to capture nonlinear relationships
- Frequency band features (alpha, beta, etc.) showed strong importance

---

## Key Learnings

- Raw EEG signals require extensive feature engineering to become usable
- Frequency-domain features are critical for representing brain activity
- Machine learning models can identify patterns not visible to the human eye
- End-to-end pipelines (data → features → model → evaluation) are essential for real-world applications

---

## Project Structure

├── features_raw.csv
├── features_engineered.csv
├── main.py
├── plots/
│   ├── class_balance.png
│   ├── distributions.png
│   ├── correlation.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SciPy (signal processing)

---

## Future Improvements

- Use real DEAP labels (valence/arousal)
- Apply deep learning models (CNN / LSTM)
- Perform cross-subject validation
- Use advanced signal transforms (Wavelet Transform)

---

## Conclusion

This project demonstrates how EEG signals can be transformed from raw time-series data into structured features and used for machine learning classification. It highlights the intersection of neuroscience, signal processing, and data science in solving real-world problems.
