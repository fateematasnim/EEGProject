import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (accuracy_score,
                                    confusion_matrix,
                                    classification_report)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_PATH   = "/Users/fateematasnim/Downloads/features_raw.csv"
PLOTS_DIR   = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 128
RANDOM_SEED = 42

# Load and Clean

def load_signal(path):
    df = pd.read_csv(path)
    print("Raw shape:", df.shape)

    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]


    df = df.ffill().bfill()

    print("Cleaned shape:", df.shape)
    print("Channels:", list(df.columns))
    return df

signal_df = load_signal(DATA_PATH)

def band_power(signal, fs, band):
    """Power in a frequency band using Welch's method."""
    from scipy.signal import welch
    f, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(psd[mask], f[mask])

BANDS = {"delta": (0.5, 4), "theta": (4, 8),
         "alpha": (8, 13),  "beta":  (13, 30), "gamma": (30, 45)}

# Feature extraction

def extract_features(signal_df, fs=SAMPLE_RATE):
    """Returns a dict {feature_name: value} for one trial."""
    feats = {}
    for ch in signal_df.columns:
        x = signal_df[ch].values
        feats[f"{ch}_mean"] = x.mean()
        feats[f"{ch}_std"]  = x.std()
        feats[f"{ch}_min"]  = x.min()
        feats[f"{ch}_max"]  = x.max()
        feats[f"{ch}_skew"] = skew(x)
        feats[f"{ch}_kurt"] = kurtosis(x)
        for band_name, band in BANDS.items():
            feats[f"{ch}_{band_name}"] = band_power(x, fs, band)
    return feats

sample_feats = extract_features(signal_df)
print(f"Extracted {len(sample_feats)} features for one trial.")
print("First 5 features:", dict(list(sample_feats.items())[:5]))

# Build the training table

def build_dataset_from_windows(signal_df, window_sec=4, fs=SAMPLE_RATE):
    """
    Split one long signal into non-overlapping windows, extract features from
    each, and generate DEMO labels so the ML pipeline has something to learn.

    In a real DEAP project you'd:
      for each (participant, trial) CSV:
          feats = extract_features(load_signal(path))
          rows.append(feats); labels.append(labels_df.loc[(p, t)])
    """
    win = window_sec * fs
    rows, labels = [], []
    for start in range(0, len(signal_df) - win + 1, win):
        chunk = signal_df.iloc[start:start + win]
        feats = extract_features(chunk)
        rows.append(feats)
        labels.append(chunk[["Fp1", "F3", "AF3"]].std().mean())
    X = pd.DataFrame(rows)
    y_cont = np.array(labels)
    median = np.median(y_cont)
    y = np.where(y_cont > median, "POSITIVE", "NEGATIVE")
    return X, y

X, y = build_dataset_from_windows(signal_df)
print("Feature table:", X.shape, "| Label distribution:", pd.Series(y).value_counts().to_dict())
X.to_csv("features_engineered.csv", index=False)

# EDA

def run_eda(X, y):
    print(X.describe().iloc[:, :5])  # summary of first 5 features

    plt.figure(figsize=(5, 3))
    sns.countplot(x=y)
    plt.title("Class balance")
    plt.savefig(PLOTS_DIR / "01_class_balance.png", dpi=120); plt.close()

    X.iloc[:, :6].hist(bins=20, figsize=(10, 6))
    plt.suptitle("Distributions (first 6 features)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_distributions.png", dpi=120); plt.close()

    plt.figure(figsize=(9, 7))
    sns.heatmap(X.iloc[:, :25].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation (first 25 features)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_correlation.png", dpi=120); plt.close()

run_eda(X, y)

# Preprocess and Split

def preprocess_split(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc,
        test_size=0.20,
        stratify=y_enc,
        random_state=RANDOM_SEED,
    )
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    return X_tr, X_te, y_tr, y_te, le

X_tr, X_te, y_tr, y_te, le = preprocess_split(X, y)

# Train two models
def train_models(X_tr, y_tr):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=300,
                                                      random_state=RANDOM_SEED,
                                                      n_jobs=-1),
    }
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        print(f"{name}: trained.")
    return models

models = train_models(X_tr, y_tr)


# Evaluate
def evaluate(models, X_te, y_te, class_names):
    results = {}
    for name, m in models.items():
        y_pred = m.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        cm     = confusion_matrix(y_te, y_pred)
        report = classification_report(y_te, y_pred, target_names=class_names, digits=3)
        results[name] = acc

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(report)

        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion matrix — {name}")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        plt.savefig(PLOTS_DIR / f"04_cm_{fname}.png", dpi=120); plt.close()
    return results

results = evaluate(models, X_te, y_te, le.classes_)
best = max(results, key=results.get)
print(f"\nBest model: {best} (accuracy {results[best]:.3f})")

# feature importance

def plot_importance(rf, feature_names, top_n=20):
    imp = pd.Series(rf.feature_importances_, index=feature_names) \
        .sort_values(ascending=False)
    print(imp.head(top_n))

    plt.figure(figsize=(7, 6))
    imp.head(top_n)[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} features — Random Forest")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=120); plt.close()

plot_importance(models["Random Forest"], X.columns)
