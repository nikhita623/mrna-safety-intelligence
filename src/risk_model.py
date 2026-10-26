"""
Week 1 starter: inspect, clean, train and compare baseline classifiers.

Safety note:
This supplied dataset appears to be a teaching/case-study dataset. The
Reaction_Score label is treated only as a dataset label, NOT as a validated
clinical outcome. Do not use predictions to make vaccine or dosage decisions.
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "patient_records.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CLEAN_PATH = ROOT / "data" / "patient_records_clean.csv"
TARGET = "Reaction_Score"

NUMERIC_FEATURES = [
    "Age", "Biomarker_Level", "Pre_Existing_Conditions", "Vaccine_Dose_mcg"
]
CATEGORICAL_FEATURES = ["Prior_Reaction_History"]


def load_and_profile():
    """Load data, print a basic profile, and create a cleaned CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put patient_records.csv in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("\n=== DATASET PROFILE ===")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print("\nColumn types:\n", df.dtypes.to_string())
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("\nSummary statistics:\n", df.describe(include="all").to_string())
    print("\nTarget counts:\n", df[TARGET].value_counts(dropna=False).to_string())

    # Preserve the raw source; this cleaned copy is for inspection only.
    clean = df.copy()
    if "Biomarker_Level" in clean:
        clean["Biomarker_Level"] = clean["Biomarker_Level"].fillna(
            clean["Biomarker_Level"].median()
        )
    if "Prior_Reaction_History" in clean:
        clean["Prior_Reaction_History"] = (
            clean["Prior_Reaction_History"].fillna("Unknown").astype(str)
        )
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned inspection copy: {CLEAN_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: target class balance.
    plt.figure(figsize=(6, 4))
    counts = clean[TARGET].value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Dataset label distribution (Reaction_Score)")
    plt.xlabel("Reaction_Score label")
    plt.ylabel("Record count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # Plot 2: age distribution.
    plt.figure(figsize=(6, 4))
    plt.hist(clean["Age"].dropna(), bins=20, edgecolor="black")
    plt.title("Age distribution in supplied dataset")
    plt.xlabel("Age")
    plt.ylabel("Record count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "age_distribution.png", dpi=150)
    plt.close()

    # Plot 3: numeric-feature correlation.
    numeric = clean[NUMERIC_FEATURES + [TARGET]].copy()
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, interpolation="nearest", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Numeric feature correlation (descriptive only)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "numeric_correlation.png", dpi=150)
    plt.close(fig)

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def train_and_compare(df):
    """Compare two baseline classifiers against the dataset's binary label."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    if df[TARGET].isna().any():
        raise ValueError("Target has missing values; review labels before training.")
    if df[TARGET].nunique() != 2:
        raise ValueError("This starter expects a binary Reaction_Score label.")

    # Patient_ID is intentionally excluded: it is an identifier, not a feature.
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"
            )),
        ]),
    }

    results = {}
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(
                y_test, predictions, average="macro", zero_division=0
            )),
            "recall_macro": float(recall_score(
                y_test, predictions, average="macro", zero_division=0
            )),
            "f1_macro": float(f1_score(
                y_test, predictions, average="macro", zero_division=0
            )),
        }
        results[name] = metrics
        fitted[name] = pipeline
        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2))
        print("\nClassification report:\n",
              classification_report(y_test, predictions, zero_division=0))

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Label 0", "Label 1"]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion matrix — {name}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()

        (OUTPUT_DIR / f"classification_report_{name}.txt").write_text(
            classification_report(y_test, predictions, zero_division=0),
            encoding="utf-8"
        )

    # Select by macro F1 for this educational baseline comparison.
    best_name = max(results, key=lambda key: results[key]["f1_macro"])
    best_model = fitted[best_name]
    joblib.dump(best_model, OUTPUT_DIR / "risk_model.joblib")
    (OUTPUT_DIR / "model_comparison.json").write_text(
        json.dumps({"metrics": results, "selected_baseline": best_name}, indent=2),
        encoding="utf-8"
    )
    print(f"\nSelected baseline by test macro-F1: {best_name}")
    print(f"Saved model: {OUTPUT_DIR / 'risk_model.joblib'}")
    print("Reminder: scores are dataset-specific and are not clinical validation.")
    return results


if __name__ == "__main__":
    data = load_and_profile()
    train_and_compare(data)
