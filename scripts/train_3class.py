"""
train_3class.py

Trains Random Forest and XGBoost as 3-class classifiers using the source
column to distinguish sub-types of malicious URLs:

  0 = Benign        (majestic_million, tranco_top10k, kaggle_malicious_urls,
                     brand_subdomains)
  1 = Phishing      (phishtank, openphish, urlhaus)
  2 = Typosquat/DGA (generated_typosquat, generated_dga)

Outputs:
  output/figures/cm_rf_3class.png
  output/figures/cm_xgb_3class.png
  output/rf_3class_report.txt
  output/xgb_3class_report.txt
  models/random_forest_3class.joblib
  models/xgboost_3class.joblib

Usage:
    python scripts/train_3class.py
    python scripts/train_3class.py --features data/features_clean_v2.csv
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "features_clean_v2.csv"
OUT          = ROOT / "output"
FIGS         = OUT / "figures"
MODELS       = ROOT / "models"
RANDOM_SEED  = 42

CLASS_NAMES = ["Benign", "Phishing", "Typo/DGA"]

PHISHING_SOURCES  = {"phishtank", "openphish", "urlhaus"}
TYPO_DGA_SOURCES  = {"generated_typosquat", "generated_dga"}
BENIGN_SOURCES    = {"majestic_million", "tranco_top10k",
                     "kaggle_malicious_urls", "brand_subdomains"}

FEATURE_COLS = [
    "domain_length", "subdomain_count", "digit_count", "hyphen_count",
    "non_alnum_count", "vowel_consonant_ratio", "entropy",
    "char_continuity_rate", "digit_ratio", "tld_risk_score",
    "min_lev_distance", "homoglyph_count", "bigram_log_prob",
    "url_length", "path_length", "path_depth", "has_query",
    "query_length", "path_entropy", "at_in_url", "double_slash_in_path",
]


def make_3class_label(source: str) -> int | None:
    if source in BENIGN_SOURCES:
        return 0
    if source in PHISHING_SOURCES:
        return 1
    if source in TYPO_DGA_SOURCES:
        return 2
    return None


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    """Save a heatmap confusion matrix figure styled like the reference image."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize for colour intensity, keep raw counts as labels
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Prediction",
        ylabel="Actual",
        title=title,
    )
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    thresh = cm_norm.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {out_path}")


def print_cm_table(cm: np.ndarray, label: str) -> list[str]:
    lines = [
        f"  3-class Confusion Matrix [{label}]",
        f"  (rows = actual, cols = predicted)",
        f"  {'':20s}  {'Pred Benign':>12}  {'Pred Phishing':>13}  {'Pred Typo/DGA':>13}",
        "  " + "-" * 65,
    ]
    row_labels = ["Actual Benign   ", "Actual Phishing ", "Actual Typo/DGA "]
    for i, rl in enumerate(row_labels):
        lines.append(
            f"  {rl}  {cm[i,0]:>12,}  {cm[i,1]:>13,}  {cm[i,2]:>13,}"
        )
    lines.append("")

    # Per-class recall
    lines.append("  Per-class recall:")
    for i, cn in enumerate(CLASS_NAMES):
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        lines.append(f"    {cn:<14}  {rec:.1%}")
    return lines


def train_rf(X_train, y_train, X_test, y_test, suffix: str) -> tuple[np.ndarray, list[str]]:
    print("\nTraining Random Forest (3-class) ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    plot_confusion_matrix(
        cm, "Random Forest — 3-Class Confusion Matrix",
        FIGS / f"cm_rf_3class{suffix}.png",
    )

    lines = [
        "=" * 65,
        "  Random Forest — 3-Class Evaluation Report",
        "=" * 65,
        f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}",
        "",
    ]
    lines += print_cm_table(cm, "Random Forest")
    lines.append("")
    for line in classification_report(
        y_test, y_pred, target_names=CLASS_NAMES
    ).splitlines():
        lines.append("  " + line)
    lines += ["", "=" * 65]

    joblib.dump({"model": clf}, MODELS / f"random_forest_3class{suffix}.joblib")
    print(f"  RF model saved -> {MODELS / f'random_forest_3class{suffix}.joblib'}")
    return cm, lines


def train_xgb(X_train, y_train, X_test, y_test, suffix: str) -> tuple[np.ndarray, list[str]]:
    print("\nTraining XGBoost (3-class) ...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    plot_confusion_matrix(
        cm, "XGBoost — 3-Class Confusion Matrix",
        FIGS / f"cm_xgb_3class{suffix}.png",
    )

    lines = [
        "=" * 65,
        "  XGBoost — 3-Class Evaluation Report",
        "=" * 65,
        f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}",
        "",
    ]
    lines += print_cm_table(cm, "XGBoost")
    lines.append("")
    for line in classification_report(
        y_test, y_pred, target_names=CLASS_NAMES
    ).splitlines():
        lines.append("  " + line)
    lines += ["", "=" * 65]

    joblib.dump({"model": clf}, MODELS / f"xgboost_3class{suffix}.joblib")
    print(f"  XGB model saved -> {MODELS / f'xgboost_3class{suffix}.joblib'}")
    return cm, lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURES_CSV))
    args = parser.parse_args()

    features_path = Path(args.features)
    tag    = features_path.stem.replace("features", "").strip("_")
    suffix = f"_{tag}" if tag else ""

    print(f"Loading {features_path.name} ...")
    df = pd.read_csv(features_path)

    # Build 3-class label from source column
    df["label3"] = df["source"].map(make_3class_label)
    dropped = df["label3"].isna().sum()
    if dropped:
        print(f"  Dropping {dropped} rows with unknown source")
        df = df.dropna(subset=["label3"])
    df["label3"] = df["label3"].astype(int)

    counts = df["label3"].value_counts().sort_index()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts.get(i, 0):,}")
    print(f"  Total: {len(df):,}")

    X = df[FEATURE_COLS].values
    y = df["label3"].values

    # 60/20/20 split stratified by 3-class label
    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val,   y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"\n  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    rf_cm,  rf_lines  = train_rf (X_train, y_train, X_test, y_test, suffix)
    xgb_cm, xgb_lines = train_xgb(X_train, y_train, X_test, y_test, suffix)

    # Save reports
    (OUT / f"rf_3class{suffix}_report.txt").write_text(
        "\n".join(rf_lines), encoding="utf-8"
    )
    (OUT / f"xgb_3class{suffix}_report.txt").write_text(
        "\n".join(xgb_lines), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("  Random Forest")
    print("\n".join(rf_lines))
    print("\n  XGBoost")
    print("\n".join(xgb_lines))


if __name__ == "__main__":
    main()
