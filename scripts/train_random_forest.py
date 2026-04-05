"""
train_random_forest.py

Loads data/features.csv, trains a Random Forest classifier,
evaluates on a held-out test set, and saves artefacts to
models/random_forest.joblib and output/rf_report.txt.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "features.csv"
MODEL_PATH   = ROOT / "models" / "random_forest.joblib"
REPORT_PATH  = ROOT / "output" / "rf_report.txt"

RANDOM_SEED  = 42
TEST_SIZE    = 0.20

FEATURE_COLS = [
    "domain_length",
    "subdomain_count",
    "digit_count",
    "hyphen_count",
    "non_alnum_count",
    "vowel_consonant_ratio",
    "entropy",
    "char_continuity_rate",
    "digit_ratio",
    "tld_risk_score",
    "min_lev_distance",
]


def build_report(
    y_test,
    y_pred,
    feature_names: list[str],
    importances,
) -> str:
    lines: list[str] = []

    lines += [
        "=" * 60,
        "  Random Forest — Evaluation Report",
        "=" * 60,
        "",
        f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}",
        f"  Precision : {precision_score(y_test, y_pred):.4f}",
        f"  Recall    : {recall_score(y_test, y_pred):.4f}",
        f"  F1 Score  : {f1_score(y_test, y_pred):.4f}",
        "",
        "  Classification Report",
        "  " + "-" * 40,
    ]
    for line in classification_report(
        y_test, y_pred, target_names=["benign", "malicious"]
    ).splitlines():
        lines.append("  " + line)

    cm = confusion_matrix(y_test, y_pred)
    lines += [
        "",
        "  Confusion Matrix (rows=actual, cols=predicted)",
        "  " + "-" * 40,
        "                 Pred benign  Pred malicious",
        f"  Actual benign       {cm[0,0]:>6}          {cm[0,1]:>6}",
        f"  Actual malicious    {cm[1,0]:>6}          {cm[1,1]:>6}",
        "",
        "  Feature Importances (high -> low)",
        "  " + "-" * 40,
    ]
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for rank, (name, imp) in enumerate(ranked, start=1):
        bar = "#" * int(imp * 60)
        lines.append(f"  {rank:>2}. {name:<25}  {imp:.4f}  {bar}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


def main() -> None:
    # ── load ─────────────────────────────────────────────────────────────
    print("Loading features ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    label_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"  Train class distribution — benign: {label_counts.get(0, 0):,}  "
          f"malicious: {label_counts.get(1, 0):,}")

    # ── train ─────────────────────────────────────────────────────────────
    print("\nTraining Random Forest (class_weight='balanced') ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")

    # ── evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    y_pred = clf.predict(X_test)

    report = build_report(y_test, y_pred, FEATURE_COLS, clf.feature_importances_)

    # ── print to console ──────────────────────────────────────────────────
    print()
    print(report)

    # ── save model ────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved  -> {MODEL_PATH}")

    # ── save report ───────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
