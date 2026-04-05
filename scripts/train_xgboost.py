"""
train_xgboost.py

Loads data/features.csv, trains an XGBClassifier, evaluates on the same
80/20 stratified split as the Random Forest, and prints a side-by-side
comparison of both models.

Artefacts saved:
  models/xgboost.joblib
  output/xgb_report.txt
"""

import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
MODEL_PATH   = ROOT / "models" / "xgboost.joblib"
REPORT_PATH  = ROOT / "output" / "xgb_report.txt"
RF_REPORT    = ROOT / "output" / "rf_report.txt"

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


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_report(
    y_test,
    y_pred,
    feature_names: list[str],
    importances,
    model_label: str = "XGBoost",
) -> str:
    lines: list[str] = []
    lines += [
        "=" * 60,
        f"  {model_label} — Evaluation Report",
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


def parse_metrics(report_text: str) -> dict[str, float]:
    """Extract the four scalar metrics from a saved report file."""
    patterns = {
        "accuracy":  r"Accuracy\s+:\s+([0-9.]+)",
        "precision": r"Precision\s+:\s+([0-9.]+)",
        "recall":    r"Recall\s+:\s+([0-9.]+)",
        "f1":        r"F1 Score\s+:\s+([0-9.]+)",
    }
    metrics: dict[str, float] = {}
    for key, pat in patterns.items():
        m = re.search(pat, report_text)
        metrics[key] = float(m.group(1)) if m else float("nan")
    return metrics


def print_comparison(rf_metrics: dict, xgb_metrics: dict) -> None:
    keys   = ["accuracy", "precision", "recall", "f1"]
    header = f"  {'Metric':<12}  {'Random Forest':>14}  {'XGBoost':>10}  {'Delta':>8}"
    sep    = "  " + "-" * (len(header) - 2)
    print()
    print("=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(header)
    print(sep)
    for k in keys:
        rf_val  = rf_metrics[k]
        xgb_val = xgb_metrics[k]
        delta   = xgb_val - rf_val
        arrow   = "+" if delta >= 0 else ""
        print(f"  {k.capitalize():<12}  {rf_val:>14.4f}  {xgb_val:>10.4f}  {arrow}{delta:>7.4f}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── load ─────────────────────────────────────────────────────────────
    print("Loading features ...")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── split (identical seed/size to RF script) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    n_benign    = int((y_train == 0).sum())
    n_malicious = int((y_train == 1).sum())
    scale_pos_weight = n_benign / n_malicious
    print(f"  Train class distribution — benign: {n_benign:,}  malicious: {n_malicious:,}")
    print(f"  scale_pos_weight = {scale_pos_weight:.4f}")

    # ── train ─────────────────────────────────────────────────────────────
    print("\nTraining XGBClassifier ...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    print("  Training complete.")

    # ── evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    y_pred = clf.predict(X_test)

    report = build_report(y_test, y_pred, FEATURE_COLS, clf.feature_importances_)
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

    # ── side-by-side comparison ───────────────────────────────────────────
    if RF_REPORT.exists():
        rf_text    = RF_REPORT.read_text(encoding="utf-8")
        rf_metrics = parse_metrics(rf_text)
        xgb_metrics = parse_metrics(report)
        print_comparison(rf_metrics, xgb_metrics)
    else:
        print(f"\n[warn] {RF_REPORT} not found — skipping comparison table")


if __name__ == "__main__":
    main()
