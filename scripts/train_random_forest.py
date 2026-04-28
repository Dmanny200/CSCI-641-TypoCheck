"""
train_random_forest.py

Loads data/features.csv, trains a Random Forest classifier, tunes the
decision threshold on a validation set to minimise false positives on
benign domains, and evaluates on a held-out test set.

Artefacts saved:
  models/random_forest.joblib  — dict {"model": clf, "threshold": float}
                                  Bundled so callers always get the right threshold.
  output/rf_report.txt
"""

from pathlib import Path

import joblib
import numpy as np
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
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
MODEL_PATH   = ROOT / "models" / "random_forest.joblib"
REPORT_PATH  = ROOT / "output" / "rf_report.txt"

RANDOM_SEED  = 42

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

THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(13)]  # 0.30 … 0.90


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def predict_at(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def threshold_sweep(proba_val: np.ndarray, y_val: np.ndarray) -> list[dict]:
    """Return per-threshold metrics focused on the benign class (label=0)."""
    rows = []
    for thr in THRESHOLDS:
        y_pred = predict_at(proba_val, thr)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold":        thr,
            "benign_precision": precision_score(y_val, y_pred, pos_label=0, zero_division=0),
            "benign_recall":    recall_score(y_val, y_pred, pos_label=0, zero_division=0),
            "benign_f1":        f1_score(y_val, y_pred, pos_label=0, zero_division=0),
            # false-positive rate: benign domains called malicious / all benign
            "fpr":              fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        })
    return rows


def best_threshold(sweep_rows: list[dict]) -> float:
    """Threshold that maximises F1 on the benign class."""
    return max(sweep_rows, key=lambda r: r["benign_f1"])["threshold"]


def metrics_block(y_true, y_pred, label: str) -> list[str]:
    cm = confusion_matrix(y_true, y_pred)
    lines = [
        f"  [{label}]",
        f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}",
        f"  Precision : {precision_score(y_true, y_pred):.4f}  (malicious class)",
        f"  Recall    : {recall_score(y_true, y_pred):.4f}  (malicious class)",
        f"  F1 Score  : {f1_score(y_true, y_pred):.4f}  (malicious class)",
        "",
        "  Classification Report",
        "  " + "-" * 40,
    ]
    for line in classification_report(
        y_true, y_pred, target_names=["benign", "malicious"]
    ).splitlines():
        lines.append("  " + line)
    lines += [
        "",
        "  Confusion Matrix (rows=actual, cols=predicted)",
        "  " + "-" * 40,
        "                 Pred benign  Pred malicious",
        f"  Actual benign       {cm[0,0]:>6}          {cm[0,1]:>6}",
        f"  Actual malicious    {cm[1,0]:>6}          {cm[1,1]:>6}",
    ]
    return lines


def build_report(
    y_test: np.ndarray,
    proba_test: np.ndarray,
    tuned_thr: float,
    sweep_rows: list[dict],
    feature_names: list[str],
    importances: np.ndarray,
) -> str:
    y_default = predict_at(proba_test, 0.50)
    y_tuned   = predict_at(proba_test, tuned_thr)

    lines: list[str] = [
        "=" * 60,
        "  Random Forest — Evaluation Report",
        "=" * 60,
        "",
    ]

    # ── default threshold ────────────────────────────────────────────────
    lines += metrics_block(y_test, y_default, "Default threshold = 0.50")
    lines += [""]

    # ── tuned threshold ──────────────────────────────────────────────────
    lines += [
        "-" * 60,
        f"  Tuned threshold = {tuned_thr:.2f}",
        "  (selected to maximise F1 on the benign class on the validation set,",
        "   reducing false positives — benign domains labelled malicious)",
        "-" * 60,
        "",
    ]
    lines += metrics_block(y_test, y_tuned, f"Tuned threshold = {tuned_thr:.2f}")
    lines += [""]

    # ── threshold sweep table ────────────────────────────────────────────
    lines += [
        "=" * 60,
        "  Threshold Sweep (validation set, benign-class metrics)",
        "=" * 60,
        f"  {'Threshold':>9}  {'Prec(ben)':>9}  {'Rec(ben)':>8}  {'F1(ben)':>7}  {'FPR':>6}",
        "  " + "-" * 48,
    ]
    for r in sweep_rows:
        marker = " <-- tuned" if r["threshold"] == tuned_thr else ""
        lines.append(
            f"  {r['threshold']:>9.2f}  {r['benign_precision']:>9.4f}  "
            f"{r['benign_recall']:>8.4f}  {r['benign_f1']:>7.4f}  "
            f"{r['fpr']:>6.4f}{marker}"
        )
    lines += [""]

    # ── feature importances ──────────────────────────────────────────────
    lines += [
        "=" * 60,
        "  Feature Importances (high -> low)",
        "  " + "-" * 40,
    ]
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for rank, (name, imp) in enumerate(ranked, start=1):
        bar = "#" * int(imp * 60)
        lines.append(f"  {rank:>2}. {name:<25}  {imp:.4f}  {bar}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


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

    # ── three-way split: 60% train / 20% val / 20% test ──────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    vc = pd.Series(y_train).value_counts().sort_index()
    print(f"  Train class distribution — benign: {vc.get(0, 0):,}  malicious: {vc.get(1, 0):,}")

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

    # ── threshold tuning on validation set ────────────────────────────────
    print("\nTuning threshold on validation set ...")
    proba_val  = clf.predict_proba(X_val)[:, 1]
    sweep_rows = threshold_sweep(proba_val, y_val)
    tuned_thr  = best_threshold(sweep_rows)
    print(f"  Best threshold (max benign F1): {tuned_thr:.2f}")

    # ── evaluate on test set ──────────────────────────────────────────────
    print("\nEvaluating on test set ...")
    proba_test = clf.predict_proba(X_test)[:, 1]

    report = build_report(
        y_test, proba_test, tuned_thr, sweep_rows,
        FEATURE_COLS, clf.feature_importances_,
    )
    print()
    print(report)

    # ── save model (bundled with tuned threshold) ─────────────────────────
    MODEL_PATH.parent.mkdir(exist_ok=True)
    # Saved as a dict so callers always load the right threshold atomically.
    joblib.dump({"model": clf, "threshold": tuned_thr}, MODEL_PATH)
    print(f"\nModel saved  -> {MODEL_PATH}  (threshold={tuned_thr:.2f})")

    # ── save report ───────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
