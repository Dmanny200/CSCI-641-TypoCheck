"""
train_balanced.py

Creates a 50/50 balanced subset of data/features.csv (all malicious rows +
an equal random sample of benign rows), then trains both Random Forest and
XGBoost on it.  Reports are written to:

  output/rf_balanced_report.txt
  output/xgb_balanced_report.txt
  output/balanced_comparison.txt   ← side-by-side vs. original imbalanced run
"""

import argparse
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
from xgboost import XGBClassifier

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
OUT          = ROOT / "output"
OUT.mkdir(exist_ok=True)

RANDOM_SEED = 42

FEATURE_COLS = [
    # ── domain-level ────────────────────────────────────────────────────
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
    "homoglyph_count",
    "bigram_log_prob",
    # ── URL-level ────────────────────────────────────────────────────────
    "url_length",
    "path_length",
    "path_depth",
    "has_query",
    "query_length",
    "path_entropy",
    "at_in_url",
    "double_slash_in_path",
]

THRESHOLDS = [round(0.30 + i * 0.05, 2) for i in range(13)]


# ── helpers (shared) ──────────────────────────────────────────────────────────

def predict_at(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def threshold_sweep(proba_val, y_val):
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
            "fpr":              fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        })
    return rows


def best_threshold(sweep_rows):
    return max(sweep_rows, key=lambda r: r["benign_f1"])["threshold"]


def metrics_block(y_true, y_pred, label):
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


def build_report(y_test, proba_test, tuned_thr, sweep_rows, feature_names, importances, model_label):
    y_default = predict_at(proba_test, 0.50)
    y_tuned   = predict_at(proba_test, tuned_thr)

    lines = [
        "=" * 60,
        f"  {model_label} (50/50 balanced) — Evaluation Report",
        "=" * 60,
        "",
    ]
    lines += metrics_block(y_test, y_default, "Default threshold = 0.50")
    lines += [""]
    lines += [
        "-" * 60,
        f"  Tuned threshold = {tuned_thr:.2f}",
        "  (selected to maximise F1 on the benign class on the validation set)",
        "-" * 60,
        "",
    ]
    lines += metrics_block(y_test, y_tuned, f"Tuned threshold = {tuned_thr:.2f}")
    lines += [""]
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


def extract_scalar(text, key):
    import re
    patterns = {
        "accuracy":  r"Accuracy\s+:\s+([0-9.]+)",
        "precision": r"Precision\s+:\s+([0-9.]+)",
        "recall":    r"Recall\s+:\s+([0-9.]+)",
        "f1":        r"F1 Score\s+:\s+([0-9.]+)",
    }
    m = re.search(patterns[key], text)
    return float(m.group(1)) if m else float("nan")


def parse_metrics(text):
    return {k: extract_scalar(text, k) for k in ["accuracy", "precision", "recall", "f1"]}


# ── comparison report ─────────────────────────────────────────────────────────

def build_comparison(rf_orig, rf_bal, xgb_orig, xgb_bal):
    keys = ["accuracy", "precision", "recall", "f1"]
    lines = [
        "=" * 72,
        "  Imbalanced (70/30) vs. Balanced (50/50) — Default threshold = 0.50",
        "=" * 72,
        f"  {'Metric':<12}  {'RF orig':>9}  {'RF bal':>9}  {'RF +/-':>7}  "
        f"{'XGB orig':>9}  {'XGB bal':>9}  {'XGB +/-':>8}",
        "  " + "-" * 70,
    ]
    for k in keys:
        ro = rf_orig[k];  rb = rf_bal[k]
        xo = xgb_orig[k]; xb = xgb_bal[k]
        def fmt(v): return f"{v:.4f}"
        def delta(a, b): d = b - a; return f"{'+'if d>=0 else ''}{d:.4f}"
        lines.append(
            f"  {k.capitalize():<12}  {fmt(ro):>9}  {fmt(rb):>9}  {delta(ro,rb):>7}  "
            f"{fmt(xo):>9}  {fmt(xb):>9}  {delta(xo,xb):>7}"
        )
    lines += ["", "=" * 72]
    return "\n".join(lines)


# ── train one model ───────────────────────────────────────────────────────────

def run_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        # no class_weight needed — data is already balanced
    )
    clf.fit(X_train, y_train)
    proba_val  = clf.predict_proba(X_val)[:, 1]
    sweep      = threshold_sweep(proba_val, y_val)
    tuned_thr  = best_threshold(sweep)
    proba_test = clf.predict_proba(X_test)[:, 1]
    report     = build_report(y_test, proba_test, tuned_thr, sweep,
                               FEATURE_COLS, clf.feature_importances_, "Random Forest")
    bundle = {"model": clf, "threshold": tuned_thr}
    return report, bundle


def run_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        # scale_pos_weight=1 because classes are balanced
    )
    clf.fit(X_train, y_train)
    proba_val  = clf.predict_proba(X_val)[:, 1]
    sweep      = threshold_sweep(proba_val, y_val)
    tuned_thr  = best_threshold(sweep)
    proba_test = clf.predict_proba(X_test)[:, 1]
    report     = build_report(y_test, proba_test, tuned_thr, sweep,
                               FEATURE_COLS, clf.feature_importances_, "XGBoost")
    bundle = {"model": clf, "threshold": tuned_thr}
    return report, bundle


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURES_CSV), help="Path to features CSV")
    args = parser.parse_args()

    features_path = Path(args.features)
    tag = features_path.stem.replace("features", "").strip("_")
    suffix = f"_{tag}" if tag else ""

    rf_orig_report  = OUT / f"rf{suffix}_report.txt"
    xgb_orig_report = OUT / f"xgb{suffix}_report.txt"

    print(f"Loading {features_path.name} ...")
    df = pd.read_csv(features_path)
    n_mal = int((df.label == 1).sum())
    n_ben = int((df.label == 0).sum())
    print(f"  Full dataset: {n_ben:,} benign  |  {n_mal:,} malicious")

    # ── build balanced 50/50 subset ──────────────────────────────────────
    mal_df = df[df.label == 1]
    ben_df = df[df.label == 0].sample(n=n_mal, random_state=RANDOM_SEED)
    balanced = pd.concat([ben_df, mal_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Balanced subset: {len(balanced):,} rows  ({n_mal:,} benign + {n_mal:,} malicious)")

    X = balanced[FEATURE_COLS].values
    y = balanced["label"].values

    # ── 60/20/20 split ────────────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── Random Forest ─────────────────────────────────────────────────────
    print("\nTraining Random Forest on balanced data ...")
    rf_report, rf_bundle = run_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    rf_path = OUT / f"rf{suffix}_balanced_report.txt"
    rf_path.write_text(rf_report, encoding="utf-8")
    joblib.dump(rf_bundle, ROOT / "models" / f"random_forest{suffix}_balanced.joblib")
    print(rf_report)

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("\nTraining XGBoost on balanced data ...")
    xgb_report, xgb_bundle = run_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    xgb_path = OUT / f"xgb{suffix}_balanced_report.txt"
    xgb_path.write_text(xgb_report, encoding="utf-8")
    joblib.dump(xgb_bundle, ROOT / "models" / f"xgboost{suffix}_balanced.joblib")
    print(xgb_report)

    # ── comparison ────────────────────────────────────────────────────────
    rf_orig_m  = parse_metrics(rf_orig_report.read_text(encoding="utf-8"))  if rf_orig_report.exists()  else {}
    xgb_orig_m = parse_metrics(xgb_orig_report.read_text(encoding="utf-8")) if xgb_orig_report.exists() else {}
    rf_bal_m   = parse_metrics(rf_report)
    xgb_bal_m  = parse_metrics(xgb_report)

    if rf_orig_m and xgb_orig_m:
        comp = build_comparison(rf_orig_m, rf_bal_m, xgb_orig_m, xgb_bal_m)
        print()
        print(comp)
        comp_path = OUT / f"balanced{suffix}_comparison.txt"
        comp_path.write_text(comp, encoding="utf-8")
        print(f"\nComparison saved -> {comp_path}")

    print(f"\nRF  report -> {rf_path}  (threshold={rf_bundle['threshold']:.2f})")
    print(f"XGB report -> {xgb_path}  (threshold={xgb_bundle['threshold']:.2f})")


if __name__ == "__main__":
    main()
