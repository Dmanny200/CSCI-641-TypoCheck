"""
gui.py

Tkinter GUI for typosquatting / phishing domain detection.
Models are loaded once at startup; subsequent checks are fast.

Usage:
    python scripts/gui.py
"""

import math
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

import joblib
import numpy as np
import pandas as pd
from Levenshtein import distance as lev_distance

# ── paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
RF_MODEL     = ROOT / "models" / "random_forest.joblib"
XGB_MODEL    = ROOT / "models" / "xgboost.joblib"

# ── constants (must stay in sync with extract_features.py) ───────────────────
TLD_RISK: dict[str, int] = {
    ".com": 0, ".org": 0, ".net": 0, ".edu": 0, ".gov": 0,
    ".io":  1, ".co":  1, ".info": 1,
}
VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

FEATURE_COLS = [
    "domain_length", "subdomain_count", "digit_count", "hyphen_count",
    "non_alnum_count", "vowel_consonant_ratio", "entropy",
    "char_continuity_rate", "digit_ratio", "tld_risk_score", "min_lev_distance",
]

TIER_COLORS = {
    "ALLOW":   {"bg": "#d4edda", "fg": "#155724", "label": "ALLOW"},
    "WARNING": {"bg": "#fff3cd", "fg": "#856404", "label": "WARNING"},
    "BLOCK":   {"bg": "#f8d7da", "fg": "#721c24", "label": "BLOCK"},
}


# ═══════════════════════════════════════════════════════════════════════════
# feature extraction (mirrors extract_features.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_sld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-2] if len(parts) >= 2 else domain

def get_tld(domain: str) -> str:
    idx = domain.rfind(".")
    return domain[idx:] if idx != -1 else ""

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def char_continuity_rate(s: str) -> float:
    if len(s) < 2:
        return 0.0
    def cls(c: str) -> str:
        if c.isalpha():  return "a"
        if c.isdigit():  return "d"
        return "o"
    pairs = len(s) - 1
    same  = sum(1 for i in range(pairs) if cls(s[i]) == cls(s[i + 1]) != "o")
    return same / pairs

def vowel_consonant_ratio(s: str) -> float:
    v = sum(1 for c in s.lower() if c in VOWELS)
    c = sum(1 for c in s.lower() if c in CONSONANTS)
    return v / c if c else 0.0

def extract_features(domain: str, top100_slds: list[str]) -> dict[str, float]:
    length    = len(domain)
    digits    = sum(1 for c in domain if c.isdigit())
    hyphens   = domain.count("-")
    non_alnum = sum(1 for c in domain if not c.isalnum() and c != ".")
    tld = get_tld(domain)
    sld = get_sld(domain)
    min_lev = min(lev_distance(sld, ref) for ref in top100_slds)
    return {
        "domain_length":         length,
        "subdomain_count":       max(domain.count(".") - 1, 0),
        "digit_count":           digits,
        "hyphen_count":          hyphens,
        "non_alnum_count":       non_alnum,
        "vowel_consonant_ratio": round(vowel_consonant_ratio(domain), 4),
        "entropy":               round(shannon_entropy(domain), 4),
        "char_continuity_rate":  round(char_continuity_rate(domain), 4),
        "digit_ratio":           round(digits / length, 4) if length else 0.0,
        "tld_risk_score":        TLD_RISK.get(tld, 2),
        "min_lev_distance":      min_lev,
    }

def risk_tier(confidence: float) -> str:
    if confidence < 0.50:  return "ALLOW"
    if confidence < 0.80:  return "WARNING"
    return "BLOCK"


# ═══════════════════════════════════════════════════════════════════════════
# app
# ═══════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Domain Threat Detector")
        self.resizable(True, True)
        self.minsize(600, 460)

        # ── fonts ─────────────────────────────────────────────────────────
        self._f_ui     = tkfont.Font(family="Segoe UI", size=10)
        self._f_mono   = tkfont.Font(family="Consolas",  size=10)
        self._f_domain = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self._f_tier   = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self._f_head   = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self._build_ui()

        # ── load models and reference data ────────────────────────────────
        self._status("Loading models ...")
        self.update_idletasks()

        df = pd.read_csv(FEATURES_CSV, dtype=str)
        tranco = df[df["source"] == "tranco_top10k"]["domain"]
        self.top100_slds: list[str] = [get_sld(d) for d in tranco.head(100)]

        self.rf  = joblib.load(RF_MODEL)
        self.xgb = joblib.load(XGB_MODEL)

        # precompute average importances for ranking
        self.avg_imp = {
            f: (ri + xi) / 2
            for f, ri, xi in zip(
                FEATURE_COLS,
                self.rf.feature_importances_,
                self.xgb.feature_importances_,
            )
        }
        self.feat_rank = sorted(FEATURE_COLS, key=lambda f: self.avg_imp[f], reverse=True)

        self._status("Ready. Enter a domain and click Check.")
        self._entry.focus_set()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.configure(bg="#f0f0f0")
        pad = {"padx": 16, "pady": 8}

        # ── header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#2c3e50", pady=10)
        hdr.pack(fill="x")
        tk.Label(
            hdr, text="Domain Threat Detector",
            font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
            bg="#2c3e50", fg="white",
        ).pack()

        # ── input row ─────────────────────────────────────────────────────
        input_frame = tk.Frame(self, bg="#f0f0f0", pady=12)
        input_frame.pack(fill="x", **{"padx": 16})

        tk.Label(
            input_frame, text="Domain:", font=self._f_ui,
            bg="#f0f0f0",
        ).pack(side="left")

        self._entry = tk.Entry(
            input_frame, font=self._f_mono, width=38,
            relief="solid", bd=1,
        )
        self._entry.pack(side="left", padx=(8, 10), ipady=4)
        self._entry.bind("<Return>", lambda _e: self._check())

        self._btn = tk.Button(
            input_frame, text="Check Domain",
            font=self._f_ui, bg="#2980b9", fg="white",
            activebackground="#1f618d", activeforeground="white",
            relief="flat", padx=14, pady=5, cursor="hand2",
            command=self._check,
        )
        self._btn.pack(side="left")

        tk.Button(
            input_frame, text="Clear",
            font=self._f_ui, bg="#95a5a6", fg="white",
            activebackground="#7f8c8d", activeforeground="white",
            relief="flat", padx=10, pady=5, cursor="hand2",
            command=self._clear,
        ).pack(side="left", padx=(6, 0))

        # ── results card ──────────────────────────────────────────────────
        self._card = tk.Frame(self, bg="#ffffff", relief="solid", bd=1)
        self._card.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # placeholder
        self._placeholder = tk.Label(
            self._card,
            text="Results will appear here.",
            font=self._f_ui, bg="#ffffff", fg="#aaaaaa",
        )
        self._placeholder.pack(expand=True)

        # ── status bar ────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Initialising ...")
        tk.Label(
            self, textvariable=self._status_var,
            font=tkfont.Font(family="Segoe UI", size=9),
            bg="#dde1e4", fg="#555555", anchor="w", pady=3, padx=8,
        ).pack(fill="x", side="bottom")

    # ── actions ────────────────────────────────────────────────────────────

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)

    def _clear(self) -> None:
        self._entry.delete(0, "end")
        self._entry.focus_set()
        for w in self._card.winfo_children():
            w.destroy()
        self._placeholder = tk.Label(
            self._card,
            text="Results will appear here.",
            font=self._f_ui, bg="#ffffff", fg="#aaaaaa",
        )
        self._placeholder.pack(expand=True)
        self._status("Ready. Enter a domain and click Check.")

    def _check(self) -> None:
        domain = self._entry.get().strip().lower()
        if not domain:
            self._status("Please enter a domain.")
            return
        if "." not in domain:
            self._status("Invalid domain — must contain at least one dot.")
            return

        self._btn.configure(state="disabled", text="Checking ...")
        self._status(f"Analysing {domain} ...")
        self.update_idletasks()

        try:
            self._run_prediction(domain)
        finally:
            self._btn.configure(state="normal", text="Check Domain")

    def _run_prediction(self, domain: str) -> None:
        # ── extract features ──────────────────────────────────────────────
        feat = extract_features(domain, self.top100_slds)
        X = np.array([[feat[col] for col in FEATURE_COLS]])

        # ── predict ───────────────────────────────────────────────────────
        rf_conf  = self.rf.predict_proba(X)[0][1]
        xgb_conf = self.xgb.predict_proba(X)[0][1]
        max_conf = max(rf_conf, xgb_conf)
        tier     = risk_tier(max_conf)
        colors   = TIER_COLORS[tier]

        # ── rebuild card ──────────────────────────────────────────────────
        for w in self._card.winfo_children():
            w.destroy()

        bg = colors["bg"]
        fg = colors["fg"]
        self._card.configure(bg=bg)

        # tier banner
        banner = tk.Frame(self._card, bg=bg, pady=14)
        banner.pack(fill="x")

        tk.Label(
            banner, text=tier,
            font=self._f_tier, bg=bg, fg=fg,
        ).pack()

        tk.Label(
            banner,
            text=f"Max confidence: {max_conf:.1%}  |  Domain: {domain}",
            font=self._f_ui, bg=bg, fg=fg,
        ).pack()

        # ── divider ───────────────────────────────────────────────────────
        tk.Frame(self._card, bg=fg, height=1).pack(fill="x", padx=16)

        # ── two-column body ───────────────────────────────────────────────
        body = tk.Frame(self._card, bg=bg)
        body.pack(fill="both", expand=True, padx=16, pady=10)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)

        # left — model predictions
        left = tk.Frame(body, bg=bg)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        tk.Label(left, text="Model predictions", font=self._f_head,
                 bg=bg, fg=fg, anchor="w").pack(fill="x", pady=(0, 6))

        for model_name, conf in [("Random Forest", rf_conf), ("XGBoost", xgb_conf)]:
            pred  = "malicious" if conf >= 0.5 else "benign"
            frame = tk.Frame(left, bg=bg)
            frame.pack(fill="x", pady=2)
            tk.Label(frame, text=f"{model_name}:", font=self._f_ui,
                     bg=bg, fg=fg, width=15, anchor="w").pack(side="left")
            pred_col = fg if pred == "malicious" else "#2d6a2d"
            tk.Label(frame, text=f"{pred}  ({conf:.1%})",
                     font=self._f_mono, bg=bg, fg=pred_col).pack(side="left")

        # right — top 5 features
        right = tk.Frame(body, bg=bg)
        right.grid(row=0, column=1, sticky="nsew")

        tk.Label(right, text="Top 5 feature signals", font=self._f_head,
                 bg=bg, fg=fg, anchor="w").pack(fill="x", pady=(0, 4))

        # header row
        hrow = tk.Frame(right, bg=bg)
        hrow.pack(fill="x")
        for col_text, w in [("Feature", 22), ("Value", 8), ("Avg imp", 8)]:
            tk.Label(hrow, text=col_text, font=self._f_head,
                     bg=bg, fg=fg, width=w, anchor="w").pack(side="left")

        tk.Frame(right, bg=fg, height=1).pack(fill="x", pady=(2, 4))

        rf_imp  = dict(zip(FEATURE_COLS, self.rf.feature_importances_))
        xgb_imp = dict(zip(FEATURE_COLS, self.xgb.feature_importances_))

        for feat_name in self.feat_rank[:5]:
            val     = feat[feat_name]
            avg_imp = self.avg_imp[feat_name]
            val_str = str(int(val)) if val == int(val) else f"{val:.4f}"
            bar_len = int(avg_imp * 120)
            bar_str = "|" * bar_len

            row = tk.Frame(right, bg=bg)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=feat_name, font=self._f_mono,
                     bg=bg, fg=fg, width=22, anchor="w").pack(side="left")
            tk.Label(row, text=val_str, font=self._f_mono,
                     bg=bg, fg=fg, width=8, anchor="e").pack(side="left")
            tk.Label(row, text=f" {avg_imp:.3f}  {bar_str}",
                     font=self._f_mono, bg=bg, fg=fg, anchor="w").pack(side="left")

        self._status(f"Done: {domain} -> {tier}  (RF {rf_conf:.1%}  /  XGB {xgb_conf:.1%})")


# ═══════════════════════════════════════════════════════════════════════════
# entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
