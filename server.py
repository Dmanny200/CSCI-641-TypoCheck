"""
server.py

FastAPI backend for the domain threat detector browser extension.
Run with:  uvicorn server:app --host 127.0.0.1 --port 8000
"""

import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from domain_checker import (
    FEATURE_COLS, WHOIS_BAND_LO,
    load_model, run_inference, risk_tier, whois_features, adjust_tier, get_sld,
)

ROOT         = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
RF_MODEL     = ROOT / "models" / "random_forest.joblib"
XGB_MODEL    = ROOT / "models" / "xgboost.joblib"

# ── cache ─────────────────────────────────────────────────────────────────────

PREDICT_TTL = timedelta(hours=1)
WHOIS_TTL   = timedelta(hours=6)

@dataclass
class _Entry:
    data:    dict
    expires: datetime

_predict_cache: dict[str, _Entry] = {}
_whois_cache:   dict[str, _Entry] = {}
_cache_lock = threading.Lock()


def _cache_get(store: dict[str, _Entry], key: str) -> dict | None:
    with _cache_lock:
        entry = store.get(key)
        if entry and datetime.now(timezone.utc) < entry.expires:
            return entry.data
        store.pop(key, None)
        return None


def _cache_set(store: dict[str, _Entry], key: str, data: dict, ttl: timedelta) -> None:
    with _cache_lock:
        store[key] = _Entry(data, datetime.now(timezone.utc) + ttl)


# Domains pre-warmed at startup so first visits are instant.
PREWARM_DOMAINS = [
    # ── major benign sites ────────────────────────────────────────────────
    "google.com", "youtube.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "reddit.com", "wikipedia.org",
    "amazon.com", "ebay.com", "etsy.com", "walmart.com",
    "netflix.com", "spotify.com", "twitch.tv", "tiktok.com",
    "apple.com", "microsoft.com", "github.com", "stackoverflow.com",
    "linkedin.com", "discord.com", "slack.com", "zoom.us",
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "cnn.com", "bbc.com", "nytimes.com", "theguardian.com",
    "cloudflare.com", "dropbox.com", "notion.so", "figma.com",
    # ── common phishing / typosquat targets ──────────────────────────────
    "g00gle.com", "gooogle.com", "googie.com", "gogle.com",
    "arnazon.com", "amaz0n.com", "amazonn.com",
    "paypa1.com", "paypai.com", "paypaI.com",
    "micros0ft.com", "micosoft.com", "microsofft.com",
    "faceb00k.com", "facebok.com", "faceboook.com",
    "netfl1x.com", "netfllx.com",
    "appleid-verify.com", "apple-support-login.com",
    "secure-paypal-login.com", "paypal-secure-login.com",
    "amazon-security-alert.com", "amazon-prime-verify.com",
    "login-microsoft-secure.com", "microsoft-account-verify.com",
    "bankofamerica-secure.com", "chase-secure-login.com",
]


def _prewarm(domains: list[str]) -> None:
    """Run inference on each domain and populate the predict cache."""
    hits = 0
    for domain in domains:
        try:
            feat, rf_conf, xgb_conf = run_inference(
                domain, state.top100_slds,
                state.rf_clf, state.rf_thr,
                state.xgb_clf, state.xgb_thr,
            )
            rf_label  = "malicious" if rf_conf  >= state.rf_thr  else "benign"
            xgb_label = "malicious" if xgb_conf >= state.xgb_thr else "benign"
            max_conf  = max(rf_conf, xgb_conf)
            ml_tier   = risk_tier(max_conf)
            avg_imp   = {f: float(state.rf_imps[f] + state.xgb_imps[f]) / 2 for f in FEATURE_COLS}
            features  = [
                {
                    "name":    f,
                    "value":   float(feat[f]),
                    "rf_imp":  round(float(state.rf_imps[f]), 4),
                    "xgb_imp": round(float(state.xgb_imps[f]), 4),
                    "avg_imp": round(avg_imp[f], 4),
                }
                for f in sorted(FEATURE_COLS, key=lambda f: avg_imp[f], reverse=True)
            ]
            payload = {
                "domain":         domain,
                "rf_conf":        round(rf_conf, 4),
                "rf_label":       rf_label,
                "rf_threshold":   state.rf_thr,
                "xgb_conf":       round(xgb_conf, 4),
                "xgb_label":      xgb_label,
                "xgb_threshold":  state.xgb_thr,
                "max_conf":       round(max_conf, 4),
                "ml_tier":        ml_tier,
                "final_tier":     ml_tier,
                "features":       features,
                "whois_eligible": max_conf >= WHOIS_BAND_LO,
                "cached":         True,
            }
            _cache_set(_predict_cache, domain, payload, PREDICT_TTL)
            hits += 1
        except Exception:
            pass
    print(f"Cache pre-warm complete: {hits}/{len(domains)} domains cached")


# ── app state ─────────────────────────────────────────────────────────────────

class State:
    rf_clf = None;  rf_thr:  float = 0.5
    xgb_clf = None; xgb_thr: float = 0.5
    rf_imps:  dict = {}
    xgb_imps: dict = {}
    top100_slds: list[str] = []

state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.rf_clf,  state.rf_thr  = load_model(RF_MODEL)
    state.xgb_clf, state.xgb_thr = load_model(XGB_MODEL)
    state.rf_imps  = dict(zip(FEATURE_COLS, state.rf_clf.feature_importances_))
    state.xgb_imps = dict(zip(FEATURE_COLS, state.xgb_clf.feature_importances_))

    df = pd.read_csv(FEATURES_CSV, dtype=str)
    tranco = df[df["source"] == "tranco_top10k"]["domain"]
    state.top100_slds = [get_sld(d) for d in tranco.head(100)]
    print("Models loaded. Listening on http://127.0.0.1:8000")

    # Pre-warm cache in background — doesn't block the server from accepting requests
    threading.Thread(target=_prewarm, args=(PREWARM_DOMAINS,), daemon=True).start()
    print(f"Pre-warming cache for {len(PREWARM_DOMAINS)} domains in background...")

    yield


app = FastAPI(title="Domain Threat Detector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _cache_lock:
        return {
            "status":         "ok",
            "predict_cached": len(_predict_cache),
            "whois_cached":   len(_whois_cache),
        }


@app.get("/predict")
async def predict(domain: str = Query(...)):
    domain = domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if "." not in domain or len(domain) < 4:
        raise HTTPException(status_code=400, detail="Invalid domain")

    cached = _cache_get(_predict_cache, domain)
    if cached:
        return cached

    feat, rf_conf, xgb_conf = await run_in_threadpool(
        run_inference, domain, state.top100_slds,
        state.rf_clf, state.rf_thr, state.xgb_clf, state.xgb_thr,
    )

    rf_label  = "malicious" if rf_conf  >= state.rf_thr  else "benign"
    xgb_label = "malicious" if xgb_conf >= state.xgb_thr else "benign"
    max_conf  = max(rf_conf, xgb_conf)
    ml_tier   = risk_tier(max_conf)

    avg_imp = {f: float(state.rf_imps[f] + state.xgb_imps[f]) / 2 for f in FEATURE_COLS}
    features = [
        {
            "name":    f,
            "value":   float(feat[f]),
            "rf_imp":  round(float(state.rf_imps[f]), 4),
            "xgb_imp": round(float(state.xgb_imps[f]), 4),
            "avg_imp": round(avg_imp[f], 4),
        }
        for f in sorted(FEATURE_COLS, key=lambda f: avg_imp[f], reverse=True)
    ]

    whois_eligible = max_conf >= WHOIS_BAND_LO

    result = {
        "domain":          domain,
        "rf_conf":         round(rf_conf, 4),
        "rf_label":        rf_label,
        "rf_threshold":    state.rf_thr,
        "xgb_conf":        round(xgb_conf, 4),
        "xgb_label":       xgb_label,
        "xgb_threshold":   state.xgb_thr,
        "max_conf":        round(max_conf, 4),
        "ml_tier":         ml_tier,
        "final_tier":      ml_tier,
        "features":        features,
        "whois_eligible":  whois_eligible,
        "cached":          False,
    }
    _cache_set(_predict_cache, domain, result, PREDICT_TTL)
    return result


@app.get("/whois")
async def whois_lookup(domain: str = Query(...)):
    domain = domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if "." not in domain or len(domain) < 4:
        raise HTTPException(status_code=400, detail="Invalid domain")

    cached = _cache_get(_whois_cache, domain)
    if cached:
        return cached

    feat, rf_conf, xgb_conf = await run_in_threadpool(
        run_inference, domain, state.top100_slds,
        state.rf_clf, state.rf_thr, state.xgb_clf, state.xgb_thr,
    )
    max_conf = max(rf_conf, xgb_conf)
    ml_tier  = risk_tier(max_conf)

    wf = await run_in_threadpool(whois_features, domain)
    final_tier, reasons = adjust_tier(ml_tier, wf)

    result = {
        "domain":                 domain,
        "available":              wf.get("available", False),
        "domain_age_days":        wf.get("domain_age_days"),
        "has_privacy_protection": wf.get("has_privacy_protection"),
        "registrar_known":        wf.get("registrar_known"),
        "expires_soon":           wf.get("expires_soon"),
        "original_tier":          ml_tier,
        "final_tier":             final_tier,
        "reasons":                reasons,
    }
    _cache_set(_whois_cache, domain, result, WHOIS_TTL)
    return result


@app.get("/cache")
def cache_status():
    """Lists all currently cached domains and their expiry times."""
    now = datetime.now(timezone.utc)
    with _cache_lock:
        predict = [
            {
                "domain":     k,
                "tier":       v.data.get("final_tier"),
                "max_conf":   v.data.get("max_conf"),
                "expires_in": str(v.expires - now).split(".")[0],
            }
            for k, v in _predict_cache.items()
        ]
        whois = [
            {
                "domain":     k,
                "expires_in": str(v.expires - now).split(".")[0],
            }
            for k, v in _whois_cache.items()
        ]
    return {
        "predict": sorted(predict, key=lambda x: x["domain"]),
        "whois":   sorted(whois,   key=lambda x: x["domain"]),
    }
