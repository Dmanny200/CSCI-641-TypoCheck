"""
filter_dataset.py

Two-stage cleaning pass on features.csv:

  Stage 1 — drop kaggle_malicious_urls (label=1).
            These contain defacement/SEO-spam URLs whose feature profile
            looks identical to normal browsing, poisoning the classifier.

  Stage 2 — drop malicious entries hosted on major legitimate platforms
            (e.g. docs.google.com, sharepoint.com, github.io).
            Phishers abuse these services, but the model should not learn
            that "google" or "microsoft" SLDs are malicious.

  Stage 3 — downsample benign to restore the target 70/30 ratio.

Output: data/features_clean.csv

Usage:
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --input data/features.csv --ratio 0.70
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data" / "features.csv"
RANDOM_SEED  = 42

# Registered domains (sld.tld) whose subdomains are routinely abused for
# phishing hosting but whose presence in the malicious set corrupts the model.
LEGIT_HOSTING_DOMAINS = {
    # Google infrastructure
    "google.com", "googleapis.com", "googleusercontent.com",
    "googledrive.com", "blogspot.com", "blogger.com",
    # Microsoft infrastructure
    "microsoft.com", "live.com", "sharepoint.com",
    "onedrive.com", "office.com", "outlook.com",
    # Cloud / CDN platforms abused for phishing pages
    "amazonaws.com", "cloudfront.net", "azurewebsites.net",
    "azure.com", "windows.net",
    # Site builders
    "github.com", "github.io", "gitlab.io",
    "wordpress.com", "wix.com", "weebly.com",
    "squarespace.com", "webflow.io",
    "dropbox.com", "box.com",
}


def _registered_domain(url: str) -> str:
    """Return sld.tld from a URL (e.g. 'docs.google.com/...' → 'google.com')."""
    try:
        host = urlparse(url).hostname or ""
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        return f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else host
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FEATURES_CSV))
    parser.add_argument("--ratio", type=float, default=0.70,
                        help="Target fraction of benign rows (default 0.70)")
    args = parser.parse_args()

    src = Path(args.input)
    print(f"Loading {src.name} ...")
    df = pd.read_csv(src)

    print("\nBefore filtering:")
    print(df.groupby(["source", "label"]).size().to_string())
    print(f"Total rows: {len(df):,}")

    # ── Stage 1: drop Kaggle malicious ────────────────────────────────────────
    mask_kagmal = (df["source"] == "kaggle_malicious_urls") & (df["label"] == 1)
    n_kagmal = mask_kagmal.sum()
    df = df[~mask_kagmal].copy()
    print(f"\nStage 1 — dropped {n_kagmal:,} kaggle_malicious_urls (label=1)")

    # ── Stage 2: drop malicious entries on major legitimate platforms ─────────
    mal_mask = df["label"] == 1
    reg_domains = df.loc[mal_mask, "url"].apply(_registered_domain)
    mask_legit_host = mal_mask & reg_domains.isin(LEGIT_HOSTING_DOMAINS)
    n_legit = mask_legit_host.sum()
    df = df[~mask_legit_host].copy()
    print(f"Stage 2 — dropped {n_legit:,} malicious entries on legit hosting platforms")
    print(f"          (e.g. docs.google.com, sharepoint.com, github.io)")

    # ── Stage 3: downsample benign to target ratio ────────────────────────────
    n_mal = int((df["label"] == 1).sum())
    n_ben_target = round(n_mal * args.ratio / (1 - args.ratio))

    ben_df = df[df["label"] == 0]
    mal_df = df[df["label"] == 1]

    print(f"\nMalicious rows remaining  : {n_mal:,}")
    print(f"Benign rows available     : {len(ben_df):,}")
    print(f"Benign rows target (70/30): {n_ben_target:,}")

    if len(ben_df) > n_ben_target:
        ben_df = ben_df.sample(n=n_ben_target, random_state=RANDOM_SEED)
        print(f"Downsampled benign to     : {len(ben_df):,}")
    else:
        print("Benign pool smaller than target — keeping all benign rows")

    out_df = (pd.concat([ben_df, mal_df])
                .sample(frac=1, random_state=RANDOM_SEED)
                .reset_index(drop=True))

    out_path = src.parent / "features_clean.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nAfter filtering:")
    print(out_df.groupby(["source", "label"]).size().to_string())
    actual_ratio = (out_df["label"] == 0).sum() / len(out_df)
    print(f"\nActual benign ratio: {actual_ratio:.1%}  ({len(out_df):,} total rows)")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
