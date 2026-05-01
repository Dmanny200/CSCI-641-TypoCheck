"""
collect_data.py

Builds a labelled dataset for URL-based typosquatting / phishing detection.
Output: data/urls.csv  (columns: url, label, source)
  label 0 = benign
  label 1 = malicious

Sources
-------
Benign  (1) Tranco top 10 000            — direct download, domains → http://domain/
        (2) Majestic Million top 500 000  — direct download, domains → http://domain/
        (3) Brand subdomains              — hardcoded list, → https://subdomain/
        (4) Kaggle Malicious URLs benign  — see KAGGLE SETUP below
Malicious (5) OpenPhish                  — full URLs from live feed
          (6) PhishTank                  — full URLs
          (7) URLhaus                    — full URLs
          (8) Kaggle Malicious URLs phishing/malware/defacement — see KAGGLE SETUP
          (9) Kaggle ISCX URL 2016 malicious — see KAGGLE SETUP
         (10) Generated typosquats        — domain → http://domain/
         (11) Generated DGA domains       — domain → http://domain/

KAGGLE SETUP
------------
1. Create a free Kaggle account at kaggle.com
2. Go to Account → API → Create New Token → downloads kaggle.json
3. mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
4. Re-run this script

Alternatively, manually download the CSVs and place them in data/external/:
  data/external/malicious_urls.csv   — from kaggle.com/datasets/sid321axn/malicious-urls-dataset
  data/external/iscx_urls.csv        — from kaggle.com/datasets/tejeswinisharma/url-dataset
"""

import csv
import io
import os
import random
import string
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import pandas as pd
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
EXTERNAL    = DATA_DIR / "external"
DATA_DIR.mkdir(exist_ok=True)
EXTERNAL.mkdir(exist_ok=True)
OUTPUT_CSV  = DATA_DIR / "urls.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def normalize_url(raw: str) -> str | None:
    """Normalize a URL, preserving scheme + host + path + query + fragment.
    Returns None if the URL has no valid hostname.
    """
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if not host or "." not in host or " " in host:
            return None
        scheme = parsed.scheme.lower()
        path   = parsed.path if parsed.path else "/"
        port   = f":{parsed.port}" if parsed.port else ""
        query  = f"?{parsed.query}" if parsed.query else ""
        frag   = f"#{parsed.fragment}" if parsed.fragment else ""
        return f"{scheme}://{host}{port}{path}{query}{frag}"
    except Exception:
        return None


def _bare_host(url: str) -> str:
    """Hostname without leading www., for dedup/collision checks."""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        h = (urlparse(url).hostname or "").lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def fetch_text(url: str, timeout: int = 30) -> str | None:
    headers = {"User-Agent": "Mozilla/5.0 (research-dataset-builder/1.0)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] could not fetch {url}: {exc}")
        return None


def fetch_bytes(url: str, timeout: int = 60) -> bytes | None:
    headers = {"User-Agent": "Mozilla/5.0 (research-dataset-builder/1.0)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        print(f"  [warn] could not fetch {url}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. BENIGN — Tranco top 10 000
# ═══════════════════════════════════════════════════════════════════════════

def get_tranco(n: int = 10_000) -> list[str]:
    """Download the latest Tranco list and return top-n as normalized URLs."""
    print(f"\n[1/11] Downloading Tranco top {n:,} …")
    data = fetch_bytes("https://tranco-list.eu/top-1m.csv.zip", timeout=120)
    if not data:
        print("  [warn] Tranco download failed — returning empty list")
        return []

    urls: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            for rank, row in enumerate(reader, start=1):
                if rank > n:
                    break
                if row:
                    domain = row[1].lower().strip()
                    urls.append(f"http://{domain}/")
    print(f"  -> {len(urls):,} Tranco URLs loaded")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 2. BENIGN — Majestic Million
# ═══════════════════════════════════════════════════════════════════════════

def get_majestic_million(n: int = 500_000) -> list[str]:
    """Download the Majestic Million and return top-n domains as http:// URLs.

    Provides benign domains ranked by referring subnets — good coverage of
    real-world traffic beyond Tranco's top 10k.
    """
    print(f"\n[2/11] Downloading Majestic Million (top {n:,}) …")
    data = fetch_bytes("https://downloads.majestic.com/majestic_million.csv", timeout=120)
    if not data:
        print("  [warn] Majestic Million download failed — skipping")
        return []

    urls: list[str] = []
    try:
        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for i, row in enumerate(reader):
            if i >= n:
                break
            domain = (row.get("Domain") or "").lower().strip()
            if domain and "." in domain:
                urls.append(f"http://{domain}/")
    except Exception as exc:
        print(f"  [warn] Majestic parse error: {exc}")

    urls = list(dict.fromkeys(urls))
    print(f"  -> {len(urls):,} Majestic Million URLs loaded")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 3. BENIGN — brand subdomains
# ═══════════════════════════════════════════════════════════════════════════

BRAND_SUBDOMAINS: list[str] = [
    # Google
    "mail.google.com", "drive.google.com", "docs.google.com",
    "maps.google.com", "news.google.com", "play.google.com",
    "cloud.google.com", "meet.google.com", "calendar.google.com",
    # Amazon / AWS
    "aws.amazon.com", "console.aws.amazon.com", "s3.amazonaws.com",
    "signin.aws.amazon.com", "payments.amazon.com",
    # Microsoft
    "login.microsoftonline.com", "outlook.live.com", "teams.microsoft.com",
    "onedrive.live.com", "azure.microsoft.com", "portal.azure.com",
    "support.microsoft.com", "store.steampowered.com",
    # Apple
    "support.apple.com", "developer.apple.com", "icloud.com",
    "appleid.apple.com",
    # Meta / Facebook
    "m.facebook.com", "developers.facebook.com", "business.facebook.com",
    "help.instagram.com",
    # Other majors
    "api.twitter.com", "developer.twitter.com",
    "www.youtube.com", "studio.youtube.com",
    "secure.paypal.com", "developer.paypal.com",
    "help.github.com", "gist.github.com", "docs.github.com",
    "account.live.com", "login.yahoo.com", "mail.yahoo.com",
    "accounts.google.com", "myaccount.google.com",
    "signin.ebay.com", "pages.ebay.com",
    "my.netflix.com", "help.netflix.com",
    "open.spotify.com", "accounts.spotify.com",
    "shop.app", "checkout.shopify.com",
]

def get_brand_subdomains() -> list[str]:
    print("\n[3/11] Adding brand subdomains …")
    urls = [f"https://{d.lower()}/" for d in BRAND_SUBDOMAINS]
    print(f"  -> {len(urls):,} brand subdomain URLs")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 4 & 8 & 9. Kaggle datasets
#   4 = Kaggle benign URLs (from malicious_urls.csv "benign" type)
#   8 = Kaggle malicious URLs (phishing/malware/defacement from same file)
#   9 = Kaggle ISCX URL 2016
# ═══════════════════════════════════════════════════════════════════════════

def _try_kaggle_download(dataset_slug: str, dest_dir: Path) -> bool:
    """Attempt to download a Kaggle dataset using stored API credentials.
    Returns True if the download succeeded.
    """
    try:
        import kaggle  # noqa: F401 — just checks import works
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True, quiet=False)
        return True
    except Exception as exc:
        print(f"  [warn] Kaggle download failed for '{dataset_slug}': {exc}")
        return False


def _kaggle_setup_instructions(filename: str, dataset_slug: str) -> None:
    print(f"""
  [info] To include this dataset, either:

  Option A — Kaggle API (automated):
    1. Sign up at kaggle.com, go to Account → API → Create New Token
    2. Save the downloaded kaggle.json:
         mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    3. Re-run this script.

  Option B — Manual download:
    1. Go to kaggle.com/datasets/{dataset_slug}
    2. Download and unzip into:  {EXTERNAL / filename}
""")


def get_kaggle_malicious_urls() -> tuple[list[str], list[str]]:
    """Load sid321axn/malicious-urls-dataset.

    Returns (benign_urls, malicious_urls).
    Expected file: data/external/malicious_urls.csv
    Columns: url, type   (type values: benign / phishing / defacement / malware)
    """
    print("\n[4+8/11] Loading Kaggle Malicious URLs dataset (sid321axn) …")
    slug     = "sid321axn/malicious-urls-dataset"
    filename = "malicious_urls.csv"
    fpath    = EXTERNAL / filename

    if not fpath.exists():
        print(f"  [info] {fpath} not found — attempting Kaggle API download …")
        ok = _try_kaggle_download(slug, EXTERNAL)
        if not ok or not fpath.exists():
            _kaggle_setup_instructions(filename, slug)
            return [], []

    try:
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        # normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        # find url and type columns flexibly
        url_col  = next((c for c in df.columns if "url" in c), None)
        type_col = next((c for c in df.columns if c in ("type", "label", "class", "result")), None)

        if url_col is None:
            print(f"  [warn] Could not find url column in {filename} — columns: {list(df.columns)}")
            return [], []

        benign_urls:    list[str] = []
        malicious_urls: list[str] = []

        for _, row in df.iterrows():
            raw  = str(row.get(url_col, "") or "").strip()
            kind = str(row.get(type_col, "") or "").strip().lower() if type_col else "malicious"
            u    = normalize_url(raw)
            if not u:
                continue
            if kind == "benign":
                benign_urls.append(u)
            else:
                malicious_urls.append(u)

        benign_urls    = list(dict.fromkeys(benign_urls))
        malicious_urls = list(dict.fromkeys(malicious_urls))
        print(f"  -> {len(benign_urls):,} benign  |  {len(malicious_urls):,} malicious (Kaggle)")
        return benign_urls, malicious_urls

    except Exception as exc:
        print(f"  [warn] Failed to parse {filename}: {exc}")
        return [], []


def get_kaggle_iscx() -> tuple[list[str], list[str]]:
    """Load ISCX URL 2016 dataset (tejeswinisharma/url-dataset).

    Returns (benign_urls, malicious_urls).
    Expected file: data/external/iscx_urls.csv
    Common columns: url + label/class/type/result  (good/bad or 0/1)
    """
    print("\n[9/11] Loading Kaggle ISCX URL dataset …")
    slug     = "tejeswinisharma/url-dataset"
    filename = "iscx_urls.csv"
    fpath    = EXTERNAL / filename

    if not fpath.exists():
        # also look for common alternative filenames that Kaggle might unzip to
        alternatives = [
            EXTERNAL / "url_dataset.csv",
            EXTERNAL / "dataset.csv",
            EXTERNAL / "data.csv",
        ]
        for alt in alternatives:
            if alt.exists():
                fpath = alt
                break
        else:
            print(f"  [info] {EXTERNAL / filename} not found — attempting Kaggle API download …")
            ok = _try_kaggle_download(slug, EXTERNAL)
            # after download, look for whatever CSV appeared
            if not ok:
                _kaggle_setup_instructions(filename, slug)
                return [], []
            csvs = list(EXTERNAL.glob("*.csv"))
            if not csvs:
                _kaggle_setup_instructions(filename, slug)
                return [], []
            fpath = csvs[0]

    try:
        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]

        url_col  = next((c for c in df.columns if "url" in c), None)
        type_col = next(
            (c for c in df.columns if c in ("label", "class", "type", "result", "category")),
            None,
        )

        if url_col is None:
            print(f"  [warn] Could not find url column in {fpath.name} — columns: {list(df.columns)}")
            return [], []

        BENIGN_VALUES = {"benign", "good", "legitimate", "0", "false", "safe"}

        benign_urls:    list[str] = []
        malicious_urls: list[str] = []

        for _, row in df.iterrows():
            raw  = str(row.get(url_col, "") or "").strip()
            kind = str(row.get(type_col, "") or "").strip().lower() if type_col else "malicious"
            u    = normalize_url(raw)
            if not u:
                continue
            if kind in BENIGN_VALUES:
                benign_urls.append(u)
            else:
                malicious_urls.append(u)

        benign_urls    = list(dict.fromkeys(benign_urls))
        malicious_urls = list(dict.fromkeys(malicious_urls))
        print(f"  -> {len(benign_urls):,} benign  |  {len(malicious_urls):,} malicious (ISCX)")
        return benign_urls, malicious_urls

    except Exception as exc:
        print(f"  [warn] Failed to parse {fpath.name}: {exc}")
        return [], []


# ═══════════════════════════════════════════════════════════════════════════
# 5. MALICIOUS — OpenPhish  (full URLs kept)
# ═══════════════════════════════════════════════════════════════════════════

def get_openphish() -> list[str]:
    print("\n[5/11] Fetching OpenPhish feed …")
    text = fetch_text("https://openphish.com/feed.txt")
    if not text:
        return []
    urls = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            u = normalize_url(line)
            if u:
                urls.append(u)
    urls = list(dict.fromkeys(urls))
    print(f"  -> {len(urls):,} unique URLs from OpenPhish")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 6. MALICIOUS — PhishTank  (full URLs kept)
# ═══════════════════════════════════════════════════════════════════════════

def get_phishtank() -> list[str]:
    print("\n[6/11] Fetching PhishTank …")
    url = "https://data.phishtank.com/data/online-valid.csv"
    text = fetch_text(url, timeout=120)
    if not text:
        return []
    urls: list[str] = []
    try:
        df = pd.read_csv(io.StringIO(text), usecols=["url"], dtype=str)
        for raw_url in df["url"].dropna():
            u = normalize_url(str(raw_url))
            if u:
                urls.append(u)
    except Exception as exc:
        print(f"  [warn] PhishTank parse error: {exc}")
    urls = list(dict.fromkeys(urls))
    print(f"  -> {len(urls):,} unique URLs from PhishTank")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 7. MALICIOUS — URLhaus  (full URLs kept)
# ═══════════════════════════════════════════════════════════════════════════

def get_urlhaus() -> list[str]:
    print("\n[7/11] Fetching URLhaus …")
    url = "https://urlhaus.abuse.ch/downloads/csv/"
    data = fetch_bytes(url, timeout=120)
    if not data:
        return []

    urls: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    # columns: id, dateadded, url, url_status, ...
                    if len(row) >= 3:
                        u = normalize_url(row[2])
                        if u:
                            urls.append(u)
    except Exception:
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                u = normalize_url(parts[2].strip('"'))
                if u:
                    urls.append(u)

    urls = list(dict.fromkeys(urls))
    print(f"  -> {len(urls):,} unique URLs from URLhaus")
    return urls


# ═══════════════════════════════════════════════════════════════════════════
# 10. GENERATED — typosquatting variations  (domain → http://domain/)
# ═══════════════════════════════════════════════════════════════════════════

HOMOGLYPHS: dict[str, list[str]] = {
    "o": ["0"], "0": ["o"], "l": ["1", "i"], "i": ["1", "l"],
    "1": ["l", "i"], "a": ["@", "4"], "e": ["3"], "s": ["5", "$"],
    "t": ["7"], "g": ["9"], "b": ["6"],
}

KEYBOARD_ADJACENCY: dict[str, list[str]] = {
    "a": ["q","w","s","z"],  "b": ["v","g","h","n"],
    "c": ["x","d","f","v"],  "d": ["s","e","r","f","c","x"],
    "e": ["w","s","d","r"],  "f": ["d","r","t","g","v","c"],
    "g": ["f","t","y","h","b","v"],  "h": ["g","y","u","j","n","b"],
    "i": ["u","j","k","o"],  "j": ["h","u","i","k","n","m"],
    "k": ["j","i","o","l","m"],  "l": ["k","o","p"],
    "m": ["n","j","k"],  "n": ["b","h","j","m"],
    "o": ["i","k","l","p"],  "p": ["o","l"],
    "q": ["w","a"],  "r": ["e","d","f","t"],
    "s": ["a","w","e","d","x","z"],  "t": ["r","f","g","y"],
    "u": ["y","h","j","i"],  "v": ["c","f","g","b"],
    "w": ["q","a","s","e"],  "x": ["z","s","d","c"],
    "y": ["t","g","h","u"],  "z": ["a","s","x"],
}

ALT_TLDS = [".co", ".net", ".org", ".xyz", ".info", ".biz",
            ".online", ".site", ".club", ".shop", ".io"]


def _sld_tld(domain: str) -> tuple[str, str]:
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:-1]), "." + parts[-1]
    return domain, ""


def generate_typos(domain: str) -> list[str]:
    sld, tld = _sld_tld(domain)
    variants: list[str] = []

    for i in range(len(sld)):
        v = sld[:i] + sld[i+1:]
        if v:
            variants.append(v + tld)
    for i in range(len(sld) - 1):
        v = sld[:i] + sld[i+1] + sld[i] + sld[i+2:]
        variants.append(v + tld)
    for i in range(len(sld)):
        v = sld[:i] + sld[i] + sld[i] + sld[i+1:]
        variants.append(v + tld)
    for i, ch in enumerate(sld):
        if ch in HOMOGLYPHS:
            for repl in HOMOGLYPHS[ch]:
                v = sld[:i] + repl + sld[i+1:]
                variants.append(v + tld)
    for i, ch in enumerate(sld):
        if ch in KEYBOARD_ADJACENCY:
            for repl in KEYBOARD_ADJACENCY[ch]:
                v = sld[:i] + repl + sld[i+1:]
                variants.append(v + tld)
    for i in range(1, len(sld)):
        variants.append(sld[:i] + "." + sld[i:] + tld)
    for alt in ALT_TLDS:
        if alt != tld:
            variants.append(sld + alt)
    for i in range(1, len(sld)):
        variants.append(sld[:i] + "-" + sld[i:] + tld)

    clean: list[str] = []
    for v in variants:
        v = v.lower().strip(".")
        if v and len(v) >= 4 and v != domain:
            clean.append(v)
    return list(dict.fromkeys(clean))


def get_typosquatting_variants(tranco_urls: list[str], top_n: int = 200) -> list[str]:
    print(f"\n[10/11] Generating typosquatting variants for top {top_n} Tranco domains …")
    seed_hosts = [_bare_host(u) for u in tranco_urls[:top_n]]
    all_variants: list[str] = []
    for d in tqdm(seed_hosts, unit="domain"):
        all_variants.extend(generate_typos(d))
    all_variants = list(dict.fromkeys(all_variants))
    print(f"  -> {len(all_variants):,} unique typosquatting variants")
    return [f"http://{d}/" for d in all_variants]


# ═══════════════════════════════════════════════════════════════════════════
# 11. GENERATED — DGA domains  (domain → http://domain/)
# ═══════════════════════════════════════════════════════════════════════════

DGA_TLDS = [".com", ".net", ".org", ".ru", ".cn", ".info", ".biz",
            ".xyz", ".top", ".club", ".pw", ".tk"]

DICT_WORDS = [
    "account","action","admin","agent","alert","amazon","apple","bank",
    "base","board","boot","box","brand","bulk","card","cash","check",
    "click","cloud","code","conf","connect","content","control","corp",
    "credit","crypto","data","deal","desk","direct","disk","domain",
    "drive","drop","email","event","extra","fast","feed","file","find",
    "fire","first","flag","flash","flow","font","force","form","free",
    "fund","game","gate","give","global","gold","good","grant","grid",
    "group","guard","guide","hash","help","hide","host","hub","info",
    "init","item","join","jump","just","keep","key","kill","kit","land",
    "launch","layer","lead","learn","left","level","link","list","live",
    "load","lock","log","loop","mail","main","make","mark","media","mesh",
    "meta","micro","mine","mode","move","name","net","new","next","node",
    "note","notify","open","order","page","pass","path","pay","ping",
    "pipe","plan","play","plug","point","pool","port","post","power",
    "press","price","prime","print","pro","proxy","push","quick","rank",
    "rate","read","real","reply","report","reset","root","route","rule",
    "run","safe","save","scan","search","secure","send","serve","set",
    "shop","sign","simple","site","skip","smart","sort","speed","split",
    "start","stat","step","store","stream","style","sync","system","tag",
    "target","task","team","text","time","token","tool","top","trace",
    "track","trade","trust","type","unit","update","upload","user","valid",
    "value","view","visit","vote","web","wire","work","world","write","zone",
]

def _random_dga(length: int) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def get_dga_domains(n_random: int = 1000, n_dict: int = 500) -> list[str]:
    print(f"\n[11/11] Generating {n_random} random + {n_dict} dictionary DGA domains …")
    domains: list[str] = []
    for _ in range(n_random):
        length = random.randint(8, 25)
        tld = random.choice(DGA_TLDS)
        domains.append(_random_dga(length) + tld)
    for _ in range(n_dict):
        n_words = random.randint(2, 3)
        sld = "".join(random.choices(DICT_WORDS, k=n_words))
        tld = random.choice(DGA_TLDS)
        domains.append(sld + tld)
    domains = list(dict.fromkeys(domains))
    print(f"  -> {len(domains):,} unique DGA domains")
    return [f"http://{d}/" for d in domains]


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Typosquatting / Phishing URL Dataset Builder")
    print("=" * 60)

    rows: list[dict] = []

    # ── benign ───────────────────────────────────────────────────────────
    tranco_urls = get_tranco(10_000)
    tranco_host_set = set(_bare_host(u) for u in tranco_urls)
    for u in tranco_urls:
        rows.append({"url": u, "label": 0, "source": "tranco_top10k"})

    majestic_urls = get_majestic_million(500_000)
    for u in majestic_urls:
        rows.append({"url": u, "label": 0, "source": "majestic_million"})

    for u in get_brand_subdomains():
        rows.append({"url": u, "label": 0, "source": "brand_subdomains"})

    # ── Kaggle (benign + malicious) ───────────────────────────────────────
    kaggle_benign, kaggle_malicious = get_kaggle_malicious_urls()
    for u in kaggle_benign:
        rows.append({"url": u, "label": 0, "source": "kaggle_malicious_urls"})
    for u in kaggle_malicious:
        rows.append({"url": u, "label": 1, "source": "kaggle_malicious_urls"})

    iscx_benign, iscx_malicious = get_kaggle_iscx()
    for u in iscx_benign:
        rows.append({"url": u, "label": 0, "source": "kaggle_iscx"})
    for u in iscx_malicious:
        rows.append({"url": u, "label": 1, "source": "kaggle_iscx"})

    # ── malicious (live feeds) ────────────────────────────────────────────
    for u in get_openphish():
        rows.append({"url": u, "label": 1, "source": "openphish"})

    for u in get_phishtank():
        rows.append({"url": u, "label": 1, "source": "phishtank"})

    for u in get_urlhaus():
        rows.append({"url": u, "label": 1, "source": "urlhaus"})

    # ── malicious (generated) ────────────────────────────────────────────
    for u in get_typosquatting_variants(tranco_urls, top_n=200):
        rows.append({"url": u, "label": 1, "source": "generated_typosquat"})

    for u in get_dga_domains(n_random=1000, n_dict=500):
        rows.append({"url": u, "label": 1, "source": "generated_dga"})

    # ── build dataframe ──────────────────────────────────────────────────
    print("\nBuilding dataframe …")
    df = pd.DataFrame(rows)
    df["url"] = df["url"].str.strip()
    df = df[df["url"].str.contains(r"://", regex=False, na=False)]
    df = df[df["url"].str.len() >= 10]

    # remove malicious rows whose hostname appears in Tranco top 10k
    df["_host"] = df["url"].apply(_bare_host)
    mask_mal_in_tranco = (df["label"] == 1) & (df["_host"].isin(tranco_host_set))
    n_removed = mask_mal_in_tranco.sum()
    if n_removed:
        print(f"  Removed {n_removed:,} malicious URLs whose host collides with Tranco top 10k")
    df = df[~mask_mal_in_tranco].drop(columns=["_host"])

    # dedup on exact URL (benign sources listed first keeps them benign on collision)
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    print(f"  Removed {before - len(df):,} duplicate URL rows")

    # ── rebalance to ~70% benign / 30% malicious ─────────────────────────
    TARGET_BENIGN_FRAC = 0.70
    TARGET_MAL_FRAC    = 0.30

    df_benign    = df[df["label"] == 0]
    df_malicious = df[df["label"] == 1]
    n_benign_orig    = len(df_benign)
    n_malicious_orig = len(df_malicious)

    # balance both directions: cap whichever class is oversized
    target_mal    = int(n_benign_orig * TARGET_MAL_FRAC / TARGET_BENIGN_FRAC)
    target_benign = int(n_malicious_orig * TARGET_BENIGN_FRAC / TARGET_MAL_FRAC)

    print(f"\nRebalancing: {n_benign_orig:,} benign, {n_malicious_orig:,} malicious")
    if n_malicious_orig > target_mal and n_benign_orig <= target_benign:
        df_malicious = df_malicious.sample(n=target_mal, random_state=RANDOM_SEED)
        print(f"  Downsampled malicious: {n_malicious_orig:,} -> {target_mal:,}")
    elif n_benign_orig > target_benign and n_malicious_orig <= target_mal:
        df_benign = df_benign.sample(n=target_benign, random_state=RANDOM_SEED)
        print(f"  Downsampled benign: {n_benign_orig:,} -> {target_benign:,}")
    elif n_malicious_orig > target_mal and n_benign_orig > target_benign:
        # both oversized — downsample malicious first, then benign to match
        df_malicious = df_malicious.sample(n=target_mal, random_state=RANDOM_SEED)
        df_benign    = df_benign.sample(n=target_benign, random_state=RANDOM_SEED)
        print(f"  Downsampled both classes to maintain 70/30 split")
    else:
        print(f"  No downsampling needed")

    df = (pd.concat([df_benign, df_malicious])
            .sample(frac=1, random_state=RANDOM_SEED)
            .reset_index(drop=True))

    # ── summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset summary")
    print("=" * 60)
    summary = (
        df.groupby(["source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[True, False])
    )
    for _, row in summary.iterrows():
        tag = "benign   " if row["label"] == 0 else "malicious"
        print(f"  {tag}  {row['source']:<30}  {row['count']:>7,}")
    print("-" * 60)
    label_counts = df["label"].value_counts().sort_index()
    n_benign  = label_counts.get(0, 0)
    n_mal     = label_counts.get(1, 0)
    n_total   = len(df)
    print(f"  Total benign:     {n_benign:>7,}  ({100 * n_benign / n_total:.1f}%)")
    print(f"  Total malicious:  {n_mal:>7,}  ({100 * n_mal / n_total:.1f}%)")
    print(f"  Grand total:      {n_total:>7,}")
    print("=" * 60)

    df.to_csv(OUTPUT_CSV, index=False, escapechar='\\')
    print(f"\nSaved -> {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
