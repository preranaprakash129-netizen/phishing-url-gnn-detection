"""
collect_redirect_chains.py

Purpose:
    Your 3 usable datasets (kaggle_cleaned.csv, cleaned_github_dataset1.csv,
    cleaned_github_dataset_2.json) only contain a single final URL per row —
    no redirect-chain info. This script visits each URL, follows every
    redirect hop (HTTP 3xx), and records the full chain (domain, IP, status
    code per hop) so you have graph-ready data for the GNN.

    uci_cleaned.csv is NOT used here — it has no raw URL column.

How to use:
    1. Put this script in the SAME folder as your 3 dataset files
       (your repo folder: phishing-url-gnn-detection).
    2. Run: python collect_redirect_chains.py
    3. It processes a SMALL_SAMPLE_SIZE first from each dataset (default 20)
       so you can sanity-check the output before committing to a full run.
    4. Output: redirect_chains_output.csv (one row per hop, chain_id +
       source_dataset link hops together per original URL).

IMPORTANT — read before running:
    - This is a REAL network operation. It visits every URL, including
      live/dead phishing URLs. Many WILL fail (ConnectionError, Timeout,
      SSLError, 404) — this is expected: a lot of phishing infrastructure
      dies within days of being reported. is_reachable=False rows are
      still useful data, not something to discard.
    - At 3000 URLs per dataset (9000 total) with a 1-second delay between
      requests, this run takes roughly 2.5-4 hours depending on how many
      URLs time out (timeouts take longer than successful requests).
      Leave your laptop plugged in and awake (disable sleep/screen-lock
      auto-suspend) and let it run in the background — overnight is ideal.
    - Progress is saved incrementally (flushed every 50 chains), so if it
      gets interrupted partway, redirect_chains_output.csv will still have
      everything collected up to that point — just not a clean "resume"
      from where it stopped, so avoid closing the window unless needed.
    - Only visiting URLs from public phishing-research datasets, which is
      standard practice in this research area. Never point this at anything
      outside these datasets.
"""

import pandas as pd
import requests
from urllib.parse import urlparse
import socket
import time
import csv
import json

# ============ CONFIG - EDIT IF NEEDED ============
SMALL_SAMPLE_SIZE = 3000        # per dataset. set to None to run on the FULL dataset (careful!)
OUTPUT_FILE = "redirect_chains_output.csv"
TIMEOUT_SECONDS = 8
DELAY_BETWEEN_REQUESTS = 1.0    # seconds, be polite to servers

# Each entry: (filename, format, url_column, label_column)
DATASETS = [
    ("kaggle_cleaned.csv", "csv", "URL", "label"),
    ("cleaned_github_dataset1.csv", "csv", "url", "label"),
    ("cleaned_github_dataset_2.json", "json", "url", "label"),
]
# ===================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


def resolve_ip(hostname):
    try:
        return socket.gethostbyname(hostname)
    except Exception:
        return None


def trace_redirect_chain(url, timeout=TIMEOUT_SECONDS):
    """Follows a URL through all its redirects and returns a list of hop dicts."""
    hops = []
    try:
        response = requests.get(
            url, headers=HEADERS, timeout=timeout,
            allow_redirects=True, stream=True
        )
        chain = list(response.history) + [response]
        for i, resp in enumerate(chain):
            hostname = urlparse(resp.url).hostname
            hops.append({
                "hop_index": i,
                "url": resp.url,
                "domain": hostname,
                "ip": resolve_ip(hostname) if hostname else None,
                "status_code": resp.status_code,
                "is_reachable": True,
            })
        response.close()
    except requests.exceptions.RequestException as e:
        hops.append({
            "hop_index": 0,
            "url": url,
            "domain": urlparse(url).hostname,
            "ip": None,
            "status_code": f"ERROR: {type(e).__name__}",
            "is_reachable": False,
        })
    return hops


def load_dataset(filename, fmt, url_col, label_col):
    """Load a dataset file and return a DataFrame with standardized url/label columns."""
    if fmt == "csv":
        df = pd.read_csv(filename)
    elif fmt == "json":
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if url_col not in df.columns:
        raise ValueError(f"Column '{url_col}' not found in {filename}. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["url"] = df[url_col]
    out["label"] = df[label_col] if label_col in df.columns else ""
    return out


def main():
    all_rows = []
    for filename, fmt, url_col, label_col in DATASETS:
        print(f"\nLoading {filename} ...")
        try:
            df = load_dataset(filename, fmt, url_col, label_col)
        except FileNotFoundError:
            print(f"  SKIPPED: {filename} not found in this folder.")
            continue
        except ValueError as e:
            print(f"  SKIPPED: {e}")
            continue

        if SMALL_SAMPLE_SIZE:
            df = df.head(SMALL_SAMPLE_SIZE)

        df["source_dataset"] = filename
        all_rows.append(df)
        print(f"  Loaded {len(df)} rows (sample) from {filename}")

    if not all_rows:
        print("\nNo datasets loaded. Check filenames/paths and try again.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\nTotal URLs to trace: {len(combined)}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chain_id", "source_dataset", "original_url", "label",
            "hop_index", "hop_url", "domain", "ip", "status_code", "is_reachable"
        ])

        for chain_id, row in combined.iterrows():
            url = row["url"]
            label = row["label"]
            source = row["source_dataset"]
            print(f"[{chain_id + 1}/{len(combined)}] ({source}) Tracing: {url}")

            hops = trace_redirect_chain(url)
            for hop in hops:
                writer.writerow([
                    chain_id, source, url, label,
                    hop["hop_index"], hop["url"], hop["domain"],
                    hop["ip"], hop["status_code"], hop["is_reachable"]
                ])

            # flush periodically so partial progress is saved if interrupted
            if chain_id % 50 == 0:
                f.flush()

            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nDone. Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
