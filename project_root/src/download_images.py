"""Download product images from URLs and cache locally.

CLI example:
    python src/download_images.py \
        --labels_csv data/processed_labels.csv \
        --out_dir data/images \
        --seed 42

This script:
- Reads `data/processed_labels.csv` (which has image URLs in `image_path` column)
- Downloads the first valid URL for each row to `data/images/<uniq_id>.jpg`
- Caches: skips re-downloading existing files
- Handles redirects, timeouts, retries (3 attempts per URL)
- Updates `data/processed_labels.csv` to replace URLs with local file paths
- Marks failed downloads (missing images in final CSV)

Design decisions:
- Robust to network failures: retries with exponential backoff
- Filters out placeholder/transparent images before download
- Uses requests library with proper User-Agent
- Saves images as JPEG (lossy but memory-efficient)
- Logs download progress and failures for debugging
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image

LOGGER = logging.getLogger("download_images")

# Default timeout and retry settings
TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5


def seed_everything(seed: int) -> None:
    """Set seed for python, numpy and random to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def download_image(url: str, output_path: str, timeout: int = TIMEOUT) -> bool:
    """Download image from URL and save to output_path.

    Returns True if successful, False otherwise.
    Handles retries, redirects, and timeouts.
    """
    # Basic validation
    if not url or not isinstance(url, str):
        return False

    # Skip obvious placeholders
    url_lower = url.lower()
    if any(x in url_lower for x in ["placeholder", "spacer.gif", "data:"]):
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url, timeout=timeout, headers=headers, allow_redirects=True
            )
            response.raise_for_status()

            # Validate image
            img = Image.open(io.BytesIO(response.content))
            if img.size[0] < 50 or img.size[1] < 50:
                # Skip tiny images (likely placeholders)
                LOGGER.warning("Image too small: %s (%s)", url, img.size)
                return False

            # Save as JPEG
            img_rgb = img.convert("RGB")
            img_rgb.save(output_path, quality=95)
            LOGGER.debug("Downloaded: %s -> %s", url, output_path)
            return True

        except (requests.RequestException, IOError, OSError) as e:
            wait = BACKOFF_FACTOR * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                LOGGER.debug(
                    "Retry %d/%d for %s after %.1fs (error: %s)",
                    attempt + 1,
                    MAX_RETRIES,
                    url,
                    wait,
                    str(e)[:50],
                )
                time.sleep(wait)
            else:
                LOGGER.warning(
                    "Failed to download after %d retries: %s (error: %s)",
                    MAX_RETRIES,
                    url,
                    str(e)[:50],
                )
                return False

    return False


def process_labels_and_download(
    labels_csv: str, out_dir: str, seed: int = 42
) -> pd.DataFrame:
    """Load processed_labels.csv, download images, update paths, return updated df."""
    seed_everything(seed)

    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load processed labels
    df = pd.read_csv(labels_csv)
    LOGGER.info("Loaded %d rows from %s", len(df), labels_csv)

    # Download images
    downloaded = 0
    failed = 0
    cached = 0

    new_paths = []
    for idx, row in df.iterrows():
        uniq_id = row["uniq_id"]
        image_url = row["image_path"]  # Currently a URL
        output_path = os.path.join(out_dir, f"{uniq_id}.jpg")

        # Check if already downloaded
        if os.path.exists(output_path):
            LOGGER.debug("Cached: %s", output_path)
            new_paths.append(output_path)
            cached += 1
            continue

        # Download
        if image_url and isinstance(image_url, str) and image_url.startswith("http"):
            success = download_image(image_url, output_path)
            if success:
                new_paths.append(output_path)
                downloaded += 1
            else:
                LOGGER.warning(
                    "Failed to download for uniq_id %s: %s", uniq_id, image_url
                )
                new_paths.append(None)  # Mark as failed
                failed += 1
        else:
            LOGGER.warning("Invalid image URL for uniq_id %s: %s", uniq_id, image_url)
            new_paths.append(None)
            failed += 1

        if (idx + 1) % 100 == 0:
            LOGGER.info("Progress: %d/%d", idx + 1, len(df))

    # Update dataframe
    df["image_path"] = new_paths

    # Remove rows with missing images
    df_clean = df[df["image_path"].notna()].copy()
    dropped = len(df) - len(df_clean)

    LOGGER.info(
        "Download summary: %d downloaded, %d cached, %d failed, %d dropped (missing images)",
        downloaded,
        cached,
        failed,
        dropped,
    )
    LOGGER.info("Final rows: %d (was %d)", len(df_clean), len(df))

    return df_clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download product images and update processed_labels.csv"
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="Path to processed_labels.csv (contains image URLs)",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save downloaded images"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: verbose logging"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max images to download (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s:%(levelname)s: %(message)s",
    )

    df_updated = process_labels_and_download(
        labels_csv=args.labels_csv, out_dir=args.out_dir, seed=args.seed
    )

    # Save updated CSV (overwrite original or to new file)
    out_csv = args.labels_csv  # Overwrite original
    df_updated.to_csv(out_csv, index=False)
    LOGGER.info("Updated labels CSV saved: %s", out_csv)


if __name__ == "__main__":
    main()
