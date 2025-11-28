#!/usr/bin/env python3
"""
Download ESC-50 audio dataset.

The ESC-50 dataset is available from GitHub as a ZIP file.
This script downloads and extracts the audio files.

Usage:
    python scripts/download_esc50_audio.py --data-root data/raw
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    logger.info(f"Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(output_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"Downloaded to {output_path}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file."""
    logger.info(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

    logger.info(f"Extracted to {extract_to}")


def download_esc50_audio(data_root: Path):
    """
    Download ESC-50 audio files.

    Args:
        data_root: Root directory for data storage
    """
    esc50_dir = data_root / "esc50"
    esc50_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = esc50_dir / "audio"

    # Check if already extracted
    if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
        logger.info(f"ESC-50 audio already downloaded")
        logger.info(f"Found {len(list(audio_dir.glob('*.wav')))} WAV files")
        return

    zip_path = esc50_dir / "ESC-50-master.zip"

    # Download ZIP
    if not zip_path.exists():
        try:
            download_file(ESC50_URL, zip_path)
        except Exception as e:
            logger.error(f"Failed to download ESC-50: {e}")
            return
    else:
        logger.info(f"ZIP already exists: {zip_path}")

    # Extract ZIP
    try:
        extract_zip(zip_path, esc50_dir)

        # Move audio files from extracted directory
        extracted_audio = esc50_dir / "ESC-50-master" / "audio"
        if extracted_audio.exists():
            audio_dir.mkdir(parents=True, exist_ok=True)
            for wav_file in extracted_audio.glob("*.wav"):
                new_path = audio_dir / wav_file.name
                wav_file.rename(new_path)

            # Count files
            wav_files = list(audio_dir.glob("*.wav"))
            logger.info(f"Moved {len(wav_files)} WAV files to {audio_dir}")

        logger.info("ESC-50 download complete!")

    except Exception as e:
        logger.error(f"Failed to extract ESC-50: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download ESC-50 audio dataset")
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw"), help="Data root directory"
    )

    args = parser.parse_args()
    download_esc50_audio(args.data_root)


if __name__ == "__main__":
    main()
