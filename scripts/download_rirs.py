#!/usr/bin/env python3
"""
Download Room Impulse Responses (RIRs) from OpenSLR SLR28.

Dataset: https://www.openslr.org/28/
Contains simulated and real RIRs for reverberation experiments.

Usage:
    python scripts/download_rirs.py --output_dir data/rirs
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

# OpenSLR SLR28 dataset info
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
RIRS_ALT_URL = "https://us.openslr.org/resources/28/rirs_noises.zip"

# T60 metadata for common RIR categories based on OpenSLR documentation
# These are approximate values based on room size categories
T60_METADATA = {
    # Simulated RIRs - categorized by room type
    "smallroom": {"T60_range": (0.2, 0.4), "T60_typical": 0.3},
    "mediumroom": {"T60_range": (0.4, 0.8), "T60_typical": 0.6},
    "largeroom": {"T60_range": (0.8, 1.5), "T60_typical": 1.0},
    # For very reverberant conditions, we'll select specific RIRs
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress indicator."""
    try:
        print(f"Downloading from {url}...")
        with urllib.request.urlopen(url, timeout=60) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 / total_size
                        print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="")

            print()  # newline after progress
            return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract zip or tar archive."""
    import zipfile

    print(f"Extracting {archive_path.name}...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
    print(f"  Extracted to {output_dir}")


def create_t60_metadata(rir_dir: Path) -> dict:
    """
    Create T60 metadata for RIRs based on directory structure.

    OpenSLR SLR28 structure:
      RIRS_NOISES/
        simulated_rirs/
          smallroom/
          mediumroom/
          largeroom/
        real_rirs_isotropic_noises/
          ...
    """
    metadata = {}
    rir_root = rir_dir / "RIRS_NOISES"

    if not rir_root.exists():
        # Try without RIRS_NOISES subdirectory
        rir_root = rir_dir

    # Simulated RIRs
    sim_dir = rir_root / "simulated_rirs"
    if sim_dir.exists():
        for room_type in ["smallroom", "mediumroom", "largeroom"]:
            room_dir = sim_dir / room_type
            if room_dir.exists():
                t60_info = T60_METADATA.get(room_type, {"T60_typical": 0.5})
                for wav_file in room_dir.rglob("*.wav"):
                    rel_path = str(wav_file.relative_to(rir_root))
                    metadata[rel_path] = {
                        "T60": t60_info["T60_typical"],
                        "room_type": room_type,
                        "type": "simulated",
                    }

    # Real RIRs - assign approximate T60 values
    real_dir = rir_root / "real_rirs_isotropic_noises"
    if real_dir.exists():
        for wav_file in real_dir.glob("*.wav"):
            rel_path = str(wav_file.relative_to(rir_root))
            # Real RIRs typically have medium reverberation
            metadata[rel_path] = {
                "T60": 0.6,  # Approximate for real recordings
                "type": "real",
            }

    return metadata


def select_rirs_by_t60(metadata: dict, target_t60_values: list[float], tolerance: float = 0.2) -> dict:
    """
    Select representative RIRs for each target T60 value.

    Args:
        metadata: Full RIR metadata
        target_t60_values: List of T60 values to select (e.g., [0.3, 1.0, 2.5])
        tolerance: Acceptable deviation from target T60

    Returns:
        Dictionary mapping T60 values to list of RIR paths
    """
    selected = {t60: [] for t60 in target_t60_values}

    for rir_path, info in metadata.items():
        rir_t60 = info.get("T60", 0)
        for target in target_t60_values:
            if abs(rir_t60 - target) <= tolerance:
                selected[target].append(rir_path)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Download OpenSLR SLR28 RIRs")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/rirs"),
        help="Output directory for RIRs",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, only regenerate metadata",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / "rirs_noises.zip"

    # Download
    if not args.skip_download:
        if archive_path.exists():
            print(f"Archive already exists: {archive_path}")
        else:
            success = download_file(RIRS_URL, archive_path)
            if not success:
                print("Trying alternate URL...")
                success = download_file(RIRS_ALT_URL, archive_path)

            if not success:
                print("[ERROR] Failed to download RIRs")
                return 1

        # Extract
        extract_archive(archive_path, output_dir)

    # Create metadata
    print("Creating T60 metadata...")
    metadata = create_t60_metadata(output_dir)

    metadata_path = output_dir / "rir_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    # Select RIRs for our target T60 values
    target_t60s = [0.3, 1.0, 2.5]
    selected = select_rirs_by_t60(metadata, target_t60s, tolerance=0.3)

    print("\nSelected RIRs by T60:")
    for t60, rirs in selected.items():
        print(f"  T60={t60}s: {len(rirs)} RIRs")
        if rirs:
            print(f"    Example: {rirs[0]}")

    # Save selection
    selection_path = output_dir / "rir_selection.json"
    with open(selection_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved selection: {selection_path}")

    print(f"\n[DONE] RIRs ready in {output_dir}")
    print(f"  Total RIRs: {len(metadata)}")

    return 0


if __name__ == "__main__":
    exit(main())
