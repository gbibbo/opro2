#!/usr/bin/env python3
"""
Generate Experimental Variants (4 Independent Dimensions)

Takes base 1000ms clips and generates variants along 4 INDEPENDENT dimensions:
1. Duration: 20, 40, 60, 80, 100, 200, 500, 1000ms (8 values)
2. SNR: -10, -5, 0, +5, +10, +20 dB (6 values)
3. Reverb (T60): none, 0.3s, 1.0s, 2.5s (4 values)
4. Filter: none, bandpass, lowpass, highpass (4 values)

Total = 22 variants per base clip (NOT 8×6×4×4 = 768 cross-product)

Each dimension is varied independently with "neutral" values for other dimensions:
- Neutral duration: 1000ms
- Neutral SNR: no noise added (just padding noise)
- Neutral reverb: none
- Neutral filter: none

Usage:
    python scripts/generate_experimental_variants.py \
        --input_base data/processed/base_1000ms \
        --output_dir data/processed/experimental_variants \
        --rir_dir data/raw/rirs

References:
- Sprint 5 psychoacoustic conditions
- Independent dimensions allow measuring effect of each factor separately
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import audio processing modules
from src.qsm.audio.filters import apply_bandpass, apply_lowpass, apply_highpass
from src.qsm.audio.reverb import RIRDatabase, apply_rir


# =============================================================================
# Configuration
# =============================================================================

# Default dimension values
DEFAULT_DURATIONS_MS = [20, 40, 60, 80, 100, 200, 500, 1000]
DEFAULT_SNR_LEVELS_DB = [-10, -5, 0, 5, 10, 20]
DEFAULT_T60_VALUES = [0.0, 0.3, 1.0, 2.5]  # 0.0 = no reverb
DEFAULT_FILTER_TYPES = ['none', 'bandpass', 'lowpass', 'highpass']

# Neutral values (used when varying other dimensions)
NEUTRAL_DURATION_MS = 1000
NEUTRAL_SNR_DB = None  # None = no noise added
NEUTRAL_T60 = 0.0  # No reverb
NEUTRAL_FILTER = 'none'

# Padding configuration
CONTAINER_DURATION_MS = 2000
PADDING_NOISE_AMPLITUDE = 0.0001


# =============================================================================
# Audio Processing Functions
# =============================================================================

def trim_audio_to_duration(audio: np.ndarray, target_duration_ms: int, sr: int) -> np.ndarray:
    """Trim audio to target duration (centered extraction)."""
    target_samples = int(target_duration_ms * sr / 1000)
    current_samples = len(audio)

    if target_samples >= current_samples:
        return audio

    start_idx = (current_samples - target_samples) // 2
    end_idx = start_idx + target_samples
    return audio[start_idx:end_idx]


def pad_audio_to_container(audio: np.ndarray, container_duration_ms: int, sr: int,
                          noise_amplitude: float = PADDING_NOISE_AMPLITUDE) -> np.ndarray:
    """Pad audio to container duration with centered placement and low-amplitude noise."""
    container_samples = int(container_duration_ms * sr / 1000)
    audio_samples = len(audio)

    if audio_samples >= container_samples:
        return audio[:container_samples]

    total_padding = container_samples - audio_samples
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    left_noise = np.random.normal(0, noise_amplitude, left_padding).astype(np.float32)
    right_noise = np.random.normal(0, noise_amplitude, right_padding).astype(np.float32)

    return np.concatenate([left_noise, audio, right_noise])


def add_noise_at_snr(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise to audio at specified SNR."""
    signal_rms = np.sqrt(np.mean(audio ** 2))

    if signal_rms < 1e-6:
        return audio

    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = np.random.normal(0, noise_rms, len(audio)).astype(np.float32)
    noisy_audio = audio + noise

    return np.clip(noisy_audio, -1.0, 1.0)


def apply_filter_by_type(audio: np.ndarray, sr: int, filter_type: str) -> np.ndarray:
    """Apply filter based on filter type string."""
    if filter_type == 'none':
        return audio
    elif filter_type == 'bandpass':
        return apply_bandpass(audio, sr, lowcut=300.0, highcut=3400.0)
    elif filter_type == 'lowpass':
        return apply_lowpass(audio, sr, highcut=3400.0)
    elif filter_type == 'highpass':
        return apply_highpass(audio, sr, lowcut=300.0)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


# =============================================================================
# Variant Generation (Independent Dimensions)
# =============================================================================

def generate_duration_variant(audio: np.ndarray, sr: int, duration_ms: int) -> np.ndarray:
    """
    Generate duration variant.

    - Trim to target duration
    - Pad to container with noise
    - NO SNR noise added (neutral)
    - NO reverb (neutral)
    - NO filter (neutral)
    """
    trimmed = trim_audio_to_duration(audio, duration_ms, sr)
    padded = pad_audio_to_container(trimmed, CONTAINER_DURATION_MS, sr)
    return padded


def generate_snr_variant(audio: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    """
    Generate SNR variant.

    - Keep full duration (1000ms = neutral)
    - Pad to container with noise
    - Add noise at target SNR
    - NO reverb (neutral)
    - NO filter (neutral)
    """
    padded = pad_audio_to_container(audio, CONTAINER_DURATION_MS, sr)
    noisy = add_noise_at_snr(padded, snr_db)
    return noisy


def generate_reverb_variant(audio: np.ndarray, sr: int, rir: np.ndarray | None) -> np.ndarray:
    """
    Generate reverb variant.

    - Keep full duration (1000ms = neutral)
    - Apply RIR convolution (if rir is not None)
    - Pad to container with noise
    - NO SNR noise added (neutral)
    - NO filter (neutral)
    """
    if rir is not None:
        reverbed = apply_rir(audio, rir, normalize=True)
    else:
        reverbed = audio

    padded = pad_audio_to_container(reverbed, CONTAINER_DURATION_MS, sr)
    return padded


def generate_filter_variant(audio: np.ndarray, sr: int, filter_type: str) -> np.ndarray:
    """
    Generate filter variant.

    - Keep full duration (1000ms = neutral)
    - Apply filter
    - Pad to container with noise
    - NO SNR noise added (neutral)
    - NO reverb (neutral)
    """
    filtered = apply_filter_by_type(audio, sr, filter_type)
    padded = pad_audio_to_container(filtered, CONTAINER_DURATION_MS, sr)
    return padded


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_variants_for_split(
    base_csv: Path,
    output_dir: Path,
    split_name: str,
    durations_ms: list,
    snr_levels_db: list,
    t60_values: list,
    filter_types: list,
    rir_database: RIRDatabase | None,
    target_sr: int
) -> pd.DataFrame:
    """Generate all variants for a single split."""

    base_df = pd.read_csv(base_csv)

    n_variants = len(durations_ms) + len(snr_levels_db) + len(t60_values) + len(filter_types)

    print(f"\n{split_name.upper()} split:")
    print(f"  Base clips: {len(base_df)}")
    print(f"  Variants per clip: {len(durations_ms)} dur + {len(snr_levels_db)} snr + {len(t60_values)} reverb + {len(filter_types)} filter = {n_variants}")
    print(f"  Total variants: {len(base_df) * n_variants}")

    # Create output audio directory
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load RIRs for each T60 level
    rirs_by_t60 = {}
    if rir_database is not None:
        for t60 in t60_values:
            if t60 > 0:
                # Get RIRs in T60 range (±0.2s tolerance)
                rir_ids = rir_database.get_by_t60(t60 - 0.2, t60 + 0.2)
                if rir_ids:
                    # Use first matching RIR
                    rirs_by_t60[t60] = rir_database.get_rir(rir_ids[0], sr=target_sr)
                else:
                    print(f"  WARNING: No RIR found for T60={t60}s")
                    rirs_by_t60[t60] = None
            else:
                rirs_by_t60[t60] = None  # No reverb
    else:
        for t60 in t60_values:
            rirs_by_t60[t60] = None

    variants = []

    for _, row in tqdm(base_df.iterrows(), total=len(base_df), desc=f"Processing {split_name}"):
        # Load base audio
        base_audio, sr = sf.read(row['audio_path'])

        if sr != target_sr:
            import librosa
            base_audio = librosa.resample(base_audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        base_clip_id = row['clip_id']

        # =====================================================================
        # 1. DURATION VARIANTS (8)
        # =====================================================================
        for duration_ms in durations_ms:
            variant_audio = generate_duration_variant(base_audio, sr, duration_ms)
            variant_id = f"{base_clip_id}_dur{duration_ms}ms"
            output_path = audio_dir / f"{variant_id}.wav"
            sf.write(output_path, variant_audio, sr)

            variants.append({
                'clip_id': base_clip_id,
                'variant_id': variant_id,
                'variant_type': 'duration',
                'duration_ms': duration_ms,
                'snr_db': None,  # No noise added
                'T60': None,  # No reverb
                'band_filter': 'none',
                'audio_path': str(output_path),
                'label': row['ground_truth'],
                'ground_truth': row['ground_truth'],
                'dataset': row['dataset'],
                'group_id': row['group_id'],
                'sr': sr,
                'rms': np.sqrt(np.mean(variant_audio ** 2)),
                'container_duration_ms': CONTAINER_DURATION_MS,
            })

        # =====================================================================
        # 2. SNR VARIANTS (6)
        # =====================================================================
        for snr_db in snr_levels_db:
            variant_audio = generate_snr_variant(base_audio, sr, snr_db)
            variant_id = f"{base_clip_id}_snr{snr_db:+d}dB"
            output_path = audio_dir / f"{variant_id}.wav"
            sf.write(output_path, variant_audio, sr)

            variants.append({
                'clip_id': base_clip_id,
                'variant_id': variant_id,
                'variant_type': 'snr',
                'duration_ms': NEUTRAL_DURATION_MS,  # Neutral
                'snr_db': snr_db,
                'T60': None,  # No reverb
                'band_filter': 'none',
                'audio_path': str(output_path),
                'label': row['ground_truth'],
                'ground_truth': row['ground_truth'],
                'dataset': row['dataset'],
                'group_id': row['group_id'],
                'sr': sr,
                'rms': np.sqrt(np.mean(variant_audio ** 2)),
                'container_duration_ms': CONTAINER_DURATION_MS,
            })

        # =====================================================================
        # 3. REVERB VARIANTS (4)
        # =====================================================================
        for t60 in t60_values:
            rir = rirs_by_t60.get(t60)
            variant_audio = generate_reverb_variant(base_audio, sr, rir)

            t60_str = "none" if t60 == 0 else f"{t60:.1f}s"
            variant_id = f"{base_clip_id}_reverb{t60_str}"
            output_path = audio_dir / f"{variant_id}.wav"
            sf.write(output_path, variant_audio, sr)

            variants.append({
                'clip_id': base_clip_id,
                'variant_id': variant_id,
                'variant_type': 'reverb',
                'duration_ms': NEUTRAL_DURATION_MS,  # Neutral
                'snr_db': None,  # No noise added
                'T60': t60 if t60 > 0 else None,
                'band_filter': 'none',
                'audio_path': str(output_path),
                'label': row['ground_truth'],
                'ground_truth': row['ground_truth'],
                'dataset': row['dataset'],
                'group_id': row['group_id'],
                'sr': sr,
                'rms': np.sqrt(np.mean(variant_audio ** 2)),
                'container_duration_ms': CONTAINER_DURATION_MS,
            })

        # =====================================================================
        # 4. FILTER VARIANTS (4)
        # =====================================================================
        for filter_type in filter_types:
            variant_audio = generate_filter_variant(base_audio, sr, filter_type)
            variant_id = f"{base_clip_id}_filter{filter_type}"
            output_path = audio_dir / f"{variant_id}.wav"
            sf.write(output_path, variant_audio, sr)

            variants.append({
                'clip_id': base_clip_id,
                'variant_id': variant_id,
                'variant_type': 'filter',
                'duration_ms': NEUTRAL_DURATION_MS,  # Neutral
                'snr_db': None,  # No noise added
                'T60': None,  # No reverb
                'band_filter': filter_type,
                'audio_path': str(output_path),
                'label': row['ground_truth'],
                'ground_truth': row['ground_truth'],
                'dataset': row['dataset'],
                'group_id': row['group_id'],
                'sr': sr,
                'rms': np.sqrt(np.mean(variant_audio ** 2)),
                'container_duration_ms': CONTAINER_DURATION_MS,
            })

    # Create DataFrame
    variants_df = pd.DataFrame(variants)

    # Save metadata CSV
    metadata_path = output_dir / f"{split_name}_metadata.csv"
    variants_df.to_csv(metadata_path, index=False)

    print(f"  ✓ Saved {len(variants_df)} variants")
    print(f"    Audio: {audio_dir}")
    print(f"    Metadata: {metadata_path}")

    return variants_df


def main():
    parser = argparse.ArgumentParser(description="Generate experimental variants (4 independent dimensions)")
    parser.add_argument("--input_base", type=Path, required=True,
                       help="Input directory with base clips")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for variants")
    parser.add_argument("--rir_dir", type=Path, default=None,
                       help="Directory with RIR files (OpenSLR SLR28 structure)")
    parser.add_argument("--rir_metadata", type=Path, default=None,
                       help="RIR metadata JSON with T60 annotations")
    parser.add_argument("--durations", type=int, nargs='+',
                       default=DEFAULT_DURATIONS_MS,
                       help="Durations in milliseconds")
    parser.add_argument("--snr_levels", type=int, nargs='+',
                       default=DEFAULT_SNR_LEVELS_DB,
                       help="SNR levels in dB")
    parser.add_argument("--t60_values", type=float, nargs='+',
                       default=DEFAULT_T60_VALUES,
                       help="T60 values in seconds (0 = no reverb)")
    parser.add_argument("--filter_types", type=str, nargs='+',
                       default=DEFAULT_FILTER_TYPES,
                       help="Filter types")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="Target sampling rate")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load RIR database if available
    rir_database = None
    if args.rir_dir and args.rir_dir.exists():
        print(f"Loading RIR database from {args.rir_dir}...")
        rir_database = RIRDatabase(args.rir_dir, args.rir_metadata)
        print(f"  Loaded {len(rir_database.list_all())} RIRs")
    else:
        print("WARNING: No RIR directory specified. Reverb variants will have no effect.")

    n_variants = len(args.durations) + len(args.snr_levels) + len(args.t60_values) + len(args.filter_types)

    print("=" * 80)
    print("GENERATE EXPERIMENTAL VARIANTS (4 INDEPENDENT DIMENSIONS)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input_base}")
    print(f"  Output: {args.output_dir}")
    print(f"  Durations: {args.durations} ms ({len(args.durations)} values)")
    print(f"  SNR levels: {args.snr_levels} dB ({len(args.snr_levels)} values)")
    print(f"  T60 values: {args.t60_values} s ({len(args.t60_values)} values)")
    print(f"  Filter types: {args.filter_types} ({len(args.filter_types)} values)")
    print(f"  Total per clip: {n_variants} variants (independent, NOT cross-product)")
    print(f"  Target SR: {args.target_sr}Hz")

    # Process each split
    for split_name in ['train', 'dev', 'test']:
        base_csv = args.input_base / f"{split_name}_base.csv"

        if not base_csv.exists():
            print(f"\n⚠️  Warning: {base_csv} not found, skipping {split_name}")
            continue

        print(f"\n{'=' * 80}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'=' * 80}")

        variants_df = generate_variants_for_split(
            base_csv,
            args.output_dir,
            split_name,
            args.durations,
            args.snr_levels,
            args.t60_values,
            args.filter_types,
            rir_database,
            args.target_sr
        )

        # Print statistics
        print(f"\n  Statistics for {split_name}:")
        print(f"    Total variants: {len(variants_df)}")
        print(f"    Unique base clips: {variants_df['clip_id'].nunique()}")
        print(f"    By variant_type:")
        for vt, count in variants_df['variant_type'].value_counts().items():
            print(f"      {vt}: {count}")
        print(f"    Ground truth distribution:")
        for gt, count in variants_df['ground_truth'].value_counts().items():
            print(f"      {gt}: {count}")

    print(f"\n{'=' * 80}")
    print(f"✓ EXPERIMENTAL VARIANTS GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  train_metadata.csv: All train variants")
    print(f"  dev_metadata.csv: All dev variants")
    print(f"  test_metadata.csv: All test variants")
    print(f"\n✓ Independent dimensions: {len(args.durations)} dur + {len(args.snr_levels)} snr + {len(args.t60_values)} reverb + {len(args.filter_types)} filter = {n_variants} per clip")


if __name__ == "__main__":
    main()
