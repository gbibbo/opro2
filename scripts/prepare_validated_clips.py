#!/usr/bin/env python3
"""
Prepare Validated Base Clips with Silero VAD

Extracts 1000ms clips from VoxConverse (SPEECH) and ESC-50 (NONSPEECH).
Each SPEECH clip is validated with Silero VAD to ensure >= 80% speech content.

Usage:
    python scripts/prepare_validated_clips.py \
        --voxconverse_dir data/raw/voxconverse/audio/dev \
        --esc50_dir data/raw/esc50/audio \
        --output_dir data/processed/base_validated \
        --n_speech 500 \
        --n_nonspeech 500 \
        --dev_ratio 0.03 \
        --seed 42
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

DURATION_MS = 1000
TARGET_SR = 16000
MIN_SPEECH_RATIO = 0.80  # Minimum 80% speech in clip
MAX_ATTEMPTS_PER_FILE = 10  # Max attempts to find valid clip per file


# =============================================================================
# Silero VAD
# =============================================================================

def load_silero_vad():
    """Load Silero VAD model."""
    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, _, _, _ = utils
    print("  Silero VAD loaded!")
    return model, get_speech_timestamps


def get_speech_ratio(audio: np.ndarray, sr: int, model, get_speech_timestamps) -> float:
    """
    Calculate speech ratio in audio clip using Silero VAD.

    Returns:
        Speech ratio (0.0 to 1.0)
    """
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)

    # Calculate total speech duration
    total_speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
    speech_ratio = total_speech_samples / len(audio) if len(audio) > 0 else 0

    return speech_ratio


# =============================================================================
# Audio Processing
# =============================================================================

def normalize_audio_peak(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio by peak level."""
    current_peak = np.abs(audio).max()
    if current_peak < 1e-6:
        return audio
    gain = target_peak / current_peak
    return np.clip(audio * gain, -1.0, 1.0)


def extract_random_clip(audio: np.ndarray, sr: int, duration_ms: int) -> np.ndarray:
    """Extract a random clip of specified duration."""
    samples_needed = int(duration_ms * sr / 1000)

    if len(audio) <= samples_needed:
        # Pad if too short
        pad_length = samples_needed - len(audio)
        return np.pad(audio, (0, pad_length), mode='constant')

    # Random start position
    max_start = len(audio) - samples_needed
    start_idx = np.random.randint(0, max_start + 1)
    return audio[start_idx:start_idx + samples_needed]


# =============================================================================
# VoxConverse Processing (SPEECH)
# =============================================================================

def extract_validated_speech_clips(
    voxconverse_dir: Path,
    n_clips: int,
    model,
    get_speech_timestamps,
    seed: int
) -> list[dict]:
    """
    Extract validated speech clips from VoxConverse.
    Each clip is validated with Silero VAD (>= 80% speech).
    """
    np.random.seed(seed)

    # Find all VoxConverse files
    audio_files = list(voxconverse_dir.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No audio files found in {voxconverse_dir}")

    print(f"\nExtracting {n_clips} validated SPEECH clips from {len(audio_files)} VoxConverse files...")
    print(f"  Validation: >= {MIN_SPEECH_RATIO*100:.0f}% speech content (Silero VAD)")

    clips = []
    files_used = set()
    rejected_count = 0

    # Shuffle files for random selection
    np.random.shuffle(audio_files)
    file_idx = 0

    pbar = tqdm(total=n_clips, desc="Extracting SPEECH clips")

    while len(clips) < n_clips and file_idx < len(audio_files) * MAX_ATTEMPTS_PER_FILE:
        # Cycle through files
        audio_file = audio_files[file_idx % len(audio_files)]
        file_idx += 1

        try:
            # Load full audio
            audio, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)

            if len(audio) < TARGET_SR:  # Less than 1 second
                continue

            # Try to extract a valid clip
            for attempt in range(MAX_ATTEMPTS_PER_FILE):
                clip = extract_random_clip(audio, sr, DURATION_MS)

                # Validate with Silero VAD
                speech_ratio = get_speech_ratio(clip, sr, model, get_speech_timestamps)

                if speech_ratio >= MIN_SPEECH_RATIO:
                    # Valid clip!
                    clip_normalized = normalize_audio_peak(clip)

                    # Create unique ID
                    speaker_id = audio_file.stem
                    clip_id = f"voxconverse_{speaker_id}_{len(clips):04d}_{DURATION_MS}ms"

                    clips.append({
                        'clip_id': clip_id,
                        'audio': clip_normalized,
                        'ground_truth': 'SPEECH',
                        'dataset': 'voxconverse',
                        'group_id': speaker_id,  # Group by speaker for split
                        'source_file': audio_file.name,
                        'speech_ratio': speech_ratio,
                        'sr': sr
                    })

                    files_used.add(audio_file.name)
                    pbar.update(1)
                    break
                else:
                    rejected_count += 1

        except Exception as e:
            print(f"\n  Error processing {audio_file.name}: {e}")
            continue

    pbar.close()

    print(f"\n  Extracted: {len(clips)} valid clips")
    print(f"  Rejected: {rejected_count} clips (< {MIN_SPEECH_RATIO*100:.0f}% speech)")
    print(f"  Files used: {len(files_used)}")
    print(f"  Mean speech ratio: {np.mean([c['speech_ratio'] for c in clips])*100:.1f}%")

    if len(clips) < n_clips:
        print(f"\n  WARNING: Could only extract {len(clips)}/{n_clips} clips!")

    return clips


# =============================================================================
# ESC-50 Processing (NONSPEECH)
# =============================================================================

def extract_nonspeech_clips(
    esc50_dir: Path,
    n_clips: int,
    seed: int
) -> list[dict]:
    """Extract nonspeech clips from ESC-50."""
    np.random.seed(seed + 1)

    # Find all ESC-50 files
    audio_files = list(esc50_dir.glob("*.wav"))
    if not audio_files:
        raise ValueError(f"No audio files found in {esc50_dir}")

    print(f"\nExtracting {n_clips} NONSPEECH clips from {len(audio_files)} ESC-50 files...")

    # Sample files
    if len(audio_files) < n_clips:
        print(f"  Warning: Only {len(audio_files)} files available")
        n_clips = len(audio_files)

    selected_files = np.random.choice(audio_files, size=n_clips, replace=False)

    clips = []
    for idx, audio_file in enumerate(tqdm(selected_files, desc="Extracting NONSPEECH clips")):
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=TARGET_SR, mono=True)

            # Extract clip (ESC-50 files are 5 seconds, extract random 1s)
            clip = extract_random_clip(audio, sr, DURATION_MS)
            clip_normalized = normalize_audio_peak(clip)

            # ESC-50 filename format: fold-class-clip.wav
            clip_identifier = audio_file.stem
            clip_id = f"esc50_{clip_identifier}_{idx:04d}_{DURATION_MS}ms"

            clips.append({
                'clip_id': clip_id,
                'audio': clip_normalized,
                'ground_truth': 'NONSPEECH',
                'dataset': 'esc50',
                'group_id': clip_identifier,
                'source_file': audio_file.name,
                'speech_ratio': 0.0,
                'sr': sr
            })

        except Exception as e:
            print(f"\n  Error processing {audio_file.name}: {e}")
            continue

    print(f"\n  Extracted: {len(clips)} clips")

    return clips


# =============================================================================
# Dataset Splitting
# =============================================================================

def split_clips(clips: list[dict], dev_ratio: float, seed: int) -> tuple[list, list]:
    """
    Split clips into dev and test sets.
    Uses group-based splitting to avoid leakage.
    """
    np.random.seed(seed)

    # Get unique groups
    groups = list(set(c['group_id'] for c in clips))
    np.random.shuffle(groups)

    # Calculate split
    n_dev_groups = max(1, int(len(groups) * dev_ratio))
    dev_groups = set(groups[:n_dev_groups])

    dev_clips = [c for c in clips if c['group_id'] in dev_groups]
    test_clips = [c for c in clips if c['group_id'] not in dev_groups]

    return dev_clips, test_clips


# =============================================================================
# Save Functions
# =============================================================================

def save_clips(clips: list[dict], output_dir: Path, split_name: str):
    """Save clips to disk and create metadata CSV."""
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for clip_data in tqdm(clips, desc=f"Saving {split_name} clips"):
        # Save audio
        output_path = audio_dir / f"{clip_data['clip_id']}.wav"
        sf.write(output_path, clip_data['audio'], clip_data['sr'])

        # Compute RMS
        rms = np.sqrt(np.mean(clip_data['audio'] ** 2))

        metadata.append({
            'clip_id': clip_data['clip_id'],
            'audio_path': str(output_path),
            'ground_truth': clip_data['ground_truth'],
            'dataset': clip_data['dataset'],
            'group_id': clip_data['group_id'],
            'source_file': clip_data['source_file'],
            'speech_ratio': clip_data['speech_ratio'],
            'duration_ms': DURATION_MS,
            'sr': clip_data['sr'],
            'rms': rms
        })

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / f"{split_name}_base.csv"
    metadata_df.to_csv(metadata_path, index=False)

    # Print summary
    speech_count = sum(1 for m in metadata if m['ground_truth'] == 'SPEECH')
    nonspeech_count = len(metadata) - speech_count

    print(f"\n  Saved {len(metadata)} {split_name} clips")
    print(f"    SPEECH: {speech_count}")
    print(f"    NONSPEECH: {nonspeech_count}")
    print(f"    Metadata: {metadata_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare validated base clips")
    parser.add_argument("--voxconverse_dir", type=Path, required=True)
    parser.add_argument("--esc50_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_speech", type=int, default=500, help="Number of speech clips")
    parser.add_argument("--n_nonspeech", type=int, default=500, help="Number of nonspeech clips")
    parser.add_argument("--dev_ratio", type=float, default=0.03, help="Ratio for dev set (default: 3%)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 70)
    print("PREPARE VALIDATED BASE CLIPS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  VoxConverse: {args.voxconverse_dir}")
    print(f"  ESC-50: {args.esc50_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Speech clips: {args.n_speech}")
    print(f"  NonSpeech clips: {args.n_nonspeech}")
    print(f"  Dev ratio: {args.dev_ratio*100:.0f}%")
    print(f"  Duration: {DURATION_MS}ms")
    print(f"  Min speech ratio: {MIN_SPEECH_RATIO*100:.0f}%")
    print(f"  Seed: {args.seed}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Silero VAD
    model, get_speech_timestamps = load_silero_vad()

    # Extract SPEECH clips (validated)
    print("\n" + "=" * 70)
    print("EXTRACTING SPEECH CLIPS (VoxConverse + Silero VAD validation)")
    print("=" * 70)
    speech_clips = extract_validated_speech_clips(
        args.voxconverse_dir,
        args.n_speech,
        model,
        get_speech_timestamps,
        args.seed
    )

    # Extract NONSPEECH clips
    print("\n" + "=" * 70)
    print("EXTRACTING NONSPEECH CLIPS (ESC-50)")
    print("=" * 70)
    nonspeech_clips = extract_nonspeech_clips(
        args.esc50_dir,
        args.n_nonspeech,
        args.seed
    )

    # Split into dev and test
    print("\n" + "=" * 70)
    print("SPLITTING INTO DEV AND TEST")
    print("=" * 70)

    speech_dev, speech_test = split_clips(speech_clips, args.dev_ratio, args.seed)
    nonspeech_dev, nonspeech_test = split_clips(nonspeech_clips, args.dev_ratio, args.seed)

    dev_clips = speech_dev + nonspeech_dev
    test_clips = speech_test + nonspeech_test

    # Shuffle
    np.random.seed(args.seed)
    np.random.shuffle(dev_clips)
    np.random.shuffle(test_clips)

    print(f"\n  Dev set: {len(dev_clips)} clips ({len(speech_dev)} speech + {len(nonspeech_dev)} nonspeech)")
    print(f"  Test set: {len(test_clips)} clips ({len(speech_test)} speech + {len(nonspeech_test)} nonspeech)")

    # Save
    print("\n" + "=" * 70)
    print("SAVING CLIPS")
    print("=" * 70)

    save_clips(dev_clips, args.output_dir, "dev")
    save_clips(test_clips, args.output_dir, "test")

    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {args.output_dir}")
    print(f"  dev_base.csv: {len(dev_clips)} clips")
    print(f"  test_base.csv: {len(test_clips)} clips")
    print(f"\nAll SPEECH clips validated with Silero VAD (>= {MIN_SPEECH_RATIO*100:.0f}% speech)")


if __name__ == "__main__":
    main()
