#!/usr/bin/env python3
"""
Quick verification of VoxConverse files:
1. Check durations (minimum 3 seconds)
2. Run Silero VAD to verify speech content
"""

import os
import sys
from pathlib import Path
import numpy as np

# Try to import required libraries
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch/torchaudio not available")

def get_duration(audio_path):
    """Get audio duration in seconds."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate

def analyze_with_silero(audio_path, model, get_speech_timestamps):
    """Analyze audio with Silero VAD."""
    wav, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.squeeze()

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)

    # Calculate speech duration
    total_speech_ms = sum(ts['end'] - ts['start'] for ts in speech_timestamps) / sr * 1000
    total_duration_ms = len(wav) / sr * 1000
    speech_ratio = total_speech_ms / total_duration_ms if total_duration_ms > 0 else 0

    # Find longest silence
    silences = []
    prev_end = 0
    for ts in speech_timestamps:
        silence_start = prev_end
        silence_end = ts['start']
        if silence_end > silence_start:
            silences.append((silence_end - silence_start) / sr * 1000)
        prev_end = ts['end']
    # Final silence
    if prev_end < len(wav):
        silences.append((len(wav) - prev_end) / sr * 1000)

    max_silence_ms = max(silences) if silences else 0

    return {
        'duration_ms': total_duration_ms,
        'speech_ms': total_speech_ms,
        'speech_ratio': speech_ratio,
        'n_segments': len(speech_timestamps),
        'max_silence_ms': max_silence_ms
    }

def main():
    vox_dir = Path("/mnt/fast/nobackup/users/gb0048/opro2/data/raw/voxconverse/audio/dev")

    if not vox_dir.exists():
        print(f"ERROR: Directory not found: {vox_dir}")
        sys.exit(1)

    # Find all wav files
    wav_files = list(vox_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} VoxConverse files")

    if len(wav_files) == 0:
        print("No files found!")
        sys.exit(1)

    # Load Silero VAD
    print("\nLoading Silero VAD...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, _, _, _ = utils
    print("Silero VAD loaded!")

    # Analyze files
    print(f"\nAnalyzing {min(20, len(wav_files))} files (sample)...\n")

    results = []
    short_files = []
    low_speech_files = []
    long_silence_files = []

    # Sample up to 20 files for quick check
    sample_files = wav_files[:20]

    for wav_file in sample_files:
        try:
            duration = get_duration(wav_file)

            # Check minimum duration
            if duration < 3.0:
                short_files.append((wav_file.name, duration))

            # Run Silero VAD
            vad_result = analyze_with_silero(wav_file, model, get_speech_timestamps)

            results.append({
                'file': wav_file.name,
                'duration_s': duration,
                **vad_result
            })

            # Check speech ratio
            if vad_result['speech_ratio'] < 0.3:
                low_speech_files.append((wav_file.name, vad_result['speech_ratio']))

            # Check max silence
            if vad_result['max_silence_ms'] > 2000:
                long_silence_files.append((wav_file.name, vad_result['max_silence_ms']))

            print(f"  {wav_file.name}: {duration:.1f}s, speech={vad_result['speech_ratio']*100:.0f}%, "
                  f"segments={vad_result['n_segments']}, max_silence={vad_result['max_silence_ms']:.0f}ms")

        except Exception as e:
            print(f"  ERROR {wav_file.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    durations = [r['duration_s'] for r in results]
    speech_ratios = [r['speech_ratio'] for r in results]

    print(f"\nDuration stats (n={len(durations)}):")
    print(f"  Min: {min(durations):.1f}s")
    print(f"  Max: {max(durations):.1f}s")
    print(f"  Mean: {np.mean(durations):.1f}s")

    print(f"\nSpeech ratio stats:")
    print(f"  Min: {min(speech_ratios)*100:.0f}%")
    print(f"  Max: {max(speech_ratios)*100:.0f}%")
    print(f"  Mean: {np.mean(speech_ratios)*100:.0f}%")

    if short_files:
        print(f"\n⚠️  Files < 3s ({len(short_files)}):")
        for name, dur in short_files[:5]:
            print(f"    {name}: {dur:.1f}s")
    else:
        print(f"\n✓ All files >= 3s")

    if low_speech_files:
        print(f"\n⚠️  Files with < 30% speech ({len(low_speech_files)}):")
        for name, ratio in low_speech_files[:5]:
            print(f"    {name}: {ratio*100:.0f}%")
    else:
        print(f"\n✓ All files have >= 30% speech")

    if long_silence_files:
        print(f"\n⚠️  Files with silence > 2s ({len(long_silence_files)}):")
        for name, ms in long_silence_files[:5]:
            print(f"    {name}: {ms:.0f}ms")
    else:
        print(f"\n✓ No files with silence > 2s")

    # Estimate how many 1s clips we can extract
    total_speech_s = sum(r['speech_ms'] for r in results) / 1000
    estimated_clips_per_file = total_speech_s / len(results)
    print(f"\nEstimated 1s speech clips per file: {estimated_clips_per_file:.1f}")
    print(f"With {len(wav_files)} files: ~{int(estimated_clips_per_file * len(wav_files))} total clips possible")

if __name__ == "__main__":
    main()
