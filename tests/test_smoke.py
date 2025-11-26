#!/usr/bin/env python3
"""
Smoke test for opro2 core modules.

Run without GPU: python tests/test_smoke.py
Only tests imports and basic functionality - does not load models.

Note: This test can run in minimal environments. If audio libraries
(librosa, soundfile, etc.) are not installed, those tests are skipped.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")

    # Normalization utilities (no heavy dependencies)
    from src.qsm.utils.normalize import normalize_to_binary, detect_format
    print("  - normalize: OK")

    # Try to import audio modules (may fail without librosa/soundfile)
    try:
        from src.qsm.models.qwen_audio import Qwen2AudioClassifier, PredictionResult
        print("  - qwen_audio: OK")
    except ImportError as e:
        print(f"  - qwen_audio: SKIPPED (missing dependency: {e.name})")

    try:
        from src.qsm.audio.slicing import extract_centered_segment
        from src.qsm.audio.noise import mix_with_noise
        from src.qsm.audio.filters import bandpass_filter
        from src.qsm.audio.reverb import apply_reverb
        print("  - audio modules: OK")
    except ImportError as e:
        print(f"  - audio modules: SKIPPED (missing dependency: {e.name})")

    try:
        from src.qsm.data.loaders import load_manifest
        print("  - data loaders: OK")
    except ImportError as e:
        print(f"  - data loaders: SKIPPED (missing dependency: {e.name})")

    print("Import tests completed!")


def test_normalize():
    """Test response normalization."""
    print("\nTesting normalize_to_binary...")
    from src.qsm.utils.normalize import normalize_to_binary, detect_format

    # Test A/B mapping
    label, conf = normalize_to_binary("A", mapping={"A": "SPEECH", "B": "NONSPEECH"})
    assert label == "SPEECH", f"Expected SPEECH, got {label}"
    print("  - A -> SPEECH: OK")

    # Test B -> NONSPEECH
    label, conf = normalize_to_binary("B", mapping={"A": "SPEECH", "B": "NONSPEECH"})
    assert label == "NONSPEECH", f"Expected NONSPEECH, got {label}"
    print("  - B -> NONSPEECH: OK")

    # Test format detection
    fmt = detect_format("Choose A or B:\nA) Speech\nB) Non-speech")
    assert fmt == "ab", f"Expected ab, got {fmt}"
    print("  - Format detection: OK")

    print("Normalization tests passed!")


def test_prediction_result():
    """Test PredictionResult dataclass."""
    print("\nTesting PredictionResult...")
    try:
        from src.qsm.models.qwen_audio import PredictionResult

        result = PredictionResult(
            label="SPEECH",
            confidence=0.95,
            raw_output="A",
            latency_ms=123.4,
            probs={"A": 0.95, "B": 0.05},
            text="A"
        )

        assert result.label == "SPEECH"
        assert result.confidence == 0.95
        assert result.probs["A"] == 0.95
        print("  - PredictionResult creation: OK")
        print("PredictionResult tests passed!")
    except ImportError as e:
        print(f"  - SKIPPED (missing dependency: {e.name})")


if __name__ == "__main__":
    print("=" * 50)
    print("OPRO2 Smoke Test (no GPU required)")
    print("=" * 50)

    try:
        test_imports()
        test_normalize()
        test_prediction_result()
        print("\n" + "=" * 50)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
