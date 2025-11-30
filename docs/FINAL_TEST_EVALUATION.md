# Final Test Set Evaluation Results

**Date**: 2024-11-30
**Test Set**: 21,340 samples (970 base clips × 22 conditions)
**Evaluation Script**: `scripts/evaluate_simple.py`

---

## Summary Results

| Configuration | BA_clip | BA_conditions | Speech Acc | NonSpeech Acc |
|---------------|---------|---------------|------------|---------------|
| BASE + Baseline | 77.68% | 76.55% | 65.90% | 89.45% |
| BASE + OPRO | **86.91%** | **87.49%** | 90.11% | 83.70% |
| LoRA + Baseline | *pending* | *pending* | *pending* | *pending* |
| LoRA + OPRO | **93.66%** | **93.93%** | 95.73% | 91.59% |

### Key Improvements:
- **OPRO on BASE**: +9.23 pp (77.68% → 86.91%) - prompt optimization alone
- **LoRA + OPRO vs BASE + Baseline**: +15.98 pp (77.68% → 93.66%)

---

## Methodology

### Dataset Structure

```
data/processed/variants_validated_1000/
├── dev_metadata.csv    # 660 samples (30 base clips × 22 conditions)
└── test_metadata.csv   # 21,340 samples (970 base clips × 22 conditions)
```

**Base clips**: 1,000 validated clips from Common Voice (≥80% speech via Silero VAD)
- Dev: 30 clips for OPRO optimization
- Test: 970 clips for final evaluation

### 22 Psychoacoustic Conditions

| Dimension | Conditions | Count |
|-----------|------------|-------|
| **Duration** | 20ms, 40ms, 60ms, 80ms, 100ms, 200ms, 500ms, 1000ms | 8 |
| **SNR** | -10dB, -5dB, 0dB, 5dB, 10dB, 20dB | 6 |
| **Reverb** | none, 0.3s RT60, 1.0s RT60, 2.5s RT60 | 4 |
| **Filter** | none, bandpass, lowpass, highpass | 4 |

**Total**: 8 + 6 + 4 + 4 = 22 independent conditions

### Prompts

**Baseline prompt**:
```
Does this audio contain human speech?
Reply with ONLY one word: SPEECH or NON-SPEECH.
```

**OPRO-optimized prompts** (obtained via 30-iteration optimization on dev set):

- **BASE model**: "Listen briefly; is this clip human speech or not? Reply: SPEECH or NON-SPEECH."
- **LoRA model**: "Pay attention to this clip, is it human speech? Just answer: SPEECH or NON-SPEECH."

### OPRO Configuration

- **Iterations**: 30
- **Candidates per iteration**: 3
- **Early stopping**: 5 iterations without improvement
- **Evaluation samples**: 500 stratified (from 660 dev samples)
- **Optimizer LLM**: Qwen2.5-7B-Instruct

### LoRA Fine-tuning

- **Checkpoint**: `checkpoints/qwen_lora_large_seed42/final`
- **Target modules**: Attention + MLP layers
- **Training data**: 5,000 samples (separate from dev/test)
- **Seed**: 42

---

## Detailed Results by Condition

### BASE + Baseline (77.68% BA_clip)

| Dimension | BA | Best Condition | Worst Condition |
|-----------|-----|----------------|-----------------|
| Duration | 79.87% | dur_100ms (88.0%) | dur_20ms (65.5%) |
| SNR | 82.27% | snr_-10dB (88.0%) | snr_20dB (74.8%) |
| Reverb | 72.96% | reverb_1.0s (74.6%) | reverb_2.5s (71.2%) |
| Filter | 71.11% | filter_highpass (73.8%) | filter_lowpass (66.7%) |

### BASE + OPRO (86.91% BA_clip)

| Dimension | BA | Best Condition | Worst Condition |
|-----------|-----|----------------|-----------------|
| Duration | 85.03% | dur_200ms (90.2%) | dur_20ms (71.6%) |
| SNR | 86.03% | snr_20dB (88.0%) | snr_-10dB (80.2%) |
| Reverb | 89.48% | reverb_1.0s (90.2%) | reverb_none (88.9%) |
| Filter | 89.41% | filter_lowpass (90.1%) | filter_none (89.2%) |

### LoRA + OPRO (93.66% BA_clip)

| Dimension | BA | Best Condition | Worst Condition |
|-----------|-----|----------------|-----------------|
| Duration | 90.76% | dur_500ms (98.1%) | dur_20ms (83.1%) |
| SNR | 97.30% | snr_20dB (98.8%) | snr_-10dB (94.6%) |
| Reverb | 93.79% | reverb_1.0s (94.4%) | reverb_0.3s (93.1%) |
| Filter | 93.87% | filter_bandpass (94.2%) | filter_lowpass (93.6%) |

---

## Reproduction Instructions

### 1. Generate Test Variants

```bash
# Already done - variants in data/processed/variants_validated_1000/
python scripts/generate_variants.py \
    --input_csv data/processed/validated_1000/metadata.csv \
    --output_dir data/processed/variants_validated_1000 \
    --split test
```

### 2. Run Evaluation Jobs

```bash
# BASE + Baseline
sbatch slurm/eval_test_base_baseline.job

# BASE + OPRO
sbatch slurm/eval_test_base.job

# LoRA + Baseline
sbatch slurm/eval_test_lora_baseline.job

# LoRA + OPRO
sbatch slurm/eval_test_lora.job
```

### 3. Evaluation Script

```bash
python scripts/evaluate_simple.py \
    --manifest data/processed/variants_validated_1000/test_metadata.csv \
    --prompt "YOUR_PROMPT_HERE" \
    --output_dir results/your_output_dir \
    --checkpoint path/to/lora/checkpoint  # optional, for LoRA
    --batch_size 50
```

---

## Files Generated

```
results/
├── eval_test_base_baseline/
│   ├── metrics.json          # Summary metrics
│   └── predictions.csv       # Per-sample predictions
├── eval_test_base_opro/
│   ├── metrics.json
│   └── predictions.csv
├── eval_test_lora_baseline/  # (pending)
│   ├── metrics.json
│   └── predictions.csv
└── eval_test_lora_opro/
    ├── metrics.json
    └── predictions.csv
```

---

## Metrics Explanation

- **BA_clip**: Balanced accuracy where each sample has equal weight
- **BA_conditions**: Average balanced accuracy across 22 conditions (each condition has equal weight)
- **Speech Acc**: Accuracy on speech samples (true positives)
- **NonSpeech Acc**: Accuracy on non-speech samples (true negatives)

---

## Hardware

- **GPU**: NVIDIA RTX 3090 (24GB)
- **Partition**: 3090
- **Time per evaluation**: ~3 hours for 21,340 samples

---

## Contact

For questions about reproduction, contact the repository maintainer.
