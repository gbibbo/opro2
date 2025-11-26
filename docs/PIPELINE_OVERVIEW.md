# Pipeline Overview

This document describes the complete OPRO pipeline for Qwen2-Audio speech detection optimization.

## Pipeline Blocks

### A: Infrastructure / Environment

- **Cluster**: QMUL Apptainer-based environment with A100 GPUs
- **Container**: `apptainer_qwen.def` defines the environment
- **Base repo**: `/mnt/fast/nobackup/users/gb0048/opro2`
- **Local dev**: Windows with VSCode (for code edits, no GPU inference)

### B: Dataset + Preprocessing

- **Source**: VoxConverse + ESC-50 (speech vs. non-speech)
- **Windowing**: 2-second segments centered on speech onset
- **Padding**: Low-amplitude noise padding for short clips
- **Augmentations**: Duration, SNR, Reverb, Bandpass filter variations
- **Splits**: train/dev/test (stratified by group to prevent leakage)

### C: Base Evaluation

- **Model**: Qwen2-Audio-7B-Instruct (base, no fine-tuning)
- **Metrics**: BA_clip (balanced accuracy), BA_conditions (per-condition BA)
- **Script**: `scripts/evaluate_with_logits.py` or inline in slurm jobs
- **Output**: CSV predictions + JSON metrics

### D: Unified Metric / Reward

- **Primary metric**: Balanced accuracy (BA_clip)
- **Secondary**: Per-condition BA breakdown (duration, SNR, etc.)
- **Reward formula**: `BA_clip` (no prompt length penalty)
- **Note**: Psychoacoustic conditions weighted equally

### E: OPRO on Base Model

- **Optimizer**: `scripts/opro_classic_optimize.py`
- **LLM**: Local Qwen2 model for prompt generation
- **Objective**: Maximize BA_clip on dev split
- **Iterations**: Configurable (e.g., 20-50 rounds)
- **Output**: Best prompt text saved to JSON

### F: Fine-tuning (LoRA)

- **Script**: `scripts/finetune_qwen_audio.py`
- **Method**: LoRA adapters on Qwen2-Audio
- **Training data**: Augmented train split
- **Shell wrapper**: `train_single_seed.sh`, `train_augmented.sh`
- **Checkpoints**: Saved under `checkpoints/` (gitignored)

### G: OPRO on Fine-tuned Model

- **Optimizer**: `scripts/opro_post_ft_v2.py`
- **Same approach as E, but uses LoRA-adapted model
- **Goal**: Re-optimize prompt for fine-tuned weights
- **Output**: Best prompt for finetuned model

### H: Threshold Optimization

- **Script**: slurm jobs with threshold sweep
- **Method**: Sweep p_first_token threshold for optimal BA
- **Output**: Optimal threshold value (e.g., 0.50)

### I: Analysis + Plots

- **Psychometric curves**: Duration vs. accuracy, SNR vs. accuracy
- **ROC/PR curves**: `scripts/compute_roc_pr_curves.py`
- **Statistical comparison**: McNemar test, bootstrap CI
- **Figures**: Saved to `results/figures/`

## Directory Structure

```
opro2/
├── src/qsm/           # Core library (models, utils, data loaders)
├── scripts/           # Main scripts (OPRO, evaluation, analysis)
├── slurm/             # Cluster job definitions
├── data/              # Dataset manifests (raw data gitignored)
├── results/           # Evaluation outputs (large files gitignored)
├── configs/           # YAML configurations
├── docs/              # Documentation
└── legacy/            # Archived/deprecated code
```

## Key Interfaces

### Qwen2AudioClassifier (`src/qsm/models/qwen_audio.py`)

```python
classifier = Qwen2AudioClassifier(
    model_name="Qwen/Qwen2-Audio-7B-Instruct",
    device="cuda",
    constrained_decoding=True,  # A/B format
)
classifier.set_prompt(system_prompt="...", user_prompt="...")
result = classifier.predict(audio_path)
# result.label: "SPEECH" | "NONSPEECH" | "UNKNOWN"
# result.confidence: float
# result.probs: {"A": float, "B": float, "p_first_token": float}
```

### Response Normalization (`src/qsm/utils/normalize.py`)

```python
from qsm.utils.normalize import normalize_to_binary

label, confidence = normalize_to_binary(
    text="A",
    mapping={"A": "SPEECH", "B": "NONSPEECH"},
    probs={"A": 0.8, "B": 0.2, "p_first_token": 0.8}
)
# Returns: ("SPEECH", 0.8)
```

## Execution Flow

1. **Setup**: Download model (`download_qwen_model.py`)
2. **Data prep**: Create splits, apply augmentations
3. **Baseline**: Evaluate base model → `results/eval_base_*/`
4. **OPRO base**: Optimize prompt on dev → best prompt JSON
5. **Fine-tune**: Train LoRA adapters on train split
6. **OPRO post-FT**: Re-optimize prompt for LoRA model
7. **Threshold**: Sweep threshold for optimal decision boundary
8. **Final eval**: Test set evaluation with best prompt + threshold
9. **Analysis**: Generate plots, statistical tests, paper figures
