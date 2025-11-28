# Code Map

This document maps pipeline functionality to scripts and their status.

## Directory Structure (Cleaned)

```
opro2/
├── src/qsm/              # Core library
├── scripts/              # Pipeline scripts
├── slurm/                # Cluster job definitions
├── configs/              # Configuration files
├── data/                 # Dataset structure (gitignored content)
├── prompts/              # Prompt templates
├── results/              # Output directory (empty, gitignored)
├── docs/                 # Documentation (PIPELINE_OVERVIEW.md, CODE_MAP.md)
├── tests/                # Smoke tests
├── legacy/               # Archived code and results
├── *.sh                  # Shell wrappers for cluster jobs
├── config.yaml           # Global configuration
├── pyproject.toml        # Python project config
└── requirements.txt      # Dependencies
```

## Core Scripts

| Functionality | Main Script | Shell Wrapper | Slurm Job |
|---------------|------------|---------------|-----------|
| Model download | `download_qwen_model.py` | - | - |
| Fine-tuning (LoRA) | `scripts/finetune_qwen_audio.py` | `train_single_seed.sh` | - |
| Fine-tuning (augmented) | `scripts/finetune_qwen_audio.py` | `train_augmented.sh` | - |
| OPRO base model | `scripts/opro_classic_optimize.py` | `run_opro_base.sh` | `slurm/opro_classic_base.job` |
| OPRO post-LoRA | `scripts/opro_post_ft_v2.py` | `run_opro_finetuned.sh` | `slurm/opro_classic_lora.job` |
| Evaluation | `scripts/evaluate_with_generation.py` | `eval_model.sh` | `slurm/eval_test.job` |
| Threshold sweep | (inline in job) | - | `slurm/eval_threshold_sweep.job` |
| Prompt testing | `scripts/test_prompts_quick.py` | `test_prompts.sh` | - |

## Source Modules

| Module | Purpose |
|--------|---------|
| `src/qsm/models/qwen_audio.py` | Qwen2AudioClassifier with constrained decoding |
| `src/qsm/utils/normalize.py` | Response normalization to binary labels |
| `src/qsm/audio/slicing.py` | Audio segment extraction, padding |
| `src/qsm/audio/noise.py` | Noise augmentation (SNR mixing) |
| `src/qsm/audio/reverb.py` | Reverb augmentation (RIR convolution) |
| `src/qsm/audio/filters.py` | Bandpass filter augmentation |
| `src/qsm/data/loaders.py` | Dataset loading utilities |
| `src/qsm/vad/silero.py` | Silero VAD baseline |

## Data Preparation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_base_clips.py` | Extract base audio clips |
| `scripts/generate_experimental_variants.py` | Apply augmentations (SNR, reverb, filter) |
| `scripts/create_group_stratified_split.py` | Create train/dev/test splits |
| `scripts/download_voxconverse_audio.py` | Download VoxConverse dataset |
| `scripts/clean_esc50_dataset.py` | Clean ESC-50 dataset |

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compute_psychometric_curves.py` | Duration/SNR vs. accuracy curves |
| `scripts/compute_roc_pr_curves.py` | ROC and PR curve generation |
| `scripts/compare_models_statistical.py` | Statistical comparison (McNemar, bootstrap) |
| `scripts/generate_final_plots.py` | Publication-ready figures |
| `scripts/analyze_per_condition.py` | Per-condition breakdown analysis |
| `scripts/baseline_silero_vad.py` | Run Silero VAD baseline |

## Legacy/Archived (in `legacy/`)

All historical code, results, and documentation has been moved to `legacy/`:

| Location | Contents |
|----------|----------|
| `legacy/scripts/` | Deprecated script versions |
| `legacy/slurm/` | Old job definitions |
| `legacy/results/` | Historical results |
| `legacy/docs/` | Sprint reports, old guides |
| `legacy/notes/` | Historical markdown notes |
| `legacy/root/` | Old shell scripts |
| `legacy/prompts/` | Experimental prompt templates |
| `legacy/assets/` | Old figures and spectrograms |

---

*Last updated: Deep cleanup pass on cleanup/core-refactor branch*
