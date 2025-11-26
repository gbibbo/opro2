# Code Map

This document maps pipeline functionality to scripts and their status.

## Core Scripts

| Functionality | Main Script | Shell Wrapper | Slurm Job | Status |
|---------------|------------|---------------|-----------|--------|
| Model download | `download_qwen_model.py` | - | - | CORE |
| Fine-tuning (LoRA) | `scripts/finetune_qwen_audio.py` | `train_single_seed.sh` | - | CORE |
| Fine-tuning (augmented) | `scripts/finetune_qwen_audio.py` | `train_augmented.sh` | - | CORE |
| OPRO base model | `scripts/opro_classic_optimize.py` | `run_opro_base.sh` | `slurm/opro_classic_base.job` | CORE |
| OPRO post-LoRA | `scripts/opro_post_ft_v2.py` | `run_opro_finetuned.sh` | `slurm/opro_classic_lora.job` | CORE |
| Evaluation (test) | `scripts/evaluate_with_logits.py` | `eval_model.sh` | `slurm/eval_test.job` | CORE |
| Threshold sweep | (inline in job) | - | `slurm/eval_threshold_sweep.job` | CORE |
| Prompt testing | `scripts/test_prompts_quick.py` | `test_prompts.sh` | - | CORE |

## Source Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `src/qsm/models/qwen_audio.py` | Qwen2AudioClassifier with constrained decoding | CORE |
| `src/qsm/utils/normalize.py` | Response normalization to binary labels | CORE |
| `src/qsm/audio/slicing.py` | Audio segment extraction, padding | CORE |
| `src/qsm/audio/noise.py` | Noise augmentation (SNR mixing) | CORE |
| `src/qsm/audio/reverb.py` | Reverb augmentation (RIR convolution) | CORE |
| `src/qsm/audio/filters.py` | Bandpass filter augmentation | CORE |
| `src/qsm/data/loaders.py` | Dataset loading utilities | CORE |
| `src/qsm/data/slicing.py` | Data slicing helpers | CORE |
| `src/qsm/vad/silero.py` | Silero VAD baseline | CORE |

## Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/compute_psychometric_curves.py` | Duration/SNR vs. accuracy curves | CORE |
| `scripts/compute_roc_pr_curves.py` | ROC and PR curve generation | CORE |
| `scripts/compare_models_statistical.py` | Statistical comparison (McNemar, bootstrap) | CORE |
| `scripts/generate_final_plots.py` | Publication-ready figures | CORE |
| `scripts/analyze_per_condition.py` | Per-condition breakdown analysis | CORE |

## Data Preparation Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/prepare_base_clips.py` | Extract base audio clips | CORE |
| `scripts/generate_experimental_variants.py` | Apply augmentations (SNR, reverb, filter) | CORE |
| `scripts/create_group_stratified_split.py` | Create train/dev/test splits | CORE |
| `scripts/augment_with_musan.py` | MUSAN noise augmentation | CORE |

## Legacy/Archived (in `legacy/` or `scripts/archive_*`)

| Location | Contents |
|----------|----------|
| `scripts/archive_obsolete/` | Old OPRO optimizer versions |
| `scripts/archive_opro_legacy/` | Legacy OPRO post-FT scripts |
| `legacy/root/` | Old shell scripts from root |
| `legacy/scripts/` | Deprecated script versions |
| `legacy/slurm/` | Old job definitions |
| `legacy/results/` | Historical results (not needed) |

## Script Version History

| Script | Restored From | Commit Hash | Notes |
|--------|---------------|-------------|-------|
| `scripts/opro_classic_optimize.py` | main | (current) | Latest with psychoacoustic metrics |
| `scripts/opro_post_ft_v2.py` | main | (current) | Dict-safe prompt saving |

---

*Last updated: During cleanup/core-refactor branch work*
