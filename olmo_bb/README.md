# OLMo + Budgeted Broadcast Mid-Training

This directory contains scripts and configurations for mid-training OLMo models with Budgeted Broadcast (BB) pruning on the Harvard FASRC / Kempner H200 cluster.

## Directory Structure

```
olmo_bb/
├── bb_linear.py           # BB Linear layer implementation with activity-dependent pruning
├── olmo_bb_train_hf.py    # Main training script using HuggingFace Transformers
├── olmo_bb_midtrain.py    # Mid-training script for Dolmino recipe
├── __init__.py            # Package init
├── configs/
│   └── OLMo-BB-1000step.yaml  # Training configuration
└── slurm_scripts/
    ├── olmo-bb-8gpu-bb.sh         # 2-node BB training (8 GPUs)
    ├── olmo-bb-8gpu-dense.sh      # 2-node Dense training (8 GPUs)
    ├── olmo-bb-8gpu-bb-large.sh   # 2-node BB with large batch
    ├── olmo-bb-8gpu-dense-large.sh # 2-node Dense with large batch
    ├── olmo-bb-8gpu-dolma-bb.sh   # Dolmino BB training
    ├── olmo-bb-8gpu-dolma-dense.sh # Dolmino Dense training
    ├── olmo-bb-multinode.sh       # Multi-node template
    └── olmo-bb-single-node.sh     # Single-node template
```

## Quick Start

### 1. Environment Setup

```bash
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

python -m venv ~/.venv/olmo_bb
source ~/.venv/olmo_bb/bin/activate

pip install torch==2.3.0 transformers datasets accelerate wandb
```

### 2. Run BB Training (2 nodes, 8 GPUs)

```bash
# Enable BB pruning
export BB_ENABLE=1
export BB_REFRESH_INTERVAL=50
export BB_K_MIN=64

# Submit job
sbatch slurm_scripts/olmo-bb-8gpu-bb.sh
```

### 3. Run Dense Baseline

```bash
export BB_ENABLE=0
sbatch slurm_scripts/olmo-bb-8gpu-dense.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BB_ENABLE` | Enable/disable BB pruning (0 or 1) | 0 |
| `BB_REFRESH_INTERVAL` | Steps between mask refreshes | 50 |
| `BB_K_MIN` | Minimum neurons to keep active | 64 |

## References

- [OLMo Paper](https://arxiv.org/abs/2402.00838) - Open Language Model
- [OLMo-2 Paper](https://arxiv.org/abs/2501.00656) - Dolmino mid-training recipe
- [Budgeted Broadcast Paper](https://arxiv.org/abs/2510.01263) - Activity-dependent pruning rule

