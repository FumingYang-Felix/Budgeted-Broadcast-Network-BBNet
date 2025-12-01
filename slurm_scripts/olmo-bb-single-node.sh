#!/bin/bash
#===============================================================================
# OLMo Mid-Training with BB - Single Node (2 GPUs)
# Reference: https://arxiv.org/pdf/2510.01263
#===============================================================================

#SBATCH -p gpu_h200                  # H200 partition (faster)
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=16           # CPUs
#SBATCH --mem=100G                   # Memory (reduced)
#SBATCH --gres=gpu:2                 # 2 GPUs
#SBATCH --time=03:00:00              # 3 hours
#SBATCH --account=lichtman_lab       # Your account
#SBATCH -o slurm_out/olmo-bb-%j.out  # Output
#SBATCH -e slurm_out/olmo-bb-%j.err  # Error
#SBATCH --job-name=olmo-bb-1node     # Job name

#===============================================================================
# BB Configuration
#===============================================================================
export BB_ENABLE=1                   # Enable BB pruning
export BB_REFRESH_INTERVAL=50        # Steps between mask refresh
export BB_K_MIN=64                   # Minimum fan-in
export BB_EMA_BETA=0.99              # EMA coefficient
export BB_D0=1.0                     # k ~= d0 / p_on
export BB_RESCALE=1                  # Variance-preserving rescale

#===============================================================================
# Training Configuration
#===============================================================================
MAX_STEPS=1000
BATCH_SIZE=2                         # Per-GPU batch size
SEQ_LEN=512
LR=1e-5
LOG_INTERVAL=10
MODEL_NAME="allenai/OLMo-1B"
NUM_GPUS=2

#===============================================================================
# Setup
#===============================================================================
set -ex  # Exit on error, print commands

mkdir -p slurm_out
cd $HOME/olmo_bb

# Load modules
module purge
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Add user bin to PATH FIRST
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUSERBASE="$HOME/.local"

# Install dependencies with explicit error checking
echo "=== Installing Dependencies ==="
pip install --user --upgrade pip

# Install packages one by one to catch errors
pip install --user torch || { echo "Failed to install torch"; exit 1; }
pip install --user transformers || { echo "Failed to install transformers"; exit 1; }
pip install --user accelerate || { echo "Failed to install accelerate"; exit 1; }
pip install --user datasets || { echo "Failed to install datasets"; exit 1; }

# Install OLMo packages - MUST succeed
echo "=== Installing OLMo packages ==="

# Method 1: Try pip install ai2-olmo (includes hf_olmo)
pip install --user ai2-olmo && echo "ai2-olmo installed" || echo "ai2-olmo failed"

# Method 2: Install hf_olmo directly from GitHub
pip install --user "git+https://github.com/allenai/OLMo.git#subdirectory=hf_olmo" && \
    echo "hf_olmo from GitHub installed" || echo "GitHub install failed"

# Method 3: Try PyPI package names
pip install --user hf-olmo 2>/dev/null || pip install --user hf_olmo 2>/dev/null || true

# Verify hf_olmo is installed - EXIT if not
echo "=== Verifying Installations ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Critical check for hf_olmo
python -c "import hf_olmo; print('hf_olmo: OK')" || {
    echo "ERROR: hf_olmo is NOT installed!"
    echo "Attempting final fix..."
    
    # Clone and install manually
    cd /tmp
    rm -rf OLMo
    git clone --depth 1 https://github.com/allenai/OLMo.git
    cd OLMo/hf_olmo
    pip install --user -e .
    cd $HOME/olmo_bb
    
    # Verify again
    python -c "import hf_olmo; print('hf_olmo: OK after manual install')" || {
        echo "FATAL: Cannot install hf_olmo. Exiting."
        exit 1
    }
}

#===============================================================================
# Print Job Info
#===============================================================================
echo ""
echo "==============================================================================="
echo "                    SLURM JOB REPORT - OLMo + BB Mid-Training"
echo "==============================================================================="
echo ""
echo "=== Job Information ==="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Account:          $SLURM_JOB_ACCOUNT"
echo "Node:             $(hostname)"
echo "Start Time:       $(date)"
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "=== Configuration ==="
echo "Model:            $MODEL_NAME"
echo "Num GPUs:         $NUM_GPUS"
echo "Max Steps:        $MAX_STEPS"
echo "Batch Size/GPU:   $BATCH_SIZE"
echo "Sequence Length:  $SEQ_LEN"
echo "Learning Rate:    $LR"
echo "BB Enabled:       $BB_ENABLE"
echo ""

#===============================================================================
# Run Training
#===============================================================================
echo "=== Starting Training ==="

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    olmo_bb_train_hf.py \
    --model_name $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --lr $LR \
    --log_interval $LOG_INTERVAL \
    --save_report slurm_out/report-$SLURM_JOB_ID.json

EXIT_CODE=$?

#===============================================================================
# Post-Training
#===============================================================================
echo ""
echo "==============================================================================="
echo "                         TRAINING COMPLETE"
echo "==============================================================================="
echo "End Time:         $(date)"
echo "Exit Code:        $EXIT_CODE"
echo ""

echo "=== Final GPU Memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

if [ -f "slurm_out/report-$SLURM_JOB_ID.json" ]; then
    echo "=== Training Report ==="
    cat slurm_out/report-$SLURM_JOB_ID.json
fi

exit $EXIT_CODE
