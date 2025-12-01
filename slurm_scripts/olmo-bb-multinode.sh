#!/bin/bash
#===============================================================================
# OLMo Mid-Training with Budgeted Broadcast (BB) - Multi-Node SLURM Script
# Reference: https://arxiv.org/pdf/2510.01263
#===============================================================================

#SBATCH -p gpu_h200                  # H200 partition
#SBATCH --nodes=2                    # 2 nodes for multi-node test
#SBATCH --ntasks-per-node=1          # One task per node (torchrun handles GPUs)
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:2                 # 2 GPUs per node (total 4 GPUs)
#SBATCH --time=03:00:00              # 3 hours
#SBATCH --account=lichtman_lab       # Your account
#SBATCH -o slurm_out/olmo-bb-multi-%j.out  # Standard output
#SBATCH -e slurm_out/olmo-bb-multi-%j.err  # Standard error
#SBATCH --job-name=olmo-bb-multi     # Job name

#===============================================================================
# Configuration
#===============================================================================

# Training parameters
MAX_STEPS=1000
BATCH_SIZE=2
SEQ_LEN=512
LR=1e-5
LOG_INTERVAL=10
MODEL_NAME="allenai/OLMo-1B"
GPUS_PER_NODE=2

# BB (Budgeted Broadcast) Configuration
export BB_ENABLE=1                   # Enable BB pruning
export BB_REFRESH_INTERVAL=50        # Steps between mask refresh
export BB_K_MIN=64                   # Minimum fan-in per neuron
export BB_K_MAX=-1                   # Maximum fan-in (-1 = no limit)
export BB_EMA_BETA=0.99              # EMA coefficient for on-rate
export BB_D0=1.0                     # Heuristic: k ~= d0 / p_on
export BB_RESCALE=1                  # Enable variance-preserving rescale

# NCCL Configuration for multi-node
export NCCL_DEBUG=WARN               # Set to INFO for debugging
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0             # Enable InfiniBand

#===============================================================================
# Setup
#===============================================================================

set -ex  # Enable command tracing, exit on error

# Create output directory
mkdir -p slurm_out
cd $HOME/olmo_bb

# Load modules
module purge
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Add user bin to PATH
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUSERBASE="$HOME/.local"

# Dependencies should already be installed from single-node run
# Verify hf_olmo
python -c "import hf_olmo; print('hf_olmo: OK')" || {
    echo "ERROR: hf_olmo not installed. Run single-node script first."
    exit 1
}

#===============================================================================
# Print Job Information (SLURM Report)
#===============================================================================

echo ""
echo "==============================================================================="
echo "              SLURM JOB REPORT - OLMo + BB Multi-Node Training                "
echo "==============================================================================="
echo ""
echo "=== Job Information ==="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Account:          $SLURM_JOB_ACCOUNT"
echo "Start Time:       $(date)"
echo ""

echo "=== Node Configuration ==="
echo "Node List:        $SLURM_NODELIST"
echo "Number of Nodes:  $SLURM_NNODES"
echo "GPUs per Node:    $GPUS_PER_NODE"
echo "Total GPUs:       $((SLURM_NNODES * GPUS_PER_NODE))"
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "=== Configuration ==="
echo "Model:            $MODEL_NAME"
echo "Max Steps:        $MAX_STEPS"
echo "Batch Size/GPU:   $BATCH_SIZE"
echo "BB Enabled:       $BB_ENABLE"
echo ""

#===============================================================================
# Distributed Setup
#===============================================================================

# Get master node address
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
MASTER_NODE=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address | head -n1)
MASTER_PORT=29500

echo "=== Distributed Configuration ==="
echo "Master Node:      $MASTER_NODE"
echo "Master IP:        $MASTER_ADDR"
echo "Master Port:      $MASTER_PORT"
echo ""

#===============================================================================
# Launch Training
#===============================================================================

echo "=== Starting Multi-Node Training ==="

NNODES=$SLURM_NNODES
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

srun python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    olmo_bb_train_hf.py \
    --model_name $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --lr $LR \
    --log_interval $LOG_INTERVAL \
    --save_report slurm_out/report-multi-$SLURM_JOB_ID.json

EXIT_CODE=$?

#===============================================================================
# Post-Training Report
#===============================================================================

echo ""
echo "==============================================================================="
echo "                         Training Complete                                     "
echo "==============================================================================="
echo "End Time:         $(date)"
echo "Exit Code:        $EXIT_CODE"
echo ""

echo "=== Final GPU Memory Usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

if [ -f "slurm_out/report-multi-$SLURM_JOB_ID.json" ]; then
    echo "=== Training Report ==="
    cat slurm_out/report-multi-$SLURM_JOB_ID.json
fi

exit $EXIT_CODE

