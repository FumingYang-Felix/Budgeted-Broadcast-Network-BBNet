#!/bin/bash
#===============================================================================
# OLMo Dense Baseline - 8 GPU (2 nodes × 4 GPUs) - LARGE SCALE
# batch=512/GPU (global batch=4096), seq=1024, 500 steps
# Tokens per step: 4096 × 1024 = 4.2M
# Total tokens: 4096 × 1024 × 500 = 2.1B tokens
# BB_ENABLE=0 (no pruning)
#===============================================================================

#SBATCH -p gpu_h200                  # H200 partition
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --mem=400G                   # Memory per node (increased)
#SBATCH --gres=gpu:4                 # 4 GPUs per node (total 8 GPUs)
#SBATCH --time=06:00:00              # 6 hours (longer run)
#SBATCH --account=lichtman_lab       # Account
#SBATCH -o slurm_out/dense-8gpu-large-%j.out
#SBATCH -e slurm_out/dense-8gpu-large-%j.err
#SBATCH --job-name=olmo-dense-large

#===============================================================================
# Configuration - LARGE SCALE
#===============================================================================

MAX_STEPS=500
BATCH_SIZE=512                       # Per GPU batch size (global batch = 512 × 8 = 4096)
SEQ_LEN=1024
# Total batch: 512 * 8 = 4096
# Tokens per step: 4096 * 1024 = 4.2M
# Total tokens: 4096 * 1024 * 500 = 2.1B tokens
LR=1e-5
LOG_INTERVAL=10
MODEL_NAME="allenai/OLMo-1B"
GPUS_PER_NODE=4

# BB DISABLED for dense baseline
export BB_ENABLE=0

# NCCL Configuration
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0

#===============================================================================
# Setup
#===============================================================================

set -ex

mkdir -p slurm_out
cd $HOME/olmo_bb

module purge
module load python/3.10.13-fasrc01
module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUSERBASE="$HOME/.local"

python -c "import hf_olmo; print('hf_olmo: OK')" || exit 1

#===============================================================================
# Job Report
#===============================================================================

echo ""
echo "==============================================================================="
echo "        DENSE BASELINE - 8 GPU LARGE SCALE (2.1B tokens)                      "
echo "==============================================================================="
echo ""
echo "=== Job Information ==="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Start Time:       $(date)"
echo ""
echo "=== Configuration ==="
echo "Nodes:            $SLURM_NNODES"
echo "GPUs per Node:    $GPUS_PER_NODE"
echo "Total GPUs:       $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Model:            $MODEL_NAME"
echo "Max Steps:        $MAX_STEPS"
echo "Batch Size/GPU:   $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * SLURM_NNODES * GPUS_PER_NODE))"
echo "Sequence Length:  $SEQ_LEN"
echo "Tokens per Step:  $((BATCH_SIZE * SLURM_NNODES * GPUS_PER_NODE * SEQ_LEN))"
echo "Total Tokens:     $((BATCH_SIZE * SLURM_NNODES * GPUS_PER_NODE * SEQ_LEN * MAX_STEPS))"
echo "BB Enabled:       $BB_ENABLE (DENSE)"
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

#===============================================================================
# Distributed Setup
#===============================================================================

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

echo "=== Starting Dense Training (8 GPUs, 2.1B tokens) ==="
START_TIME=$(date +%s)

CKPT_DIR="checkpoints/olmo-dense-8gpu-large-$SLURM_JOB_ID"

srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    olmo_bb_train_hf.py \
    --model_name $MODEL_NAME \
    --use_dolma \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --lr $LR \
    --log_interval $LOG_INTERVAL \
    --save_report slurm_out/report-dense-8gpu-large-$SLURM_JOB_ID.json \
    --save_checkpoint $CKPT_DIR

echo "Checkpoint saved to: $CKPT_DIR"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

#===============================================================================
# Post-Training Report
#===============================================================================

echo ""
echo "==============================================================================="
echo "                         Training Complete                                     "
echo "==============================================================================="
echo "End Time:         $(date)"
echo "Elapsed Time:     ${ELAPSED}s ($((ELAPSED / 60)) min)"
echo "Exit Code:        $EXIT_CODE"
echo ""

echo "=== Final GPU Memory Usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

if [ -f "slurm_out/report-dense-8gpu-large-$SLURM_JOB_ID.json" ]; then
    echo "=== Training Report ==="
    cat slurm_out/report-dense-8gpu-large-$SLURM_JOB_ID.json
fi

exit $EXIT_CODE

