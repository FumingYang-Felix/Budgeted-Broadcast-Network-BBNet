#!/bin/bash
#===============================================================================
# OLMo-1B Mid-Training with Dolma Dataset - BB Pruning
# Using official OLMo configuration:
#   - Sequence length: 2048
#   - Global batch size: ~2M tokens
#   - Model: allenai/OLMo-1B (pretrained checkpoint)
#   - BB Pruning enabled
#===============================================================================

#SBATCH -p gpu_h200                  # H200 partition
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --mem=200G                   # Memory per node
#SBATCH --gres=gpu:4                 # 4 GPUs per node (total 8 GPUs)
#SBATCH --time=01:00:00              # 1 hour
#SBATCH --account=lichtman_lab       # Account
#SBATCH -o slurm_out/dolma-bb-8gpu-%j.out
#SBATCH -e slurm_out/dolma-bb-8gpu-%j.err
#SBATCH --job-name=olmo-dolma-bb

#===============================================================================
# OLMo Official Configuration + BB Pruning
#===============================================================================

MAX_STEPS=100
# OLMo official config:
# global_train_batch_size: 512
# device_train_microbatch_size: 4
BATCH_SIZE=4                         # OLMo official micro-batch
GRAD_ACCUM=16                        # To achieve 64 effective per GPU
# Total effective batch: 64 * 8 = 512 (OLMo official)

SEQ_LEN=2048                         # OLMo official sequence length
LR=3e-5                              # OLMo official mid-training LR
LOG_INTERVAL=10
MODEL_NAME="allenai/OLMo-1B"
GPUS_PER_NODE=4

# BB ENABLED
export BB_ENABLE=1
export BB_REFRESH_INTERVAL=10        # Refresh mask every 10 steps (more frequent for 100 steps)
export BB_K_MIN=128                  # Higher k_min for stability
export BB_K_MAX=-1                   # No maximum
export BB_EMA_BETA=0.99              # EMA for on-rate
export BB_D0=1.0                     # Heuristic coefficient
export BB_RESCALE=1                  # Variance-preserving rescale

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

# Verify dependencies
python -c "import hf_olmo; print('hf_olmo: OK')" || exit 1
python -c "from datasets import load_dataset; print('datasets: OK')" || exit 1

#===============================================================================
# Job Report
#===============================================================================

echo ""
echo "==============================================================================="
echo "        OLMo-1B Mid-Training with Dolma - BB PRUNING                          "
echo "==============================================================================="
echo ""
echo "=== Job Information ==="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Start Time:       $(date)"
echo ""
echo "=== OLMo Official Configuration ==="
echo "Nodes:            $SLURM_NNODES"
echo "GPUs per Node:    $GPUS_PER_NODE"
echo "Total GPUs:       $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Model:            $MODEL_NAME (pretrained checkpoint)"
echo "Dataset:          Dolma (streaming)"
echo "Max Steps:        $MAX_STEPS"
echo "Micro-batch/GPU:  $BATCH_SIZE"
echo "Grad Accum:       $GRAD_ACCUM"
echo "Effective/GPU:    $((BATCH_SIZE * GRAD_ACCUM))"
echo "Total Batch:      $((BATCH_SIZE * GRAD_ACCUM * SLURM_NNODES * GPUS_PER_NODE)) instances"
echo "Sequence Length:  $SEQ_LEN (OLMo official)"
echo "Tokens per Step:  $((BATCH_SIZE * GRAD_ACCUM * SLURM_NNODES * GPUS_PER_NODE * SEQ_LEN))"
echo "Learning Rate:    $LR (OLMo official)"
echo ""
echo "=== BB Configuration ==="
echo "BB_ENABLE:           $BB_ENABLE"
echo "BB_REFRESH_INTERVAL: $BB_REFRESH_INTERVAL"
echo "BB_K_MIN:            $BB_K_MIN"
echo "BB_EMA_BETA:         $BB_EMA_BETA"
echo "BB_D0:               $BB_D0"
echo "BB_RESCALE:          $BB_RESCALE"
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

CKPT_DIR="checkpoints/olmo-dolma-bb-$SLURM_JOB_ID"

echo "=== Starting OLMo Mid-Training with Dolma (BB Pruning) ==="
START_TIME=$(date +%s)

srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    olmo_bb_train_hf.py \
    --model_name $MODEL_NAME \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --seq_len $SEQ_LEN \
    --lr $LR \
    --log_interval $LOG_INTERVAL \
    --use_dolma \
    --save_report slurm_out/report-dolma-bb-$SLURM_JOB_ID.json \
    --save_checkpoint $CKPT_DIR

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
echo "Checkpoint:       $CKPT_DIR"
echo ""

echo "=== Final GPU Memory Usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

exit $EXIT_CODE

