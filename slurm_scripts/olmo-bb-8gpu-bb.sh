#!/bin/bash
#===============================================================================
# OLMo BB Pruning - 8 GPU (2 nodes × 4 GPUs) for Technical Readiness
# BB_ENABLE=1 (with periodic pruning)
#===============================================================================

#SBATCH -p gpu_h200                  # H200 partition
#SBATCH --nodes=2                    # 2 nodes
#SBATCH --ntasks-per-node=1          # One task per node
#SBATCH --cpus-per-task=32           # CPUs per task
#SBATCH --mem=200G                   # Memory per node
#SBATCH --gres=gpu:4                 # 4 GPUs per node (total 8 GPUs)
#SBATCH --time=02:00:00              # 2 hours
#SBATCH --account=kempner_dev        # Kempner account
#SBATCH -o slurm_out/bb-8gpu-%j.out
#SBATCH -e slurm_out/bb-8gpu-%j.err
#SBATCH --job-name=olmo-bb-8gpu

#===============================================================================
# Configuration
#===============================================================================

MAX_STEPS=1000
BATCH_SIZE=2                         # Per GPU, total batch = 2 * 8 = 16
SEQ_LEN=512
LR=1e-5
LOG_INTERVAL=10
MODEL_NAME="allenai/OLMo-1B"
GPUS_PER_NODE=4

# BB ENABLED with production settings
export BB_ENABLE=1
export BB_REFRESH_INTERVAL=50        # Refresh mask every 50 steps
export BB_K_MIN=64                   # Minimum fan-in per neuron
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

python -c "import hf_olmo; print('hf_olmo: OK')" || exit 1

#===============================================================================
# Job Report
#===============================================================================

echo ""
echo "==============================================================================="
echo "        BB PRUNING - 8 GPU Technical Readiness Test                           "
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

echo "=== Starting BB Training (8 GPUs) ==="
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
    --seq_len $SEQ_LEN \
    --lr $LR \
    --log_interval $LOG_INTERVAL \
    --save_report slurm_out/report-bb-8gpu-$SLURM_JOB_ID.json

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

if [ -f "slurm_out/report-bb-8gpu-$SLURM_JOB_ID.json" ]; then
    echo "=== Training Report ==="
    cat slurm_out/report-bb-8gpu-$SLURM_JOB_ID.json
fi

exit $EXIT_CODE

