#!/usr/bin/env python3
"""
OLMo Mid-Training with Budgeted Broadcast (BB) Pruning
=======================================================

This script integrates BB pruning into OLMo's mid-training pipeline.
It's designed for multi-node distributed training on SLURM clusters.

Usage:
    # Single node test
    BB_ENABLE=1 torchrun --nproc_per_node=4 olmo_bb_midtrain.py --config config.yaml
    
    # Multi-node (via SLURM)
    sbatch olmo-bb-multinode.sh

Environment Variables for BB:
    BB_ENABLE=1              # Enable BB pruning
    BB_REFRESH_INTERVAL=50   # Steps between mask refresh
    BB_K_MIN=64              # Minimum fan-in per neuron
    BB_K_MAX=-1              # Maximum fan-in (-1 = no limit)
    BB_EMA_BETA=0.99         # EMA coefficient for on-rate
    BB_D0=1.0                # Heuristic: k ~= d0 / p_on
    BB_RESCALE=1             # Enable variance-preserving rescale

Reference: https://arxiv.org/pdf/2510.01263
"""

import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bb_linear import BBLinear, patch_olmo_ffn_with_bb, log_bb_stats


def get_slurm_info() -> Dict[str, Any]:
    """Collect SLURM job information for logging."""
    info = {
        "job_id": os.getenv("SLURM_JOB_ID", "N/A"),
        "job_name": os.getenv("SLURM_JOB_NAME", "N/A"),
        "nodelist": os.getenv("SLURM_NODELIST", "N/A"),
        "nnodes": os.getenv("SLURM_NNODES", "1"),
        "ntasks": os.getenv("SLURM_NTASKS", "1"),
        "cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK", "1"),
        "gpus_per_node": os.getenv("SLURM_GPUS_PER_NODE", "N/A"),
        "mem_per_node": os.getenv("SLURM_MEM_PER_NODE", "N/A"),
        "partition": os.getenv("SLURM_JOB_PARTITION", "N/A"),
        "account": os.getenv("SLURM_JOB_ACCOUNT", "N/A"),
    }
    return info


def get_gpu_info() -> Dict[str, Any]:
    """Collect GPU information."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    info = {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        })
    
    return info


def get_distributed_info() -> Dict[str, Any]:
    """Collect distributed training information."""
    if not dist.is_initialized():
        return {"distributed": False}
    
    return {
        "distributed": True,
        "backend": dist.get_backend(),
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "local_rank": int(os.getenv("LOCAL_RANK", 0)),
    }


def log_system_report(output_path: Optional[str] = None):
    """Generate and log a comprehensive system report for SLURM."""
    rank = int(os.getenv("RANK", 0))
    
    # Only rank 0 logs the full report
    if rank != 0:
        return
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "slurm": get_slurm_info(),
        "gpu": get_gpu_info(),
        "distributed": get_distributed_info(),
        "bb_config": {
            "BB_ENABLE": os.getenv("BB_ENABLE", "0"),
            "BB_REFRESH_INTERVAL": os.getenv("BB_REFRESH_INTERVAL", "50"),
            "BB_K_MIN": os.getenv("BB_K_MIN", "64"),
            "BB_K_MAX": os.getenv("BB_K_MAX", "-1"),
            "BB_EMA_BETA": os.getenv("BB_EMA_BETA", "0.99"),
            "BB_D0": os.getenv("BB_D0", "1.0"),
            "BB_RESCALE": os.getenv("BB_RESCALE", "1"),
        },
    }
    
    print("\n" + "="*80)
    print("SLURM JOB REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Hostname: {report['hostname']}")
    print(f"PyTorch Version: {report['pytorch_version']}")
    print()
    
    print("--- SLURM Info ---")
    for k, v in report["slurm"].items():
        print(f"  {k}: {v}")
    print()
    
    print("--- GPU Info ---")
    gpu = report["gpu"]
    if gpu["cuda_available"]:
        print(f"  CUDA Version: {gpu['cuda_version']}")
        print(f"  cuDNN Version: {gpu['cudnn_version']}")
        print(f"  Device Count: {gpu['device_count']}")
        for dev in gpu["devices"]:
            print(f"    GPU {dev['index']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")
    else:
        print("  CUDA not available")
    print()
    
    print("--- Distributed Info ---")
    dist_info = report["distributed"]
    if dist_info.get("distributed"):
        print(f"  Backend: {dist_info['backend']}")
        print(f"  World Size: {dist_info['world_size']}")
        print(f"  Rank: {dist_info['rank']}")
        print(f"  Local Rank: {dist_info['local_rank']}")
    else:
        print("  Not using distributed training")
    print()
    
    print("--- BB Config ---")
    for k, v in report["bb_config"].items():
        print(f"  {k}: {v}")
    print("="*80 + "\n")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")
    
    return report


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            dist.init_process_group(backend="gloo", init_method="env://")
        
        if rank == 0:
            print(f"[Distributed] Initialized with world_size={world_size}")
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description="OLMo Mid-Training with BB Pruning")
    parser.add_argument("--config", type=str, default=None, help="Path to OLMo config YAML")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_report", type=str, default=None, help="Path to save system report JSON")
    parser.add_argument("--dry_run", action="store_true", help="Only log system info, don't train")
    args = parser.parse_args()
    
    # Setup distributed
    device, rank, world_size, local_rank = setup_distributed()
    
    # Log system report
    log_system_report(args.save_report)
    
    if args.dry_run:
        print("[Dry Run] Exiting without training.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    
    # =========================================================================
    # TODO: Replace this section with actual OLMo loading and training
    # =========================================================================
    
    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING LOOP (Placeholder)")
        print("="*80)
        print("To integrate with OLMo:")
        print("  1. Load OLMo model from checkpoint")
        print("  2. Call patch_olmo_ffn_with_bb(model) to enable BB")
        print("  3. Run training loop with log_bb_stats() for monitoring")
        print("="*80 + "\n")
    
    # Example: Simulate training loop with dummy model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if rank == 0:
            print("Loading OLMo model...")
        
        # You would replace this with your actual OLMo loading code
        # model = OLMo.from_pretrained("allenai/OLMo-1B")
        # For now, create a dummy model for testing
        
        class DummyFFN(torch.nn.Module):
            def __init__(self, d_model=1024, d_ff=4096):
                super().__init__()
                self.ff_proj = torch.nn.Linear(d_ff, d_model)
                self.ff_out = torch.nn.Linear(d_model, d_ff)
            
            def forward(self, x):
                return self.ff_proj(torch.relu(self.ff_out(x)))
        
        class DummyModel(torch.nn.Module):
            def __init__(self, n_layers=4):
                super().__init__()
                self.layers = torch.nn.ModuleList([DummyFFN() for _ in range(n_layers)])
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        model = DummyModel().to(device)
        
        # Apply BB patching
        bb_enabled = os.getenv("BB_ENABLE", "0") == "1"
        if bb_enabled:
            patched = patch_olmo_ffn_with_bb(model, verbose=(rank == 0))
            if rank == 0:
                print(f"[BB] Patched {patched} layers with BBLinear")
        
        # Wrap with DDP if distributed
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
        
        # Dummy optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for step in range(1, args.max_steps + 1):
            # Dummy forward/backward
            x = torch.randn(2, 128, 1024, device=device)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            if step % args.log_interval == 0 and rank == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                
                # GPU utilization
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated(device) / (1024**3)
                    mem_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                    gpu_util = (mem_used / mem_total) * 100
                else:
                    mem_used = mem_total = gpu_util = 0
                
                print(f"[Step {step:5d}/{args.max_steps}] "
                      f"loss={loss.item():.4f} | "
                      f"steps/s={steps_per_sec:.2f} | "
                      f"GPU mem={mem_used:.1f}/{mem_total:.1f}GB ({gpu_util:.1f}%)")
                
                # Log BB stats
                if bb_enabled:
                    base_model = model.module if hasattr(model, 'module') else model
                    log_bb_stats(base_model, step)
        
        if rank == 0:
            total_time = time.time() - start_time
            print(f"\n[Training Complete] {args.max_steps} steps in {total_time:.1f}s "
                  f"({args.max_steps/total_time:.2f} steps/s)")
    
    except Exception as e:
        if rank == 0:
            print(f"[Error] {e}")
            print("Note: This is a placeholder. Replace with actual OLMo training code.")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

