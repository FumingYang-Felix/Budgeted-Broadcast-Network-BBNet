#!/usr/bin/env python3
"""
OLMo Mid-Training with Budgeted Broadcast (BB) Pruning
=======================================================
Loads OLMo from HuggingFace and runs 1000 steps of mid-training with BB.

Usage:
    BB_ENABLE=1 torchrun --nproc_per_node=4 olmo_bb_train_hf.py

Reference: https://arxiv.org/pdf/2510.01263

BB Pruning Strategy:
- Traffic Budget: t_i = a_i * k_i (on-rate × fan-in)
- Dynamic fan-in: k ≈ d0 / p_on (selective neurons get more connections)
- Top-K Magnitude: keep top-k weights per row by |W|
- Variance Rescaling: multiply by sqrt(in_features / k)
"""

import os
import sys
import time
import json
import argparse
import datetime
import threading
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Import BB
from bb_linear import BBLinear, patch_olmo_ffn_with_bb, log_bb_stats, collect_bb_stats


class GPUMonitor:
    """Background GPU utilization monitor."""
    
    def __init__(self, device, interval=1.0):
        self.device = device
        self.interval = interval
        self.running = False
        self.thread = None
        self.samples = []
        
    def _monitor_loop(self):
        import subprocess
        device_id = self.device.index if hasattr(self.device, 'index') else 0
        
        while self.running:
            try:
                # Get GPU utilization via nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                     '--format=csv,noheader,nounits', f'--id={device_id}'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 5:
                        self.samples.append({
                            'timestamp': time.time(),
                            'gpu_util': float(parts[0].strip()),
                            'mem_util': float(parts[1].strip()),
                            'mem_used_mb': float(parts[2].strip()),
                            'mem_total_mb': float(parts[3].strip()),
                            'power_w': float(parts[4].strip()) if parts[4].strip() != '[N/A]' else 0,
                        })
            except Exception:
                pass
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def get_summary(self):
        if not self.samples:
            return {}
        
        gpu_utils = [s['gpu_util'] for s in self.samples]
        mem_utils = [s['mem_util'] for s in self.samples]
        powers = [s['power_w'] for s in self.samples if s['power_w'] > 0]
        
        return {
            'avg_gpu_util': sum(gpu_utils) / len(gpu_utils),
            'max_gpu_util': max(gpu_utils),
            'min_gpu_util': min(gpu_utils),
            'avg_mem_util': sum(mem_utils) / len(mem_utils),
            'avg_power_w': sum(powers) / len(powers) if powers else 0,
            'num_samples': len(self.samples),
        }
    
    def print_summary(self, phase="Training"):
        summary = self.get_summary()
        if summary:
            print(f"\n=== GPU Utilization Summary ({phase}) ===")
            print(f"  Samples collected: {summary['num_samples']}")
            print(f"  GPU Util: avg={summary['avg_gpu_util']:.1f}%, "
                  f"max={summary['max_gpu_util']:.1f}%, min={summary['min_gpu_util']:.1f}%")
            print(f"  Memory Util: avg={summary['avg_mem_util']:.1f}%")
            if summary['avg_power_w'] > 0:
                print(f"  Power: avg={summary['avg_power_w']:.1f}W")
        return summary


def get_slurm_info():
    """Collect SLURM job information."""
    return {
        "job_id": os.getenv("SLURM_JOB_ID", "N/A"),
        "job_name": os.getenv("SLURM_JOB_NAME", "N/A"),
        "nodelist": os.getenv("SLURM_NODELIST", "N/A"),
        "nnodes": os.getenv("SLURM_NNODES", "1"),
        "partition": os.getenv("SLURM_JOB_PARTITION", "N/A"),
        "account": os.getenv("SLURM_JOB_ACCOUNT", "N/A"),
    }


def print_slurm_report(rank):
    """Print SLURM report (only rank 0)."""
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("                    SLURM JOB REPORT - OLMo + BB Mid-Training")
    print("="*80)
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"Hostname: {os.uname().nodename}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    slurm = get_slurm_info()
    print("--- SLURM Info ---")
    for k, v in slurm.items():
        print(f"  {k}: {v}")
    print()
    
    print("--- GPU Info ---")
    print(f"  Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()
    
    print("--- BB Config ---")
    print(f"  BB_ENABLE: {os.getenv('BB_ENABLE', '0')}")
    print(f"  BB_REFRESH_INTERVAL: {os.getenv('BB_REFRESH_INTERVAL', '50')}")
    print(f"  BB_K_MIN: {os.getenv('BB_K_MIN', '64')}")
    print(f"  BB_EMA_BETA: {os.getenv('BB_EMA_BETA', '0.99')}")
    print(f"  BB_D0: {os.getenv('BB_D0', '1.0')}")
    print("="*80 + "\n")


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        if rank == 0:
            print(f"[Distributed] Initialized: world_size={world_size}")
    
    device = torch.device(f"cuda:{local_rank}")
    return device, rank, world_size, local_rank


def create_dolma_dataset(tokenizer, seq_len=2048, max_samples=None):
    """Load streaming dataset for OLMo mid-training.
    
    Uses allenai/c4 as a substitute for Dolma (similar web text data).
    Dolma's loading script is deprecated in newer datasets versions.
    """
    from datasets import load_dataset
    from torch.utils.data import IterableDataset
    
    class DolmaIterableDataset(IterableDataset):
        def __init__(self, tokenizer, seq_len, max_samples=None):
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.max_samples = max_samples
            
        def __iter__(self):
            # Use C4 as substitute (similar web text data)
            # Other options: "HuggingFaceFW/fineweb", "wikipedia"
            dataset = load_dataset(
                "allenai/c4", 
                "en",
                split="train",
                streaming=True
            )
            
            buffer = []
            count = 0
            
            for example in dataset:
                # Tokenize text
                text = example.get("text", "")
                if not text:
                    continue
                    
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(tokens)
                buffer.append(self.tokenizer.eos_token_id)  # Add EOS after each doc
                
                # Yield chunks of seq_len
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[:self.seq_len + 1]
                    buffer = buffer[self.seq_len:]
                    
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    
                    yield {"input_ids": input_ids, "labels": labels}
                    
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        return
    
    return DolmaIterableDataset(tokenizer, seq_len, max_samples)


def create_dummy_dataset(tokenizer, num_samples=10000, seq_len=512):
    """Create a dummy dataset for testing (random tokens)."""
    # In real mid-training, you'd use Dolma or other datasets
    # This is just for testing the multi-GPU setup
    
    vocab_size = tokenizer.vocab_size
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, seq_len, vocab_size):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random tokens (for testing only)
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {"input_ids": input_ids, "labels": input_ids.clone()}
    
    return DummyDataset(num_samples, seq_len, vocab_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-1B", 
                        help="HuggingFace model name")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Per-GPU batch size")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--use_dolma", action="store_true",
                        help="Use Dolma dataset instead of dummy data")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_report", type=str, default=None)
    parser.add_argument("--save_checkpoint", type=str, default=None,
                        help="Path to save final checkpoint (e.g., checkpoints/olmo-bb-final)")
    args = parser.parse_args()
    
    # Setup distributed
    device, rank, world_size, local_rank = setup_distributed()
    
    # Print SLURM report
    print_slurm_report(rank)
    
    # =========================================================================
    # Load Model from HuggingFace
    # =========================================================================
    if rank == 0:
        print(f"[Step 1] Loading model: {args.model_name}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Try to load OLMo, fallback to GPT-2 if it fails
    model_loaded = False
    model_name_used = args.model_name
    
    # Import hf_olmo - REQUIRED for OLMo models
    if "olmo" in args.model_name.lower():
        try:
            import hf_olmo
            if rank == 0:
                print("  hf_olmo imported successfully")
        except ImportError as e:
            if rank == 0:
                print(f"  FATAL: hf_olmo not found: {e}")
                print("  Cannot load OLMo without hf_olmo. Exiting.")
            raise SystemExit(1)
    
    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name_used, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_used,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    model = model.to(device)
    
    # Enable gradient checkpointing to save memory (OLMo official uses this)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("  Gradient checkpointing: ENABLED (saves memory)")
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded: {model_name_used} ({total_params / 1e9:.2f}B parameters)")
    
    # =========================================================================
    # Apply BB Pruning
    # =========================================================================
    bb_enabled = os.getenv("BB_ENABLE", "0") == "1"
    
    if bb_enabled:
        if rank == 0:
            print(f"\n[Step 2] Applying BB Pruning...")
            print(f"  Scanning model architecture...")
        
        # First, print model structure to understand naming
        # Collect all linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append((name, module.in_features, module.out_features))
        
        if rank == 0:
            print(f"  Found {len(linear_layers)} Linear layers")
            # Show all to understand naming
            for name, in_f, out_f in linear_layers[:10]:
                print(f"    - {name}: [{in_f} -> {out_f}]")
            if len(linear_layers) > 10:
                print(f"    ... and {len(linear_layers) - 10} more")
        
        # Patch all Linear layers that look like MLP down-projection
        # These typically have: large_dim -> small_dim (e.g., 4096 -> 1024)
        # Common names: down_proj, c_proj, ff_out, fc2, wo, w2
        patched = 0
        target_names = ['down_proj', 'c_proj', 'ff_out', 'fc2', 'wo', 'w2', 'ff_proj']
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if this looks like a down-projection (name contains target patterns)
                layer_name = name.split('.')[-1]
                is_target = any(t in layer_name.lower() for t in target_names)
                
                # Also check by shape: in_features > out_features (down projection)
                is_down_proj = module.in_features > module.out_features
                
                # For GPT-2: transformer.h.X.mlp.c_proj
                is_gpt2_mlp = 'mlp' in name.lower() and 'c_proj' in name.lower()
                
                # For OLMo: model.transformer.blocks.X.ff_proj or similar
                is_olmo_mlp = 'ff' in name.lower() and 'proj' in name.lower()
                
                if is_target or is_gpt2_mlp or is_olmo_mlp or (is_down_proj and ('mlp' in name.lower() or 'ff' in name.lower())):
                    old_linear = module
                    new_linear = BBLinear(
                        in_features=old_linear.in_features,
                        out_features=old_linear.out_features,
                        bias=old_linear.bias is not None,
                        device=old_linear.weight.device,
                        dtype=old_linear.weight.dtype,
                    )
                    new_linear.weight.data.copy_(old_linear.weight.data)
                    if old_linear.bias is not None:
                        new_linear.bias.data.copy_(old_linear.bias.data)
                    
                    # Replace the module
                    # Navigate to parent and replace
                    parts = name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_name, child_name = parts
                        parent = model
                        for p in parent_name.split('.'):
                            parent = getattr(parent, p)
                        setattr(parent, child_name, new_linear)
                        patched += 1
                        if rank == 0:
                            print(f"  Patched: {name} [{old_linear.in_features} -> {old_linear.out_features}]")
                    elif len(parts) == 1:
                        setattr(model, name, new_linear)
                        patched += 1
                        if rank == 0:
                            print(f"  Patched: {name} [{old_linear.in_features} -> {old_linear.out_features}]")
        
        if rank == 0:
            if patched > 0:
                print(f"  Total patched layers: {patched}")
            else:
                print(f"  WARNING: No layers patched! Trying fallback method...")
                # Fallback: patch ANY linear layer in MLP blocks
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and 'mlp' in name.lower():
                        print(f"    Found MLP layer: {name}")
    else:
        if rank == 0:
            print(f"\n[Step 2] BB Pruning DISABLED (set BB_ENABLE=1 to enable)")
    
    # =========================================================================
    # Wrap with DDP
    # =========================================================================
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if rank == 0:
            print(f"\n[Step 3] Wrapped model with DDP (world_size={world_size})")
    
    # =========================================================================
    # Create Dataset & DataLoader
    # =========================================================================
    if args.use_dolma:
        if rank == 0:
            print(f"\n[Step 4] Loading Dolma dataset for mid-training...")
            print(f"  Sequence length: {args.seq_len}")
        
        dataset = create_dolma_dataset(tokenizer, seq_len=args.seq_len)
        
        # For streaming dataset, no sampler needed
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
        )
        sampler = None
    else:
        if rank == 0:
            print(f"\n[Step 4] Creating dummy dataset for testing...")
            print(f"  (Use --use_dolma for real mid-training with Dolma dataset)")
        
        dataset = create_dummy_dataset(tokenizer, num_samples=10000, seq_len=args.seq_len)
        
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    
    # =========================================================================
    # Optimizer & Training Loop
    # =========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    if rank == 0:
        print(f"\n[Step 5] Starting training loop...")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Batch size per GPU: {args.batch_size} (micro-batch)")
        print(f"  Gradient accumulation: {args.grad_accum}")
        print(f"  Effective batch per GPU: {args.batch_size * args.grad_accum}")
        print(f"  Total effective batch: {args.batch_size * args.grad_accum * world_size}")
        print(f"  Learning rate: {args.lr}")
        print()
        print("  BB Pruning Strategy:")
        print("    - Traffic Budget: t_i = a_i * k_i (on-rate × fan-in)")
        print("    - Dynamic fan-in: k ≈ d0 / p_on")
        print("    - Top-K Magnitude: keep top-k weights per row")
        print("    - Refresh interval: every", os.getenv("BB_REFRESH_INTERVAL", "50"), "steps")
        print("="*80 + "\n")
    
    # Start GPU monitor
    gpu_monitor = None
    if rank == 0:
        gpu_monitor = GPUMonitor(device, interval=2.0)
        gpu_monitor.start()
        print("[GPU Monitor] Started background monitoring (2s interval)")
    
    model.train()
    step = 0
    start_time = time.time()
    data_iter = iter(dataloader)
    pruning_times = []
    grad_accum = args.grad_accum
    
    while step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        # Gradient accumulation loop
        for accum_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(step * grad_accum + accum_step)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / grad_accum  # Scale loss
            
            # Backward (accumulate gradients)
            loss.backward()
            accum_loss += loss.item()  # loss was divided by grad_accum, sum of 64 = original loss
        
        # Optimizer step after accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        loss_value = accum_loss  # This is the correct averaged loss
        
        # Logging
        if step % args.log_interval == 0:
            if rank == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                
                # GPU memory
                mem_used = torch.cuda.max_memory_allocated(device) / 1024**3
                mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                
                # Get current GPU utilization
                gpu_util_str = ""
                if gpu_monitor and gpu_monitor.samples:
                    latest = gpu_monitor.samples[-1]
                    gpu_util_str = f" | GPU={latest['gpu_util']:.0f}%"
                
                print(f"[Step {step:5d}/{args.max_steps}] "
                      f"loss={loss_value:.4f} | "
                      f"steps/s={steps_per_sec:.2f} | "
                      f"mem={mem_used:.1f}/{mem_total:.1f}GB{gpu_util_str}")
                
                # Log BB stats
                if bb_enabled:
                    base_model = model.module if hasattr(model, 'module') else model
                    stats = collect_bb_stats(base_model)
                    if stats:
                        sparsities = [s["sparsity"] for s in stats.values() if s.get("bb_enabled")]
                        avg_ks = [s["avg_k"] for s in stats.values() if s.get("bb_enabled")]
                        on_rates = [s["avg_on_rate"] for s in stats.values() if s.get("bb_enabled")]
                        
                        if sparsities:
                            print(f"  [BB] sparsity={sum(sparsities)/len(sparsities)*100:.2f}% | "
                                  f"avg_k={sum(avg_ks)/len(avg_ks):.0f} | "
                                  f"on_rate={sum(on_rates)/len(on_rates):.4f}")
    
    # =========================================================================
    # Final Report
    # =========================================================================
    if rank == 0:
        # Stop GPU monitor
        gpu_summary = {}
        if gpu_monitor:
            gpu_monitor.stop()
            gpu_summary = gpu_monitor.print_summary("Training")
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("                         TRAINING COMPLETE")
        print("="*80)
        print(f"Total steps: {args.max_steps}")
        print(f"Total time: {total_time:.1f}s ({args.max_steps/total_time:.2f} steps/s)")
        print(f"Final loss: {loss_value:.4f}")
        
        # Final GPU stats
        print(f"\nGPU Memory (max allocated): {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
        
        # BB stats
        bb_final_stats = {}
        if bb_enabled:
            base_model = model.module if hasattr(model, 'module') else model
            stats = collect_bb_stats(base_model)
            if stats:
                sparsities = [s["sparsity"] for s in stats.values() if s.get("bb_enabled")]
                avg_ks = [s["avg_k"] for s in stats.values() if s.get("bb_enabled")]
                on_rates = [s["avg_on_rate"] for s in stats.values() if s.get("bb_enabled")]
                
                if sparsities:
                    bb_final_stats = {
                        "final_sparsity": sum(sparsities)/len(sparsities),
                        "final_avg_k": sum(avg_ks)/len(avg_ks),
                        "final_on_rate": sum(on_rates)/len(on_rates),
                        "num_bb_layers": len(sparsities),
                    }
                    print(f"\n=== BB Pruning Final Stats ===")
                    print(f"  Layers with BB: {bb_final_stats['num_bb_layers']}")
                    print(f"  Final Sparsity: {bb_final_stats['final_sparsity']*100:.2f}%")
                    print(f"  Final Avg Fan-in (k): {bb_final_stats['final_avg_k']:.0f}")
                    print(f"  Final Avg On-Rate: {bb_final_stats['final_on_rate']:.4f}")
        
        print("="*80)
        
        # Save checkpoint
        if args.save_checkpoint:
            ckpt_path = args.save_checkpoint
            os.makedirs(ckpt_path, exist_ok=True)
            
            # Get the base model (unwrap DDP)
            base_model = model.module if hasattr(model, 'module') else model
            
            # Save model and tokenizer
            print(f"\n[Saving checkpoint to {ckpt_path}]")
            base_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            
            # Save BB state if enabled
            if bb_enabled:
                bb_state = {}
                for name, module in base_model.named_modules():
                    if hasattr(module, 'mask') and module.mask is not None:
                        bb_state[name] = {
                            'mask': module.mask.cpu(),
                            'on_rate_ema': module.on_rate_ema.cpu() if hasattr(module, 'on_rate_ema') else None,
                        }
                if bb_state:
                    torch.save(bb_state, os.path.join(ckpt_path, "bb_state.pt"))
                    print(f"  BB state saved ({len(bb_state)} layers)")
            
            print(f"  Checkpoint saved to: {ckpt_path}")
        
        # Save report
        if args.save_report:
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": args.model_name,
                "max_steps": args.max_steps,
                "total_time_sec": total_time,
                "steps_per_sec": args.max_steps / total_time,
                "final_loss": loss_value,
                "world_size": world_size,
                "slurm": get_slurm_info(),
                "bb_enabled": bb_enabled,
                "bb_stats": bb_final_stats,
                "gpu_utilization": gpu_summary,
            }
            with open(args.save_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.save_report}")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

