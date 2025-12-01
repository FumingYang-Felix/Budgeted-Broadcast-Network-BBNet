"""
Budgeted Broadcast (BB) Linear Layer for OLMo Integration
Based on: https://arxiv.org/pdf/2510.01263

Traffic Budget: t_i = a_i * k_i (on-rate × fan-out)
Selectivity-Audience Balance: log((1-a_i)/a_i) ≈ β * k_i

This is a fan-in pruner that dynamically adjusts per-row top-k connections
based on EMA of activation on-rate.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    v = os.getenv(name, str(default))
    try:
        return float(v)
    except Exception:
        return default


class BBLinear(nn.Linear):
    """
    A minimal BB-style fan-in pruner for a Linear layer:
      - Maintains a boolean mask per output neuron (row-wise).
      - Tracks EMA of "on-rate" per output neuron using (pre-activation > 0).
      - Periodically refreshes per-row top-k fan-in based on |W|.
    This is meant as a *smoke-test integration* into OLMo's FFN ff_proj.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        self.bb_enabled = os.getenv("BB_ENABLE", "0") == "1"
        self.refresh_interval = max(1, _get_int("BB_REFRESH_INTERVAL", 50))
        self.k_min = max(1, _get_int("BB_K_MIN", 64))
        k_max_env = _get_int("BB_K_MAX", -1)
        self.k_max = in_features if k_max_env <= 0 else min(in_features, k_max_env)

        self.ema_beta = _get_float("BB_EMA_BETA", 0.99)
        self.d0 = _get_float("BB_D0", 1.0)  # heuristic: k ~= d0 / p_on
        self.eps = 1e-6
        self.rescale = os.getenv("BB_RESCALE", "1") == "1"

        # Use same device and dtype as weight for buffers
        self.register_buffer("_mask", torch.ones((out_features, in_features), dtype=torch.bool, device=device), persistent=False)
        self.register_buffer("_ema_on", torch.zeros((out_features,), dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("_k", torch.full((out_features,), in_features, dtype=torch.int32, device=device), persistent=False)
        self.register_buffer("_scale", torch.ones((out_features,), dtype=torch.float32, device=device), persistent=False)

        self._calls = 0
        if self.bb_enabled:
            self._refresh_mask(force=True)

    @torch.no_grad()
    def _compute_k(self) -> torch.Tensor:
        # heuristic controller (smoke-test): p_on -> target fan-in
        p = self._ema_on.clamp(min=self.eps, max=1.0)
        k = torch.round(self.d0 / p).to(torch.int32)
        k = torch.clamp(k, min=self.k_min, max=self.k_max)
        return k

    @torch.no_grad()
    def _refresh_mask(self, force: bool = False):
        if not self.bb_enabled:
            return
        k = self._compute_k()
        k_max = int(k.max().item())
        k_max = max(1, min(self.in_features, k_max))

        w = self.weight.detach().abs().to(torch.float32)  # stable
        _, idx = torch.topk(w, k=k_max, dim=1, largest=True, sorted=False)

        new_mask = torch.zeros_like(self._mask)
        # Fill per-row variable-k using the top-k_max indices
        for r in range(self.out_features):
            kr = int(k[r].item())
            if kr <= 0:
                continue
            new_mask[r, idx[r, :kr]] = True

        self._mask = new_mask
        self._k = k

        if self.rescale:
            # variance-ish preservation
            scale = torch.sqrt(torch.tensor(float(self.in_features), device=self._k.device) / self._k.to(torch.float32))
            self._scale = scale
        else:
            self._scale.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bb_enabled:
            return F.linear(x, self.weight, self.bias)

        w = self.weight * self._mask.to(self.weight.dtype)
        y = F.linear(x, w, self.bias)

        # apply per-neuron rescale
        if self.rescale:
            y = y * self._scale.to(y.dtype).view(*([1] * (y.ndim - 1)), -1)

        if self.training:
            with torch.no_grad():
                # EMA of on-rate (pre-activation heuristic)
                dims = tuple(range(y.ndim - 1))  # average over batch/seq dims
                on = (y > 0).to(torch.float32).mean(dim=dims)
                self._ema_on.mul_(self.ema_beta).add_(on * (1.0 - self.ema_beta))

                self._calls += 1
                if (self._calls % self.refresh_interval) == 0:
                    self._refresh_mask()

        return y

    def get_sparsity_stats(self) -> dict:
        """Return current sparsity statistics for logging."""
        if not self.bb_enabled:
            return {"bb_enabled": False}
        
        mask = self._mask.float()
        total = mask.numel()
        alive = mask.sum().item()
        sparsity = 1.0 - (alive / total)
        
        return {
            "bb_enabled": True,
            "sparsity": sparsity,
            "avg_k": self._k.float().mean().item(),
            "min_k": self._k.min().item(),
            "max_k": self._k.max().item(),
            "avg_on_rate": self._ema_on.mean().item(),
        }


def patch_olmo_ffn_with_bb(model: nn.Module, verbose: bool = True):
    """
    Monkey-patch OLMo's FFN layers to use BBLinear.
    This replaces ff_proj (the down-projection) in each transformer block.
    
    Usage:
        from olmo_bb.bb_linear import patch_olmo_ffn_with_bb
        model = OLMo.from_pretrained(...)
        patch_olmo_ffn_with_bb(model)
    """
    import copy
    
    patched_count = 0
    
    for name, module in model.named_modules():
        # OLMo uses "ff_proj" for the down-projection in FFN
        # We want to replace this with BBLinear
        if hasattr(module, 'ff_proj') and isinstance(module.ff_proj, nn.Linear):
            old_linear = module.ff_proj
            
            # Create BBLinear with same dimensions
            new_linear = BBLinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                bias=old_linear.bias is not None,
                device=old_linear.weight.device,
                dtype=old_linear.weight.dtype,
            )
            
            # Copy weights
            new_linear.weight.data.copy_(old_linear.weight.data)
            if old_linear.bias is not None:
                new_linear.bias.data.copy_(old_linear.bias.data)
            
            # Replace
            module.ff_proj = new_linear
            patched_count += 1
            
            if verbose:
                print(f"[BB] Patched {name}.ff_proj -> BBLinear")
    
    if verbose:
        print(f"[BB] Total patched layers: {patched_count}")
    
    return patched_count


def collect_bb_stats(model: nn.Module) -> dict:
    """Collect BB sparsity statistics from all BBLinear layers."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, BBLinear):
            stats[name] = module.get_sparsity_stats()
    return stats


def log_bb_stats(model: nn.Module, step: int, logger=None):
    """Log BB statistics (for wandb or console)."""
    stats = collect_bb_stats(model)
    if not stats:
        return
    
    # Aggregate
    sparsities = [s["sparsity"] for s in stats.values() if s.get("bb_enabled")]
    avg_ks = [s["avg_k"] for s in stats.values() if s.get("bb_enabled")]
    on_rates = [s["avg_on_rate"] for s in stats.values() if s.get("bb_enabled")]
    
    if sparsities:
        summary = {
            "bb/avg_sparsity": sum(sparsities) / len(sparsities),
            "bb/avg_k": sum(avg_ks) / len(avg_ks),
            "bb/avg_on_rate": sum(on_rates) / len(on_rates),
        }
        
        if logger:
            logger.log(summary, step=step)
        else:
            print(f"[BB Step {step}] sparsity={summary['bb/avg_sparsity']:.4f}, "
                  f"avg_k={summary['bb/avg_k']:.1f}, on_rate={summary['bb/avg_on_rate']:.4f}")

