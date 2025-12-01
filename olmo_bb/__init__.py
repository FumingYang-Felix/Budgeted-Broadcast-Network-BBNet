"""
OLMo + Budgeted Broadcast (BB) Integration Package
https://arxiv.org/pdf/2510.01263
"""

from .bb_linear import (
    BBLinear,
    patch_olmo_ffn_with_bb,
    collect_bb_stats,
    log_bb_stats,
)

__all__ = [
    "BBLinear",
    "patch_olmo_ffn_with_bb",
    "collect_bb_stats",
    "log_bb_stats",
]

