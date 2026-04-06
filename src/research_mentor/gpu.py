"""NVIDIA GPU detection for vLLM compatibility checks."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUInfo:
    """Summary of the best available NVIDIA GPU."""

    name: str
    vram_mb: int
    compute_capability: float  # e.g. 8.9

    @property
    def fp8_native(self) -> bool:
        """FP8 native support requires compute capability >= 8.9 (Ada Lovelace+)."""
        return self.compute_capability >= 8.9

    @property
    def vllm_compatible(self) -> bool:
        """vLLM needs >= 16 GB VRAM."""
        return self.vram_mb >= 16000


def detect_gpu() -> GPUInfo | None:
    """Detect the best NVIDIA GPU. Returns None if no suitable GPU found."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        best: GPUInfo | None = None
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[0]
            vram_mb = int(parts[1])
            compute_cap = float(parts[2])
            gpu = GPUInfo(name=name, vram_mb=vram_mb, compute_capability=compute_cap)
            if best is None or vram_mb > best.vram_mb:
                best = gpu
        return best
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return None
