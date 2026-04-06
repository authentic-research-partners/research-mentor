"""vLLM model switching: process management + status tracking.

Provides a background task that stops the running vLLM server, updates config,
starts a new server with the selected model profile, and polls until ready.
"""

from __future__ import annotations

import asyncio
import subprocess
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from loguru import logger


class SwitchStatus(StrEnum):
    IDLE = "idle"
    STOPPING = "stopping"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


_state: dict[str, Any] = {"status": SwitchStatus.IDLE, "model": None, "error": None}


def get_switch_status() -> dict[str, Any]:
    """Return current switch state."""
    return dict(_state)


def get_available_profiles() -> dict[str, dict[str, Any]]:
    """Return model profiles from bundled config."""
    from research_mentor.config import _VLLM_DEFAULTS

    return dict(_VLLM_DEFAULTS.get("models", {}))


def _kill_vllm_processes() -> bool:
    """Kill any running vLLM server processes. Returns True if something was killed."""
    result = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True)
    if result.returncode == 0:
        logger.info("Stopping existing vLLM server processes")
        subprocess.run(["pkill", "-f", "vllm serve"])
        return True
    return False


async def switch_model(profile_name: str) -> None:
    """Background task: kill vLLM, update config, start new vLLM, poll until ready."""
    import shutil

    from research_mentor.config import (
        _VLLM_DEFAULTS,
        VLLMConfig,
        _resolve_vllm_from_toml,
        invalidate_config_cache,
        load_config,
        set_runtime_override,
        update_user_config_active_model,
    )

    models = _VLLM_DEFAULTS.get("models", {})
    if profile_name not in models:
        _state["status"] = SwitchStatus.FAILED
        _state["error"] = f"Unknown model profile: {profile_name}"
        return

    try:
        # --- Stop existing vLLM ---
        _state["status"] = SwitchStatus.STOPPING
        _state["model"] = profile_name
        _state["error"] = None

        killed = _kill_vllm_processes()
        if killed:
            await asyncio.sleep(3)

        # --- Update config on disk ---
        update_user_config_active_model(profile_name)
        invalidate_config_cache()

        # --- Apply runtime override so load_config() picks up new profile immediately ---
        vllm_raw: dict[str, Any] = {**_VLLM_DEFAULTS, "active_model": profile_name}
        resolved = _resolve_vllm_from_toml(vllm_raw)
        set_runtime_override("vllm", VLLMConfig(**resolved))

        # --- Start new vLLM process ---
        _state["status"] = SwitchStatus.STARTING

        if not shutil.which("vllm"):
            _state["status"] = SwitchStatus.FAILED
            _state["error"] = "vllm command not found. Install with: pip install vllm"
            return

        config = load_config()
        vllm_cfg = config.vllm
        profile = models[profile_name]
        hf_model = profile["hf_model"]
        parsed_url = urlparse(vllm_cfg.api_base)
        serve_port = parsed_url.port or 5001
        gpu_mem = vllm_cfg.gpu_memory_utilization
        context_len = vllm_cfg.max_input_tokens + vllm_cfg.max_completion_tokens + 256

        cmd = [
            "vllm", "serve",
            "--host", "0.0.0.0",
            "--port", str(serve_port),
            "--model", hf_model,
            "--served-model-name", profile_name,
            "--trust-remote-code",
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-model-len", str(context_len),
            "--kv-cache-dtype", "fp8",
            "--enable-chunked-prefill",
        ]

        logger.info("Starting vLLM: {}", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # --- Poll until ready ---
        poll_url = f"http://localhost:{serve_port}/v1/models"
        max_wait = 120
        poll_interval = 3
        elapsed = 0

        while elapsed < max_wait:
            # Check if process crashed
            if proc.poll() is not None:
                _state["status"] = SwitchStatus.FAILED
                _state["error"] = f"vLLM process exited with code {proc.returncode}"
                return

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            try:
                import httpx

                resp = httpx.get(poll_url, timeout=3.0)
                if resp.status_code == 200:
                    logger.info("vLLM ready with model profile: {}", profile_name)
                    _state["status"] = SwitchStatus.READY
                    _state["error"] = None
                    _schedule_auto_reset()
                    return
            except Exception:
                logger.debug("vLLM not ready yet at {}", poll_url)
                continue

        # Timeout
        _state["status"] = SwitchStatus.FAILED
        _state["error"] = f"vLLM failed to become ready within {max_wait}s"
        _schedule_auto_reset()

    except Exception as e:
        logger.exception("Model switch failed")
        _state["status"] = SwitchStatus.FAILED
        _state["error"] = str(e)
        _schedule_auto_reset()


def _schedule_auto_reset() -> None:
    """Reset state to idle after 10s (so UI shows the result briefly)."""

    async def _reset() -> None:
        await asyncio.sleep(10)
        _state["status"] = SwitchStatus.IDLE
        _state["model"] = None
        _state["error"] = None

    asyncio.create_task(_reset())
