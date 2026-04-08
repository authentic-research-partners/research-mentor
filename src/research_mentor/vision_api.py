"""Vision API backend — OpenAI-compatible vision endpoint.

Sends files (images, PDFs) to a remote vision API for interpretation.
Independent of the main chat LLM backend — has its own credentials.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import time
from pathlib import Path

import openai
from loguru import logger

from research_mentor.config import load_config


async def interpret_file(
    file_path: str | Path,
    *,
    prompt: str | None = None,
    research_context: str | None = None,
) -> str | None:
    """Interpret a file using a remote OpenAI-compatible vision API.

    Reads the file, base64-encodes it, and sends it as a vision message.

    Args:
        file_path: Path to the file to interpret.
        prompt: Pre-built vision prompt (from build_vision_prompt). Takes priority.
        research_context: Deprecated — use prompt instead. Kept for backward compat.

    Returns the model's text interpretation, or None if the file doesn't exist.
    Raises on API errors (fail-fast).
    """

    path = Path(file_path)
    if not path.is_file():
        logger.warning("vision_api: file not found: {}", path)
        return None

    config = load_config()
    vcfg = config.vision

    # Read API key
    key_path = Path(vcfg.api_key_file).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(
            f"Vision API key file not found: {key_path}. "
            "Create it with: echo 'sk-...' > ~/.research-mentor/vision_api_key"
        )
    api_key = (await asyncio.to_thread(key_path.read_text)).strip()
    if not api_key:
        raise ValueError(f"Vision API key file is empty: {key_path}")

    # Determine MIME type and base64-encode
    mime_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_bytes = await asyncio.to_thread(path.read_bytes)
    b64_data = base64.b64encode(file_bytes).decode("ascii")
    data_uri = f"data:{mime_type};base64,{b64_data}"

    if not prompt:
        prompt = (
            (
                "This file is from a research project investigating: "
                f"{research_context}. "
                if research_context
                else ""
            )
            + "Examine this file. Transcribe all text. "
            "Describe diagrams, charts, tables, visual elements."
        )

    client = openai.AsyncOpenAI(
        base_url=vcfg.api_base_url,
        api_key=api_key,
        timeout=vcfg.api_timeout,
    )

    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=vcfg.api_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        max_tokens=vcfg.api_max_tokens,
    )
    elapsed = time.monotonic() - t0

    result = response.choices[0].message.content
    if not result or not result.strip():
        return None

    result = result.strip()
    logger.info(
        "vision_api: interpreted {} ({} chars) in {:.1f}s",
        path.name, len(result), elapsed,
    )

    # Track usage
    usage = response.usage
    from research_mentor.db.crud import store_llm_usage_batch

    try:
        await store_llm_usage_batch(
            [{
                "backend": "api",
                "provider": vcfg.api_provider or "vision-api",
                "model": vcfg.api_model,
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "call_type": "vision",
                "elapsed_seconds": elapsed,
                "error": 0,
            }],
            purpose="vision",
        )
    except Exception:
        logger.warning("Failed to record vision API usage")

    return result
