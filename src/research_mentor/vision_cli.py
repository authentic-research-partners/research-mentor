"""Vision CLI backend — Claude Code CLI for file interpretation.

Uses the Claude CLI to read and interpret uploaded files.
Claude has native support for images and PDFs via the Read tool.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from research_mentor.claude_cli import DEFAULT_DISALLOWED_TOOLS, ClaudeCLI
from research_mentor.config import load_config


async def interpret_file(
    file_path: str | Path,
    *,
    prompt: str | None = None,
    research_context: str | None = None,
) -> str | None:
    """Interpret a file using Claude Code CLI.

    Claude reads the file via its Read tool and provides text extraction
    with descriptions of visual elements.

    Args:
        file_path: Path to the file to interpret.
        prompt: Pre-built vision prompt (from build_vision_prompt). Takes priority.
        research_context: Deprecated — use prompt instead. Kept for backward compat.

    Returns the interpretation text, or None if the file doesn't exist.
    Raises on CLI errors (fail-fast).
    """

    path = Path(file_path).resolve()
    if not path.is_file():
        logger.warning("vision_cli: file not found: {}", path)
        return None

    config = load_config()
    cli = ClaudeCLI(
        model=config.claude_cli.model,
        effort="low",
        timeout=config.claude_cli.timeout,
    )

    if prompt:
        cli_prompt = f"Read the file at {path}.\n\n{prompt}"
    else:
        context_prefix = (
            f"This file is from a research project investigating: {research_context}. "
            if research_context
            else ""
        )
        cli_prompt = (
            f"{context_prefix}"
            f"Read the file at {path} and provide detailed text extraction. "
            "Transcribe all text. Describe diagrams, charts, tables, visual elements. "
            "Return ONLY the content."
        )

    t0 = time.monotonic()
    result = await cli.arun(
        cli_prompt,
        disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
    )
    elapsed = time.monotonic() - t0

    text = result.stdout.strip()
    if not text:
        return None

    logger.info(
        "vision_cli: interpreted {} ({} chars) in {:.1f}s",
        path.name, len(text), elapsed,
    )

    # Track usage (estimated tokens, same as ChatClaudeCLI)
    from research_mentor.db.crud import store_llm_usage_batch

    est_in = len(cli_prompt) // 4
    est_out = len(text) // 4
    try:
        await store_llm_usage_batch(
            [{
                "backend": "claude-cli",
                "provider": "anthropic",
                "model": config.claude_cli.model,
                "prompt_tokens": est_in,
                "completion_tokens": est_out,
                "call_type": "vision",
                "elapsed_seconds": elapsed,
                "error": 0,
            }],
            purpose="vision",
        )
    except Exception:
        logger.warning("Failed to record vision CLI usage")

    return text
