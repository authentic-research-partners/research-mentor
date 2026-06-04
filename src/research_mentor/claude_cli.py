"""Unified async wrapper for Claude CLI subprocess calls.

Provides consistent error handling, timeout management, flag construction,
and global rate limiting for all code that calls the Claude CLI.

Rate limiting is shared across all ClaudeCLI instances in the same process:
- asyncio.Semaphore caps parallel subprocess count (memory protection)
- MonotonicRateLimiter caps requests/minute (API rate limit protection)

Both limits are configured in config.toml under [llm.claude_cli].

Usage:
    from research_mentor.claude_cli import ClaudeCLI

    cli = ClaudeCLI(model="sonnet", effort="medium", timeout=120)
    result = await cli.arun("Analyze this code...")
    data = await cli.arun_json("Return JSON with fields x, y")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from research_mentor.tools.rate_limiter import MonotonicRateLimiter

if TYPE_CHECKING:
    from asyncio.subprocess import Process

# Tools blocked for automated Claude calls (prevent runaway agents)
DEFAULT_DISALLOWED_TOOLS = [
    "Task",
    "WebSearch",
    "WebFetch",
    "Bash",
    "Write",
    "Edit",
    "NotebookEdit",
]

# All tools blocked (read-only not needed either)
DISALLOWED_TOOLS_ALL = DEFAULT_DISALLOWED_TOOLS + ["Read", "Glob", "Grep"]


# --- Rate limiting defaults (overridden by config.toml [llm.claude_cli]) ---

_DEFAULT_MAX_PARALLEL_PROCESSES = 4
_DEFAULT_MAX_REQUESTS_PER_MINUTE = 20


# --- Global rate limiting state (shared across all ClaudeCLI instances) ---

_global_semaphore: asyncio.Semaphore | None = None
_global_rate_limiter: MonotonicRateLimiter | None = None
_global_limits_loop: asyncio.AbstractEventLoop | None = None


def _load_claude_cli_config() -> tuple[int, int]:
    """Load [llm.claude_cli] config from config.toml.

    Returns:
        Tuple of (max_parallel_processes, max_requests_per_minute).
    """
    try:
        from research_mentor.config import load_config_from_toml

        config = load_config_from_toml()
        cli_config = config.get("llm", {}).get("claude_cli", {})
        max_parallel = cli_config.get(
            "max_parallel_processes",
            _DEFAULT_MAX_PARALLEL_PROCESSES,
        )
        max_rpm = cli_config.get(
            "max_requests_per_minute",
            _DEFAULT_MAX_REQUESTS_PER_MINUTE,
        )
        return int(max_parallel), int(max_rpm)
    except (FileNotFoundError, RuntimeError):
        return _DEFAULT_MAX_PARALLEL_PROCESSES, _DEFAULT_MAX_REQUESTS_PER_MINUTE


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global process semaphore (lazy init, per event loop)."""
    global _global_semaphore, _global_rate_limiter, _global_limits_loop
    loop = asyncio.get_running_loop()
    if _global_semaphore is None or _global_limits_loop is not loop:
        max_parallel, max_rpm = _load_claude_cli_config()
        _global_semaphore = asyncio.Semaphore(max_parallel)
        _global_rate_limiter = MonotonicRateLimiter(max_rpm, 60)
        _global_limits_loop = loop
        logger.info(
            "Claude CLI rate limits initialized: "
            "max_parallel={}, max_rpm={}",
            max_parallel,
            max_rpm,
        )
    return _global_semaphore


def _get_global_rate_limiter() -> MonotonicRateLimiter:
    """Get or create the global rate limiter (lazy init, per event loop)."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _get_global_semaphore()  # initializes both
    else:
        # Ensure limiter belongs to current loop
        loop = asyncio.get_running_loop()
        if _global_limits_loop is not loop:
            _get_global_semaphore()  # reinitializes for new loop
    assert _global_rate_limiter is not None
    return _global_rate_limiter


def reset_global_limits(
    *,
    max_parallel_processes: int | None = None,
    max_requests_per_minute: int | None = None,
) -> None:
    """Reset global rate limits. Useful for tests or runtime reconfiguration."""
    global _global_semaphore, _global_rate_limiter, _global_limits_loop

    if max_parallel_processes is None or max_requests_per_minute is None:
        cfg_parallel, cfg_rpm = _load_claude_cli_config()
        if max_parallel_processes is None:
            max_parallel_processes = cfg_parallel
        if max_requests_per_minute is None:
            max_requests_per_minute = cfg_rpm

    _global_semaphore = asyncio.Semaphore(max_parallel_processes)
    _global_rate_limiter = MonotonicRateLimiter(max_requests_per_minute, 60)
    _global_limits_loop = asyncio.get_running_loop()
    logger.info(
        "Claude CLI rate limits reset: max_parallel={}, max_rpm={}",
        max_parallel_processes,
        max_requests_per_minute,
    )


# --- Error types ---


class ClaudeCLIError(Exception):
    """Base error for Claude CLI operations."""


class ClaudeCLINotFoundError(ClaudeCLIError):
    """Claude CLI binary not found. Install: npm install -g @anthropic-ai/claude-code"""


class ClaudeCLITimeoutError(ClaudeCLIError):
    """Claude CLI process timed out."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Claude CLI timed out after {timeout}s")
        self.timeout = timeout


class ClaudeCLIProcessError(ClaudeCLIError):
    """Claude CLI returned non-zero exit code."""

    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(f"Claude CLI exited with code {returncode}: {stderr[:200]}")
        self.returncode = returncode
        self.stderr = stderr


class ClaudeCLIParseError(ClaudeCLIError):
    """Failed to parse Claude CLI output as JSON."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


# --- Result type ---


@dataclass(frozen=True)
class ClaudeCLIResult:
    """Raw result from a Claude CLI invocation."""

    stdout: str
    stderr: str


# --- Main class ---


class ClaudeCLI:
    """Unified async wrapper for Claude CLI subprocess calls.

    All instances share global rate limits (semaphore + RPM limiter)
    configured in config.toml under [llm.claude_cli].

    Args:
        model: Default Claude model (sonnet, opus, haiku).
        effort: Default thinking effort (low, medium, high).
        timeout: Default timeout in seconds.
    """

    def __init__(
        self,
        model: str = "sonnet",
        effort: str = "medium",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def _build_args(
        self,
        prompt: str,
        *,
        agent: str | None = None,
        system_prompt: str | None = None,
        disallowed_tools: list[str] | None = None,
        effort: str | None = None,
        output_format: str | None = None,
    ) -> list[str]:
        """Build the CLI argument list."""
        args: list[str] = []

        if agent:
            args.extend(["--agent", agent])

        args.extend(["--model", self.model])
        args.extend(["--effort", effort or self.effort])
        args.append("--print")

        if system_prompt:
            args.extend(["--system-prompt", system_prompt])

        if output_format:
            args.extend(["--output-format", output_format])

        if disallowed_tools:
            args.extend(["--disallowed-tools", ",".join(disallowed_tools)])

        if output_format == "json":
            args.extend(["-p", prompt])
        else:
            args.extend(["--", prompt])

        return args

    async def _exec(
        self,
        args: list[str],
        *,
        timeout: float,
        prompt_len: int,
        label: str,
    ) -> ClaudeCLIResult:
        """Execute Claude CLI subprocess with rate limiting and logging."""
        est_input_tokens = prompt_len // 4

        semaphore = _get_global_semaphore()
        rate_limiter = _get_global_rate_limiter()

        async with semaphore:
            await rate_limiter.acquire()

            logger.debug(
                "claude call start | model={} effort={} label={} est_input_tokens=~{}",
                self.model,
                self.effort,
                label,
                est_input_tokens,
            )

            t0 = time.monotonic()
            try:
                proc: Process = await asyncio.create_subprocess_exec(
                    "claude",
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                raise ClaudeCLINotFoundError(
                    "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
                ) from None

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "claude call timeout | label={} elapsed={:.1f}s timeout={}s",
                    label,
                    elapsed,
                    timeout,
                )
                proc.kill()
                await proc.wait()
                raise ClaudeCLITimeoutError(timeout) from None

        elapsed = time.monotonic() - t0
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        if proc.returncode != 0:
            logger.error(
                "claude call failed | label={} code={} elapsed={:.1f}s stderr={}",
                label,
                proc.returncode,
                elapsed,
                stderr[:200],
            )
            raise ClaudeCLIProcessError(proc.returncode or 1, stderr)

        est_output_tokens = len(stdout) // 4
        logger.info(
            "claude call done | model={} label={} elapsed={:.1f}s "
            "est_tokens=~{}in/~{}out output_chars={}",
            self.model,
            label,
            elapsed,
            est_input_tokens,
            est_output_tokens,
            len(stdout),
        )

        return ClaudeCLIResult(stdout=stdout, stderr=stderr)

    async def arun(
        self,
        prompt: str,
        *,
        agent: str | None = None,
        system_prompt: str | None = None,
        disallowed_tools: list[str] | None = None,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> ClaudeCLIResult:
        """Run Claude CLI asynchronously and return raw result.

        Rate limiting is applied automatically via global semaphore + RPM limiter.
        """
        args = self._build_args(
            prompt,
            agent=agent,
            system_prompt=system_prompt,
            disallowed_tools=disallowed_tools,
            effort=effort,
        )
        label = agent or "text"
        return await self._exec(
            args,
            timeout=timeout or self.timeout,
            prompt_len=len(prompt),
            label=label,
        )

    async def arun_json(
        self,
        prompt: str,
        *,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run Claude CLI with --output-format json and parse the response.

        Handles two-layer JSON parsing:
        1. Outer envelope: {"result": "..."} from --output-format json
        2. Inner JSON object extracted from the result string
        """
        args = self._build_args(
            prompt,
            effort=effort,
            output_format="json",
        )
        result = await self._exec(
            args,
            timeout=timeout or self.timeout,
            prompt_len=len(prompt),
            label="json",
        )
        stdout = result.stdout

        # Layer 1: parse outer JSON envelope
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIParseError(f"Failed to parse outer JSON envelope: {e}", stdout) from e

        content = response.get("result", response.get("content", ""))

        if isinstance(content, dict):
            return content

        if not isinstance(content, str):
            raise ClaudeCLIParseError(f"Unexpected content type: {type(content)}", stdout)

        # Layer 2: extract inner JSON object from string
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ClaudeCLIParseError(f"No JSON object found in response: {content[:200]}", stdout)

        try:
            return json.loads(json_match.group())  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise ClaudeCLIParseError(f"Failed to parse inner JSON: {e}", stdout) from e
