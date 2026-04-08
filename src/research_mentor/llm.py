"""Unified LLM factory for Personal Research Mentor.

Two backends:
- vLLM: OpenAI AsyncClient with response_format json_schema for constrained decoding
- Claude CLI: Subprocess wrapper (chat_claude_cli.py)

Supports both thinking models (Qwen3 with <think> tags) and non-thinking models
(Gemma3). Controlled by config.vllm.thinking_tag (per model profile).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from loguru import logger
from pydantic import BaseModel, ValidationError

from research_mentor.config import load_config

# --- Token tracking (accumulated across a graph invocation) ---

_call_stats: dict[str, int | float] = {
    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "elapsed": 0.0,
}


def reset_call_stats() -> None:
    """Reset accumulated call stats. Call before each graph invocation."""
    _call_stats["calls"] = 0
    _call_stats["prompt_tokens"] = 0
    _call_stats["completion_tokens"] = 0
    _call_stats["elapsed"] = 0.0


def get_call_stats() -> dict[str, int | float]:
    """Get accumulated call stats for the current graph invocation."""
    return dict(_call_stats)


# --- Thinking presets (Qwen3 only — ignored for non-thinking models like Gemma) ---
#
# Presets pair reasoning_effort (soft hint) with thinking budget (hard token cap).
# The sizing rule: max_tokens >= thinking_tokens + response_tokens.
# Valley of death: budgets 512-1024 cause MORE truncation than 200 or 2048+.
# See docs/philosophy/small-model-thinking-presets.md for rationale.

type ThinkingPreset = dict[str, int | str]

THINKING_PRESETS: dict[str, ThinkingPreset] = {
    # Simple boolean/enum output (<50 response tokens). No thinking needed.
    "off": {"thinking_budget": 0, "reasoning_effort": "low"},
    # Short extraction or classification (50-200 response tokens).
    # Budget 200 is below the valley of death — model barely thinks,
    # most tokens go to the response. Reliable and fast.
    "low": {"thinking_budget": 200, "reasoning_effort": "low"},
    # Multi-field analysis, scored assessments (200-500 response tokens).
    # Medium effort is more reliable than low for classification — low sometimes
    # generates very long reasoning, medium stays concise.
    "medium": {"thinking_budget": 200, "reasoning_effort": "medium"},
    # Complex generation: lists of objects, deep analysis (500+ response tokens).
    # Budget 2048 clears the valley of death with room for deep reasoning.
    "high": {"thinking_budget": 2048, "reasoning_effort": "high"},
}


# --- Test-mode guard ---
#
# When _test_mode is True, structured_call() and get_chat_llm() raise
# RuntimeError instead of making real LLM calls. This catches unmocked
# call paths in tests — any code that reaches these functions without a
# mock in place gets a loud failure with a helpful message.
#
# Enable in conftest: monkeypatch.setattr(research_mentor.llm, "_test_mode", True)
# Since the function body reads this variable at call time, it works even
# when other modules have already imported structured_call by name.

_test_mode: bool = False


# --- Public API ---


def get_chat_llm(**overrides: Any) -> BaseChatModel:
    """Get a chat LLM for free-text generation (guides, experts, presenter)."""
    if _test_mode:
        raise RuntimeError(
            "get_chat_llm() called in test mode without a mock. "
            "Patch get_chat_llm in the calling module to prevent real LLM calls."
        )
    config = load_config()
    backend = overrides.pop("backend", config.backend)

    if backend == "claude-cli":
        from research_mentor.chat_claude_cli import ChatClaudeCLI, resolve_agent_path

        agent_name = overrides.get("agent")
        agent_path = resolve_agent_path(agent_name) if agent_name else None

        return ChatClaudeCLI(
            model_name=overrides.get("model", config.claude_cli.model),
            effort=overrides.get("effort", config.claude_cli.effort),
            timeout=overrides.get("timeout", config.claude_cli.timeout),
            agent=agent_path,
        )

    if backend == "vllm":
        return VLLMChatModel(
            api_base=overrides.get("api_base", config.vllm.api_base),
            model_name=overrides.get("model", config.vllm.model),
            temperature=overrides.get("temperature", config.vllm.temperature),
            max_completion_tokens=overrides.get(
                "max_completion_tokens", config.vllm.max_completion_tokens,
            ),
            top_p=config.vllm.top_p,
            top_k=config.vllm.top_k,
        )

    if backend == "api":
        api_key = _read_api_key_sync()
        return VLLMChatModel(
            api_base=overrides.get("api_base", config.api.base_url),
            api_key=api_key,
            model_name=overrides.get("model", config.api.model),
            temperature=overrides.get("temperature", config.api.temperature),
            max_completion_tokens=overrides.get(
                "max_completion_tokens", config.api.max_completion_tokens,
            ),
            top_p=config.api.top_p,
            top_k=0,  # not used by remote APIs, set to 0 to skip extra_body
        )

    raise ValueError(
        f"Unknown LLM backend: {backend!r}. Use 'claude-cli', 'vllm', or 'api'."
    )


def get_structured_llm(**overrides: Any) -> BaseChatModel:
    """Alias for compatibility. Use structured_call() for JSON output."""
    return get_chat_llm(**overrides)


async def structured_call[T: BaseModel](
    schema: type[T],
    messages: list[BaseMessage],
    *,
    thinking: str,
    **overrides: Any,
) -> T:
    """Structured LLM call returning a validated Pydantic model.

    vLLM: Uses response_format json_schema for constrained decoding.
          Prompt example + json_schema constraint ensures correct output.
    Claude CLI: Prompt engineering + JSON parsing.

    Raises RuntimeError in test mode (_test_mode=True) if called without a mock.
    Args:
        thinking: Preset name from THINKING_PRESETS ("off", "low", "medium", "high").
            Only affects Qwen3 thinking models — ignored for Gemma and Claude.
    """
    if thinking not in THINKING_PRESETS:
        raise ValueError(
            f"Unknown thinking preset: {thinking!r}. "
            f"Use one of: {', '.join(THINKING_PRESETS)}"
        )
    if _test_mode:
        raise RuntimeError(
            f"structured_call({schema.__name__}) called in test mode without a mock. "
            f"Patch structured_call in the calling module to prevent real LLM calls."
        )
    preset = THINKING_PRESETS[thinking]
    thinking_budget: int = preset["thinking_budget"]  # type: ignore[assignment]
    reasoning_effort: str = preset["reasoning_effort"]  # type: ignore[assignment]
    config = load_config()
    backend = overrides.pop("backend", config.backend)
    json_schema = schema.model_json_schema()

    # Build JSON instruction with example (not raw schema — models copy schema structure)
    example = _schema_to_example(json_schema)
    json_instruction = (
        "\n\nRespond with ONLY a valid JSON object like this example "
        "(no other text before or after the JSON):\n"
        + json.dumps(example, indent=2)
    )

    if backend in ("vllm", "api"):
        oai_messages = _to_openai_messages(messages)

        # Add JSON instruction to system prompt
        if oai_messages and oai_messages[0]["role"] == "system":
            oai_messages[0]["content"] += json_instruction

        # Build overrides based on backend
        if backend == "api":
            api_key = _read_api_key_sync()
            overrides.setdefault("api_base", config.api.base_url)
            overrides.setdefault("api_key", api_key)
            overrides.setdefault("model", config.api.model)
            overrides.setdefault("temperature", config.api.temperature)
            overrides.setdefault("max_completion_tokens", config.api.max_completion_tokens)
            overrides.setdefault("timeout", config.api.timeout)
            extra_body: dict[str, Any] | None = None
            thinking_tag = config.api.thinking_tag
        else:
            extra_body = {"top_k": config.vllm.top_k}
            thinking_tag = config.vllm.thinking_tag
            if thinking_tag:
                extra_body["thinking"] = {"budget": thinking_budget}
                overrides.setdefault("reasoning_effort", reasoning_effort)

                # Sizing rule: max_tokens must exceed thinking budget, otherwise the
                # model exhausts all tokens on reasoning and never produces JSON.
                # See docs/philosophy/small-model-thinking-presets.md.
                max_tok = overrides.get(
                    "max_completion_tokens", config.vllm.max_completion_tokens,
                )
                if thinking_budget > 0 and max_tok <= thinking_budget:
                    overrides["max_completion_tokens"] = thinking_budget + max_tok
                    logger.debug(
                        "Sizing rule: bumped max_completion_tokens {} → {} "
                        "(thinking_budget={})",
                        max_tok, overrides["max_completion_tokens"],
                        thinking_budget,
                    )

        raw, _usage = await _vllm_complete(
            oai_messages,
            label=f"structured:{schema.__name__}",
            # response_format for constrained JSON decoding (OpenAI-compatible API)
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                },
            },
            extra_body=extra_body,
            **overrides,
        )
        cleaned = _strip_thinking(raw, thinking_tag)
        cleaned = _strip_markdown_fences(cleaned)

        if not cleaned:
            raise RuntimeError(
                f"vLLM structured call for {schema.__name__} produced no JSON — "
                + (
                    "model likely exhausted max_tokens on thinking. "
                    if thinking_tag
                    else "model produced empty output. "
                )
                + f"Raw output ends with: ...{raw[-200:]}"
            )

        data = json.loads(cleaned)
        max_retries = config.structured_call_max_retries
        last_exc: ValidationError | None = None
        for attempt in range(max_retries + 1):
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                logger.warning(
                    "structured_call validation retry {}/{} | schema={} backend={} "
                    "error={}",
                    attempt + 1, max_retries, schema.__name__, backend,
                    str(exc)[:200],
                )
                oai_messages.append({"role": "assistant", "content": cleaned})
                oai_messages.append({
                    "role": "user",
                    "content": (
                        f"Your JSON had validation errors:\n{exc}\n\n"
                        "Return the COMPLETE fixed JSON object with ALL required fields."
                    ),
                })
                raw_r, _ = await _vllm_complete(
                    oai_messages,
                    label=f"structured:{schema.__name__}:validation_retry",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": json_schema,
                            "strict": True,
                        },
                    },
                    extra_body=extra_body,
                    **overrides,
                )
                cleaned = _strip_thinking(raw_r, thinking_tag)
                cleaned = _strip_markdown_fences(cleaned)
                data = json.loads(cleaned)
        raise last_exc  # type: ignore[misc]

    else:
        # Claude CLI: add JSON instruction to system prompt (no constrained decoding)
        llm = get_chat_llm(backend=backend, **overrides)

        enhanced = list(messages)
        for i, msg in enumerate(enhanced):
            if isinstance(msg, SystemMessage):
                enhanced[i] = SystemMessage(
                    content=str(msg.content) + json_instruction
                )
                break

        t0 = time.monotonic()
        result = await llm.ainvoke(enhanced)
        raw = str(result.content)
        elapsed = time.monotonic() - t0

        _call_stats["calls"] += 1
        _call_stats["elapsed"] += elapsed

        # Estimate tokens for Claude CLI (subscription, no real token count)
        prompt_est = sum(len(str(m.content)) for m in enhanced) // 4
        completion_est = len(raw) // 4
        _call_stats["prompt_tokens"] += prompt_est
        _call_stats["completion_tokens"] += completion_est

        _record_usage(
            backend="claude-cli",
            provider="anthropic",
            model=config.claude_cli.model,
            prompt_tokens=prompt_est,
            completion_tokens=completion_est,
            call_type="structured",
            elapsed_seconds=round(elapsed, 2),
        )

        logger.info(
            "structured_call done | schema={} backend={} elapsed={:.1f}s",
            schema.__name__, backend, elapsed,
        )

        cleaned = _strip_thinking(raw)
        cleaned = _strip_markdown_fences(cleaned)

        if not cleaned:
            raise RuntimeError(
                f"structured_call for {schema.__name__} produced no JSON — "
                f"raw output: {raw[:200]}"
            )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Retry once: feed back the bad output and insist on JSON
            logger.warning(
                "structured_call retry | schema={} first attempt returned non-JSON: {}",
                schema.__name__, cleaned[:100],
            )
            retry_messages = enhanced + [
                AIMessage(content=raw),
                HumanMessage(
                    content="Your response above was not valid JSON. "
                    "You MUST respond with ONLY a valid JSON object matching the "
                    "requested schema — no other text, no explanation, no greeting."
                ),
            ]

            t0 = time.monotonic()
            result = await llm.ainvoke(retry_messages)
            raw_retry = str(result.content)
            elapsed = time.monotonic() - t0

            _call_stats["calls"] += 1
            _call_stats["elapsed"] += elapsed

            retry_prompt_est = sum(len(str(m.content)) for m in retry_messages) // 4
            retry_completion_est = len(raw_retry) // 4
            _call_stats["prompt_tokens"] += retry_prompt_est
            _call_stats["completion_tokens"] += retry_completion_est
            _record_usage(
                backend="claude-cli",
                provider="anthropic",
                model=config.claude_cli.model,
                prompt_tokens=retry_prompt_est,
                completion_tokens=retry_completion_est,
                call_type="structured",
                elapsed_seconds=round(elapsed, 2),
            )

            cleaned = _strip_thinking(raw_retry)
            cleaned = _strip_markdown_fences(cleaned)

            if not cleaned:
                raise RuntimeError(
                    f"structured_call for {schema.__name__} produced no JSON on retry — "
                    f"raw output: {raw_retry[:200]}"
                )

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"structured_call for {schema.__name__} returned non-JSON after retry — "
                    f"cleaned output: {cleaned[:200]}"
                ) from exc

        max_retries = config.structured_call_max_retries
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return schema.model_validate(data)
            except ValidationError as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                logger.warning(
                    "structured_call validation retry {}/{} | schema={} "
                    "backend=claude-cli error={}",
                    attempt + 1, max_retries, schema.__name__, str(exc)[:200],
                )
                enhanced.append(AIMessage(content=cleaned))
                enhanced.append(HumanMessage(
                    content=(
                        f"Your JSON had validation errors:\n{exc}\n\n"
                        "Return the COMPLETE fixed JSON object with ALL required "
                        "fields — no other text."
                    ),
                ))
                t0 = time.monotonic()
                result = await llm.ainvoke(enhanced)
                raw_v = str(result.content)
                elapsed = time.monotonic() - t0
                _call_stats["calls"] += 1
                _call_stats["elapsed"] += elapsed
                cleaned = _strip_thinking(raw_v)
                cleaned = _strip_markdown_fences(cleaned)
                data = json.loads(cleaned)
        raise last_exc  # type: ignore[misc]


# --- vLLM shared completion (DRY: used by both VLLMChatModel and structured_call) ---


async def _vllm_complete(
    oai_messages: list[dict[str, str]],
    *,
    label: str = "chat",
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    **overrides: Any,
) -> tuple[str, Any]:
    """Call vLLM via openai SDK. Returns (raw_content, usage).

    Single place for all vLLM API calls — logging, timing, token tracking.
    """
    import httpx
    from openai import AsyncOpenAI

    config = load_config()
    client = AsyncOpenAI(
        base_url=overrides.get("api_base", config.vllm.api_base),
        api_key=overrides.get("api_key", "not-needed"),
        timeout=httpx.Timeout(
            overrides.get("timeout", config.vllm.timeout), connect=10.0,
        ),
    )

    t0 = time.monotonic()
    logger.debug("vLLM call start | label={} model={}", label, config.vllm.model)

    kwargs: dict[str, Any] = {
        "model": overrides.get("model", config.vllm.model),
        "messages": oai_messages,
        "temperature": overrides.get("temperature", config.vllm.temperature),
        "top_p": config.vllm.top_p,
        "max_tokens": overrides.get("max_completion_tokens", config.vllm.max_completion_tokens),
    }
    if response_format:
        kwargs["response_format"] = response_format
    if extra_body:
        kwargs["extra_body"] = extra_body
    if overrides.get("reasoning_effort"):
        kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}),
                                "reasoning_effort": overrides["reasoning_effort"]}

    completion = await client.chat.completions.create(**kwargs)

    raw = completion.choices[0].message.content or ""
    elapsed = time.monotonic() - t0
    usage = completion.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    _call_stats["calls"] += 1
    _call_stats["prompt_tokens"] += prompt_tokens
    _call_stats["completion_tokens"] += completion_tokens
    _call_stats["elapsed"] += elapsed

    model_name = kwargs.get("model", config.vllm.model)
    logger.info(
        "vLLM call done | label={} model={} elapsed={:.1f}s tokens={}in/{}out",
        label, model_name, elapsed, prompt_tokens, completion_tokens,
    )

    # Record usage for persistence
    be = config.backend
    prov = config.api.provider if be == "api" else be
    _record_usage(
        backend=be,
        provider=prov,
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        call_type="structured" if "structured:" in label else "chat",
        elapsed_seconds=round(elapsed, 2),
    )

    return raw, usage


# --- vLLM LangChain wrapper (uses _vllm_complete) ---


class VLLMChatModel(BaseChatModel):
    """LangChain chat model backed by vLLM or remote OpenAI-compatible API."""

    # Populated by get_chat_llm() from config.toml — defaults are fallbacks only
    api_base: str = "http://localhost:5001/v1"
    api_key: str = "not-needed"
    model_name: str = "default"
    temperature: float = 0.7
    max_completion_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 40

    @property
    def _llm_type(self) -> str:
        return "vllm"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, **kwargs)
        )

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: Any = None, **kwargs: Any) -> ChatResult:
        oai_messages = _to_openai_messages(messages)

        extra: dict[str, Any] = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
        }

        raw, _usage = await _vllm_complete(
            oai_messages, label="chat", **extra,
        )

        config = load_config()
        thinking_tag = (
            config.api.thinking_tag if config.backend == "api" else config.vllm.thinking_tag
        )
        text = _strip_thinking(raw, thinking_tag)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: Any = None, **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream tokens from vLLM via OpenAI streaming API.

        Buffers content during <think>...</think> tags and only yields
        post-thinking content token by token.
        """
        import httpx
        from openai import AsyncOpenAI

        config = load_config()
        timeout_val = (
            config.api.timeout if config.backend == "api" else config.vllm.timeout
        )
        client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout_val, connect=10.0),
        )
        oai_messages = _to_openai_messages(messages)

        t0 = time.monotonic()
        logger.debug("vLLM stream start | model={}", self.model_name)

        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": oai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_completion_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.top_k > 0:
            create_kwargs["extra_body"] = {"top_k": self.top_k}

        stream = await client.chat.completions.create(
            **create_kwargs,
        )

        thinking_tag = (
            config.api.thinking_tag if config.backend == "api" else config.vllm.thinking_tag
        )
        open_tag = f"<{thinking_tag}>" if thinking_tag else ""
        close_tag = f"</{thinking_tag}>" if thinking_tag else ""
        buffer = ""
        in_thinking = False
        thinking_done = not thinking_tag  # skip buffering for non-thinking models
        total_completion_tokens = 0

        async for chunk in stream:
            # Track usage from the final chunk
            if chunk.usage:
                _call_stats["calls"] += 1
                _call_stats["prompt_tokens"] += chunk.usage.prompt_tokens
                _call_stats["completion_tokens"] += chunk.usage.completion_tokens
                total_completion_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = delta.content or ""
            if not content:
                continue

            # Already past thinking block (or non-thinking model) — yield directly
            if thinking_done:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
                continue

            buffer += content

            if not in_thinking:
                if buffer.lstrip().startswith(open_tag):
                    in_thinking = True
                else:
                    # No thinking tags at all — yield buffer and switch to direct mode
                    thinking_done = True
                    yield ChatGenerationChunk(message=AIMessageChunk(content=buffer))
                    buffer = ""
                    continue

            # Inside thinking block — check for closing tag
            if close_tag in buffer:
                thinking_done = True
                _, _, after = buffer.partition(close_tag)
                after = after.strip()
                if after:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=after))
                buffer = ""

        elapsed = time.monotonic() - t0
        _call_stats["elapsed"] += elapsed
        logger.info(
            "vLLM stream done | model={} elapsed={:.1f}s completion_tokens={}",
            self.model_name, elapsed, total_completion_tokens,
        )

        # Record usage for persistence
        be = config.backend
        prov = config.api.provider if be == "api" else be
        prompt_tokens = (
            chunk.usage.prompt_tokens if chunk and chunk.usage else 0
        )
        _record_usage(
            backend=be,
            provider=prov,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
            call_type="chat",
            elapsed_seconds=round(elapsed, 2),
        )


# --- API key helper ---


def _read_api_key_sync() -> str:
    """Read API key from the configured file. Raises if not found."""
    from pathlib import Path

    config = load_config()
    key_path = Path(config.api.api_key_file).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(
            f"API key file not found: {key_path}. "
            "Create it with: echo 'sk-...' > ~/.research-mentor/api_key"
        )
    key = key_path.read_text().strip()
    if not key:
        raise ValueError(f"API key file is empty: {key_path}")
    return key


# --- Usage tracking via ContextVar (write-through to DB) ---
#
# Each LLM call records usage directly to the DB with purpose/session/project
# from the current context. No pending list, no manual draining.
#
# Entry points set context via set_usage_context() before calling LLM functions.
# asyncio.create_task() inherits context vars automatically.

_usage_purpose: ContextVar[str] = ContextVar("llm_usage_purpose")
_usage_session_id: ContextVar[str | None] = ContextVar("llm_usage_session_id", default=None)
_usage_project_id: ContextVar[str | None] = ContextVar("llm_usage_project_id", default=None)


def set_usage_context(
    *,
    purpose: str,
    session_id: str | None = None,
    project_id: str | None = None,
) -> None:
    """Set the usage tracking context for the current async task.

    Call this at the entry point of each feature. All LLM calls within
    this task (and child tasks) will be tagged with this context automatically.

    Standard purpose values:
        chat                    — main mentoring chat
        workshop_phenomenon     — Question Workshop phenomenon generation
        workshop_claims         — Question Workshop claims
        workshop_hypothesis     — Question Workshop Hypothesis interactive
        assessment              — project / student assessments
        title                   — auto-generated session titles
        vision                  — artifact vision (image/PDF)
    """
    _usage_purpose.set(purpose)
    _usage_session_id.set(session_id)
    _usage_project_id.set(project_id)


def _record_usage(
    *,
    backend: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    call_type: str = "chat",
    elapsed_seconds: float = 0.0,
    error: bool = False,
) -> None:
    """Write a usage record directly to DB (fire-and-forget).

    Reads purpose/session/project from ContextVar — no manual flush needed.
    """
    record = {
        "backend": backend,
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "call_type": call_type,
        "elapsed_seconds": elapsed_seconds,
        "error": int(error),
    }
    try:
        purpose = _usage_purpose.get()
    except LookupError:
        logger.warning(
            "LLM call without set_usage_context() — purpose unknown. "
            "Call set_usage_context(purpose=...) at the entry point.",
        )
        purpose = "unknown"
    session_id = _usage_session_id.get()
    project_id = _usage_project_id.get()

    async def _persist() -> None:
        try:
            from research_mentor.db.crud import store_llm_usage_batch

            await store_llm_usage_batch(
                [record],
                session_id=session_id,
                project_id=project_id,
                purpose=purpose,
            )
        except Exception as exc:
            logger.warning(
                "Failed to persist usage record (purpose={}): {}",
                purpose, exc,
            )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist())
    except RuntimeError:
        # No event loop (e.g. during sync tests) — skip
        pass


# --- Helpers ---


def _schema_to_example(
    schema: dict[str, Any],
    root_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert JSON schema to a concrete example with placeholder values.

    Models tend to echo the schema structure when shown raw schemas.
    Showing a concrete example produces correct output.
    """
    if root_schema is None:
        root_schema = schema
    props = schema.get("properties", {})
    example: dict[str, Any] = {}
    for key, prop in props.items():
        prop_type = prop.get("type", "string")
        if prop_type == "string":
            example[key] = "..."
        elif prop_type == "number":
            example[key] = 0.0
        elif prop_type == "integer":
            example[key] = 0
        elif prop_type == "boolean":
            example[key] = False
        elif prop_type == "array":
            items = prop.get("items", {})
            if "$ref" in items:
                ref_name = items["$ref"].split("/")[-1]
                ref_schema = root_schema.get("$defs", {}).get(ref_name, {})
                if ref_schema:
                    example[key] = [_schema_to_example(ref_schema, root_schema)]
                else:
                    example[key] = [{}]
            elif items.get("type") == "string":
                example[key] = ["..."]
            else:
                example[key] = []
        elif prop_type == "object":
            example[key] = {}
        else:
            example[key] = None
    return example


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages to OpenAI format.

    Ensures role alternation (user/assistant) required by vLLM and
    OpenAI-compatible APIs. Consecutive same-role messages are merged
    with newline separators. System messages are always kept at the start.
    """
    raw: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            raw.append({"role": "system", "content": str(msg.content)})
        elif isinstance(msg, HumanMessage):
            raw.append({"role": "user", "content": str(msg.content)})
        else:
            raw.append({"role": "assistant", "content": str(msg.content)})

    # Merge consecutive same-role messages to ensure alternation
    merged: list[dict[str, str]] = []
    for entry in raw:
        if merged and entry["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + entry["content"]
        else:
            merged.append(entry)

    # vLLM/OpenAI require first non-system message to be "user".
    # If history starts with an assistant message (prior AI response),
    # fold it into the system prompt to preserve context.
    if len(merged) >= 2 and merged[0]["role"] == "system" and merged[1]["role"] == "assistant":
        merged[0]["content"] += "\n\nYour previous response:\n" + merged[1]["content"]
        merged.pop(1)

    return merged


def _strip_thinking(text: str, tag: str = "think") -> str:
    """Strip thinking tags (e.g. <think>...</think>) from model output."""
    if not tag:
        return text.strip()
    open_tag = f"<{tag}>"
    pattern = rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()
    if text.strip().startswith(open_tag):
        return ""
    return text.strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
