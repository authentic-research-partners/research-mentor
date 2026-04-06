"""LangChain BaseChatModel wrapping the Claude Code CLI.

Allows the Claude CLI backend to be used anywhere LangChain expects a
chat model — agent graphs, structured output, chains, etc.

Identity is controlled via Claude Code's --agent flag. Each feature
provides an agent definition file (.md with frontmatter) that replaces
Claude Code's default identity entirely. Agent files live in
research_mentor/agents/ and are resolved by name via resolve_agent_path().

Usage:
    from research_mentor.chat_claude_cli import ChatClaudeCLI, resolve_agent_path

    llm = ChatClaudeCLI(
        model_name="sonnet",
        agent=resolve_agent_path("hypothesis"),
    )
    response = await llm.ainvoke("What is photosynthesis?")
"""

from __future__ import annotations

import asyncio
from importlib import resources as pkg_resources
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from research_mentor.claude_cli import DEFAULT_DISALLOWED_TOOLS, ClaudeCLI


def resolve_agent_path(agent_name: str) -> str:
    """Resolve agent name (e.g. 'hypothesis') to absolute path of the agent .md file.

    Accepts:
    - A bare name like 'hypothesis' → looks up research_mentor/agents/hypothesis.md
    - A name with .md suffix like 'hypothesis.md' → same lookup
    - An absolute or relative path → resolved directly

    Raises FileNotFoundError if not found.
    """
    path = Path(agent_name)

    # Absolute or relative path that exists on disk
    if path.is_file():
        return str(path.resolve())

    # Look in the bundled agents/ directory
    agents_dir = Path(pkg_resources.files("research_mentor") / "agents")  # type: ignore[arg-type]

    # Try with .md suffix
    agent_file = agents_dir / f"{agent_name}.md"
    if agent_file.is_file():
        return str(agent_file)

    # Try as-is (already has .md or other suffix)
    agent_file = agents_dir / agent_name
    if agent_file.is_file():
        return str(agent_file)

    msg = f"Agent file not found: {agent_name} (searched {agents_dir})"
    raise FileNotFoundError(msg)


def _split_messages(
    messages: list[BaseMessage],
) -> tuple[str | None, str]:
    """Split LangChain messages into system prompt and conversation prompt.

    Returns (system_prompt, conversation_prompt).
    System messages are extracted for --system-prompt flag.
    Remaining messages are flattened as conversation text.
    """
    system_parts: list[str] = []
    conv_parts: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            conv_parts.append(f"[User]\n{msg.content}")
        elif isinstance(msg, AIMessage):
            conv_parts.append(f"[Assistant]\n{msg.content}")
        else:
            conv_parts.append(str(msg.content))

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    conversation = "\n\n".join(conv_parts)
    return system_prompt, conversation


class ChatClaudeCLI(BaseChatModel):
    """LangChain chat model backed by the Claude Code CLI.

    Args:
        model_name: Claude model name (sonnet, opus, haiku).
        effort: Thinking effort (low, medium, high).
        timeout: Request timeout in seconds.
        disallowed_tools: Tools to block in CLI calls.
        agent: Agent file path for --agent flag (controls identity).
               Use resolve_agent_path() to convert a name to a path.
    """

    model_name: str = "sonnet"
    effort: str = "medium"
    timeout: float = 120.0
    disallowed_tools: list[str] = DEFAULT_DISALLOWED_TOOLS  # noqa: RUF012
    agent: str | None = None

    _cli: ClaudeCLI | None = None

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _get_cli(self) -> ClaudeCLI:
        if self._cli is None:
            self._cli = ClaudeCLI(
                model=self.model_name,
                effort=self.effort,
                timeout=self.timeout,
            )
        return self._cli

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation — runs the async version in an event loop."""
        return asyncio.get_event_loop().run_until_complete(
            self._agenerate(messages, stop=stop, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via Claude CLI subprocess."""
        import time

        agent = kwargs.get("agent", self.agent)
        system_prompt, prompt = _split_messages(messages)
        cli = self._get_cli()

        t0 = time.monotonic()
        result = await cli.arun(
            prompt,
            agent=agent,
            system_prompt=system_prompt,
            disallowed_tools=self.disallowed_tools,
            effort=kwargs.get("effort", None),
            timeout=kwargs.get("timeout", None),
        )
        elapsed = time.monotonic() - t0

        content = result.stdout.strip()

        # Track usage (estimates for Claude CLI — subscription, no real token count)
        from research_mentor.llm import _call_stats, _record_usage

        prompt_est = len(prompt) // 4
        completion_est = len(content) // 4
        _call_stats["calls"] += 1
        _call_stats["elapsed"] += elapsed
        _call_stats["prompt_tokens"] += prompt_est
        _call_stats["completion_tokens"] += completion_est
        _record_usage(
            backend="claude-cli",
            provider="anthropic",
            model=self.model_name,
            prompt_tokens=prompt_est,
            completion_tokens=completion_est,
            call_type="chat",
            elapsed_seconds=round(elapsed, 2),
        )

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
