"""Tool-calling loop for expert nodes.

YAML-based tool format (proven 100% reliable with Gemma 3 vs 80% JSON accuracy).
Works with both vLLM and Claude CLI backends.

Public API:
    invoke_with_tools() — LLM tool-calling loop, returns (final_content, metadata)
    SEARCH_WEB_TOOL_DEF — OpenAI-format tool definition for search_web
    TOOL_REGISTRY — maps tool names to async callables
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import yaml
from loguru import logger

from research_mentor.config import load_config
from research_mentor.tools.web_search import search_web

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------

SEARCH_WEB_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current information (MSDS sheets, safety protocols, "
            "IRB templates, regulations, citation formats)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (1-20, default 5)",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_REGISTRY: dict[str, Any] = {"search_web": search_web}


# ---------------------------------------------------------------------------
# YAML injection
# ---------------------------------------------------------------------------


def _inject_yaml_tool_instructions(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Append YAML tool-calling instructions to the system prompt."""
    if not tools:
        return messages

    tool_descriptions: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        tool_name = func.get("name", "unknown")
        description = func.get("description", "No description")
        parameters = func.get("parameters", {})

        param_props = parameters.get("properties", {})
        required_params = parameters.get("required", [])
        param_lines: list[str] = []
        for pname, pinfo in param_props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required_params else " (optional)"
            param_lines.append(f"  - {pname} ({ptype}){req}: {pdesc}")

        tool_descriptions.append(
            f"**{tool_name}**\n"
            f"Description: {description}\n"
            "Parameters:\n"
            + ("\n".join(param_lines) if param_lines else "  (no parameters)")
        )

    yaml_instructions = (
        "\n\n# Available Tools\n\n"
        + "\n\n".join(tool_descriptions)
        + "\n\n# Tool Calling Format\n\n"
        "To call tools, respond with YAML in this exact format:\n\n"
        "```yaml\n"
        "tool_calls:\n"
        "  - id: call_1\n"
        "    function:\n"
        "      name: tool_name\n"
        "      arguments:\n"
        "        param: value\n"
        "```\n\n"
        "**IMPORTANT:**\n"
        "- When using tools, output ONLY the YAML (no other text)\n"
        "- You can call multiple tools by adding more items to the list\n"
        "- After tools return results, synthesize a final response using the tool data"
    )

    modified: list[dict[str, str]] = []
    system_found = False
    for msg in messages:
        if msg.get("role") == "system":
            modified.append({
                "role": "system",
                "content": msg.get("content", "") + yaml_instructions,
            })
            system_found = True
        else:
            modified.append(dict(msg))

    if not system_found:
        modified.insert(0, {"role": "system", "content": yaml_instructions})

    return modified


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_yaml_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract tool calls from YAML in model output.

    Handles both ```yaml code blocks and bare YAML.
    """
    if not content:
        return []

    def _extract(data: Any) -> list[dict[str, Any]]:
        if not isinstance(data, dict) or "tool_calls" not in data:
            return []
        raw_calls = data["tool_calls"]
        if not isinstance(raw_calls, list):
            return []
        calls: list[dict[str, Any]] = []
        for i, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            func = call.get("function", {})
            arguments = func.get("arguments")
            if arguments is None:
                arguments = {}
            calls.append({
                "id": call.get("id", f"call_{i}"),
                "function": {
                    "name": func.get("name", "unknown"),
                    "arguments": arguments,
                },
            })
        return calls

    # Try markdown code block first
    yaml_match = re.search(r"```(?:yaml)?\s*(.+?)\s*```", content, re.DOTALL)
    if yaml_match:
        try:
            data = yaml.safe_load(yaml_match.group(1))
            calls = _extract(data)
            if calls:
                return calls
        except yaml.YAMLError:
            pass

    # Try bare YAML
    try:
        data = yaml.safe_load(content.strip())
        return _extract(data)
    except yaml.YAMLError:
        pass

    return []


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Run a single tool and return a result dict."""
    import asyncio
    import inspect

    if name not in registry:
        return {
            "tool": name,
            "status": "failed",
            "error": f"Tool '{name}' not found. Available: {list(registry.keys())}",
        }

    try:
        func = registry[name]
        if inspect.iscoroutinefunction(func):
            result = await func(**arguments)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: func(**arguments))
        return {"tool": name, "status": "success", "result": result}
    except TypeError as exc:
        return {"tool": name, "status": "failed", "error": f"Invalid arguments: {exc}"}
    except Exception as exc:
        logger.exception("Error executing tool {}", name)
        return {"tool": name, "status": "failed", "error": f"Execution failed: {exc}"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def invoke_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    tool_registry: dict[str, Any],
    *,
    label: str = "tool_call",
    max_iterations: int = 5,
) -> tuple[str, dict[str, Any]]:
    """LLM tool-calling loop with YAML format.

    Injects YAML tool instructions, calls LLM, parses tool calls,
    executes tools, appends results, repeats until final text response.

    Works with both vLLM and Claude CLI backends (auto-detected from config).

    Args:
        messages: Conversation in OpenAI format [{"role": ..., "content": ...}]
        tools: Tool definitions in OpenAI format
        tool_registry: Maps tool names to async callables
        label: Logging label for this invocation
        max_iterations: Safety limit on tool-calling rounds

    Returns:
        (final_content, metadata) where metadata includes tool_calls,
        iteration_count, and total_duration_ms
    """
    config = load_config()
    backend = config.backend
    start_time = time.monotonic()

    loop_messages = _inject_yaml_tool_instructions(list(messages), tools)
    tool_calls_metadata: list[dict[str, Any]] = []
    content = ""
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.debug("{}: tool-calling iteration {}/{}", label, iteration, max_iterations)

        # --- Call LLM ---
        if backend == "vllm":
            from research_mentor.llm import _strip_thinking, _vllm_complete

            raw, _usage = await _vllm_complete(loop_messages, label=label)
            content = _strip_thinking(raw, config.vllm.thinking_tag)
        else:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            from research_mentor.llm import get_chat_llm

            lc_messages: list[SystemMessage | AIMessage | HumanMessage] = []
            for msg in loop_messages:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=text))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=text))
                else:
                    lc_messages.append(HumanMessage(content=text))

            llm = get_chat_llm()
            result = await llm.ainvoke(lc_messages)
            content = str(result.content)

        # --- Parse tool calls ---
        parsed_calls = _parse_yaml_tool_calls(content)

        if not parsed_calls:
            logger.debug("{}: no tool calls in iteration {}, returning response", label, iteration)
            break

        # --- Execute tools ---
        tool_results_parts: list[str] = []
        for call in parsed_calls:
            func_info = call.get("function", {})
            tool_name = func_info.get("name", "unknown")
            tool_args = func_info.get("arguments", {})

            logger.info("{}: calling tool {} with {}", label, tool_name, tool_args)
            tool_start = time.monotonic()
            result_dict = await _execute_tool(tool_name, tool_args, tool_registry)
            tool_duration_ms = (time.monotonic() - tool_start) * 1000

            tool_calls_metadata.append({
                "tool": tool_name,
                "arguments": tool_args,
                "result": result_dict.get("result"),
                "success": result_dict.get("status") == "success",
                "duration_ms": tool_duration_ms,
            })

            tool_results_parts.append(
                f"Tool {tool_name} returned: {json.dumps(result_dict)}"
            )

        # Append tool results as user message (role alternation safe)
        loop_messages.append({"role": "assistant", "content": content})
        loop_messages.append({"role": "user", "content": "\n\n".join(tool_results_parts)})
    else:
        logger.warning("{}: exhausted {} iterations without final response", label, max_iterations)

    total_duration_ms = (time.monotonic() - start_time) * 1000
    metadata: dict[str, Any] = {
        "tool_calls": tool_calls_metadata,
        "iteration_count": iteration,
        "total_duration_ms": total_duration_ms,
    }

    logger.info(
        "{}: completed in {:.0f}ms, {} tool calls across {} iterations",
        label, total_duration_ms, len(tool_calls_metadata), iteration,
    )

    return content, metadata
