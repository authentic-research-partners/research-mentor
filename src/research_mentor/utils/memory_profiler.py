"""Memory profiling utility for tracking memory usage in LangGraph nodes.

ENABLED BY DEFAULT - Set MEMORY_PROFILING=false to disable (zero overhead when disabled).

Usage:
    from research_mentor.utils.memory_profiler import MemoryProfiler, memory_profile_node

    # Context manager
    async with MemoryProfiler("node_name") as profiler:
        pass
    # Logs: "Memory [node_name]: current=50MB, peak=75MB, delta=+25MB"

    # Decorator (preferred for nodes)
    @memory_profile_node("scaffolding")
    async def provide_scaffolding(state: MentorState) -> dict:
        return {"response": "..."}
"""

from __future__ import annotations

import asyncio
import functools
import os
import tracemalloc
from collections.abc import Callable
from typing import Any

from loguru import logger

MEMORY_PROFILING_ENABLED = os.getenv("MEMORY_PROFILING", "true").lower() not in (
    "false", "0", "no",
)


class MemoryProfiler:
    """Context manager for tracking memory usage during operations.

    Tracks memory using Python's tracemalloc module and logs results.
    Zero overhead when MEMORY_PROFILING=false.
    """

    def __init__(self, operation_name: str, log_level: str = "info") -> None:
        self.operation_name = operation_name
        self.log_level = log_level
        self.enabled = MEMORY_PROFILING_ENABLED

        self.start_snapshot: tracemalloc.Snapshot | None = None
        self.current_mb: float = 0.0
        self.peak_mb: float = 0.0
        self.delta_mb: float = 0.0
        self.start_current: int = 0
        self.start_peak: int = 0

    def start(self) -> None:
        """Start memory profiling."""
        if not self.enabled:
            return
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.start_current, self.start_peak = tracemalloc.get_traced_memory()
        self.start_snapshot = tracemalloc.take_snapshot()

    def stop(self) -> dict[str, Any]:
        """Stop profiling and return memory statistics."""
        if not self.enabled:
            return {
                "operation": self.operation_name,
                "current_mb": 0.0,
                "peak_mb": 0.0,
                "delta_mb": 0.0,
                "enabled": False,
            }

        end_current, end_peak = tracemalloc.get_traced_memory()

        self.current_mb = end_current / 1024 / 1024
        self.peak_mb = end_peak / 1024 / 1024
        self.delta_mb = (end_current - self.start_current) / 1024 / 1024

        log_func = getattr(logger, self.log_level, logger.info)
        log_func(
            "Memory [{}]: current={:.1f}MB, peak={:.1f}MB, delta={:+.1f}MB",
            self.operation_name, self.current_mb, self.peak_mb, self.delta_mb,
        )

        return {
            "operation": self.operation_name,
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "delta_mb": round(self.delta_mb, 2),
            "enabled": True,
        }

    def get_top_allocations(self, limit: int = 10) -> list[str]:
        """Get top memory allocations since start."""
        if not self.enabled or not self.start_snapshot:
            return []

        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(self.start_snapshot, "lineno")

        return [
            f"{stat.size_diff / 1024:.1f}KB: {stat.traceback.format()[0]}"
            for stat in top_stats[:limit]
        ]

    async def __aenter__(self) -> MemoryProfiler:
        self.start()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc_val: BaseException | None, exc_tb: Any,
    ) -> None:
        self.stop()

    def __enter__(self) -> MemoryProfiler:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_val: BaseException | None, exc_tb: Any,
    ) -> None:
        self.stop()


def memory_profile_node(node_name: str) -> Callable[..., Any]:
    """Decorator that wraps async node functions with memory profiling.

    Usage:
        @memory_profile_node("scaffolding")
        async def provide_scaffolding(state: MentorState) -> dict:
            return {"response": "..."}

    Logs: Memory [node:scaffolding]: current=150MB, peak=180MB, delta=+30MB
    Zero overhead when MEMORY_PROFILING=false.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with MemoryProfiler(f"node:{node_name}"):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with MemoryProfiler(f"node:{node_name}"):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
