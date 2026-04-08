"""Configuration for Personal Research Mentor with Pydantic validation.

Two-layer config: bundled defaults (this package's config.toml) are always
loaded first, then user overrides from <data_dir>/config.toml are deep-merged
on top.  Environment variables (RESEARCH_MENTOR_*) take highest priority.

Override order (highest → lowest):
1. Environment variables (RESEARCH_MENTOR_*)
2. User config: <data_dir>/config.toml  (default ~/.research-mentor/config.toml)
3. Bundled package defaults
"""

from __future__ import annotations

import functools
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_bundled_config() -> dict[str, Any]:
    """Load defaults from bundled config.toml (single source of truth)."""
    package_config = Path(__file__).parent / "config.toml"
    if not package_config.exists():
        raise RuntimeError(
            f"Bundled config.toml not found at {package_config}. "
            "This indicates a broken installation. Reinstall research-mentor."
        )
    with open(package_config, "rb") as f:
        return tomllib.load(f)


_BUNDLED = _load_bundled_config()
_SERVER_DEFAULTS = _BUNDLED.get("server", {})
_LLM_DEFAULTS = _BUNDLED.get("llm", {})
_CLAUDE_CLI_DEFAULTS = _LLM_DEFAULTS.get("claude_cli", {})
_VLLM_DEFAULTS = _LLM_DEFAULTS.get("vllm", {})
_API_DEFAULTS = _LLM_DEFAULTS.get("api", {})
_DB_DEFAULTS = _BUNDLED.get("database", {})
_SERVICES_DEFAULTS = _BUNDLED.get("services", {})
_BRAVE_SEARCH_DEFAULTS = _SERVICES_DEFAULTS.get("brave_search", {})
_TAVILY_DEFAULTS = _SERVICES_DEFAULTS.get("tavily", {})
_CORE_API_DEFAULTS = _SERVICES_DEFAULTS.get("core_api", {})
_NCBI_DEFAULTS = _SERVICES_DEFAULTS.get("ncbi", {})
_RATE_LIMITS_DEFAULTS = _BUNDLED.get("rate_limits", {})
_STORAGE_DEFAULTS = _BUNDLED.get("storage", {})
_LOGGING_DEFAULTS = _BUNDLED.get("logging", {})
_VISION_DEFAULTS = _BUNDLED.get("vision", {})
_GROBID_DEFAULTS = _BUNDLED.get("grobid", {})
_EMBEDDINGS_DEFAULTS = _BUNDLED.get("embeddings", {})
_UI_DEFAULTS = _BUNDLED.get("ui", {})
_BUDGET_DEFAULTS = _BUNDLED.get("budget", {})
_EVAL_DEFAULTS = _BUNDLED.get("eval", {})
_EVAL_JUDGE_DEFAULTS = _EVAL_DEFAULTS.get("judge", {})
_EVAL_SIMULATOR_DEFAULTS = _EVAL_DEFAULTS.get("student_simulator", {})


class ClaudeCLIConfig(BaseModel):
    """Configuration for Claude Code CLI backend. Defaults from bundled config.toml."""

    model: str = Field(default_factory=lambda: _CLAUDE_CLI_DEFAULTS["model"])
    effort: str = Field(default_factory=lambda: _CLAUDE_CLI_DEFAULTS["effort"])
    timeout: int = Field(default_factory=lambda: _CLAUDE_CLI_DEFAULTS["timeout"], ge=10, le=600)
    max_parallel_processes: int = Field(
        default_factory=lambda: _CLAUDE_CLI_DEFAULTS["max_parallel_processes"], ge=1, le=64,
    )
    max_requests_per_minute: int = Field(
        default_factory=lambda: _CLAUDE_CLI_DEFAULTS["max_requests_per_minute"], ge=1, le=1000,
    )


class VLLMModelProfile(BaseModel):
    """Model-specific settings (sampling, HuggingFace ID, thinking tag)."""

    hf_model: str
    thinking_tag: str = ""
    reasoning_parser: str = ""
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=1, le=100)


def _resolve_vllm_defaults() -> dict[str, Any]:
    """Resolve active model profile into flat VLLMConfig defaults."""
    active = _VLLM_DEFAULTS.get("active_model", "gemma3-12b-fp8")
    models = _VLLM_DEFAULTS.get("models", {})
    profile = models.get(active, {})
    result = {**_VLLM_DEFAULTS, "model": active, **profile}
    result.pop("thinking_model", None)
    return result


def _resolve_vllm_from_toml(vllm_raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve a TOML [llm.vllm] section into flat kwargs for VLLMConfig.

    Merges the active model profile (from [llm.vllm.models.*]) into the
    top-level vllm settings, then removes keys that VLLMConfig doesn't expect.
    """
    raw = dict(vllm_raw)
    models = raw.pop("models", {})
    active = raw.pop("active_model", None)
    if active:
        raw["model"] = active
        profile = models.get(active, {})
        raw.update(profile)
    return raw


class VLLMConfig(BaseModel):
    """Configuration for vLLM backend. Defaults from bundled config.toml.

    Model-specific settings (sampling, hf_model, thinking_tag) are defined in
    [llm.vllm.models.*] profiles. The active_model key selects which profile to use.
    Shared settings (api_base, timeout, token limits) live in [llm.vllm].

    thinking_tag: XML tag name the model wraps reasoning in (e.g. "think").
    Empty string means the model does not produce thinking blocks.
    """

    api_base: str = Field(default_factory=lambda: _VLLM_DEFAULTS["api_base"])
    model: str = Field(default_factory=lambda: _resolve_vllm_defaults()["model"])
    thinking_tag: str = Field(
        default_factory=lambda: _resolve_vllm_defaults().get("thinking_tag", ""),
    )
    reasoning_parser: str = Field(
        default_factory=lambda: _resolve_vllm_defaults().get("reasoning_parser", ""),
    )
    temperature: float = Field(
        default_factory=lambda: _resolve_vllm_defaults()["temperature"], ge=0.0, le=2.0,
    )
    top_p: float = Field(
        default_factory=lambda: _resolve_vllm_defaults()["top_p"], ge=0.0, le=1.0,
    )
    top_k: int = Field(
        default_factory=lambda: _resolve_vllm_defaults()["top_k"], ge=1, le=100,
    )
    timeout: int = Field(default_factory=lambda: _VLLM_DEFAULTS["timeout"], ge=10, le=600)
    max_input_tokens: int = Field(
        default_factory=lambda: _VLLM_DEFAULTS["max_input_tokens"], ge=1000, le=32000,
    )
    max_completion_tokens: int = Field(
        default_factory=lambda: _VLLM_DEFAULTS["max_completion_tokens"], ge=256, le=32768,
    )
    hf_model: str = Field(default_factory=lambda: _resolve_vllm_defaults()["hf_model"])
    gpu_memory_utilization: float = Field(
        default_factory=lambda: _VLLM_DEFAULTS["gpu_memory_utilization"], ge=0.1, le=1.0,
    )
    enforce_eager: bool = Field(
        default_factory=lambda: _VLLM_DEFAULTS.get("enforce_eager", False),
    )
    container_runtime: str = Field(
        default_factory=lambda: _VLLM_DEFAULTS.get("container_runtime", "podman"),
    )
    container_name: str = Field(
        default_factory=lambda: _VLLM_DEFAULTS.get("container_name", ""),
    )
    container_image: str = Field(
        default_factory=lambda: _VLLM_DEFAULTS.get(
            "container_image", "docker.io/vllm/vllm-openai:latest",
        ),
    )

    @property
    def max_model_len(self) -> int:
        """Total context length (max_input_tokens + max_completion_tokens)."""
        return self.max_input_tokens + self.max_completion_tokens

    @property
    def thinking_model(self) -> bool:
        """Whether the active model produces thinking blocks."""
        return bool(self.thinking_tag)


class APIConfig(BaseModel):
    """Configuration for remote OpenAI-compatible API backend.

    Lets users bring their own API key for OpenAI, Gemini, Anthropic, etc.
    Uses the openai SDK under the hood — just different base_url + real api_key.
    """

    provider: str = Field(default_factory=lambda: _API_DEFAULTS["provider"])
    base_url: str = Field(default_factory=lambda: _API_DEFAULTS["base_url"])
    model: str = Field(default_factory=lambda: _API_DEFAULTS["model"])
    api_key_file: str = Field(default_factory=lambda: _API_DEFAULTS["api_key_file"])
    thinking_tag: str = Field(default_factory=lambda: _API_DEFAULTS.get("thinking_tag", ""))
    temperature: float = Field(
        default_factory=lambda: _API_DEFAULTS["temperature"], ge=0.0, le=2.0,
    )
    top_p: float = Field(
        default_factory=lambda: _API_DEFAULTS["top_p"], ge=0.0, le=1.0,
    )
    max_completion_tokens: int = Field(
        default_factory=lambda: _API_DEFAULTS["max_completion_tokens"], ge=256, le=32768,
    )
    timeout: int = Field(default_factory=lambda: _API_DEFAULTS["timeout"], ge=10, le=600)

    @property
    def thinking_model(self) -> bool:
        """Whether the model produces thinking blocks."""
        return bool(self.thinking_tag)


class BudgetConfig(BaseModel):
    """Token usage budget configuration.

    Only enforced for the "api" backend (pay-per-token).
    Claude CLI (subscription) and vLLM (local) skip budget checks.
    """

    enabled: bool = Field(default_factory=lambda: _BUDGET_DEFAULTS.get("enabled", True))
    period: str = Field(default_factory=lambda: _BUDGET_DEFAULTS.get("period", "monthly"))
    global_limit: int = Field(
        default_factory=lambda: _BUDGET_DEFAULTS.get("global_limit", 0), ge=0,
    )
    providers: dict[str, int] = Field(
        default_factory=lambda: _BUDGET_DEFAULTS.get("providers", {}),
    )
    models: dict[str, int] = Field(
        default_factory=lambda: _BUDGET_DEFAULTS.get("models", {}),
    )


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    api_key_file: str = Field(
        default_factory=lambda: _BRAVE_SEARCH_DEFAULTS.get(
            "api_key_file", "~/.research-mentor/brave_search_api_key"
        ),
    )


class TavilyConfig(BaseModel):
    """Configuration for Tavily Search API."""

    api_key_file: str = Field(
        default_factory=lambda: _TAVILY_DEFAULTS.get(
            "api_key_file", "~/.research-mentor/tavily_api_key"
        ),
    )


class CoreApiConfig(BaseModel):
    """Configuration for CORE API (core.ac.uk)."""

    api_key_file: str = Field(
        default_factory=lambda: _CORE_API_DEFAULTS.get(
            "api_key_file", "~/.research-mentor/core_api_key"
        ),
    )


class NcbiConfig(BaseModel):
    """Configuration for NCBI (PubMed Central)."""

    api_key_file: str = Field(
        default_factory=lambda: _NCBI_DEFAULTS.get(
            "api_key_file", "~/.research-mentor/ncbi_api_key"
        ),
    )


class ServicesConfig(BaseModel):
    """Configuration for external service integrations."""

    polite_email: str = Field(
        default_factory=lambda: _SERVICES_DEFAULTS.get("polite_email", ""),
    )
    brave_search: BraveSearchConfig = Field(default_factory=BraveSearchConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    core_api: CoreApiConfig = Field(default_factory=CoreApiConfig)
    ncbi: NcbiConfig = Field(default_factory=NcbiConfig)


class StorageConfig(BaseModel):
    """Configuration for local artifact file storage."""

    artifacts_dir: str = Field(
        default_factory=lambda: _STORAGE_DEFAULTS["artifacts_dir"],
    )
    max_upload_bytes: int = Field(
        default_factory=lambda: _STORAGE_DEFAULTS["max_upload_bytes"],
        ge=1024,
        le=1_073_741_824,
    )

    def get_artifacts_path(self) -> Path:
        """Resolve artifacts directory to an absolute path.

        If the configured path is relative, it is resolved relative to
        ``~/.research-mentor/``.
        """
        p = Path(self.artifacts_dir)
        if not p.is_absolute():
            p = Path.home() / ".research-mentor" / p
        p.mkdir(parents=True, exist_ok=True)
        return p


class LoggingConfig(BaseModel):
    """Configuration for file-based logging with rotation."""

    file: str = Field(
        default_factory=lambda: _LOGGING_DEFAULTS.get("file", "logs/mentor.log"),
    )
    rotation: str = Field(
        default_factory=lambda: _LOGGING_DEFAULTS.get("rotation", "10 MB"),
    )
    retention: int = Field(
        default_factory=lambda: _LOGGING_DEFAULTS.get("retention", 5), ge=1, le=100,
    )
    level: str = Field(
        default_factory=lambda: _LOGGING_DEFAULTS.get("level", "DEBUG"),
    )

    def get_log_path(self) -> Path:
        """Resolve log file path. Relative paths are under ~/.research-mentor/."""
        p = Path(self.file)
        if not p.is_absolute():
            p = Path.home() / ".research-mentor" / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


class VisionConfig(BaseModel):
    """Configuration for vision / artifact interpretation.

    Four backends:
    - ``auto``: follows the chat backend (recommended)
    - ``local``: Qwen3-VL-2B on CPU (free/offline, ~15-30s per image, downloads ~4GB model)
    - ``claude-cli``: Claude Code CLI reads files directly (excellent quality)
    - ``api``: OpenAI-compatible vision API (bring your own key)
    """

    enabled: bool = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("enabled", True),
    )
    backend: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("backend", "local"),
    )
    # Local backend
    model: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("model", "Qwen/Qwen3-VL-2B-Instruct"),
    )
    max_new_tokens: int = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("max_new_tokens", 512), ge=64, le=2048,
    )
    max_images_per_artifact: int = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("max_images_per_artifact", 20),
        ge=1, le=100,
    )
    max_image_dimension: int = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("max_image_dimension", 1536),
        ge=256, le=4096,
    )
    # API backend
    api_provider: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("api_provider", ""),
    )
    api_base_url: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("api_base_url", ""),
    )
    api_model: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("api_model", ""),
    )
    api_key_file: str = Field(
        default_factory=lambda: _VISION_DEFAULTS.get(
            "api_key_file", "~/.research-mentor/vision_api_key"
        ),
    )
    api_max_tokens: int = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("api_max_tokens", 1024), ge=64, le=16384,
    )
    api_timeout: int = Field(
        default_factory=lambda: _VISION_DEFAULTS.get("api_timeout", 120), ge=10, le=600,
    )


class GrobidConfig(BaseModel):
    """Configuration for GROBID academic PDF extraction.

    GROBID provides structured extraction of academic papers with proper heading
    hierarchy, reference parsing, and metadata extraction. Requires a running
    GROBID container (CPU-only, ~4-8 GB RAM).

    Opt-in: disabled by default since it requires running a container.
    Reuses ``container_runtime`` from ``[llm.vllm]`` config.
    """

    enabled: bool = Field(
        default_factory=lambda: _GROBID_DEFAULTS.get("enabled", False),
    )
    url: str = Field(
        default_factory=lambda: _GROBID_DEFAULTS.get("url", "http://localhost:8070"),
    )
    container_image: str = Field(
        default_factory=lambda: _GROBID_DEFAULTS.get(
            "container_image", "lfoppiano/grobid:0.8.1",
        ),
    )
    container_name: str = Field(
        default_factory=lambda: _GROBID_DEFAULTS.get(
            "container_name", "research-mentor-grobid",
        ),
    )


class EmbeddingsConfig(BaseModel):
    """Configuration for local embedding model (sentence-transformers).

    Uses snowflake-arctic-embed-m-v2.0 by default (110M params, 768-dim, 8192 token context).
    Always runs on CPU to avoid competing with vLLM for GPU VRAM.
    Embedding calls run in a thread pool to avoid blocking the event loop.
    """

    enabled: bool = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("enabled", True),
    )
    model: str = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get(
            "model", "Snowflake/snowflake-arctic-embed-m-v2.0"
        ),
    )
    dimension: int = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("dimension", 768), ge=1, le=4096,
    )
    similarity_threshold: float = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("similarity_threshold", 0.65),
        ge=0.0,
        le=1.0,
    )
    max_results: int = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("max_results", 5), ge=1, le=50,
    )
    artifact_chunk_size: int = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("artifact_chunk_size", 1500),
        ge=100,
        le=10000,
    )
    artifact_chunk_overlap: int = Field(
        default_factory=lambda: _EMBEDDINGS_DEFAULTS.get("artifact_chunk_overlap", 200),
        ge=0,
        le=5000,
    )


class EvalJudgeConfig(BaseModel):
    """Configuration for the LLM-as-judge used in evaluation.

    Should be a stronger model than the system under test to avoid
    weak-model-judges-weak-model bias. Defaults from [eval.judge] in config.toml.
    """

    backend: str = Field(
        default_factory=lambda: _EVAL_JUDGE_DEFAULTS["backend"],
    )
    model: str = Field(
        default_factory=lambda: _EVAL_JUDGE_DEFAULTS["model"],
    )
    effort: str = Field(
        default_factory=lambda: _EVAL_JUDGE_DEFAULTS["effort"],
    )

    def to_overrides(self) -> dict[str, str]:
        """Return as a dict suitable for **kwargs to structured_call()."""
        return {"backend": self.backend, "model": self.model, "effort": self.effort}


class EvalStudentSimulatorConfig(BaseModel):
    """Configuration for the AI student simulator used in conversation evals.

    Auto mode uses this model (via claude-cli) for normal student profiles.
    Adversarial profiles bypass this and use the SUT backend instead.
    Defaults from [eval.student_simulator] in config.toml.
    """

    model: str = Field(
        default_factory=lambda: _EVAL_SIMULATOR_DEFAULTS.get("model", "haiku"),
    )


class EvalConfig(BaseModel):
    """Configuration for the evaluation framework."""

    judge: EvalJudgeConfig = Field(default_factory=EvalJudgeConfig)
    student_simulator: EvalStudentSimulatorConfig = Field(
        default_factory=EvalStudentSimulatorConfig,
    )


_QW_DEFAULTS = _BUNDLED.get("question_workshop", {})
_HYPOTHESIS_DEFAULTS = _QW_DEFAULTS.get("hypothesis", {})
_CLAIMS_DEFAULTS = _QW_DEFAULTS.get("claims", {})
_QUESTIONED_DEFAULTS = _QW_DEFAULTS.get("questioned", {})
_RETRACTIONS_DEFAULTS = _QW_DEFAULTS.get("retractions", {})
_GAPS_DEFAULTS = _QW_DEFAULTS.get("gaps", {})
_THEORY_DEFAULTS = _QW_DEFAULTS.get("theory", {})
_SHARING_DEFAULTS = _BUNDLED.get("sharing", {})


class HypothesisConfig(BaseModel):
    """Configuration for the Hypothesis question workshop."""

    max_message_history: int = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("max_message_history", 30),
        ge=5, le=200,
    )
    max_papers_reviewed: int = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("max_papers_reviewed", 20),
        ge=1, le=100,
    )
    max_datasets_found: int = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("max_datasets_found", 15),
        ge=1, le=100,
    )
    max_confounds: int = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("max_confounds", 10),
        ge=1, le=50,
    )
    min_confounds_for_transition: int = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("min_confounds_for_transition", 3),
        ge=1, le=20,
    )
    conversational_temperature: float = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("conversational_temperature", 0.7),
        ge=0.0, le=2.0,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _HYPOTHESIS_DEFAULTS.get("extraction_temperature", 0.0),
        ge=0.0, le=2.0,
    )


class ClaimsConfig(BaseModel):
    """Configuration for the Claims pipeline."""

    max_papers: int = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("max_papers", 15),
        ge=1, le=50,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("extraction_temperature", 0.3),
        ge=0.0, le=2.0,
    )
    analysis_temperature: float = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("analysis_temperature", 0.2),
        ge=0.0, le=2.0,
    )
    gap_temperature: float = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("gap_temperature", 0.4),
        ge=0.0, le=2.0,
    )
    framing_temperature: float = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("framing_temperature", 0.6),
        ge=0.0, le=2.0,
    )
    questions_temperature: float = Field(
        default_factory=lambda: _CLAIMS_DEFAULTS.get("questions_temperature", 0.5),
        ge=0.0, le=2.0,
    )


class QuestionedConfig(BaseModel):
    """Configuration for the Questioned pipeline (papers under scrutiny)."""

    max_papers: int = Field(
        default_factory=lambda: _QUESTIONED_DEFAULTS.get("max_papers", 10),
        ge=1, le=50,
    )
    max_curated: int = Field(
        default_factory=lambda: _QUESTIONED_DEFAULTS.get("max_curated", 5),
        ge=1, le=20,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _QUESTIONED_DEFAULTS.get("extraction_temperature", 0.3),
        ge=0.0, le=2.0,
    )
    curation_temperature: float = Field(
        default_factory=lambda: _QUESTIONED_DEFAULTS.get("curation_temperature", 0.5),
        ge=0.0, le=2.0,
    )


class RetractionsConfig(BaseModel):
    """Configuration for the Retractions workshop (interactive + pipeline)."""

    # Interactive (chat) settings
    max_message_history: int = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("max_message_history", 30),
        ge=5, le=200,
    )
    max_citing_papers: int = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("max_citing_papers", 15),
        ge=1, le=100,
    )
    conversational_temperature: float = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("conversational_temperature", 0.7),
        ge=0.0, le=2.0,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("extraction_temperature", 0.0),
        ge=0.0, le=2.0,
    )

    # Pipeline settings
    pipeline_max_browse_results: int = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("pipeline_max_browse_results", 8),
        ge=1, le=30,
    )
    pipeline_max_output_questions: int = Field(
        default_factory=lambda: _RETRACTIONS_DEFAULTS.get("pipeline_max_output_questions", 5),
        ge=1, le=10,
    )


class GapsConfig(BaseModel):
    """Configuration for the Gaps workshop (literature gap mining)."""

    paper_intro_enabled: bool = Field(
        default_factory=lambda: _GAPS_DEFAULTS.get("paper_intro_enabled", True),
    )
    paper_intro_max_papers: int = Field(
        default_factory=lambda: _GAPS_DEFAULTS.get("paper_intro_max_papers", 2),
        ge=0, le=10,
    )
    paper_intro_max_words: int = Field(
        default_factory=lambda: _GAPS_DEFAULTS.get("paper_intro_max_words", 400),
        ge=100, le=2000,
    )
    paper_intro_download_timeout: int = Field(
        default_factory=lambda: _GAPS_DEFAULTS.get("paper_intro_download_timeout", 30),
        ge=5, le=120,
    )


class TheoryConfig(BaseModel):
    """Configuration for the Theory workshop (pipeline + interactive)."""

    # Pipeline settings
    max_papers: int = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("max_papers", 25),
        ge=1, le=50,
    )
    max_consilience_domains: int = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("max_consilience_domains", 3),
        ge=1, le=5,
    )
    conversational_temperature: float = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("conversational_temperature", 0.7),
        ge=0.0, le=2.0,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("extraction_temperature", 0.0),
        ge=0.0, le=2.0,
    )

    # Interactive (chat) settings — kept for backward compat until chat is removed
    max_message_history: int = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("max_message_history", 30),
        ge=5, le=200,
    )
    max_observations: int = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("max_observations", 20),
        ge=1, le=50,
    )
    max_candidate_explanations: int = Field(
        default_factory=lambda: _THEORY_DEFAULTS.get("max_candidate_explanations", 10),
        ge=1, le=20,
    )


class SharingConfig(BaseModel):
    """Configuration for the Sharing & Publication module."""

    max_message_history: int = Field(
        default_factory=lambda: _SHARING_DEFAULTS.get("max_message_history", 30),
        ge=5, le=200,
    )
    max_venue_results: int = Field(
        default_factory=lambda: _SHARING_DEFAULTS.get("max_venue_results", 20),
        ge=1, le=100,
    )
    conversational_temperature: float = Field(
        default_factory=lambda: _SHARING_DEFAULTS.get("conversational_temperature", 0.7),
        ge=0.0, le=2.0,
    )
    extraction_temperature: float = Field(
        default_factory=lambda: _SHARING_DEFAULTS.get("extraction_temperature", 0.0),
        ge=0.0, le=2.0,
    )


_COLLABORATION_DEFAULTS = _BUNDLED.get("collaboration", {})


class CollaborationConfig(BaseModel):
    """Configuration for collaboration scout (researcher/institution search)."""

    search_cooldown_minutes: int = Field(
        default_factory=lambda: _COLLABORATION_DEFAULTS.get("search_cooldown_minutes", 30),
        ge=1, le=1440,
    )
    max_web_queries_per_search: int = Field(
        default_factory=lambda: _COLLABORATION_DEFAULTS.get("max_web_queries_per_search", 3),
        ge=0, le=10,
    )
    result_expiry_days: int = Field(
        default_factory=lambda: _COLLABORATION_DEFAULTS.get("result_expiry_days", 30),
        ge=1, le=365,
    )


class QuestionWorkshopConfig(BaseModel):
    """Configuration for Question Workshop (research question generation)."""

    candidates: int = Field(
        default_factory=lambda: _QW_DEFAULTS.get("candidates", 3),
        ge=1, le=10,
    )
    fields: dict[str, list[str]] = Field(
        default_factory=lambda: _QW_DEFAULTS.get("fields", {}),
        description="Per-workshop field lists and category definitions",
    )
    hypothesis: HypothesisConfig = Field(default_factory=HypothesisConfig)
    claims: ClaimsConfig = Field(default_factory=ClaimsConfig)
    questioned: QuestionedConfig = Field(default_factory=QuestionedConfig)
    retractions: RetractionsConfig = Field(default_factory=RetractionsConfig)
    gaps: GapsConfig = Field(default_factory=GapsConfig)
    theory: TheoryConfig = Field(default_factory=TheoryConfig)


class RateLimitsConfig(BaseModel):
    """Rate limit configuration. Each value is [max_requests, period_seconds]."""

    arxiv: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("arxiv", [18, 60]),
    )
    semantic_scholar: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("semantic_scholar", [2, 3]),
    )
    pubmed: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("pubmed", [150, 60]),
    )
    pubmed_polite: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("pubmed_polite", [480, 60]),
    )
    openalex: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("openalex", [300, 60]),
    )
    openalex_polite: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("openalex_polite", [480, 60]),
    )
    tavily_per_second: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("tavily_per_second", [1, 1]),
    )
    brave_search_per_second: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("brave_search_per_second", [1, 1]),
    )
    brave_search_monthly: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("brave_search_monthly", [1900, 2592000]),
    )
    ror_api: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("ror_api", [10, 60]),
    )
    core_api: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("core_api", [10, 60]),
    )
    unpaywall: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("unpaywall", [10, 1]),
    )
    pmc: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("pmc", [3, 1]),
    )
    europepmc: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("europepmc", [10, 1]),
    )
    orcid: list[int] = Field(
        default_factory=lambda: _RATE_LIMITS_DEFAULTS.get("orcid", [10, 1]),
    )


# --- Runtime overrides (set via API, take priority over TOML + env vars) ---

_runtime_overrides: dict[str, Any] = {}


def set_runtime_override(key: str, value: Any) -> None:
    """Set a runtime config override (e.g. from PUT /api/config/backend)."""
    _runtime_overrides[key] = value


def get_runtime_override(key: str) -> Any | None:
    """Get a runtime config override, or None if not set."""
    return _runtime_overrides.get(key)


def invalidate_config_cache() -> None:
    """Clear the TOML config LRU cache so the next load_config() re-reads from disk."""
    load_config_from_toml.cache_clear()


def get_user_config_path() -> Path:
    """Resolve the writable user config file path.

    Returns ``<data_dir>/config.toml`` — defaults to
    ``~/.research-mentor/config.toml``, overridden by
    ``RESEARCH_MENTOR_DATA_DIR`` env var (used by tests).
    """
    return get_data_dir() / "config.toml"


def update_user_config(updates: dict[str, Any]) -> Path:
    """Write arbitrary settings to the user config file.

    ``updates`` is a nested dict mirroring the TOML structure, e.g.::

        {"services": {"polite_email": "a@b.com"},
         "embeddings": {"enabled": False}}

    Only the specified keys are touched; everything else is preserved.
    Returns the path to the config file.
    """
    import tomli_w

    config_path = get_user_config_path()

    if config_path.exists():
        with open(config_path, "rb") as f:
            existing = tomllib.load(f)
    else:
        existing = {}

    _deep_merge(existing, updates)

    with open(config_path, "wb") as f:
        tomli_w.dump(existing, f)

    invalidate_config_cache()
    return config_path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` in place."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def reset_user_config() -> Path:
    """Remove the user config file to restore bundled defaults.

    Clears runtime overrides as well. Returns the config path.
    """
    config_path = get_user_config_path()
    if config_path.exists():
        config_path.unlink()
    _runtime_overrides.clear()
    invalidate_config_cache()
    return config_path


def update_user_config_active_model(profile_name: str) -> Path:
    """Write active_model to the user config file. Creates file if needed.

    Only touches the ``active_model`` line under ``[llm.vllm]``.
    Returns the path to the config file.
    """
    import re

    config_path = get_user_config_path()

    if config_path.exists():
        text = config_path.read_text()
    else:
        text = ""

    new_line = f'active_model = "{profile_name}"'

    # Case 1: active_model line exists under [llm.vllm] — replace it
    if re.search(r'^active_model\s*=', text, re.MULTILINE):
        text = re.sub(r'^active_model\s*=.*$', new_line, text, count=1, flags=re.MULTILINE)
    # Case 2: [llm.vllm] section exists but no active_model — append after section header
    elif "[llm.vllm]" in text:
        text = text.replace("[llm.vllm]", f"[llm.vllm]\n{new_line}", 1)
    # Case 3: neither exists — append the full section
    else:
        separator = "\n" if text and not text.endswith("\n") else ""
        text += f"{separator}[llm.vllm]\n{new_line}\n"

    config_path.write_text(text)
    invalidate_config_cache()
    return config_path


class SafetyConfig(BaseModel):
    """Safety and privacy settings."""

    query_heuristic_review: bool = Field(
        default=True,
        description=(
            "Fast regex-based check on outgoing search queries. "
            "Blocks queries containing emails, phone numbers, file paths, "
            "credentials, or API keys. No LLM call, no latency cost."
        ),
    )
    query_llm_review: bool = Field(
        default=True,
        description=(
            "LLM-based review of outgoing search queries. "
            "Catches subtler leaks that heuristics miss, e.g. student name + "
            "school combinations. Adds one structured LLM call per search."
        ),
    )


class AppConfig(BaseSettings):
    """Top-level application configuration.

    Override via environment variables with RESEARCH_MENTOR_ prefix:
    - RESEARCH_MENTOR_BACKEND: LLM backend ("claude-cli" or "vllm")
    - RESEARCH_MENTOR_HOST: Server host
    - RESEARCH_MENTOR_PORT: Server port
    """

    host: str = Field(default_factory=lambda: _SERVER_DEFAULTS["host"])
    port: int = Field(default_factory=lambda: _SERVER_DEFAULTS["port"], ge=1, le=65535)
    stream_inactivity_timeout: int = Field(
        default_factory=lambda: _SERVER_DEFAULTS["stream_inactivity_timeout"], ge=10, le=600,
    )
    stream_global_timeout: int = Field(
        default_factory=lambda: _SERVER_DEFAULTS["stream_global_timeout"], ge=60, le=3600,
    )
    backend: str = Field(default_factory=lambda: _LLM_DEFAULTS["backend"])
    structured_call_max_retries: int = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("structured_call_max_retries", 2),
        ge=0, le=5,
    )
    db_path: str = Field(default_factory=lambda: _DB_DEFAULTS["path"])
    db_pool_size: int = Field(
        default_factory=lambda: _DB_DEFAULTS.get("pool_size", 4), ge=1, le=16,
    )
    claude_cli: ClaudeCLIConfig = Field(default_factory=ClaudeCLIConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    grobid: GrobidConfig = Field(default_factory=GrobidConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    question_workshop: QuestionWorkshopConfig = Field(
        default_factory=QuestionWorkshopConfig,
    )
    sharing: SharingConfig = Field(default_factory=SharingConfig)
    collaboration: CollaborationConfig = Field(default_factory=CollaborationConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    ui_port: int = Field(default_factory=lambda: _UI_DEFAULTS["port"], ge=1, le=65535)

    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_MENTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@functools.lru_cache(maxsize=1)
def load_config_from_toml() -> dict[str, Any]:
    """Load configuration from bundled defaults + user overrides (cached).

    Always starts with the bundled package ``config.toml`` as the base,
    then deep-merges the user config on top (if it exists).

    User config location: ``<data_dir>/config.toml``
    (``~/.research-mentor/config.toml`` by default, overridden by
    ``RESEARCH_MENTOR_DATA_DIR`` env var — used by tests for isolation).
    """
    import copy

    merged = copy.deepcopy(_BUNDLED)

    user_config = get_data_dir() / "config.toml"
    if user_config.exists():
        with open(user_config, "rb") as f:
            user_overrides = tomllib.load(f)
        _deep_merge(merged, user_overrides)

    return merged


def get_data_dir() -> Path:
    """Get the Personal Research Mentor data directory (~/.research-mentor/).

    Override with RESEARCH_MENTOR_DATA_DIR env var (used by integration tests).
    Creates the directory if it doesn't exist.
    """
    override = os.environ.get("RESEARCH_MENTOR_DATA_DIR")
    data_dir = Path(override) if override else Path.home() / ".research-mentor"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def load_config() -> AppConfig:
    """Load application configuration with TOML + env var overrides.

    Priority (highest to lowest):
    1. Environment variables (RESEARCH_MENTOR_*)
    2. .env file
    3. config.toml (local → user → bundled)
    4. Built-in defaults
    """
    import os

    toml_config = load_config_from_toml()

    server = toml_config.get("server", {})
    llm = toml_config.get("llm", {})
    db = toml_config.get("database", {})

    # For scalar fields that map to RESEARCH_MENTOR_* env vars,
    # only use TOML value when env var is NOT set — env vars must win per
    # pydantic-settings convention. Without this guard, TOML values passed
    # as init kwargs would shadow env vars (init kwargs > env > defaults).
    prefix = "RESEARCH_MENTOR_"
    kwargs: dict[str, Any] = {}
    if "host" in server and not os.getenv(f"{prefix}HOST"):
        kwargs["host"] = server["host"]
    if "port" in server and not os.getenv(f"{prefix}PORT"):
        kwargs["port"] = server["port"]
    if "stream_inactivity_timeout" in server and not os.getenv(
        f"{prefix}STREAM_INACTIVITY_TIMEOUT"
    ):
        kwargs["stream_inactivity_timeout"] = server["stream_inactivity_timeout"]
    if "stream_global_timeout" in server and not os.getenv(f"{prefix}STREAM_GLOBAL_TIMEOUT"):
        kwargs["stream_global_timeout"] = server["stream_global_timeout"]
    if "backend" in llm and not os.getenv(f"{prefix}BACKEND"):
        kwargs["backend"] = llm["backend"]
    if "structured_call_max_retries" in llm and not os.getenv(
        f"{prefix}STRUCTURED_CALL_MAX_RETRIES"
    ):
        kwargs["structured_call_max_retries"] = llm["structured_call_max_retries"]
    if "path" in db and not os.getenv(f"{prefix}DB_PATH"):
        kwargs["db_path"] = db["path"]
    if "pool_size" in db and not os.getenv(f"{prefix}DB_POOL_SIZE"):
        kwargs["db_pool_size"] = db["pool_size"]

    ui = toml_config.get("ui", {})
    if "port" in ui and not os.getenv(f"{prefix}UI_PORT"):
        kwargs["ui_port"] = ui["port"]

    if "claude_cli" in llm:
        kwargs["claude_cli"] = ClaudeCLIConfig(**llm["claude_cli"])
    if "vllm" in llm:
        kwargs["vllm"] = VLLMConfig(**_resolve_vllm_from_toml(llm["vllm"]))

    if "api" in llm:
        kwargs["api"] = APIConfig(**llm["api"])

    budget_raw = toml_config.get("budget", {})
    if budget_raw:
        kwargs["budget"] = BudgetConfig(**budget_raw)

    services_raw = toml_config.get("services", {})
    if services_raw:
        brave_raw = services_raw.pop("brave_search", {})
        tavily_raw = services_raw.pop("tavily", {})
        svc_kwargs: dict[str, Any] = {**services_raw}
        if brave_raw:
            svc_kwargs["brave_search"] = BraveSearchConfig(**brave_raw)
        if tavily_raw:
            svc_kwargs["tavily"] = TavilyConfig(**tavily_raw)
        kwargs["services"] = ServicesConfig(**svc_kwargs)

    rate_limits_raw = toml_config.get("rate_limits", {})
    if rate_limits_raw:
        kwargs["rate_limits"] = RateLimitsConfig(**rate_limits_raw)

    storage_raw = toml_config.get("storage", {})
    if storage_raw:
        kwargs["storage"] = StorageConfig(**storage_raw)

    logging_raw = toml_config.get("logging", {})
    if logging_raw:
        kwargs["logging"] = LoggingConfig(**logging_raw)

    vision_raw = toml_config.get("vision", {})
    if vision_raw:
        kwargs["vision"] = VisionConfig(**vision_raw)

    grobid_raw = toml_config.get("grobid", {})
    if grobid_raw:
        kwargs["grobid"] = GrobidConfig(**grobid_raw)

    embeddings_raw = toml_config.get("embeddings", {})
    if embeddings_raw:
        kwargs["embeddings"] = EmbeddingsConfig(**embeddings_raw)

    qw_raw = toml_config.get("question_workshop", {})
    if qw_raw:
        hypothesis_raw = qw_raw.pop("hypothesis", {})
        claims_raw = qw_raw.pop("claims", {})
        questioned_raw = qw_raw.pop("questioned", {})
        retractions_raw = qw_raw.pop("retractions", {})
        gaps_raw = qw_raw.pop("gaps", {})
        qw_kwargs: dict[str, Any] = {**qw_raw}
        if hypothesis_raw:
            qw_kwargs["hypothesis"] = HypothesisConfig(**hypothesis_raw)
        if claims_raw:
            qw_kwargs["claims"] = ClaimsConfig(**claims_raw)
        if questioned_raw:
            qw_kwargs["questioned"] = QuestionedConfig(**questioned_raw)
        if retractions_raw:
            qw_kwargs["retractions"] = RetractionsConfig(**retractions_raw)
        if gaps_raw:
            qw_kwargs["gaps"] = GapsConfig(**gaps_raw)
        kwargs["question_workshop"] = QuestionWorkshopConfig(**qw_kwargs)

    # Runtime overrides (from API calls) take highest priority
    for key, value in _runtime_overrides.items():
        kwargs[key] = value

    return AppConfig(**kwargs)


_CLAUDE_ALIASES = {"sonnet", "opus", "haiku"}


def get_llm_model_info() -> dict[str, str]:
    """Return the active LLM backend, model name, and version for metadata tagging.

    For Claude CLI aliases ("sonnet"), marks version as "latest".
    For full model IDs ("claude-sonnet-4-6"), extracts the version ("4.6").
    """
    import re

    config = load_config()
    backend = config.backend
    if backend == "claude-cli":
        raw = config.claude_cli.model
        # Strip optional [1m] suffix for parsing
        base = re.sub(r"\[.*\]$", "", raw)
        if base in _CLAUDE_ALIASES:
            model = base
            version = "latest"
        else:
            # Full ID like "claude-sonnet-4-6" → model="sonnet", version="4.6"
            m = re.match(r"claude-(\w+)-([\d]+-[\d]+(?:-\d+)?)", base)
            if m:
                model = m.group(1)
                version = m.group(2).replace("-", ".")
            else:
                model = raw
                version = ""
    elif backend == "vllm":
        model = config.vllm.model
        version = ""
    elif backend == "api":
        model = config.api.model
        version = ""
    else:
        model = "unknown"
        version = ""
    info: dict[str, str] = {"llm_backend": backend, "llm_model": model}
    if version:
        info["llm_model_version"] = version
    return info
