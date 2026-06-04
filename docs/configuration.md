# Configuration Reference

Most settings can be configured through the **Settings page** in the web UI. Changes are saved to `~/.research-mentor/config.toml`.

For manual configuration, edit `~/.research-mentor/config.toml`. You only need to include keys you want to change - everything else uses bundled defaults.

Environment variables override both: `RESEARCH_MENTOR_BACKEND=vllm`.

---

## `[server]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Host to bind to |
| `port` | integer | `8080` | Port to listen on |

---

## `[llm]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `"claude-cli"` | LLM backend: `"claude-cli"`, `"vllm"`, or `"api"` |

### `[llm.claude_cli]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `"sonnet"` | Model: `"sonnet"`, `"opus"`, or `"haiku"` |
| `effort` | string | `"medium"` | Thinking effort: `"low"`, `"medium"`, `"high"` |
| `timeout` | integer | `120` | Request timeout in seconds |
| `max_requests_per_minute` | integer | `20` | RPM limit (Claude Pro: ~50, Max: higher) |

### `[llm.vllm]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_base` | string | `"http://localhost:5001/v1"` | vLLM API base URL |
| `active_model` | string | `"gemma3-12b-fp8"` | Active model profile |
| `gpu_memory_utilization` | float | `0.9` | GPU memory fraction (0.0-1.0) |
| `container_runtime` | string | `"podman"` | `"podman"` or `"docker"` |

To switch models:
```toml
[llm.vllm]
active_model = "qwen3-8b-fp8"  # or "gemma3-12b-fp8"
```

### `[llm.api]`

Remote OpenAI-compatible API. Bring your own key.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | `"openai"` | Display name for usage tracking |
| `base_url` | string | `"https://api.openai.com/v1"` | API endpoint |
| `model` | string | `"gpt-4o"` | Model name |
| `api_key_file` | string | `"~/.research-mentor/api_key"` | Path to API key file |

Common providers:

| Provider | `base_url` | Example model |
|----------|-----------|---------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| Anthropic | `https://api.anthropic.com/v1` | `claude-sonnet-4-6` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4` |

---

## `[budget]`

Token usage budget for the `api` backend only. Claude CLI and vLLM skip budget checks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable budget tracking |
| `period` | string | `"monthly"` | `"daily"`, `"weekly"`, `"monthly"`, or `"total"` |
| `global_limit` | integer | `0` | Total token limit (0 = unlimited) |

---

## `[services]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `polite_email` | string | `""` | Email for PubMed/OpenAlex higher rate limits |

API keys for optional services (configure via Settings UI or manually):

```toml
[services.brave_search]
api_key_file = "~/.research-mentor/brave_search_api_key"

[services.tavily]
api_key_file = "~/.research-mentor/tavily_api_key"
```

---

## `[vision]`

Controls how uploaded images and document figures are interpreted.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable vision processing |
| `backend` | string | `"local"` | `"local"`, `"claude-cli"`, or `"api"` |

The local backend uses Qwen3-VL-2B on CPU (~15-30s/image). Claude CLI uses your subscription. API backend uses any OpenAI-compatible vision endpoint.

---

## `[embeddings]`

Semantic memory for cross-session context.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable semantic memory |
| `similarity_threshold` | float | `0.65` | Minimum similarity for retrieval (0.0-1.0) |

---

## `[safety]`

Query safety review - checks search queries for personal data before sending to external APIs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `query_heuristic_review` | bool | `true` | Pattern-based check (instant) |
| `query_llm_review` | bool | `true` | AI-based check (adds one LLM call per search) |

Both can be toggled in the Settings UI.
