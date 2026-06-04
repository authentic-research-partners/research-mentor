# Usage Guide

For installation instructions, see [INSTALLATION.md](../INSTALLATION.md). For configuration reference, see [configuration.md](configuration.md).

---

## Getting Started

```bash
research-mentor         # start the server
```

Open http://localhost:8080 in your browser. The web UI handles everything from there - projects, conversations, question workshop, sharing, and settings.

Run `research-mentor doctor` at any time to check that everything is working.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `research-mentor` | Start the server (same as `serve`) |
| `research-mentor serve` | Start with options: `--backend`, `--port`, `--host`, `--model` |
| `research-mentor init` | Initialize the database (run once after install, safe to re-run) |
| `research-mentor doctor` | Check system dependencies and configuration |
| `research-mentor migrate` | Update database schema after upgrading (creates backup first) |
| `research-mentor reset` | Wipe database and start fresh (`--yes` to skip confirmation) |
| `research-mentor export` | Export all data as ZIP (`-o backup.zip`, `--project <id>` for single project) |
| `research-mentor vllm-container start` | Start vLLM in a container (requires GPU) |
| `research-mentor vllm-container stop` | Stop vLLM (releases GPU memory) |
| `research-mentor vllm-container status` | Check container state and API health |
| `research-mentor vllm-container logs -f` | Follow container logs |

Global flag: `-v` / `--verbose` enables debug logging.

---

## LLM Backends

Three backends are available. Switch via `--backend` flag, config file, or the Settings page in the UI.

### Claude Code CLI (default)

Uses your existing Claude subscription (Pro/Max) via the Claude Code CLI. No per-token billing.

**Setup:** Install Claude Code CLI (`npm install -g @anthropic-ai/claude-code`), run `claude` once to authenticate, then start the mentor.

**Models:** `sonnet` (fast, recommended), `opus` (best quality), `haiku` (cheapest).

### vLLM (local)

Runs a local LLM server. Free, fully offline, but requires an NVIDIA GPU with 16GB+ VRAM.

```bash
research-mentor vllm-container start    # start vLLM in a container
research-mentor serve --backend vllm    # start the mentor
```

See [INSTALLATION.md](../INSTALLATION.md) for GPU setup and troubleshooting.

### Remote API (bring your own key)

Uses any OpenAI-compatible API (OpenAI, Gemini, Anthropic, OpenRouter, etc.) with your own API key.

```bash
echo "sk-..." > ~/.research-mentor/api_key
research-mentor serve --backend api
```

Configure the provider and model in the Settings page or in [configuration.md](configuration.md).

---

## External Services

The mentor can search academic databases and the web during conversations. All services are optional - the mentor works without them.

| Service | Purpose | API Key |
|---------|---------|---------|
| OpenAlex | Open scholarly metadata | None |
| Semantic Scholar | Citation graph, abstracts | None |
| PubMed | Biomedical literature | None |
| arXiv | Preprints | None |
| Europe PMC | Full-text search (biomedical) | None |
| Brave Search | Web search | Required (free tier available) |
| Tavily | Web search (alternative) | Required (free tier available) |

Academic APIs work out of the box with no setup. Web search requires one API key - configure via **Settings > External Services & Tools** in the UI.

---

## Data Storage

All data is stored locally in `~/.research-mentor/`:

| File | Contents |
|------|----------|
| `research_mentor.db` | Main database (projects, sessions, memories, assessments) |
| `config.toml` | Your configuration overrides |
| `artifacts/` | Uploaded files |
| `brave_search_api_key` | Brave Search API key (optional) |
| `api_key` | Remote API key (optional) |

Use `research-mentor export` to back up your data, and `research-mentor reset` to start fresh.
