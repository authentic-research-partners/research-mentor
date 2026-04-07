# Changelog

## v1.0.2 (2026-04-06)

- Embedding model downloads in the background — server starts immediately
- Vision off by default; enable in Settings with backend choice (auto/Claude CLI/local/API)
- Local vision model (~4 GB, ~10 GB RAM) managed from Settings: download, remove, size and path info
- Fix local vision model crash (processor expected PIL Image, not file path)
- Fix version display

## v1.0.1 (2026-04-06)

- Fix assessment score display to use 0-100 scale matching backend
- Sync UI hypothesis stage names with backend graph
- Remove orphaned phase labels from stage configuration
- Add missing `aiosqlitepool` dependency

## v1.0.0 (2026-04-06)

Initial public release.

### Mentoring
- Adaptive research mentoring with 21-node LangGraph agent
- Question Workshop: 9 pipelines + Hypothesis chat
- Research sharing flow (guided publication)
- Semantic memory across sessions (per-project)
- Project and student assessment tracking
- Multiple teaching personas
- Content safety filtering

### Infrastructure and Interface
- Three LLM backends: Claude Code CLI, local vLLM, remote API
- Browser-based interface built with React: conversational office, question workshop, research sharing, project management, progress tracking, usage monitoring, and settings
- vLLM container management: start, stop, restart, status, health checks
- API usage budget tracking
- Tool usage tracking with real-time activity reporting
- Tool error reporting with status categories and user-facing warnings
- CLI: serve, init, doctor, reset, export, migrate, vllm-container

### Integration with External Resources
- Academic search: OpenAlex, Semantic Scholar, PubMed, arXiv, Europe PMC
- Researcher and institution discovery (OpenAlex, ORCID, ROR)
- Paper fetching and introduction extraction (8 open-access sources)
- Retraction Watch integration (local semantic search over retracted papers)
- Web search (Brave Search or Tavily)
