# Changelog

## v1.0.5 (2026-06-03)

- Fix vLLM throughput collapse during question generation under concurrency on the local vLLM backend (no effect on the default Claude CLI backend)
- Fix a crash that prevented generated questions from saving in the Formulating Research Question (phenomenon) workshop
- Questioned-Science workshop: show a clear error when no valid retraction cases can be curated, instead of presenting a fabricated placeholder case
- Robustness (fail-fast): removed silent fallbacks that could mask failures (empty curation, config loading, optional-import guards) and now log previously-swallowed exceptions, so errors surface instead of degrading quietly
- Update dependencies

## v1.0.4 (2026-04-10)

- Improve server responsiveness during local vision processing and file I/O - sync CPU-bound work now runs in `asyncio.to_thread`
- Improve PDF image extraction resilience - corrupt pages no longer crash artifact upload
- Skip tiny embedded images during PDF extraction (decorative icons, logos) via `vision.min_image_pixels` and `vision.min_image_bytes` thresholds
- Improve sqlite-vec KNN query accuracy - use `rowid IN (...)` pre-filtering instead of post-scan filtering
- Improve DB schema: remove stale defaults, add CHECK constraints on enum columns (migrations V30-V32)
- Process multiple embedded PDF images concurrently via `asyncio.gather`
- Use LANCZOS filter for high-quality image downscaling before vision inference
- Raise dependency version floors to reflect tested versions

## v1.0.3 (2026-04-08)

### Artifacts
- Typed uploads (7 types) with description, per-type validation and vision prompts
- Replace Docling with pdfplumber/python-docx/pypdfium2 - lighter dependencies
- Section-aware chunking for GROBID PDFs, inline description editing
- Pandas data analysis for CSV/TSV (describe, correlations, value counts)
- Statistical analysis: LLM-selected scipy/statsmodels tests (11 types), tier selection, retry with error feedback
- Pedagogical display: data summaries visible, AI descriptions reserved for mentor conversation

### Other
- Prompt injection protection for uploaded documents and user inputs
- Rewrite workshop descriptions, improve upload form and startup banner

## v1.0.2 (2026-04-06)

- Embedding model downloads in the background - server starts immediately
- Vision off by default; enable in Settings with backend choice (auto/Claude CLI/local/API)
- Local vision model (~4 GB, ~10 GB RAM) managed from Settings: download, remove, size and path info
- Improve local vision model compatibility (processor expected PIL Image, not file path)
- Improve version display

## v1.0.1 (2026-04-06)

- Improve assessment score display to use 0-100 scale matching backend
- Sync UI hypothesis stage names with backend graph
- Remove orphaned phase labels from stage configuration
- Add `aiosqlitepool` dependency

## v1.0.0 (2026-04-06)

Initial public release.

### Mentoring
- 21-node LangGraph adaptive mentoring agent
- Question Workshop: 9 pipelines + Hypothesis chat
- Research sharing flow, semantic memory, project assessment tracking
- Multiple teaching personas, content safety filtering

### Infrastructure and Interface
- Three LLM backends: Claude Code CLI, local vLLM, remote API
- React UI: office, question workshop, sharing, projects, progress, usage, settings
- vLLM container management, API budget tracking, tool activity reporting
- CLI: serve, init, doctor, reset, export, migrate, vllm-container

### External Integrations
- Academic search: OpenAlex, Semantic Scholar, PubMed, arXiv, Europe PMC
- Researcher/institution discovery (OpenAlex, ORCID, ROR)
- Paper fetching with introduction extraction (8 open-access sources)
- Retraction Watch (local semantic search), web search (Brave/Tavily)
