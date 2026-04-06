# Security

Research Mentor is a **single-user, localhost-only** application. It has no authentication by design — do not expose it to untrusted networks.

## What we protect against

- **XML attacks** (XXE, billion laughs) — all XML parsing uses `defusedxml`
- **Malicious uploads** — ZIP bomb detection, file size limits, extension allowlists, path traversal prevention
- **Download abuse** — streaming size caps on PDF and CSV downloads (100 MB / 200 MB)
- **SQL injection** — parameterized queries with column allowlists throughout
- **Command injection** — no `shell=True`, no `eval()`/`exec()`/`pickle`
- **Credential leakage** — API keys stored with restricted permissions, masked in API responses, never in source code
- **LLM query exfiltration** — two-layer outgoing query safety reviewer (regex + LLM) blocks queries containing emails, phone numbers, file paths, or credentials before they reach external search APIs

## What leaves your machine

With the `vllm` backend, conversation content stays on your machine. With `claude_cli` or `api` backends, conversation content is sent to the LLM provider (Anthropic, OpenAI, etc.).

Regardless of backend, external API calls send:

- Paper identifiers (DOIs, arXiv IDs) to metadata APIs
- Search queries to Brave Search or Tavily
- Your email (if configured) for polite API access
- API keys to their respective services over HTTPS

## Reporting a vulnerability

If you discover a security vulnerability, please report it privately. **Do not open a public issue.**

Email **hello@arpconnect.com** with:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge receipt within a week and provide a timeline for a fix.
