# Installation

This guide assumes you're starting from scratch — no tools installed yet.

Research Mentor requires Python 3.13. You don't need to install it yourself — the installer (`uv`) downloads it automatically into its own location, without affecting any Python you may already have.

## Windows

### Step 1: Install uv

Open **PowerShell** (search "PowerShell" in the Start menu) and paste this command:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Close and reopen PowerShell** after installation.

### Step 2: Install Research Mentor

```
uv tool install research-mentor --python 3.13
```

This downloads Python 3.13 automatically (does not affect any Python you may already have) and installs Research Mentor in an isolated environment.

**Close and reopen PowerShell** for the `research-mentor` command to become available.

## Mac

### Step 1: Install uv

Open **Terminal** (search "Terminal" in Spotlight) and run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal after installation.

### Step 2: Install Research Mentor

```bash
uv tool install research-mentor --python 3.13
```

## Linux

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal after installation.

### Step 2: Install Research Mentor

```bash
uv tool install research-mentor --python 3.13
```

## Start

```bash
research-mentor         # start the server (creates database automatically on first run)
```

On first launch, an embedding model (~1.2 GB) is downloaded automatically to `~/.cache/huggingface/`. This is a one-time download used for semantic search across your conversations and research.

Then open http://localhost:8080 in your browser.

Run `research-mentor doctor` at any time to check that everything is working.

---

## Choose an LLM Backend

Research Mentor needs a language model to work. Run `research-mentor doctor` — it will detect your hardware and tell you which options are available.

### Claude Code CLI (default — works on any computer)

Uses Anthropic's Claude via a subscription. No GPU needed.

1. Sign up for a [Claude Pro or Max subscription](https://claude.ai)
2. Install Node.js from https://nodejs.org/ (LTS version)
3. Install Claude Code CLI:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
4. Log in — this opens your browser to authenticate:
   ```bash
   claude
   ```
   When prompted, choose **"Use Claude subscription"** (not "API key"). Follow the prompts to connect your Claude account. Once logged in, you can close Claude Code with `Ctrl+C`.
5. Start the mentor:
   ```bash
   research-mentor
   ```

### Local vLLM (optional — if you have an NVIDIA GPU)

If `research-mentor doctor` detects a compatible NVIDIA GPU (16GB+ VRAM), you can run everything locally — nothing is sent to the internet.

The model runs in a container (Podman), so there's nothing extra to install in Python.

**One-time GPU setup (Linux / WSL2):**

```bash
# 1. Install Podman
sudo apt install podman            # Debian/Ubuntu
sudo dnf install podman            # Fedora/RHEL

# 2. Install nvidia-container-toolkit
sudo apt install nvidia-container-toolkit    # Debian/Ubuntu
sudo dnf install nvidia-container-toolkit    # Fedora/RHEL

# If not found in your distro repos (common on WSL2), add the NVIDIA repo first:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install nvidia-container-toolkit

# 3. Generate NVIDIA CDI spec (tells Podman how to access the GPU)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 4. Verify GPU access
podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
```

**Start vLLM:**

```bash
research-mentor vllm-container start      # pulls image + downloads model on first run
research-mentor vllm-container logs -f    # watch progress (model loading takes ~1-2 min)

# In another terminal:
research-mentor serve --backend vllm
```

**Managing the container:**

```bash
research-mentor vllm-container status    # check if running + API health
research-mentor vllm-container stop      # stop (fully releases GPU memory)
research-mentor vllm-container restart   # stop + start fresh
research-mentor vllm-container start --pull  # update to latest vLLM image
```

Model weights are stored on your machine (`~/.cache/huggingface/`), not inside the container.

Run `research-mentor doctor` to check that Podman, NVIDIA CDI, and the container are all configured correctly.

### Remote vLLM Server (institutional)

If your school or institution runs a shared vLLM server:

```bash
export RESEARCH_MENTOR_VLLM_API_BASE="https://vllm.your-institution.edu/v1"
research-mentor serve --backend vllm
```

### Switching Backends

You can switch backends in the UI sidebar at any time — no restart needed. Both backends use the same database and projects.

---

## System Requirements

- **Python:** 3.13 (installed automatically by uv — no manual install needed)
- **Disk space:** ~2 GB (for dependencies and embedding model)
- **RAM:** 2 GB minimum

### For local vision (optional)

Local vision lets the mentor interpret uploaded images, charts, and figures entirely on your machine — nothing is sent to the internet. It runs on CPU (no GPU needed) but requires more RAM and disk space.

- **Disk space:** ~4 GB additional (vision model weights)
- **RAM:** ~10 GB total during image processing (model is unloaded when idle)
- **Speed:** ~15–90 seconds per image on CPU (runs in the background — does not block the conversation)
- **Alternative:** If you use the Claude CLI backend, vision uses Claude instead (faster, no extra download needed)

### For local vLLM (optional)

- **NVIDIA GPU** with 16GB+ VRAM (RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, etc.)
- **Platform:** Linux, WSL2, or Windows with Podman Desktop
- **Not supported:** macOS (no NVIDIA GPU — use Claude CLI or API backend instead)
- **Disk space:** ~15GB additional (container image + model weights)

---

## Optional: Web Search

The mentor works without web search, but can give richer answers with it enabled.

1. Sign up at https://brave.com/search/api/ (free tier: 2,000 queries/month)
2. Save your API key:
   ```bash
   echo "BSA..." > ~/.research-mentor/brave_search_api_key
   ```

---

## Updating

```
uv tool upgrade research-mentor
```

---

## Uninstalling

### Step 1: Remove the package

```
uv tool uninstall research-mentor
```

### Step 2: Remove your data

Research Mentor stores all user data in `~/.research-mentor/`. This includes your database (conversations, projects, personas), uploaded files, logs, and local configuration.

**To delete everything:**

Windows (PowerShell):
```
Remove-Item -Recurse -Force "$env:USERPROFILE\.research-mentor"
```

Mac / Linux:
```
rm -rf ~/.research-mentor
```

### Step 3: Remove cached models (optional)

If you used the local embedding model or vLLM, model weights are cached in `~/.cache/huggingface/`. This folder is shared with other HuggingFace-based tools — only delete it if nothing else uses it.

```
rm -rf ~/.cache/huggingface
```

After these steps, Research Mentor is fully removed from your system.

---

## Troubleshooting

### General

- **"uv: command not found"**: Close and reopen your terminal. If still not found, re-run the uv install command from Step 1.
- **"research-mentor: command not found"**: Close and reopen your terminal. If still not found, run `uv tool update-shell` and reopen again.
- **Installation fails with dependency errors**: Try `uv tool install --force research-mentor --python 3.13` to reinstall cleanly.
- **Database errors after upgrade**: Run `research-mentor migrate` to update your database.

### vLLM Container

- **"unresolvable CDI devices"**: NVIDIA CDI spec is missing. Run: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- **"nvidia-ctk: command not found"**: Install nvidia-container-toolkit (see GPU setup above). On WSL2, you likely need to add the NVIDIA repo first.
- **"podman: command not found"**: Install Podman: `sudo apt install podman` (Debian/Ubuntu) or `sudo dnf install podman` (Fedora/RHEL).
- **Container starts but API never responds**: Check logs with `research-mentor vllm-container logs -f`. Common cause: GPU out of memory — try lowering `gpu_memory_utilization` in config.
- **Model downloading is slow**: First run downloads ~6-12GB of model weights. Subsequent starts reuse the cached weights from `~/.cache/huggingface/`.

