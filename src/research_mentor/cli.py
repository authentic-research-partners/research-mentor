"""CLI entry point for Personal Research Mentor."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

from research_mentor import __version__


def setup_logging(*, verbose: bool) -> None:
    """Configure loguru logging (stderr + rotating file)."""
    from research_mentor.config import load_config

    logger.remove()

    # Stderr sink: WARNING+ by default (banner handles user-facing info), DEBUG if --verbose
    stderr_level = "DEBUG" if verbose else "WARNING"
    logger.add(sys.stderr, level=stderr_level, format="<level>{level:<8}</level> | {message}")

    # File sink with rotation
    config = load_config()
    log_cfg = config.logging
    log_path = log_cfg.get_log_path()
    logger.add(
        str(log_path),
        level=log_cfg.level,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )
    logger.debug("Log file: {}", log_path)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="research-mentor")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Personal Research Mentor — AI-powered research mentor for individual students."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose=verbose)
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@main.command()
@click.option("--port", default=None, type=int, help="Port to listen on (default: 8080).")
@click.option("--host", default=None, help="Host to bind to (default: 127.0.0.1).")
@click.option(
    "--backend",
    type=click.Choice(["claude-cli", "vllm"]),
    default=None,
    help="LLM backend (default: from config).",
)
@click.option("--api-base", default=None, help="vLLM API base URL.")
@click.option("--model", default=None, help="Model name override.")
@click.pass_context
def serve(
    ctx: click.Context,
    port: int | None,
    host: str | None,
    backend: str | None,
    api_base: str | None,
    model: str | None,
) -> None:
    """Start the Personal Research Mentor server.

    The React UI is served automatically at the root URL (/) when built
    static files are present in the package.
    """
    import os

    import uvicorn

    # Apply CLI overrides as env vars (picked up by pydantic-settings)
    if backend:
        os.environ["RESEARCH_MENTOR_BACKEND"] = backend
    if api_base:
        os.environ["RESEARCH_MENTOR_API_BASE"] = api_base
    if model:
        os.environ["RESEARCH_MENTOR_MODEL"] = model

    # Load config after env vars are set
    from research_mentor.config import load_config

    config = load_config()
    serve_host = host or config.host

    # Auto-port: scan if user didn't pass --port explicitly, fail-fast if they did
    port_explicit = port is not None
    serve_port = _find_available_port(
        port or config.port, serve_host, explicit=port_explicit, service="server",
    )

    logger.info(
        "Starting Personal Research Mentor v{} on {}:{}", __version__, serve_host, serve_port,
    )
    logger.info("Backend: {}", config.backend)
    if config.backend == "vllm":
        logger.info("vLLM: model={} api_base={}", config.vllm.model, config.vllm.api_base)

    api_url = f"http://{serve_host}:{serve_port}"

    from research_mentor.server import app

    # Banner prints after lifespan startup completes (model download, DB init, etc.)
    app.state.startup_banner_url = api_url

    uvicorn.run(app, host=serve_host, port=serve_port, log_level="warning")

    # After uvicorn exits (Ctrl+C), show what's still running
    _print_shutdown_notice(config)


@main.command()
@click.option("--port", default=None, type=int, help="Backend API port (default: from config).")
@click.option("--host", default=None, help="Host to bind to (default: from config).")
@click.option("--ui-port", default=5173, type=int, help="Vite dev server port (default: 5173).")
@click.pass_context
def dev(ctx: click.Context, port: int | None, host: str | None, ui_port: int) -> None:
    """Start backend + React UI dev server for development.

    Runs the FastAPI backend and the Vite dev server side by side.
    The Vite dev server proxies /api/* requests to the backend.
    """
    import asyncio
    import os

    import uvicorn

    from research_mentor.config import load_config

    config = load_config()
    serve_host = host or config.host
    serve_port = _find_available_port(
        port or config.port, serve_host, explicit=port is not None, service="server",
    )

    ui_dir = Path(__file__).resolve().parent.parent.parent / "ui"
    if not ui_dir.exists():
        raise click.ClickException(
            f"React UI directory not found at {ui_dir}. "
            f"Dev mode requires a source checkout with the ui/ directory."
        )

    npm = shutil.which("npm")
    if npm is None:
        raise click.ClickException("npm is not installed. Install Node.js 18+ to use dev mode.")

    api_url = f"http://{serve_host}:{serve_port}"
    log_path = config.logging.get_log_path()

    from research_mentor.server import app

    async def _run_dev() -> None:
        server = uvicorn.Server(uvicorn.Config(
            app, host=serve_host, port=serve_port, log_level="warning",
        ))

        # Start backend
        click.echo("  Starting backend...", nl=False)
        backend_task = asyncio.create_task(server.serve())

        if not await _async_wait_for_backend(api_url, timeout=30):
            click.echo(" failed!")
            server.should_exit = True
            await backend_task
            raise click.ClickException(f"Backend did not start within 30s at {api_url}")
        click.echo(" ready.")

        # Start Vite dev server
        click.echo("  Starting Vite dev server...", nl=False)
        env = {**os.environ, "VITE_API_TARGET": api_url}
        vite_proc = await asyncio.create_subprocess_exec(
            npm, "run", "dev", "--", "--port", str(ui_port), "--host", serve_host,
            cwd=str(ui_dir),
            env=env,
        )
        click.echo(" ready.")

        ui_url = f"http://{serve_host}:{ui_port}"

        click.echo()
        click.echo(f"  Research Mentor v{__version__} (dev mode)")
        click.echo(f"  UI:   {ui_url}")
        click.echo(f"  API:  {api_url}/api/docs")
        click.echo(f"  Logs: {log_path}")
        click.echo()
        click.echo("  Press Ctrl+C to stop both servers.")
        click.echo()

        vite_task = asyncio.create_task(vite_proc.wait())

        try:
            # Wait for either to finish (Ctrl+C triggers CancelledError)
            done, _ = await asyncio.wait(
                [backend_task, vite_task], return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            pass
        finally:
            click.echo("\n  Shutting down...")
            server.should_exit = True
            if vite_proc.returncode is None:
                vite_proc.terminate()
            await asyncio.gather(backend_task, vite_task, return_exceptions=True)

    try:
        asyncio.run(_run_dev())
    except KeyboardInterrupt:
        pass


@main.command()
def init() -> None:
    """Initialize the database (creates tables, seeds personas)."""
    import asyncio

    from research_mentor.db.connection import SchemaMismatchError, init_db

    try:
        db_path = asyncio.run(init_db())
    except SchemaMismatchError as e:
        click.echo(f"Database exists but needs migration (v{e.current} → v{e.target}).")
        click.echo("Run 'research-mentor migrate' to update your database.")
        raise SystemExit(1)
    click.echo(f"Database initialized at {db_path}")


@main.command()
@click.option("--dry-run", is_flag=True, help="Show pending migrations without applying.")
def migrate(dry_run: bool) -> None:
    """Migrate the database to the latest schema version.

    Creates a backup of the current database, applies pending migrations
    to a copy, verifies each migration, and swaps the copy into place.
    The original database is never modified — it becomes a timestamped
    backup in ~/.research-mentor/backups/.
    """
    import asyncio

    from research_mentor.db.connection import _resolve_db_path
    from research_mentor.db.migration import get_migration_status, run_migration

    db_path = _resolve_db_path()
    if not db_path.exists():
        click.echo(f"No database found at {db_path}")
        click.echo("Run 'research-mentor init' to create a fresh database.")
        raise SystemExit(1)

    status = asyncio.run(get_migration_status(db_path))
    if status.pending_count == 0:
        click.echo(f"Database is already at v{status.current_version} (latest).")
        return

    click.echo(f"  Current: v{status.current_version}")
    click.echo(f"  Target:  v{status.target_version}")
    click.echo(f"  Pending: {status.pending_count} migration(s)")
    click.echo()

    if dry_run:
        click.echo("  (dry run — no changes made)")
        return

    result = asyncio.run(run_migration(db_path))
    if result.success:
        click.echo(f"  Migrated v{result.from_version} -> v{result.to_version}")
        click.echo(f"  Backup:  {result.backup_path}")
    else:
        click.echo(f"  Migration failed: {result.error}")
        click.echo("  Your database is unchanged.")
        raise SystemExit(1)


@main.command()
def doctor() -> None:
    """Check system dependencies and configuration."""
    from research_mentor.config import get_data_dir, load_config

    config = load_config()

    click.echo(f"Personal Research Mentor v{__version__}")
    click.echo()

    # Python
    click.echo("System:")
    click.echo(f"  Python:   {sys.version.split()[0]}")

    # Data directory
    data_dir = get_data_dir()
    click.echo(f"  Data dir: {data_dir}")

    # Database

    db_file = data_dir / config.db_path
    if db_file.exists():
        import asyncio as _asyncio

        from research_mentor.db.connection import get_schema_version
        from research_mentor.db.schema import SCHEMA_VERSION as _TARGET

        size_kb = db_file.stat().st_size / 1024
        current, _ = _asyncio.run(get_schema_version(db_file))
        version_info = f"schema v{current}"
        if current < _TARGET:
            version_info += f" (v{_TARGET} available — run 'research-mentor migrate')"
        click.echo(f"  Database: {db_file} ({size_kb:.0f} KB, {version_info})")
    else:
        click.echo(f"  Database: {db_file} (NOT INITIALIZED)")
        click.echo("    -> Run 'research-mentor init' to create")
    click.echo()

    # LLM backend
    click.echo(f"LLM backend: {config.backend}")
    if config.backend == "claude-cli":
        claude_path = shutil.which("claude")
        if claude_path:
            click.echo(f"  Claude CLI: {claude_path}")
        else:
            click.echo("  Claude CLI: NOT FOUND")
            click.echo("    -> Install: npm install -g @anthropic-ai/claude-code")
        click.echo(f"  Model:  {config.claude_cli.model}")
        click.echo(f"  Effort: {config.claude_cli.effort}")
    elif config.backend == "vllm":
        click.echo(f"  API base: {config.vllm.api_base}")
        click.echo(f"  Model:    {config.vllm.model}")
        click.echo(f"  Temp:     {config.vllm.temperature}")
        click.echo(
            f"  Tokens:   {config.vllm.max_input_tokens}in"
            f" / {config.vllm.max_completion_tokens}out"
        )

        # GPU check
        from research_mentor.gpu import detect_gpu

        gpu = detect_gpu()
        if gpu and gpu.vllm_compatible:
            fp8_label = "FP8 native" if gpu.fp8_native else "FP8 emulated (FP16 fallback)"
            click.echo(
                f"  GPU:      {gpu.name} ({gpu.vram_mb // 1024} GB, "
                f"compute {gpu.compute_capability}, {fp8_label})"
            )
            if not gpu.fp8_native:
                click.echo("    -> FP8 models will run in FP16 (uses ~2x VRAM)")
                click.echo("    -> Native FP8 requires RTX 40-series or newer (compute >= 8.9)")
        elif gpu:
            click.echo(f"  GPU:      {gpu.name} ({gpu.vram_mb // 1024} GB — vLLM needs >=16 GB)")
        else:
            click.echo("  GPU:      NVIDIA driver not found")
            click.echo("    -> vLLM requires an NVIDIA GPU with >=16 GB VRAM")

        # Container runtime
        runtime = config.vllm.container_runtime
        if runtime:
            name = (
                config.vllm.container_name
                or f"research-mentor-vllm-{config.vllm.model}"
            )
            click.echo(f"  Runtime:  {runtime}")
            if shutil.which(runtime):
                click.echo(f"    {runtime}: found")

                # Check CDI / GPU passthrough
                cdi_path = Path("/etc/cdi/nvidia.yaml")
                if cdi_path.exists():
                    click.echo("    GPU CDI: configured")
                else:
                    click.echo("    GPU CDI: NOT CONFIGURED")
                    click.echo(
                        "    -> sudo apt install nvidia-container-toolkit  (Debian/Ubuntu)"
                    )
                    click.echo(
                        "    -> sudo dnf install nvidia-container-toolkit  (Fedora/RHEL)"
                    )
                    click.echo(
                        "    -> sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
                    )

                if _container_running(runtime, name):
                    click.echo(f"  Container: {name} (running)")
                elif _container_exists(runtime, name):
                    click.echo(f"  Container: {name} (stopped)")
                    click.echo("    -> Start with: research-mentor vllm-container start")
                else:
                    click.echo(f"  Container: {name} (not created)")
                    click.echo("    -> Start with: research-mentor vllm-container start")
            else:
                click.echo(f"    {runtime}: NOT FOUND")
                click.echo(f"    -> sudo apt install {runtime}  (Debian/Ubuntu)")
                click.echo(f"    -> sudo dnf install {runtime}  (Fedora/RHEL)")

        # Check vLLM connectivity
        import httpx

        try:
            resp = httpx.get(f"{config.vllm.api_base}/models", timeout=5.0)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                names = [m["id"] for m in models]
                click.echo(f"  Status:   connected (models: {', '.join(names)})")
            else:
                click.echo(f"  Status:   error (HTTP {resp.status_code})")
        except httpx.ConnectError:
            click.echo(f"  Status:   NOT REACHABLE at {config.vllm.api_base}")
            from urllib.parse import urlparse

            vllm_host = urlparse(config.vllm.api_base).hostname or ""
            if vllm_host in ("localhost", "127.0.0.1", "::1"):
                click.echo("    -> Start vLLM: research-mentor vllm-container start")
            else:
                click.echo("    -> Check that the remote vLLM server is running and reachable")
        except Exception as e:
            click.echo(f"  Status:   error ({e})")
    click.echo()

    # Logging
    log_path = config.logging.get_log_path()
    click.echo("Logging:")
    click.echo(f"  File:      {log_path}")
    click.echo(f"  Rotation:  {config.logging.rotation}")
    click.echo(f"  Retention: {config.logging.retention} files")
    click.echo()

    # Server config
    click.echo(f"Server: {config.host}:{config.port}")


@main.command("vllm-server")
@click.option("--model", default=None, help="HuggingFace model ID (default: from config).")
@click.option("--served-name", default=None, help="Model name exposed via API (default: auto).")
@click.option("--port", default=None, type=int, help="Port for vLLM server (default: from config).")
@click.option(
    "--gpu-memory-utilization", default=None, type=float,
    help="GPU memory utilization 0.0–1.0 (default: from config).",
)
@click.option("--max-model-len", default=None, type=int, help="Max context length in tokens.")
@click.option("--restart", is_flag=True, help="Kill existing vLLM server before starting.")
@click.pass_context
def vllm_server(
    ctx: click.Context,
    model: str | None,
    served_name: str | None,
    port: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    restart: bool,
) -> None:
    """Start a local vLLM server with settings from config.

    When --model overrides the HuggingFace model ID, a matching served-model-name
    is derived automatically (e.g. "RedHatAI/gemma-3-12b-it-FP8-dynamic" becomes
    "gemma3-12b-fp8"). Use --served-name to set it explicitly.
    """
    import shlex
    import signal
    import subprocess
    from urllib.parse import urlparse

    from research_mentor.config import load_config

    config = load_config()
    vllm_cfg = config.vllm

    # Resolve parameters from config + CLI overrides
    hf_model = model or vllm_cfg.hf_model
    parsed_url = urlparse(vllm_cfg.api_base)
    config_port = parsed_url.port or 5001
    if port:
        serve_port = port
    else:
        # Try remembered port first, fall back to config
        last = _load_last_ports().get("vllm")
        if last is not None and last != config_port and _port_available("0.0.0.0", last):
            logger.info("Reusing last vLLM port {}", last)
            serve_port = last
        else:
            serve_port = config_port
    gpu_mem = gpu_memory_utilization or vllm_cfg.gpu_memory_utilization
    # 256-token padding for chat template markup, special tokens, tokenizer overhead
    default_len = vllm_cfg.max_input_tokens + vllm_cfg.max_completion_tokens + 256
    context_len = max_model_len or default_len

    # Derive served model name: explicit flag > config (if model not overridden) > auto
    if served_name:
        final_served_name = served_name
    elif not model:
        # No --model override, use config as-is
        final_served_name = vllm_cfg.model
    else:
        # --model was passed, derive a short name from the HuggingFace ID
        final_served_name = _derive_served_name(hf_model)

    # Check vllm is installed
    if not shutil.which("vllm"):
        click.echo("Error: vllm command not found.", err=True)
        click.echo("Install with: pip install vllm", err=True)
        raise SystemExit(1)

    # Handle --restart
    if restart:
        _kill_existing_vllm()

    # Check if already running — process-level check catches servers still loading
    vllm_procs = subprocess.run(
        ["pgrep", "-f", "vllm serve"], capture_output=True,
    )
    if vllm_procs.returncode == 0:
        click.echo("A vLLM server process is already running.")
        click.echo("Use --restart to kill it and start fresh.")
        raise SystemExit(1)

    # Also check if port is responding (e.g. externally started vLLM)
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{serve_port}/v1/models", timeout=3.0)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            click.echo(f"vLLM already running on port {serve_port} (models: {', '.join(models)})")
            click.echo("Use --restart to restart with different settings.")
            return
    except Exception:
        logger.debug("vLLM not reachable on port {}, will start", serve_port)

    cmd = [
        "vllm", "serve",
        "--host", "0.0.0.0",
        "--port", str(serve_port),
        "--model", hf_model,
        "--served-model-name", final_served_name,
        "--trust-remote-code",
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(context_len),
        "--kv-cache-dtype", "fp8",
        "--enable-chunked-prefill",
    ]
    if vllm_cfg.enforce_eager:
        cmd.append("--enforce-eager")

    _save_last_port("vllm", serve_port)
    click.echo("Starting vLLM server...")
    click.echo(f"  Model:      {hf_model}")
    click.echo(f"  Served as:  {final_served_name}")
    click.echo(f"  Port:       {serve_port}")
    click.echo(f"  Max length: {context_len} tokens")
    click.echo(f"  GPU memory: {gpu_mem}")
    if vllm_cfg.enforce_eager:
        click.echo("  Eager mode: enabled (CUDA graphs disabled)")
    click.echo()
    if final_served_name != vllm_cfg.model:
        click.echo(
            f"  NOTE: served name '{final_served_name}' differs from config "
            f"model '{vllm_cfg.model}'."
        )
        click.echo(
            f"  Update config or use: research-mentor serve --model {final_served_name}"
        )
        click.echo()
    click.echo("First run downloads the model. Subsequent runs use cache.")
    click.echo(f"  {shlex.join(cmd)}")
    click.echo()

    # Forward SIGINT/SIGTERM to the child process
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
    raise SystemExit(proc.returncode or 0)


@main.group("vllm-container")
def vllm_group() -> None:
    """Manage vLLM container (Podman/Docker)."""


def _get_container_runtime() -> tuple[str, str]:
    """Return (runtime_cmd, container_name) from config.

    Raises click.ClickException if container_runtime is not configured.
    """
    from research_mentor.config import load_config

    config = load_config()
    vllm_cfg = config.vllm
    runtime = vllm_cfg.container_runtime
    if not runtime:
        raise click.ClickException(
            "Container runtime not configured.\n"
            "Set container_runtime = \"podman\" (or \"docker\") in your config.toml "
            "under [llm.vllm]."
        )
    if runtime not in ("podman", "docker"):
        raise click.ClickException(
            f"Unknown container_runtime '{runtime}'. Must be 'podman' or 'docker'."
        )
    if not shutil.which(runtime):
        raise click.ClickException(
            f"'{runtime}' not found on PATH.\n"
            f"Install {runtime}:\n"
            f"  Debian/Ubuntu:  sudo apt install {runtime}\n"
            f"  Fedora/RHEL:    sudo dnf install {runtime}\n"
            f"  macOS:          brew install {runtime}"
        )
    # Container name: explicit config or derived from model name
    container_name = (
        vllm_cfg.container_name or f"research-mentor-vllm-{vllm_cfg.model}"
    )
    return runtime, container_name


def _container_exists(runtime: str, name: str) -> bool:
    """Check if a container with the given name exists (running or stopped)."""
    result = subprocess.run(
        [runtime, "container", "inspect", name],
        capture_output=True,
    )
    return result.returncode == 0


def _container_running(runtime: str, name: str) -> bool:
    """Check if a container with the given name is currently running."""
    result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _build_vllm_container_cmd(
    runtime: str, name: str, vllm_cfg: object,
) -> list[str]:
    """Build the `podman/docker run` command for vLLM from config."""
    from urllib.parse import urlparse

    parsed_url = urlparse(vllm_cfg.api_base)  # type: ignore[attr-defined]
    host_port = parsed_url.port or 5001

    cmd = [
        runtime, "run", "-d",
        "--name", name,
        "--device", "nvidia.com/gpu=all",
        "-p", f"{host_port}:8000",
        "-v", f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface",
        "--restart", "unless-stopped",
        vllm_cfg.container_image,  # type: ignore[attr-defined]
        "serve",
        vllm_cfg.hf_model,  # type: ignore[attr-defined]  # positional arg (vLLM 0.13+)
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", vllm_cfg.model,  # type: ignore[attr-defined]
        "--trust-remote-code",
        "--gpu-memory-utilization", str(vllm_cfg.gpu_memory_utilization),  # type: ignore[attr-defined]
        "--max-model-len", str(vllm_cfg.max_input_tokens + vllm_cfg.max_completion_tokens + 256),  # type: ignore[attr-defined]
        "--kv-cache-dtype", "fp8",
        "--enable-chunked-prefill",
    ]
    if vllm_cfg.enforce_eager:  # type: ignore[attr-defined]
        cmd.append("--enforce-eager")
    if vllm_cfg.reasoning_parser:  # type: ignore[attr-defined]
        cmd.extend(["--reasoning-parser", vllm_cfg.reasoning_parser])  # type: ignore[attr-defined]
    return cmd


@vllm_group.command("start")
@click.option("--pull", is_flag=True, help="Pull latest image before starting.")
@click.pass_context
def vllm_start(ctx: click.Context, pull: bool) -> None:
    """Start vLLM in a container."""
    from urllib.parse import urlparse

    from research_mentor.config import load_config

    runtime, name = _get_container_runtime()
    config = load_config()
    vllm_cfg = config.vllm

    # If container already exists and is running, report and exit
    if _container_running(runtime, name):
        click.echo(f"Container '{name}' is already running.")
        click.echo("Use 'research-mentor vllm-container restart' to restart.")
        return

    # If container exists but stopped, remove it (config may have changed)
    if _container_exists(runtime, name):
        click.echo(f"Removing stopped container '{name}'...")
        subprocess.run([runtime, "rm", name], capture_output=True)

    # Optionally pull latest image
    if pull:
        click.echo(f"Pulling {vllm_cfg.container_image}...")
        pull_result = subprocess.run([runtime, "pull", vllm_cfg.container_image])
        if pull_result.returncode != 0:
            raise click.ClickException(f"Failed to pull image: {vllm_cfg.container_image}")

    cmd = _build_vllm_container_cmd(runtime, name, vllm_cfg)

    parsed_url = urlparse(vllm_cfg.api_base)
    host_port = parsed_url.port or 5001

    click.echo(f"Starting vLLM container '{name}'...")
    click.echo(f"  Runtime:    {runtime}")
    click.echo(f"  Image:      {vllm_cfg.container_image}")
    click.echo(f"  Model:      {vllm_cfg.hf_model}")
    click.echo(f"  Served as:  {vllm_cfg.model}")
    click.echo(f"  Port:       {host_port}")
    click.echo(f"  GPU memory: {vllm_cfg.gpu_memory_utilization}")
    if vllm_cfg.enforce_eager:
        click.echo("  Eager mode: enabled")
    click.echo()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "invalid internal status" in stderr or "pause process" in stderr:
            raise click.ClickException(
                f"Container runtime internal state is corrupted.\n\n"
                f"Fix with:\n"
                f"  {runtime} system migrate\n\n"
                f"Then retry:\n"
                f"  research-mentor vllm-container start\n\n"
                f"Note: if other containers are running, this will briefly disrupt them."
            )
        if "CDI" in stderr or "unresolvable" in stderr:
            raise click.ClickException(
                "GPU passthrough not configured (CDI spec missing).\n\n"
                "Fix with:\n"
                "  # 1. Install nvidia-container-toolkit\n"
                "  sudo apt install nvidia-container-toolkit   # Debian/Ubuntu\n"
                "  sudo dnf install nvidia-container-toolkit   # Fedora/RHEL\n\n"
                "  # 2. Generate CDI spec\n"
                "  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml\n\n"
                "  # 3. Verify GPU access\n"
                "  podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi\n\n"
                "If nvidia-container-toolkit is not in your distro repos, add the NVIDIA repo:\n"
                "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/"
                "install-guide.html"
            )
        raise click.ClickException(f"Failed to start container:\n{stderr}")

    container_id = result.stdout.strip()[:12]
    click.echo(f"Container started: {container_id}")
    click.echo(f"  API will be available at {vllm_cfg.api_base}")
    click.echo()

    # Check if model is already cached on host
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = f"models--{vllm_cfg.hf_model.replace('/', '--')}"
    model_cached = (hf_cache_dir / model_cache_name).exists()

    if model_cached:
        click.echo("  Model weights already cached. Loading into GPU (~30-60s).")
    else:
        click.echo(
            "  First run: downloading model weights (~6-12 GB). "
            "This may take several minutes"
        )
        click.echo("  depending on your internet connection. Weights are cached for future runs.")
    if not vllm_cfg.enforce_eager:
        click.echo("  CUDA graph compilation adds ~1-3 min (set enforce_eager = true to skip).")
    click.echo()
    click.echo("  Check progress:  research-mentor vllm-container logs -f")
    click.echo("  Check status:    research-mentor vllm-container status")

    _save_last_port("vllm", host_port)


@vllm_group.command("stop")
def vllm_stop() -> None:
    """Stop the vLLM container."""
    runtime, name = _get_container_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    if not _container_running(runtime, name):
        click.echo(f"Container '{name}' is not running.")
        return

    click.echo(f"Stopping container '{name}'...")
    result = subprocess.run([runtime, "stop", name], capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to stop container:\n{result.stderr.strip()}")
    click.echo("Container stopped. GPU memory released.")


@vllm_group.command("restart")
@click.pass_context
def vllm_restart(ctx: click.Context) -> None:
    """Stop and start the vLLM container."""
    runtime, name = _get_container_runtime()

    # Stop and remove existing container
    if _container_exists(runtime, name):
        if _container_running(runtime, name):
            click.echo(f"Stopping container '{name}'...")
            subprocess.run([runtime, "stop", name], capture_output=True)
        click.echo(f"Removing container '{name}'...")
        subprocess.run([runtime, "rm", name], capture_output=True)

    # Start fresh
    ctx.invoke(vllm_start)


@vllm_group.command("status")
def vllm_status() -> None:
    """Show vLLM container status and API health."""
    runtime, name = _get_container_runtime()

    from research_mentor.config import load_config

    config = load_config()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        click.echo("Start it with: research-mentor vllm-container start")
        return

    # Container state
    inspect_result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True, text=True,
    )
    state = inspect_result.stdout.strip() if inspect_result.returncode == 0 else "unknown"
    click.echo(f"Container: {name}")
    click.echo(f"  State:  {state}")
    click.echo(f"  Image:  {config.vllm.container_image}")

    if state != "running":
        click.echo("  API:    not available (container not running)")
        return

    # Check API health
    import httpx

    try:
        resp = httpx.get(f"{config.vllm.api_base}/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            names = [m["id"] for m in models]
            click.echo(f"  API:    ready (models: {', '.join(names)})")
        else:
            click.echo(f"  API:    error (HTTP {resp.status_code})")
    except httpx.ConnectError:
        click.echo(f"  API:    loading (not responding yet at {config.vllm.api_base})")
    except Exception as e:
        click.echo(f"  API:    error ({e})")


@vllm_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f).")
@click.option("-n", "--tail", default=50, help="Number of lines to show (default: 50).")
def vllm_logs(follow: bool, tail: int) -> None:
    """Show vLLM container logs."""
    import signal

    runtime, name = _get_container_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    cmd = [runtime, "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(name)

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()


@vllm_group.command("rm")
@click.option("--force", is_flag=True, help="Force remove even if running.")
def vllm_rm(force: bool) -> None:
    """Remove the vLLM container (for image updates or cleanup)."""
    runtime, name = _get_container_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    if _container_running(runtime, name) and not force:
        click.echo(f"Container '{name}' is still running.")
        click.echo("Stop it first with 'research-mentor vllm-container stop', or use --force.")
        return

    cmd = [runtime, "rm"]
    if force:
        cmd.append("-f")
    cmd.append(name)

    click.echo(f"Removing container '{name}'...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to remove container:\n{result.stderr.strip()}")
    click.echo("Container removed.")


# ---------------------------------------------------------------------------
# GROBID container management
# ---------------------------------------------------------------------------


@main.group("grobid")
def grobid_group() -> None:
    """Manage GROBID container for academic PDF extraction (Podman/Docker)."""


def _get_grobid_runtime() -> tuple[str, str]:
    """Return (runtime_cmd, container_name) for GROBID from config."""
    from research_mentor.config import load_config

    config = load_config()
    runtime = config.vllm.container_runtime
    if not runtime:
        raise click.ClickException(
            "Container runtime not configured.\n"
            "Set container_runtime = \"podman\" (or \"docker\") in your config.toml "
            "under [llm.vllm]."
        )
    if runtime not in ("podman", "docker"):
        raise click.ClickException(
            f"Unknown container_runtime '{runtime}'. Must be 'podman' or 'docker'."
        )
    if not shutil.which(runtime):
        raise click.ClickException(
            f"'{runtime}' not found on PATH.\n"
            f"Install {runtime}:\n"
            f"  Debian/Ubuntu:  sudo apt install {runtime}\n"
            f"  Fedora/RHEL:    sudo dnf install {runtime}\n"
            f"  macOS:          brew install {runtime}"
        )
    return runtime, config.grobid.container_name


@grobid_group.command("start")
@click.option("--pull", is_flag=True, help="Pull latest image before starting.")
def grobid_start(pull: bool) -> None:
    """Start GROBID in a container (CPU-only, ~4-8 GB RAM)."""
    from urllib.parse import urlparse

    from research_mentor.config import load_config

    runtime, name = _get_grobid_runtime()
    config = load_config()
    grobid_cfg = config.grobid

    if _container_running(runtime, name):
        click.echo(f"Container '{name}' is already running.")
        click.echo("Use 'research-mentor grobid restart' to restart.")
        return

    if _container_exists(runtime, name):
        click.echo(f"Removing stopped container '{name}'...")
        subprocess.run([runtime, "rm", name], capture_output=True)

    if pull:
        click.echo(f"Pulling {grobid_cfg.container_image}...")
        pull_result = subprocess.run([runtime, "pull", grobid_cfg.container_image])
        if pull_result.returncode != 0:
            raise click.ClickException(f"Failed to pull image: {grobid_cfg.container_image}")

    parsed_url = urlparse(grobid_cfg.url)
    host_port = parsed_url.port or 8070

    cmd = [
        runtime, "run", "-d",
        "--name", name,
        "-p", f"{host_port}:8070",
        "--restart", "unless-stopped",
        grobid_cfg.container_image,
    ]

    click.echo(f"Starting GROBID container '{name}'...")
    click.echo(f"  Runtime: {runtime}")
    click.echo(f"  Image:   {grobid_cfg.container_image}")
    click.echo(f"  Port:    {host_port}")
    click.echo()

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to start container:\n{result.stderr.strip()}")

    container_id = result.stdout.strip()[:12]
    click.echo(f"Container started: {container_id}")
    click.echo(f"  GROBID API will be available at {grobid_cfg.url}")
    click.echo()
    click.echo("  GROBID takes ~30-60s to load CRF models on first start.")
    click.echo()
    click.echo("  Check progress:  research-mentor grobid logs -f")
    click.echo("  Check status:    research-mentor grobid status")


@grobid_group.command("stop")
def grobid_stop() -> None:
    """Stop the GROBID container."""
    runtime, name = _get_grobid_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    if not _container_running(runtime, name):
        click.echo(f"Container '{name}' is not running.")
        return

    click.echo(f"Stopping container '{name}'...")
    result = subprocess.run([runtime, "stop", name], capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to stop container:\n{result.stderr.strip()}")
    click.echo("Container stopped.")


@grobid_group.command("restart")
@click.pass_context
def grobid_restart(ctx: click.Context) -> None:
    """Stop and start the GROBID container."""
    runtime, name = _get_grobid_runtime()

    if _container_exists(runtime, name):
        if _container_running(runtime, name):
            click.echo(f"Stopping container '{name}'...")
            subprocess.run([runtime, "stop", name], capture_output=True)
        click.echo(f"Removing container '{name}'...")
        subprocess.run([runtime, "rm", name], capture_output=True)

    ctx.invoke(grobid_start)


@grobid_group.command("status")
def grobid_status() -> None:
    """Show GROBID container status and API health."""
    runtime, name = _get_grobid_runtime()

    from research_mentor.config import load_config

    config = load_config()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        click.echo("Start it with: research-mentor grobid start")
        return

    inspect_result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True, text=True,
    )
    state = inspect_result.stdout.strip() if inspect_result.returncode == 0 else "unknown"
    click.echo(f"Container: {name}")
    click.echo(f"  State:  {state}")
    click.echo(f"  Image:  {config.grobid.container_image}")

    if state != "running":
        click.echo("  API:    not available (container not running)")
        return

    import httpx

    try:
        resp = httpx.get(f"{config.grobid.url}/api/isalive", timeout=5.0)
        if resp.status_code == 200:
            click.echo(f"  API:    ready ({config.grobid.url})")
        else:
            click.echo(f"  API:    error (HTTP {resp.status_code})")
    except httpx.ConnectError:
        click.echo(f"  API:    loading (not responding yet at {config.grobid.url})")
    except Exception as e:
        click.echo(f"  API:    error ({e})")


@grobid_group.command("logs")
@click.option("-f", "--follow", is_flag=True, help="Follow log output (like tail -f).")
@click.option("-n", "--tail", default=50, help="Number of lines to show (default: 50).")
def grobid_logs(follow: bool, tail: int) -> None:
    """Show GROBID container logs."""
    import signal

    runtime, name = _get_grobid_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    cmd = [runtime, "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(name)

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()


@grobid_group.command("rm")
@click.option("--force", is_flag=True, help="Force remove even if running.")
def grobid_rm(force: bool) -> None:
    """Remove the GROBID container."""
    runtime, name = _get_grobid_runtime()

    if not _container_exists(runtime, name):
        click.echo(f"Container '{name}' does not exist.")
        return

    if _container_running(runtime, name) and not force:
        click.echo(f"Container '{name}' is still running.")
        click.echo("Stop it first with 'research-mentor grobid stop', or use --force.")
        return

    cmd = [runtime, "rm"]
    if force:
        cmd.append("-f")
    cmd.append(name)

    click.echo(f"Removing container '{name}'...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to remove container:\n{result.stderr.strip()}")
    click.echo("Container removed.")


@main.command()
@click.option(
    "--yes", "-y", is_flag=True, help="Skip confirmation prompt.",
)
def reset(yes: bool) -> None:
    """Wipe the database, checkpoints, and uploaded artifacts. Start fresh.

    Deletes the main database, LangGraph checkpoints, and all artifact files,
    then re-initializes with empty tables and seeded personas.
    """
    import asyncio
    import shutil

    from research_mentor.config import get_data_dir, load_config

    config = load_config()
    data_dir = get_data_dir()
    db_file = data_dir / config.db_path
    checkpoints_file = data_dir / "checkpoints.db"
    artifacts_dir = config.storage.get_artifacts_path()

    files_to_delete = [f for f in (db_file, checkpoints_file) if f.exists()]
    has_artifacts = artifacts_dir.exists() and any(artifacts_dir.iterdir())

    if not files_to_delete and not has_artifacts:
        click.echo("No database files or artifacts found. Nothing to reset.")
        return

    click.echo("This will delete:")
    for f in files_to_delete:
        size_kb = f.stat().st_size / 1024
        click.echo(f"  {f} ({size_kb:.0f} KB)")
    if has_artifacts:
        click.echo(f"  {artifacts_dir}/ (all uploaded files)")

    if not yes:
        click.confirm("Are you sure?", abort=True)

    for f in files_to_delete:
        # Also delete WAL/SHM journal files
        for suffix in ("", "-wal", "-shm"):
            p = f.parent / (f.name + suffix)
            if p.exists():
                p.unlink()
                logger.debug("Deleted {}", p)

    if has_artifacts:
        shutil.rmtree(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"  Deleted artifact files at {artifacts_dir}")

    from research_mentor.db.connection import init_db

    db_path = asyncio.run(init_db())
    click.echo(f"Database reset. Re-initialized at {db_path}")


@main.command()
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path (default: stdout).",
)
@click.option(
    "--project", "-p", "project_id", default=None,
    help="Export only a specific project by ID.",
)
def export(output: str | None, project_id: str | None) -> None:
    """Export all data as a ZIP: database JSON + uploaded artifact files."""
    import asyncio

    asyncio.run(_export_data(output, project_id))


async def _export_data(output: str | None, project_id: str | None) -> None:
    """Async implementation of export command."""
    import json as json_mod

    from research_mentor.db.connection import init_db
    from research_mentor.db.crud import (
        get_memories_for_project,
        list_artifacts,
        list_assessments,
        list_milestones,
        list_projects,
        list_sessions,
    )

    await init_db()

    projects_result = await list_projects(page_size=1000)
    projects = projects_result["data"]
    if project_id:
        projects = [p for p in projects if p["id"] == project_id]
        if not projects:
            click.echo(f"Project {project_id} not found.", err=True)
            raise SystemExit(1)

    exported: list[dict[str, Any]] = []
    for proj in projects:
        pid = proj["id"]
        proj_data = {
            "project": proj,
            "milestones": (await list_milestones(pid, page_size=1000))["data"],
            "sessions": (await list_sessions(pid, page_size=1000))["data"],
            "assessments": (await list_assessments(pid, page_size=1000))["data"],
            "artifacts": (await list_artifacts(pid, page_size=1000))["data"],
            "memories": (await get_memories_for_project(pid, page_size=1000))["data"],
        }

        # Fetch LangGraph checkpoint messages for each session
        await _export_session_messages(proj_data)

        exported.append(proj_data)

    import io
    import zipfile
    from datetime import datetime

    from research_mentor.config import load_config

    data_json = json_mod.dumps(exported, indent=2, default=str)

    # Build ZIP: data.json + artifact files
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.json", data_json)
        config = load_config()
        artifacts_root = config.storage.get_artifacts_path()
        if artifacts_root.exists():
            for file_path in artifacts_root.rglob("*"):
                if file_path.is_file():
                    arc_name = "artifacts/" + str(file_path.relative_to(artifacts_root))
                    zf.write(file_path, arc_name)
    buf.seek(0)

    if output:
        out_path = output if output.endswith(".zip") else output + ".zip"
        with open(out_path, "wb") as f:
            f.write(buf.getvalue())
        click.echo(f"Exported {len(exported)} project(s) to {out_path}")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d")
        out_path = f"research-mentor-export-{stamp}.zip"
        with open(out_path, "wb") as f:
            f.write(buf.getvalue())
        click.echo(f"Exported {len(exported)} project(s) to {out_path}")


async def _export_session_messages(proj_data: dict[str, Any]) -> None:
    """Attach LangGraph checkpoint messages to each session."""
    from research_mentor.config import get_data_dir

    checkpoints_file = get_data_dir() / "checkpoints.db"
    if not checkpoints_file.exists():
        return

    import aiosqlite

    async with aiosqlite.connect(str(checkpoints_file)) as db:
        db.row_factory = aiosqlite.Row
        for session in proj_data["sessions"]:
            thread_id = session["session_id"]
            try:
                cursor = await db.execute(
                    """
                    SELECT checkpoint
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id,),
                )
                row = await cursor.fetchone()
                if row and row["checkpoint"]:
                    import json as json_mod

                    checkpoint = json_mod.loads(row["checkpoint"])
                    # Extract messages from the channel_values
                    channel_values = checkpoint.get("channel_values", {})
                    messages = channel_values.get("messages", [])
                    session["messages"] = messages
            except Exception:
                logger.debug("Could not read checkpoints for session {}", thread_id)


_LAST_PORTS_FILE = Path.home() / ".research-mentor" / "last_ports.json"


def _load_last_ports() -> dict[str, int]:
    """Load remembered ports from ~/.research-mentor/last_ports.json."""
    try:
        return json.loads(_LAST_PORTS_FILE.read_text())  # type: ignore[no-any-return]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_last_port(service: str, port: int) -> None:
    """Remember which port a service last used successfully."""
    ports = _load_last_ports()
    if ports.get(service) == port:
        return  # already up to date
    ports[service] = port
    try:
        _LAST_PORTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_PORTS_FILE.write_text(json.dumps(ports, indent=2) + "\n")
    except OSError:
        pass  # non-critical


def _port_available(host: str, port: int) -> bool:
    """Check whether *port* can be bound on *host*.

    Uses SO_REUSEADDR so that TIME_WAIT ports (from a just-killed server) are
    considered available — matching what uvicorn does.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_available_port(
    preferred: int, host: str, *, max_attempts: int = 10, explicit: bool = False,
    service: str = "",
) -> int:
    """Find an available port starting from *preferred*.

    If *explicit* is True the user asked for this exact port — fail-fast instead
    of scanning.

    When *service* is given (e.g. ``"server"``, ``"ui"``), the last port that
    worked for this service is tried first.  If it's still free it is reused,
    keeping ports stable across restarts.
    """
    # Try the remembered port first (only when the user didn't pass --port)
    if service and not explicit:
        last = _load_last_ports().get(service)
        if last is not None and last != preferred and _port_available(host, last):
            logger.info("Reusing last {} port {}", service, last)
            return last

    for offset in range(max_attempts):
        candidate = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, candidate))
                if offset > 0:
                    logger.info("Port {} busy, using {}", preferred, candidate)
                if service:
                    _save_last_port(service, candidate)
                return candidate
        except OSError:
            if explicit:
                raise click.ClickException(
                    f"Port {preferred} is already in use. "
                    f"Free it or choose a different port with --port."
                )
    raise click.ClickException(
        f"No available port found in range {preferred}–{preferred + max_attempts - 1}."
    )


async def _async_wait_for_backend(api_url: str, timeout: int = 30) -> bool:
    """Wait until the backend responds to /api/health. Returns True if ready."""
    import asyncio
    import time

    import httpx

    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{api_url}/api/health", timeout=2.0)
                if r.status_code == 200:
                    return True
            except Exception:
                logger.debug("Health check not ready yet at {}", api_url)
            await asyncio.sleep(0.5)
    return False


def _container_status_line(runtime: str, name: str, api_url: str) -> str:
    """Return a one-line status string for a container."""
    import httpx

    if not shutil.which(runtime):
        return f"{runtime} not installed"

    # Check API first — it may be served by a different container or native process
    api_ready = False
    try:
        resp = httpx.get(api_url, timeout=3.0)
        api_ready = resp.status_code == 200
    except Exception:
        logger.debug("API not reachable at {}", api_url)

    if _container_running(runtime, name):
        return "running, API ready" if api_ready else "running, API loading..."
    if api_ready:
        return "API ready"  # served by something else, container name doesn't matter
    if _container_exists(runtime, name):
        return "stopped"
    return "not created"


def _has_gpu(min_vram_mb: int = 16000) -> bool:
    """Check if an NVIDIA GPU with sufficient VRAM is available."""
    from research_mentor.gpu import detect_gpu

    gpu = detect_gpu()
    return gpu is not None and gpu.vram_mb >= min_vram_mb


def _print_container_status(config: object) -> None:
    """Print container status for vLLM and GROBID."""
    from research_mentor.config import AppConfig

    assert isinstance(config, AppConfig)
    runtime = config.vllm.container_runtime
    if not runtime or not shutil.which(runtime):
        return  # no container runtime available — skip entire section

    show_vllm = _has_gpu()
    lines: list[str] = []

    # vLLM — only show if GPU with >=16GB VRAM is present
    if show_vllm:
        vllm_name = config.vllm.container_name or f"research-mentor-vllm-{config.vllm.model}"
        vllm_status = _container_status_line(
            runtime, vllm_name, f"{config.vllm.api_base}/models",
        )
        lines.append(f"    vLLM:   {vllm_status}")
        if "not" in vllm_status or "stopped" in vllm_status:
            lines.append("            Start with: research-mentor vllm-container start")

    # GROBID (CPU-only, enhanced PDF parsing)
    grobid_name = config.grobid.container_name
    grobid_status = _container_status_line(
        runtime, grobid_name, f"{config.grobid.url}/api/isalive",
    )
    lines.append(f"    GROBID (enhanced PDF parsing): {grobid_status}")
    if "not" in grobid_status or "stopped" in grobid_status:
        lines.append("            Start with: research-mentor grobid start")

    click.echo("  Services (optional):")
    for line in lines:
        click.echo(line)

    click.echo()
    if show_vllm:
        click.echo(
            "    Stop all: research-mentor vllm-container stop "
            "&& research-mentor grobid stop",
        )
    else:
        click.echo("    Stop: research-mentor grobid stop")
    click.echo()


def _print_startup_banner(api_url: str) -> None:
    """Print a tidy startup banner to the terminal."""
    from research_mentor.config import load_config

    config = load_config()

    click.echo()
    click.echo(f"  Research Mentor (v{__version__}) is running!")
    click.echo()
    click.echo(f"  Open in browser: {api_url}")
    click.echo()
    _print_container_status(config)
    click.echo("  Press Ctrl+C to stop the server.")
    click.echo()


def _print_shutdown_notice(config: object) -> None:
    """After server stops, warn about containers still consuming resources."""
    from research_mentor.config import AppConfig

    assert isinstance(config, AppConfig)
    runtime = config.vllm.container_runtime
    if not runtime or not shutil.which(runtime):
        return

    vllm_name = config.vllm.container_name or f"research-mentor-vllm-{config.vllm.model}"
    grobid_name = config.grobid.container_name

    running = []
    if _container_running(runtime, vllm_name):
        running.append(("vLLM", vllm_name, "research-mentor vllm-container stop"))
    if _container_running(runtime, grobid_name):
        running.append(("GROBID", grobid_name, "research-mentor grobid stop"))

    if not running:
        return

    click.echo()
    click.echo("  Note: these containers are still running:")
    for label, name, stop_cmd in running:
        click.echo(f"    {label} ({name})")
    click.echo()
    if len(running) == 1:
        click.echo(f"  Stop with: {running[0][2]}")
    else:
        click.echo("  Stop with:")
        for _, _, stop_cmd in running:
            click.echo(f"    {stop_cmd}")
    click.echo()


def _derive_served_name(hf_model: str) -> str:
    """Derive a short served-model-name from a HuggingFace model ID.

    Examples:
        "RedHatAI/Qwen3-8B-FP8-dynamic"           → "qwen3-8b-fp8"
        "RedHatAI/gemma-3-12b-it-FP8-dynamic"      → "gemma3-12b-fp8"
        "google/gemma-3-27b-it-qat"                 → "gemma3-27b-qat"
    """
    import re

    # Take the model name after the org prefix
    name = hf_model.split("/")[-1].lower()

    # Normalize: remove "dynamic", "-it", "instruct" suffixes
    name = re.sub(r"-dynamic$", "", name)
    name = re.sub(r"-it\b", "", name)
    name = re.sub(r"-instruct$", "", name)

    # Collapse "gemma-3" → "gemma3", "qwen-3" → "qwen3" etc.
    name = re.sub(r"([a-z]+)-(\d)", r"\1\2", name)

    # Clean up double dashes
    name = re.sub(r"-+", "-", name).strip("-")

    return name


def _kill_existing_vllm() -> None:
    """Kill any running vLLM server processes."""
    import subprocess
    import time

    result = subprocess.run(["pgrep", "-f", "vllm serve"], capture_output=True)
    if result.returncode == 0:
        click.echo("Stopping existing vLLM server...")
        subprocess.run(["pkill", "-f", "vllm serve"])
        time.sleep(3)
    else:
        click.echo("No existing vLLM server found.")
