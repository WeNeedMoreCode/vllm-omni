# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate a local Cosmos3-Edge snapshot, then start the A10 timing server."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
REQUIRED = ("model_index.json", "transformer", "vae", "scheduler", "text_tokenizer")


def missing_parts(path):
    return [name for name in REQUIRED if not (path / name).exists()]


def resolve_model(explicit):
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        missing = missing_parts(candidate)
        if missing:
            raise ValueError(f"Invalid local model directory {candidate}; missing: {', '.join(missing)}")
        return candidate

    candidates = []
    configured = os.environ.get("MODEL_DIR", "")
    if configured:
        if "/path/to/" in configured:
            print("Ignoring placeholder MODEL_DIR; checking existing local snapshots.", flush=True)
        else:
            candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            REPO.parent / "models/Cosmos3-Edge",
            REPO / "models/Cosmos3-Edge",
            Path.home() / "cosmos-baseline/models/Cosmos3-Edge",
        ]
    )
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    snapshot = cache / "models--nvidia--Cosmos3-Edge/snapshots/344d602b128d1bbdacb43b08d0a3626f46343e29"
    candidates.append(snapshot)
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if not missing_parts(candidate):
            return candidate
    paths = "\n".join(f"  {path}" for path in checked)
    raise ValueError(
        "No complete local Cosmos3-Edge snapshot found. Checked:\n"
        + paths
        + "\nFind the real directory and pass --model-dir followed by that directory. "
        "No weights were downloaded."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Existing local Diffusers snapshot; autodetect if omitted.")
    parser.add_argument("--work-dir", type=Path, default=Path.home() / "cosmos-physicsiq")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate paths and print the command without loading a model."
    )
    args = parser.parse_args()
    try:
        model = resolve_model(args.model_dir)
    except ValueError as error:
        parser.error(str(error))
    executable = shutil.which("vllm")
    if executable is None:
        parser.error("vllm is not on PATH. Activate the Conda environment containing vLLM-Omni.")
    work_dir = args.work_dir.expanduser().resolve()
    print(f"Python: {sys.executable}", flush=True)
    print(f"Repository: {REPO}", flush=True)
    print(f"Model directory: {model}", flush=True)
    print(f"Work directory: {work_dir}", flush=True)
    for package in ("vllm", "vllm-omni"):
        try:
            print(f"{package}: {version(package)}", flush=True)
        except PackageNotFoundError:
            print(f"{package}: distribution metadata not found", flush=True)
    command = [
        executable,
        "serve",
        str(model),
        "--omni",
        "--model-class-name",
        "Cosmos3OmniDiffusersPipeline",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--diffusion-attention-backend",
        "TORCH_SDPA",
        "--no-guardrails",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--num-gpus",
        "1",
        "--init-timeout",
        "1800",
        "--stage-init-timeout",
        "1800",
    ]
    print("Command: " + shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_OMNI_VIDEO_SYNC_TIMEOUT"] = "7200"
    log_path = work_dir / "server_a10_timing.log"
    print(f"Server log: {log_path}", flush=True)
    print("Keep this terminal open. Ctrl+C stops the server.", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
