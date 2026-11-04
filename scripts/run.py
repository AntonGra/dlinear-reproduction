"""Run the unchanged author entry point; each run gets a new output directory."""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state():
    base = ROOT / "upstream"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError("Unexpected upstream commit")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=base, check=True)
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=base).decode().split("\0")
    return {p: digest(base / p) for p in paths if p}


def environment():
    for path in (".cache/matplotlib", ".cache/pip", ".cache/torch", "tmp", "runs"):
        (ROOT / path).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2", "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"), "TORCH_HOME": str(ROOT / ".cache/torch"),
        "PIP_CACHE_DIR": str(ROOT / ".cache/pip"), "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "TMPDIR": str(ROOT / "tmp"),
    })
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="A new name, for example pilot-01")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.run_id):
        parser.error("Use a simple run name")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use this project's .venv/bin/python")
    sources = source_state()
    data_source = json.loads((ROOT / "records/data-source.json").read_text())
    for item in data_source["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Data provenance check failed")
    config = json.loads((ROOT / "configs/etth1-336-96.json").read_text())
    command = [sys.executable, "-B", "-u", str(ROOT / "upstream/run_longExp.py")]
    for key, value in config.items():
        command.extend(["--" + key, str(value)])
    command.extend(["--root_path", str(ROOT / "data")])
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment()
    metadata = {
        "upstream_commit": UPSTREAM_COMMIT, "source_hashes": sources,
        "config": config, "command": command, "cwd": str(output),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(ROOT / "configs/etth1-336-96.json"),
        "runner_sha256": digest(Path(__file__)),
        "data_sha256": digest(ROOT / "data/ETTh1.csv"),
        "environment_controls": {k: env[k] for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], env=env, text=True)
    (output / "environment.txt").write_text(freeze)
    start = time.monotonic()
    print(f"Starting {args.run_id}; log: {output / 'train.log'}", flush=True)
    with (output / "train.log").open("x") as log:
        process = subprocess.Popen(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.terminate()
            code = process.wait(timeout=30)
            metadata["timed_out"] = True
    metadata.update({"exit_code": code, "elapsed_seconds": time.monotonic() - start,
                     "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                     "sources_unchanged": source_state() == sources})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("exit_code", "elapsed_seconds", "sources_unchanged")}))
    if not metadata["sources_unchanged"]:
        raise RuntimeError("Upstream sources changed during run")
    sys.exit(code or (1 if metadata.get("timed_out") else 0))


if __name__ == "__main__":
    main()
