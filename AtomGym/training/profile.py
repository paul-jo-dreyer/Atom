"""Run py-spy on a target script, producing a flame-graph SVG.

Two targets:

  * `--target train`  — full training pipeline (env stepping + PPO update +
                        callbacks). Use this to see where ALL the time goes
                        in your real training run.
  * `--target bench`  — pure env stepping via `benchmark_throughput.py`,
                        no agent and no PPO update. Use this when you've
                        already established that env stepping is the
                        bottleneck and want to see *why*.

Examples
--------

Training-loop profile, 60 s of sampling, custom training args:

    .venv/bin/python -m AtomGym.training.profile \\
        --target train --duration 60 -- \\
        --total-timesteps 100_000 --n-envs 16 --use-subproc \\
        --batch-size 1024 --n-steps 1024

Pure env-stepping profile, 30 s:

    .venv/bin/python -m AtomGym.training.profile \\
        --target bench --duration 30 -- \\
        --n-envs 16 --use-subproc

Output lands at `training_runs/profiles/<target>_<YYYYMMDD_HHMMSS>.svg`.
Open the SVG in a browser. Click any frame to zoom in; the wider a frame,
the more time was spent in that call. Stack frames go bottom-up: leaf at
top is the function actually running when the sample fired.

Notes on py-spy
---------------
- `--subprocesses` is mandatory for SubprocVecEnv. Without it, py-spy
  only samples the parent process, which is mostly waiting on pipes
  during rollout — the flame graph will look almost empty.
- py-spy does NOT need ptrace caps / sudo when it launches the target
  itself (our case). It only needs caps when attaching with `--pid`.
- This script prefers a system py-spy (installed via `uv pip install
  py-spy` in the active venv, or globally). If not on PATH, it falls
  back to `uv run --with py-spy py-spy` for an ad-hoc invocation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROFILES_DIR = _REPO_ROOT / "training_runs" / "profiles"

_TARGETS = {
    "train": "AtomGym.training.train",
    "bench": "AtomGym.training.benchmark_throughput",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile training or env-stepping with py-spy.",
        # Don't auto-handle `--` in our parser — argparse recognises it as
        # a positional separator, which is exactly what we want for
        # passthrough.
    )
    p.add_argument(
        "--target", choices=sorted(_TARGETS.keys()), required=True,
        help="Which script to profile.",
    )
    p.add_argument(
        "--duration", type=int, default=None,
        help="Hard cap on sampling time, in seconds. Default: no cap — "
             "py-spy samples until the target exits naturally (control "
             "runtime via the target's own flags: `--duration` for "
             "`bench`, `--total-timesteps` for `train`). Setting this "
             "kills the target abruptly when reached, which can cause "
             "noisy teardown errors from SubprocVecEnv workers — usually "
             "cosmetic, the SVG still gets written.",
    )
    p.add_argument(
        "--rate", type=int, default=200,
        help="py-spy sample rate (Hz). 100-200 is plenty for most flame "
             "graphs; bump to 500 for very short hot-path investigations.",
    )
    p.add_argument(
        "--include-idle", action="store_true",
        help="Include time spent in syscalls / waiting on pipes. Default "
             "OFF: gives a CPU-time view, where workers blocked on "
             "_recv_bytes don't drown out actual compute. Turn on when "
             "investigating 'is the parent stuck waiting on something?' — "
             "but expect 16 SubprocVecEnv workers' idle pipe-waits to "
             "dominate any flame graph if you do.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=_PROFILES_DIR,
        help="Directory for the SVG. Created if missing.",
    )
    p.add_argument(
        "passthrough", nargs="*",
        help="Args to forward to the target script. Put `--` before them "
             "to separate from this script's own flags.",
    )
    args = p.parse_args()

    # Locate py-spy. Prefer a binary on PATH; fall back to ad-hoc uv-run.
    pyspy_path = shutil.which("py-spy")
    if pyspy_path is not None:
        pyspy_cmd = [pyspy_path]
    else:
        if shutil.which("uv") is None:
            print(
                "[profile] ERROR: neither py-spy nor uv on PATH. Install "
                "py-spy (`uv pip install py-spy` in the venv) or install "
                "uv so we can fall back to `uv run --with py-spy ...`.",
                file=sys.stderr,
            )
            sys.exit(1)
        pyspy_cmd = ["uv", "run", "--with", "py-spy", "py-spy"]
        print("[profile] py-spy not on PATH; using `uv run --with py-spy`.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"{args.target}_{ts}.svg"

    # Use the same Python interpreter we're running under so the target
    # inherits the venv's installed packages.
    target_cmd = [
        sys.executable,
        "-m",
        _TARGETS[args.target],
        *args.passthrough,
    ]

    cmd = [
        *pyspy_cmd,
        "record",
        "--rate", str(args.rate),
        "--subprocesses",     # capture SubprocVecEnv workers
        "--output", str(output),
    ]
    if args.include_idle:
        cmd.append("--idle")
    if args.duration is not None:
        cmd += ["--duration", str(args.duration)]
    cmd += ["--", *target_cmd]

    print(f"[profile] target  : {args.target} ({_TARGETS[args.target]})")
    duration_desc = (
        f"{args.duration}s (hard cap)"
        if args.duration is not None
        else "until target exits"
    )
    print(f"[profile] duration: {duration_desc}  rate: {args.rate}Hz")
    print(f"[profile] output  : {output}")
    print(f"[profile] cmd     : {' '.join(cmd)}")
    print()

    # Run cwd=_REPO_ROOT so `python -m AtomGym...` can find the package.
    result = subprocess.run(cmd, cwd=_REPO_ROOT)

    # py-spy can exit with a nonzero code even when the SVG was written
    # cleanly — typically a race between its post-target wait() and the
    # target's natural exit. Treat "file exists and is non-trivial" as
    # success; only fail loud if the SVG is missing or tiny.
    _MIN_VALID_SVG_BYTES = 4096
    if output.exists() and output.stat().st_size >= _MIN_VALID_SVG_BYTES:
        size_kb = output.stat().st_size / 1024
        print(f"\n[profile] flame graph saved → {output} ({size_kb:.0f} KB)")
        print("[profile] open in a browser; click any frame to zoom in.")
        if result.returncode != 0:
            print(
                f"[profile] (py-spy exited with code {result.returncode}, "
                f"likely a benign post-exit race — SVG looks complete.)"
            )
        sys.exit(0)

    print(
        f"\n[profile] py-spy exited with code {result.returncode} and the "
        f"SVG is missing or too small. Check the target script's stderr "
        f"above for the underlying failure.",
        file=sys.stderr,
    )
    sys.exit(result.returncode if result.returncode != 0 else 1)


if __name__ == "__main__":
    main()
