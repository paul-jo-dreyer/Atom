"""Benchmark env-stepping throughput, isolated from agent inference and
PPO updates.

What this measures
------------------
Pure VecEnv throughput with random actions. No policy forward pass, no
PPO update, no logging — just the cost of stepping every env in the
vector. That's what you're trying to scale when you change `--n-envs`,
`--use-subproc`, or anything else about the vectorisation strategy. The
PPO update time is separate; the training script's `time/fps` already
includes that, so it's not the right number for tuning vec-env config.

How to use it
-------------
Run individual configs and read off the fps. Sweep over (vec-env type,
n_envs) to find the knee:

    for n in 4 8 12 16 20; do
        .venv/bin/python -m AtomGym.training.benchmark_throughput \\
            --n-envs $n --use-subproc --duration 10
    done

For a flame-graph view of where time goes inside a chosen config:

    uv run --with py-spy py-spy record --rate 200 --subprocesses \\
        --output env_step_prof.svg -- \\
        .venv/bin/python -m AtomGym.training.benchmark_throughput \\
            --n-envs 14 --use-subproc --duration 30

`--subprocesses` is mandatory for SubprocVecEnv — without it py-spy only
samples the parent process, which is mostly waiting on pipes, so the
flame graph will look almost empty.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make sure `AtomGym` is importable regardless of how this script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from AtomGym.training.train import make_vec_env  # noqa: E402


def benchmark(
    *,
    n_envs: int,
    use_subproc: bool,
    duration: float,
    warmup: float,
    seed: int,
    max_episode_steps: int,
    stall_penalty: float,
    obstacle_contact_penalty: float,
) -> dict:
    """Step a vec env with random actions for `duration` seconds; return
    timing summary. `warmup` seconds of stepping happen first, unmeasured
    — that absorbs first-step costs (lazy imports finishing in workers,
    initial reset, JIT effects)."""
    vec_env = make_vec_env(
        n_envs=n_envs,
        base_seed=seed,
        use_subproc=use_subproc,
        max_episode_steps=max_episode_steps,
        stall_penalty=stall_penalty,
        obstacle_contact_penalty=obstacle_contact_penalty,
    )
    rng = np.random.default_rng(seed)

    try:
        vec_env.reset()

        # Warmup — get past first-step cost.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < warmup:
            actions = rng.uniform(-1.0, 1.0, size=(n_envs, 2)).astype(np.float32)
            vec_env.step(actions)

        # Measure.
        t_start = time.perf_counter()
        n_steps = 0
        while time.perf_counter() - t_start < duration:
            actions = rng.uniform(-1.0, 1.0, size=(n_envs, 2)).astype(np.float32)
            vec_env.step(actions)
            n_steps += 1
        elapsed = time.perf_counter() - t_start
    finally:
        vec_env.close()

    total_env_steps = n_steps * n_envs
    return {
        "n_envs": n_envs,
        "use_subproc": use_subproc,
        "elapsed_s": elapsed,
        "control_steps": n_steps,
        "total_env_steps": total_env_steps,
        "fps": total_env_steps / elapsed,
        "fps_per_env": total_env_steps / elapsed / n_envs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Pure env-stepping throughput benchmark.")
    p.add_argument("--n-envs", type=int, default=14)
    p.add_argument("--use-subproc", action="store_true")
    p.add_argument(
        "--duration", type=float, default=10.0,
        help="Wall-clock seconds to measure (after warmup).",
    )
    p.add_argument(
        "--warmup", type=float, default=1.0,
        help="Wall-clock seconds of unmeasured warmup before timing starts.",
    )
    p.add_argument("--seed", type=int, default=0)
    # Match the training script's defaults so what we measure here is the
    # same env config the training run uses.
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--stall-penalty", type=float, default=0.3)
    p.add_argument("--obstacle-contact-penalty", type=float, default=0.5)
    args = p.parse_args()

    cfg = "subproc" if args.use_subproc else "dummy"
    print(
        f"[bench] starting: n_envs={args.n_envs}  vecenv={cfg}  "
        f"warmup={args.warmup:.1f}s  measure={args.duration:.1f}s",
        flush=True,
    )

    r = benchmark(
        n_envs=args.n_envs,
        use_subproc=args.use_subproc,
        duration=args.duration,
        warmup=args.warmup,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        stall_penalty=args.stall_penalty,
        obstacle_contact_penalty=args.obstacle_contact_penalty,
    )

    # One-line summary, easy to grep / copy across runs.
    print(
        f"[bench]   result: vecenv={cfg:7s}  n_envs={r['n_envs']:2d}  "
        f"steps={r['total_env_steps']:>9,}  "
        f"fps={r['fps']:>8,.0f}  "
        f"fps_per_env={r['fps_per_env']:>6,.0f}  "
        f"elapsed={r['elapsed_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
