"""Train a PPO policy on the Level-1 SoloEnv (single robot + ball, two goals).

Usage (run from the repo root, with the venv active):

    .venv/bin/python -m AtomGym.training.train --run-name l1_baseline
    # or:
    .venv/bin/python AtomGym/training/train.py --run-name l1_baseline

What this script does:
1. Builds a vector of `AtomSoloEnv`s (DummyVecEnv by default; SubprocVecEnv
   via `--use-subproc` for genuine parallelism on bigger machines).
2. Each env is configured with the Level-1 reward composition and the
   default initial-state DR. Tweak the constants in `make_l1_env` below.
3. Creates a PPO model with a 2×128 MLP policy head (and matching value
   head) and SB3's default PPO hyperparameters.
4. Trains, logging both standard SB3 metrics and per-reward-term means to
   TensorBoard, and saving periodic checkpoints.

Outputs land under `training_runs/<run_name>/`:
    tensorboard/        — TensorBoard event files
    checkpoints/        — periodic .zip checkpoints
    final.zip           — final policy

Recommended next steps after a run:
    tensorboard --logdir training_runs/<run_name>/tensorboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Make sure `AtomGym` is importable regardless of how this script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from AtomGym.environments import AtomSoloEnv, InitialStateRanges  # noqa: E402
from AtomGym.rewards import (  # noqa: E402
    BallProgressReward,
    DistanceToBallReward,
    GoalScoredReward,
)
from AtomGym.training.gif_eval_callback import GifEvalCallback  # noqa: E402


# ---------------------------------------------------------------------------
# Env factory — Level 1 default config
# ---------------------------------------------------------------------------


def make_l1_env(
    seed: int = 0,
    max_episode_steps: int = 400,
) -> AtomSoloEnv:
    """Construct a Level-1 env with the agreed reward composition and a
    light initial-state DR config. Edit here for first-pass tuning."""
    rewards = [
        # Dense shaping: reward speed-toward-goal and proximity-to-ball.
        BallProgressReward(weight=1.0),       # ~ m/s of ball progress toward +x goal
        DistanceToBallReward(weight=-0.5),    # negative ⟹ closer = better
        # Sparse terminal: large ± reward on a goal.
        GoalScoredReward(weight=50.0),
    ]
    init_ranges = InitialStateRanges(
        # Default: random pose, zero velocities. Once the policy can score
        # from a stationary ball, broaden the velocity ranges to make the
        # policy robust to a moving ball / off-axis approach.
    )
    return AtomSoloEnv(
        rewards=rewards,
        init_state_ranges=init_ranges,
        physics_dt=1.0 / 60.0,    # sim ticks at 60 Hz
        control_dt=1.0 / 30.0,    # policy emits actions at 30 Hz (action_repeat=2)
        max_episode_steps=max_episode_steps,
        seed=seed,
    )


def make_vec_env(
    n_envs: int, base_seed: int, use_subproc: bool, max_episode_steps: int
) -> VecEnv:
    # Wrap each env in SB3's Monitor so PPO can log rollout/ep_rew_mean and
    # rollout/ep_len_mean (it pulls those from `info["episode"]` which Monitor
    # injects on episode end). Monitor inherits gym.Wrapper, so attribute
    # lookups (env.world, env.robot, env.t, ...) still pass through to the
    # underlying AtomSoloEnv.
    factories = [
        (
            lambda i=i: Monitor(
                make_l1_env(seed=base_seed + i, max_episode_steps=max_episode_steps)
            )
        )
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


# ---------------------------------------------------------------------------
# Custom callback — log per-term reward breakdown to TensorBoard
# ---------------------------------------------------------------------------


class RewardBreakdownCallback(BaseCallback):
    """Pulls `info["reward_breakdown"]` off every env step and feeds it to
    SB3's logger via `record_mean`. SB3 dumps means at each rollout, so
    you get one TensorBoard scalar per reward term per rollout — useful
    for spotting which terms are driving learning vs. doing nothing."""

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            breakdown = info.get("reward_breakdown")
            if not breakdown:
                continue
            for name, value in breakdown.items():
                self.logger.record_mean(f"reward/{name}", float(value))
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_grid(s: str) -> tuple[int, int]:
    """Parse 'RxC' (e.g. '2x2') into (rows, cols) for argparse."""
    try:
        s_clean = s.lower().strip()
        if "x" in s_clean:
            r_str, c_str = s_clean.split("x", 1)
            r, c = int(r_str), int(c_str)
        else:
            r, c = 1, int(s_clean)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"--render-grid: expected 'RxC' (e.g. '2x2'), got {s!r}"
        ) from exc
    if r < 1 or c < 1:
        raise argparse.ArgumentTypeError(
            f"--render-grid: rows and cols must be >= 1, got {r}x{c}"
        )
    return r, c


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on AtomSoloEnv (Level 1).")
    parser.add_argument(
        "--run-name", type=str, default="l1_baseline",
        help="Name of this run. Outputs land under training_runs/<run_name>/.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000,
        help="Total environment steps (across all parallel envs).",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel envs. SB3's PPO batch comes from n_envs * n_steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed; each parallel env gets seed + worker_index.",
    )
    parser.add_argument(
        "--use-subproc", action="store_true",
        help="Use SubprocVecEnv (one process per env). Default is DummyVecEnv "
             "(single-process, easier to debug).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="PPO learning rate (default: 3e-4, SB3 default).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048,
        help="PPO rollout length per env. SB3 default 2048; lower if memory tight.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=400,
        help="Episode truncation length, in CONTROL steps. At control_dt=1/30s, "
             "400 steps ≈ 13.3 s of sim time per episode. Lower = more episode "
             "boundaries per training step ⟹ faster credit assignment on sparse rewards.",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="PPO entropy bonus coefficient. SB3 default is 0.0; we raise to 0.01 "
             "to push back on premature collapse to deterministic 'do nothing' actions. "
             "Tune up (0.02-0.05) if the policy keeps stalling out; tune down (0.005-0.0) "
             "once the policy is consistently scoring and you want it to commit.",
    )
    parser.add_argument(
        "--log-std-init", type=float, default=0.0,
        help="Initial log-std of the Gaussian action distribution. SB3 default 0.0 "
             "⟹ initial std=1.0. Bump to 0.3-0.5 (std ≈ 1.35-1.65) to widen the "
             "starting distribution and increase initial exploration.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50_000,
        help="Save a checkpoint every N total env steps.",
    )
    parser.add_argument(
        "--render-every", type=int, default=None,
        help="Render the current policy as a .gif every N total env steps "
             "(written to training_runs/<run_name>/gifs/). Disabled by default.",
    )
    parser.add_argument(
        "--render-eval-seed", type=int, default=999,
        help="Fixed seed for the eval-rollout env so progress over training "
             "is directly comparable — same scenario every render.",
    )
    parser.add_argument(
        "--render-frame-stride", type=int, default=1,
        help="Keep every Nth frame in the gif (higher = smaller files, "
             "choppier playback). Playback fps adjusts to keep 1 s gif = 1 s sim.",
    )
    parser.add_argument(
        "--render-max-seconds", type=float, default=None,
        help="Cap each render rollout to N seconds of sim time. Caps both "
             "the gif's playback length AND the wall-clock the callback "
             "spends each interval. Default: no cap (run a full episode).",
    )
    parser.add_argument(
        "--render-grid", type=_parse_grid, default=(1, 1),
        help="Render a composite gif laid out as a grid of independent "
             "rollouts. Format: 'RxC' (e.g. '2x2' for a 4-cell grid). Each "
             "cell uses --render-eval-seed + cell_index, so progress is "
             "directly comparable across renders. Default: 1x1 (single cell).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=_REPO_ROOT / "training_runs",
        help="Parent directory for run outputs.",
    )
    args = parser.parse_args()

    # Output directories
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tensorboard"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Vec env
    vec_env = make_vec_env(
        args.n_envs, args.seed, args.use_subproc, args.max_episode_steps
    )

    # PPO with 2x128 MLP for both policy and value heads.
    # `log_std_init` widens the initial action distribution → more
    # exploration during early training. `ent_coef` is the entropy bonus
    # that keeps the policy from collapsing to a low-std deterministic
    # mode before it has found the good actions.
    policy_kwargs: dict[str, Any] = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=args.log_std_init,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
    )

    # Callbacks
    # CheckpointCallback's save_freq is per-env, so divide the desired total-
    # step cadence by n_envs.
    save_freq_per_env = max(1, args.checkpoint_every // args.n_envs)
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        RewardBreakdownCallback(),
    ]
    if args.render_every is not None:
        gif_dir = output_dir / "gifs"
        rows, cols = args.render_grid
        callbacks.append(
            GifEvalCallback(
                eval_env_factory=lambda: make_l1_env(
                    seed=args.render_eval_seed,
                    max_episode_steps=args.max_episode_steps,
                ),
                render_every=args.render_every,
                save_dir=gif_dir,
                eval_seed=args.render_eval_seed,
                grid_rows=rows,
                grid_cols=cols,
                frame_stride=args.render_frame_stride,
                max_seconds=args.render_max_seconds,
                verbose=1,
            )
        )

    print(f"[train] run name      : {args.run_name}")
    print(f"[train] output dir    : {output_dir}")
    print(f"[train] total timesteps: {args.total_timesteps:,}")
    print(f"[train] n_envs        : {args.n_envs} ({'SubprocVecEnv' if args.use_subproc else 'DummyVecEnv'})")
    print(f"[train] policy net    : 2x128 MLP")
    print(f"[train] episode steps : {args.max_episode_steps}")
    print(f"[train] exploration   : ent_coef={args.ent_coef}  log_std_init={args.log_std_init}")
    print(f"[train] checkpoints   : every {args.checkpoint_every:,} steps → {ckpt_dir}")
    if args.render_every is not None:
        rows, cols = args.render_grid
        grid_str = f" ({rows}x{cols} grid)" if (rows, cols) != (1, 1) else ""
        print(f"[train] gifs          : every {args.render_every:,} steps{grid_str} → {output_dir / 'gifs'}")
    print(f"[train] tensorboard   : {tb_dir}")
    print()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.run_name,
    )

    final_path = output_dir / "final.zip"
    model.save(final_path)
    print(f"\n[train] training complete. Final model → {final_path}")
    print(f"[train] view metrics  : tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
