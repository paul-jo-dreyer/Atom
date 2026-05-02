"""Train a 1v1 self-play PPO policy on AtomTeamEnv (Level 2).

Usage (from repo root, venv active):

    .venv/bin/python -m AtomGym.training.train_team --run-name l2_baseline \
        --n-envs 16 --use-subproc

What this script does
---------------------
1. Builds N `AtomTeamEnv` workers, each wrapped in a `TeamWorkerWrapper`
   that holds its own CPU-only `OpponentRunner` (architecturally
   matching the learner). The runner's `predict` is bound as the env's
   `opponent_policy`; the env queries it on every step.
2. Constructs a master `SnapshotPool`, plus the curriculum `ReferenceOpponent`
   and `WinRateTracker`. These live on the main process — workers only
   ever see pool replicas pushed via `env_method`.
3. Creates the PPO learner with the agreed-upon arch (2×128 MLP for both
   pi and vf heads). The same `policy_kwargs` is also passed to every
   worker's runner and to the reference, so all three architectures
   stay in sync.
4. Wires callbacks:
     * `PoolSyncCallback` — every `--snapshot-every` steps, snapshots the
       learner's weights, adds to pool, broadcasts to workers.
     * `RefEvalCallback` — every `--eval-every` steps, plays K
       deterministic episodes against the reference, logs win rate to
       TB, promotes when threshold met.
     * `GifEvalCallback` (existing, now team-aware) — periodic render.
     * `CheckpointCallback` (existing) — periodic .zip dumps.
     * `RewardBreakdownCallback` (existing) — per-term reward logs.
5. Trains.

Resume support
--------------
`--resume PATH` works the same way as in `train.py` (absolute-target
semantics for `--total-timesteps`). After loading, the script bootstraps
the snapshot pool with the loaded policy at `iteration=loaded_steps`,
so workers have a usable opponent immediately on resume rather than
waiting until the first `--snapshot-every` boundary.

Outputs land under `training_runs/<run_name>/`:
    tensorboard/        — TensorBoard event files
    checkpoints/        — periodic .zip checkpoints
    gifs/               — periodic episode renders (if --render-every set)
    final.zip           — final policy
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
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from AtomGym.environments import AtomTeamEnv, InitialStateRanges  # noqa: E402
from AtomGym.rewards import (  # noqa: E402
    BallAlignmentReward,
    BallProgressReward,
    DistanceToBallReward,
    GoalScoredReward,
    ObstacleContactPenalty,
    StallPenaltyReward,
    StaticFieldPenalty,
)
from AtomGym.training.gif_eval_callback import GifEvalCallback  # noqa: E402
from AtomGym.training._shadow_policy import state_dict_to_numpy  # noqa: E402
from AtomGym.training.opponent_runner import OpponentRunner  # noqa: E402
from AtomGym.training.pool_sync_callback import PoolSyncCallback  # noqa: E402
from AtomGym.training.ref_eval_callback import RefEvalCallback  # noqa: E402
from AtomGym.training.reference_opponent import ReferenceOpponent  # noqa: E402
from AtomGym.training.snapshot_pool import SnapshotPool  # noqa: E402
from AtomGym.training.team_worker_wrapper import TeamWorkerWrapper  # noqa: E402
from AtomGym.training.train import (  # noqa: E402
    RewardBreakdownCallback,
    _parse_grid,
)
from AtomGym.training.win_rate_tracker import WinRateTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Env factory — Level 2 default config
# ---------------------------------------------------------------------------


def make_l2_env(
    seed: int = 0,
    max_episode_steps: int = 400,
    stall_penalty: float = 0.3,
    obstacle_contact_penalty: float = 0.5,
    ball_alignment: float = 0.3,
    static_field_penalty: float = 0.0,
) -> AtomTeamEnv:
    """Construct a Level-2 (1v1) env with the agreed reward composition.
    Mirrors `make_l1_env` from train.py — same reward weights, same
    init-state DR — so a solo-trained policy transfers cleanly via
    weight transfer (step 7) without reward shifts confusing the
    objective."""
    rewards: list = [
        BallProgressReward(weight=1.0),
        DistanceToBallReward(weight=-0.5),
        GoalScoredReward(weight=50.0),
    ]
    if stall_penalty > 0.0:
        rewards.append(StallPenaltyReward(weight=-stall_penalty))
    if obstacle_contact_penalty > 0.0:
        # Now also penalises robot-robot contact (CATEGORY_ROBOT is in
        # _OBSTACLE_CATEGORIES) — useful in 1v1 to discourage purely
        # defensive collision strategies.
        rewards.append(ObstacleContactPenalty(weight=-obstacle_contact_penalty))
    if ball_alignment > 0.0:
        rewards.append(BallAlignmentReward(weight=ball_alignment))
    if static_field_penalty > 0.0:
        # Anticipatory shaping for collision-free motion + opposing
        # goalie-box rule (penalty 0 at box edge, ramps with intrusion).
        # Geometry + sigmoid params live in StaticFieldPenalty's class
        # defaults; edit there to retune. Each subproc builds its own
        # grid in __init__ — the cost is ~ms, dwarfed by torch + sim_py
        # imports each worker already pays.
        rewards.append(StaticFieldPenalty(weight=-static_field_penalty))
    init_ranges = InitialStateRanges(
        ball_speed=(0.0, 0.3),
    )
    return AtomTeamEnv(
        rewards=rewards,
        init_state_ranges=init_ranges,
        physics_dt=1.0 / 60.0,
        control_dt=1.0 / 30.0,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )


def _make_team_worker(
    seed: int,
    policy_kwargs: dict[str, Any],
    eps_latest: float,
    max_episode_steps: int,
    stall_penalty: float,
    obstacle_contact_penalty: float,
    ball_alignment: float,
    static_field_penalty: float,
) -> TeamWorkerWrapper:
    """Per-worker factory: builds AtomTeamEnv + OpponentRunner + wrapper.

    Constructed *inside the subproc* (one call per worker), so the
    OpponentRunner's torch import + CPU shadow policy live in the
    worker's address space. Main process never imports torch on the
    workers' behalf."""
    team_env = make_l2_env(
        seed=seed,
        max_episode_steps=max_episode_steps,
        stall_penalty=stall_penalty,
        obstacle_contact_penalty=obstacle_contact_penalty,
        ball_alignment=ball_alignment,
        static_field_penalty=static_field_penalty,
    )
    runner = OpponentRunner(
        observation_space=team_env.observation_space,
        action_space=team_env.action_space,
        policy_kwargs=policy_kwargs,
        eps_latest=eps_latest,
        seed=seed,  # worker-local RNG seed
    )
    return TeamWorkerWrapper(team_env, runner)


def make_vec_env(
    n_envs: int,
    base_seed: int,
    use_subproc: bool,
    policy_kwargs: dict[str, Any],
    eps_latest: float,
    max_episode_steps: int,
    stall_penalty: float,
    obstacle_contact_penalty: float,
    ball_alignment: float,
    static_field_penalty: float,
) -> VecEnv:
    factories = [
        (
            lambda i=i: Monitor(
                _make_team_worker(
                    seed=base_seed + i,
                    policy_kwargs=policy_kwargs,
                    eps_latest=eps_latest,
                    max_episode_steps=max_episode_steps,
                    stall_penalty=stall_penalty,
                    obstacle_contact_penalty=obstacle_contact_penalty,
                    ball_alignment=ball_alignment,
                    static_field_penalty=static_field_penalty,
                )
            )
        )
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO 1v1 self-play on AtomTeamEnv (Level 2)."
    )
    # ---- basic run config (mirrors train.py) ----------------------------
    parser.add_argument("--run-name", type=str, default="l2_baseline")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-subproc", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--stall-penalty", type=float, default=0.3)
    parser.add_argument("--obstacle-contact-penalty", type=float, default=0.5)
    parser.add_argument("--ball-alignment", type=float, default=0.3)
    parser.add_argument(
        "--static-field-penalty",
        type=float,
        default=0.0,
        help="Weight (positive magnitude) on the StaticFieldPenalty term — "
        "anticipatory shaping for walls + opposing goalie box. Per-step "
        "value is bounded in [0, 1]; weight is comparable to the "
        "obstacle-contact penalty per second of saturated proximity. "
        "0 disables. Edit StaticFieldPenalty class defaults to retune "
        "geometry / sigmoid params.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--disable-reward-breakdown", action="store_true")
    parser.add_argument("--render-every", type=int, default=None)
    parser.add_argument("--render-eval-seed", type=int, default=999)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--render-max-seconds", type=float, default=None)
    parser.add_argument("--render-grid", type=_parse_grid, default=(1, 1))
    parser.add_argument(
        "--output-root", type=Path, default=_REPO_ROOT / "training_runs"
    )
    parser.add_argument("--resume", type=Path, default=None)

    # ---- self-play / pool config ---------------------------------------
    parser.add_argument(
        "--pool-capacity",
        type=int,
        default=20,
        help="Maximum number of past snapshots kept in the pool. FIFO "
        "eviction once full.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=500_000,
        help="Add a snapshot of the learner to the pool every N total "
        "env steps and broadcast to workers.",
    )
    parser.add_argument(
        "--eps-latest",
        type=float,
        default=0.5,
        help="Probability that a worker samples the LATEST pool entry "
        "as its opponent for the next rollout (vs. uniform from pool). "
        "0.5 is a balanced default — keeps gradient signal sharp at the "
        "frontier while spreading rollouts across history.",
    )

    # ---- reference / win-rate config -----------------------------------
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250_000,
        help="Run a curriculum-eval cycle every N total env steps "
        "(deterministic episodes vs. the current reference, win rate "
        "logged to TB, promotion gate evaluated).",
    )
    parser.add_argument(
        "--eval-episodes-per-cycle",
        type=int,
        default=10,
        help="Eval episodes per cycle. With window=50 the window fills "
        "after ~5 cycles ≈ --eval-every × 5 timesteps.",
    )
    parser.add_argument(
        "--win-rate-window",
        type=int,
        default=50,
        help="Sliding-window size for win-rate-vs-reference. Larger ⟹ "
        "lower variance, slower promotion. The window also acts as the "
        "post-promotion cooldown — no further promotion can fire until "
        "the new reference has accumulated this many fresh outcomes.",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.80,
        help="Win rate (chess-style: W=1, T=0.5, L=0) at or above which "
        "the reference is promoted to pool.latest(). 0.80 is the "
        "starting value; tune up if the curriculum advances too quickly.",
    )

    args = parser.parse_args()

    # Output directories
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tensorboard"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Architecture spec — shared between learner, every worker's runner,
    # and the reference opponent.
    policy_kwargs: dict[str, Any] = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=args.log_std_init,
    )

    # Vec env — each worker holds its own OpponentRunner + CPU shadow.
    vec_env = make_vec_env(
        n_envs=args.n_envs,
        base_seed=args.seed,
        use_subproc=args.use_subproc,
        policy_kwargs=policy_kwargs,
        eps_latest=args.eps_latest,
        max_episode_steps=args.max_episode_steps,
        stall_penalty=args.stall_penalty,
        obstacle_contact_penalty=args.obstacle_contact_penalty,
        ball_alignment=args.ball_alignment,
        static_field_penalty=args.static_field_penalty,
    )

    # Master pool + reference + tracker. Live on main; workers see only
    # pool replicas pushed via env_method.
    pool = SnapshotPool(capacity=args.pool_capacity)
    # Construct reference + sample obs/action spaces from a one-shot env
    # so we can build the reference's shadow policy without a full env
    # round-trip. The training envs in vec_env are already constructed
    # with the same spaces, so this is just a metadata read.
    reference_proto = make_l2_env()  # discarded after spaces lookup
    reference = ReferenceOpponent(
        observation_space=reference_proto.observation_space,
        action_space=reference_proto.action_space,
        policy_kwargs=policy_kwargs,
    )
    del reference_proto
    tracker = WinRateTracker(window_size=args.win_rate_window)

    # Model — load on resume, else build fresh.
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(
                f"--resume: checkpoint file not found: {args.resume}"
            )
        model = PPO.load(
            str(args.resume),
            env=vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            tensorboard_log=str(tb_dir),
            # Explicit verbose=1 — guarantees SB3's stdout tabular
            # output (rollout/, train/, time/) shows after resume.
            verbose=1,
        )
        loaded_steps = int(model.num_timesteps)
        remaining_steps = max(0, args.total_timesteps - loaded_steps)
        if remaining_steps == 0:
            print(
                f"[train_team] resume target {args.total_timesteps:,} already "
                f"reached (checkpoint at {loaded_steps:,}). Nothing to do."
            )
            return
        # Bootstrap pool with the loaded policy so workers have an
        # opponent immediately on resume — otherwise they'd run zero-
        # action opponents until the next snapshot_every boundary.
        # Convert to numpy first (matches PoolSyncCallback's pattern;
        # avoids torch tensor FD leaks across SubprocVec env_method).
        sd = state_dict_to_numpy(model.policy.state_dict())
        pool.add(sd, iteration=loaded_steps)
        vec_env.env_method("update_opponent_pool", pool)
        print(
            f"[train_team] bootstrapped pool with resumed policy at "
            f"iter {loaded_steps:,}; workers synced."
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(tb_dir),
            seed=args.seed,
        )
        loaded_steps = 0
        remaining_steps = args.total_timesteps

    # Callbacks
    save_freq_per_env = max(1, args.checkpoint_every // args.n_envs)
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=save_freq_per_env,
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        PoolSyncCallback(
            pool=pool,
            vec_env=vec_env,
            snapshot_every=args.snapshot_every,
            verbose=1,
        ),
        RefEvalCallback(
            eval_env_factory=lambda: make_l2_env(
                seed=args.seed + 10_000,  # eval RNG separated from training
                max_episode_steps=args.max_episode_steps,
                stall_penalty=args.stall_penalty,
                obstacle_contact_penalty=args.obstacle_contact_penalty,
                ball_alignment=args.ball_alignment,
                static_field_penalty=args.static_field_penalty,
            ),
            pool=pool,
            reference=reference,
            tracker=tracker,
            eval_every=args.eval_every,
            episodes_per_cycle=args.eval_episodes_per_cycle,
            promotion_threshold=args.promotion_threshold,
            verbose=1,
        ),
    ]
    if not args.disable_reward_breakdown:
        callbacks.append(RewardBreakdownCallback())
    if args.render_every is not None:
        gif_dir = output_dir / "gifs"
        rows, cols = args.render_grid

        # GIF eval envs need an opponent too — without this they fall
        # back to the env's default (zero-action) opponent and the
        # render shows the learner playing against a stationary body
        # forever. We bind the curriculum reference's `predict` so the
        # GIF shows the same matchup the win-rate gate is tracking;
        # progress is then visually comparable across renders.
        # Early training: reference isn't set yet → predict returns
        # zeros (same as today), and GIFs become real once the first
        # snapshot lands in the pool and the first eval cycle promotes
        # it to the reference.
        def _make_gif_eval_env() -> AtomTeamEnv:
            env = make_l2_env(
                seed=args.render_eval_seed,
                max_episode_steps=args.max_episode_steps,
                stall_penalty=args.stall_penalty,
                obstacle_contact_penalty=args.obstacle_contact_penalty,
                ball_alignment=args.ball_alignment,
                static_field_penalty=args.static_field_penalty,
            )
            env.set_opponent_policy(reference.predict)
            return env

        callbacks.append(
            GifEvalCallback(
                eval_env_factory=_make_gif_eval_env,
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

    # Header
    print(f"[train_team] run name        : {args.run_name}")
    print(f"[train_team] output dir      : {output_dir}")
    if args.resume is not None:
        print(f"[train_team] resume from     : {args.resume}")
        print(f"[train_team] resumed at      : {loaded_steps:,} env steps")
        print(
            f"[train_team] remaining       : {remaining_steps:,} steps "
            f"to reach target {args.total_timesteps:,}"
        )
    else:
        print(f"[train_team] total timesteps : {args.total_timesteps:,}")
    print(
        f"[train_team] n_envs          : {args.n_envs} "
        f"({'SubprocVecEnv' if args.use_subproc else 'DummyVecEnv'})"
    )
    print(f"[train_team] policy net      : 2x128 MLP")
    print(f"[train_team] episode steps   : {args.max_episode_steps}")
    print(
        f"[train_team] exploration     : ent_coef={args.ent_coef}  "
        f"log_std_init={args.log_std_init}"
    )
    if args.stall_penalty > 0:
        print(f"[train_team] stall penalty   : weight=-{args.stall_penalty}")
    if args.obstacle_contact_penalty > 0:
        print(
            f"[train_team] obstacle pen.   : weight=-{args.obstacle_contact_penalty}"
        )
    if args.ball_alignment > 0:
        print(f"[train_team] ball alignment  : weight=+{args.ball_alignment}")
    print(
        f"[train_team] pool            : capacity={args.pool_capacity}, "
        f"snapshot every {args.snapshot_every:,} steps, "
        f"eps_latest={args.eps_latest}"
    )
    print(
        f"[train_team] ref eval        : every {args.eval_every:,} steps, "
        f"{args.eval_episodes_per_cycle} eps/cycle, window={args.win_rate_window}, "
        f"promote@{args.promotion_threshold:.2f}"
    )
    print(
        f"[train_team] checkpoints     : every {args.checkpoint_every:,} steps "
        f"→ {ckpt_dir}"
    )
    if args.render_every is not None:
        rows, cols = args.render_grid
        grid_str = f" ({rows}x{cols} grid)" if (rows, cols) != (1, 1) else ""
        print(
            f"[train_team] gifs            : every {args.render_every:,} "
            f"steps{grid_str} → {output_dir / 'gifs'}"
        )
    print(f"[train_team] tensorboard     : {tb_dir}")
    print()

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        # Explicit log_interval=1 — dump rollout/train/time tables
        # every rollout (default; pinned to be explicit across resume).
        log_interval=1,
        reset_num_timesteps=(args.resume is None),
    )

    final_path = output_dir / "final.zip"
    model.save(final_path)
    print(f"\n[train_team] training complete. Final model → {final_path}")
    print(f"[train_team] view metrics    : tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
