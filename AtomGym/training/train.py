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
from collections import deque
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
from AtomGym.rewards import RewardTerm  # noqa: E402
from AtomGym.training.config import (  # noqa: E402
    TrainingConfig,
    load_training_config,
)
from AtomGym.training.gif_eval_callback import GifEvalCallback  # noqa: E402

# Default config path — used when `--config` isn't passed. Lives next
# to AtomGym so users can `--config AtomGym/configs/my_run.yaml` after
# copying the default.
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "AtomGym" / "configs" / "default_solo.yaml"


# ---------------------------------------------------------------------------
# Env factory — Level 1 default config
# ---------------------------------------------------------------------------


def make_l1_env(
    seed: int = 0,
    rewards: list[RewardTerm] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> AtomSoloEnv:
    """Construct a Level-1 env. Reward composition + env shaping params
    come from a `TrainingConfig` (see `config.py`); this factory just
    merges the per-worker `seed` in.

    `env_kwargs` is the dict from `TrainingConfig.env_kwargs` — a subset
    of `AtomSoloEnv.__init__` kwargs. Sensible defaults for `physics_dt`
    (60 Hz) and `control_dt` (30 Hz, action_repeat=2) are filled in if
    not specified, preserving the legacy CLI-default behaviour.

    `rewards=None` is supported as an empty composite — useful for shape-
    lookup callers like `transfer_extend_obs` that build an env purely
    to read its observation_space."""
    rewards = list(rewards) if rewards is not None else []
    base_kwargs: dict[str, Any] = {
        "physics_dt": 1.0 / 60.0,  # sim ticks at 60 Hz
        "control_dt": 1.0 / 30.0,  # policy emits actions at 30 Hz (action_repeat=2)
    }
    if env_kwargs is not None:
        base_kwargs.update(env_kwargs)
    init_ranges = InitialStateRanges(
        # Default: random pose, zero velocities. Once the policy can score
        # from a stationary ball, broaden the velocity ranges to make the
        # policy robust to a moving ball / off-axis approach.
        # TODO: lift InitialStateRanges into the YAML once we want to A/B
        # different DR schedules across runs.
        ball_speed=(0.0, 0.3)
    )
    return AtomSoloEnv(
        rewards=rewards,
        init_state_ranges=init_ranges,
        seed=seed,
        **base_kwargs,
    )


def make_vec_env(
    n_envs: int,
    base_seed: int,
    use_subproc: bool,
    config: TrainingConfig,
) -> VecEnv:
    """Wrap N envs (each freshly constructed from `config`) in
    `Monitor` and stack them into a SB3 vec env. Each worker gets its
    own freshly-built reward composite — reward terms are stateless
    per `_base_reward.py`'s contract, but constructing per-worker is
    safer than sharing instances across processes."""

    def _factory(i: int):
        # Re-build the rewards inside the factory so each worker (under
        # SubprocVecEnv) has its own instances. The list-of-instances
        # in `config.rewards` is fine for DummyVecEnv but pickling a
        # closure that references it through SubprocVecEnv would also
        # work; constructing fresh keeps semantics uniform.
        rewards = [type(r)(**_term_kwargs(r, config)) for r in config.rewards]
        return Monitor(
            make_l1_env(
                seed=base_seed + i,
                rewards=rewards,
                env_kwargs=config.env_kwargs,
            )
        )

    factories = [(lambda i=i: _factory(i)) for i in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def _term_kwargs(term: RewardTerm, config: TrainingConfig) -> dict[str, Any]:
    """Recover the YAML-supplied kwargs for a given constructed term so
    we can rebuild it inside each worker. Falls back to {weight} when
    the term came from an unknown source (defensive)."""
    raw_rewards = config.raw.get("rewards", {})
    return raw_rewards.get(term.name, {"weight": term.weight})


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


class GoalRateCallback(BaseCallback):
    """Sliding-window goal-rate per episode, decoupled from reward weight.

    `reward/goal_scored` mixes magnitude and frequency: a doubled
    `goal_scored.weight` looks like learning even if scoring rate is
    flat. This callback strips the weight out and reports the raw
    "fraction of episodes that ended in a goal" over the last `window`
    completed episodes — the same window-mean style SB3 already uses
    for `rollout/ep_rew_mean`.

    Metrics:
        goals/for_us_rate     — fraction of episodes where this team scored
        goals/against_us_rate — fraction where the other side scored
                                (own-goal in solo, opponent goal in team)
        goals/net_rate        — for_us minus against_us

    The env emits `info["scored_for_us"]` / `info["scored_against_us"]`
    edge-detected on the terminating step, and Monitor sets
    `info["episode"]` on episode end — we latch the goal flags during
    the episode and push to the deque when Monitor signals close.
    """

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.window = int(window)
        self._for_us: deque[float] = deque(maxlen=self.window)
        self._against_us: deque[float] = deque(maxlen=self.window)
        # Per-env latches; sized lazily on first step (we don't know
        # n_envs until SB3 binds the callback to the model).
        self._scored_for: list[bool] = []
        self._scored_against: list[bool] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        if len(self._scored_for) != len(infos):
            self._scored_for = [False] * len(infos)
            self._scored_against = [False] * len(infos)
        for i, info in enumerate(infos):
            if info.get("scored_for_us", False):
                self._scored_for[i] = True
            if info.get("scored_against_us", False):
                self._scored_against[i] = True
            # Monitor sets info["episode"] exactly on the step the
            # episode terminated/truncated. Push then reset.
            if "episode" in info:
                self._for_us.append(1.0 if self._scored_for[i] else 0.0)
                self._against_us.append(1.0 if self._scored_against[i] else 0.0)
                self._scored_for[i] = False
                self._scored_against[i] = False
        return True

    def _on_rollout_end(self) -> None:
        if not self._for_us:
            return
        for_rate = sum(self._for_us) / len(self._for_us)
        against_rate = sum(self._against_us) / len(self._against_us)
        self.logger.record("goals/for_us_rate", for_rate)
        self.logger.record("goals/against_us_rate", against_rate)
        self.logger.record("goals/net_rate", for_rate - against_rate)
        self.logger.record("goals/window_episodes", len(self._for_us))


class EpisodeOutcomeCallback(BaseCallback):
    """Sliding-window classification of how each episode ended.

    `goals/*_rate` answers "did the policy score?" but not "what
    happened on the episodes it didn't?" A flat 60% scoring rate could
    be 40% timeouts (approach-geometry problem), 40% own-goals
    (defensive failure), 40% box-violations (rule-budget exhaustion),
    or any mix — and each points to a different fix. This callback
    splits the negative space.

    Every episode terminates with exactly one of:

        scored_for_us       — learner scored on the opposing goal
        scored_against_us   — own-goal in solo, opponent goal in team
        box_violation_self  — learner exhausted the goalie-box budget
                              (only fires when env.goalie_box_terminal_time > 0)
        box_violation_opp   — opponent exhausted the goalie-box budget
                              (team only; learner-favourable termination)
        timeout             — episode hit max_episode_steps (truncated)

    Reports each as a fraction of the last `window` completed episodes
    under `outcomes/<name>_rate`. The five rates sum to 1.0 once the
    window has any episodes — read the block as a stacked breakdown of
    where the policy is ending up. Resolution rule when multiple
    terminal flags fire on the same substep: scoring beats box
    violation (the goal conceptually completes first).
    """

    _OUTCOMES = (
        "scored_for_us",
        "scored_against_us",
        "box_violation_self",
        "box_violation_opp",
        "timeout",
    )

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.window = int(window)
        self._counts: dict[str, deque[float]] = {
            name: deque(maxlen=self.window) for name in self._OUTCOMES
        }
        # Per-env latches for terminal info flags. Sized lazily on first
        # step (n_envs isn't known until SB3 binds the callback).
        self._latched_for: list[bool] = []
        self._latched_against: list[bool] = []
        self._latched_box_self: list[bool] = []
        self._latched_box_opp: list[bool] = []

    def _ensure_capacity(self, n: int) -> None:
        if len(self._latched_for) != n:
            self._latched_for = [False] * n
            self._latched_against = [False] * n
            self._latched_box_self = [False] * n
            self._latched_box_opp = [False] * n

    def _classify(self, i: int) -> str:
        if self._latched_for[i]:
            return "scored_for_us"
        if self._latched_against[i]:
            return "scored_against_us"
        if self._latched_box_self[i]:
            return "box_violation_self"
        if self._latched_box_opp[i]:
            return "box_violation_opp"
        return "timeout"

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        self._ensure_capacity(len(infos))
        for i, info in enumerate(infos):
            if info.get("scored_for_us", False):
                self._latched_for[i] = True
            if info.get("scored_against_us", False):
                self._latched_against[i] = True
            if info.get("box_violation_self", False):
                self._latched_box_self[i] = True
            if info.get("box_violation_opp", False):
                self._latched_box_opp[i] = True
            if "episode" in info:
                outcome = self._classify(i)
                for name in self._OUTCOMES:
                    self._counts[name].append(1.0 if name == outcome else 0.0)
                self._latched_for[i] = False
                self._latched_against[i] = False
                self._latched_box_self[i] = False
                self._latched_box_opp[i] = False
        return True

    def _on_rollout_end(self) -> None:
        any_window = self._counts["scored_for_us"]
        if not any_window:
            return
        n = len(any_window)
        for name in self._OUTCOMES:
            rate = sum(self._counts[name]) / n
            self.logger.record(f"outcomes/{name}_rate", rate)
        self.logger.record("outcomes/window_episodes", n)


class LifetimeStepsCallback(BaseCallback):
    """Logs cumulative env-step counts that span across resume boundaries.

    Used in tandem with `--reset-step-counter`, where the run's own
    `time/total_timesteps` chart restarts at 0 (so the new run gets its
    own clean TB scalars) but we still want a "lifetime" view that
    includes any pretraining steps. Records on rollout end so it lands
    in the same dump as SB3's built-in `time/total_timesteps`."""

    def __init__(self, pretrain_offset: int) -> None:
        super().__init__()
        self.pretrain_offset = int(pretrain_offset)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("time/pretrain_steps", self.pretrain_offset)
        self.logger.record(
            "time/lifetime_steps", self.pretrain_offset + self.num_timesteps
        )


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
        "--run-name",
        type=str,
        default="l1_baseline",
        help="Name of this run. Outputs land under training_runs/<run_name>/.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps (across all parallel envs).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel envs. SB3's PPO batch comes from n_envs * n_steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; each parallel env gets seed + worker_index.",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv (one process per env). Default is DummyVecEnv "
        "(single-process, easier to debug).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4, SB3 default).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout length per env. SB3 default 2048; lower if memory tight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size for the update. SB3 default 64. With small "
        "MLPs the per-minibatch Python/PyTorch overhead dominates the "
        "actual matmul, so the update phase is *much* faster at 512-2048 "
        "even though the number of gradient steps drops. Ensure n_envs * "
        "n_steps is divisible by batch_size.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to a YAML training config (reward weights + env shaping). "
        f"Default: {_DEFAULT_CONFIG_PATH}. Schema: see "
        "AtomGym/training/config.py — the TL;DR is `env:` and `rewards:` "
        "top-level keys, each `rewards` sub-key matches a RewardTerm.name "
        "and accepts the term's __init__ kwargs. Copy the default file "
        "into your run directory and edit there for an experiment, then "
        "diff between runs to see what changed.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy bonus coefficient. SB3 default is 0.0; we raise to 0.01 "
        "to push back on premature collapse to deterministic 'do nothing' actions. "
        "Tune up (0.02-0.05) if the policy keeps stalling out; tune down (0.005-0.0) "
        "once the policy is consistently scoring and you want it to commit.",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=0.0,
        help="Initial log-std of the Gaussian action distribution. SB3 default 0.0 "
        "⟹ initial std=1.0. Bump to 0.3-0.5 (std ≈ 1.35-1.65) to widen the "
        "starting distribution and increase initial exploration.",
    )
    # Reward weights, env shaping, and `--manipulator` moved to the YAML
    # config (see `--config`). To override for a single experiment, copy
    # the default YAML and edit it; commit the YAML alongside the run so
    # the diff between experiments is a single readable file.
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50_000,
        help="Save a checkpoint every N total env steps.",
    )
    parser.add_argument(
        "--disable-reward-breakdown",
        action="store_true",
        help="Skip the per-term reward breakdown callback. The callback "
        "iterates over every env's info dict every step and feeds "
        "each reward term to logger.record_mean — at 16 envs × 1024 "
        "steps × ~5 terms = 80K record_mean calls per rollout, the "
        "per-call overhead is non-negligible. Use this flag to "
        "isolate its cost in a profile / throughput-tuning run.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=None,
        help="Render the current policy as a .gif every N total env steps "
        "(written to training_runs/<run_name>/gifs/). Disabled by default.",
    )
    parser.add_argument(
        "--render-eval-seed",
        type=int,
        default=999,
        help="Fixed seed for the eval-rollout env so progress over training "
        "is directly comparable — same scenario every render.",
    )
    parser.add_argument(
        "--render-frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame in the gif (higher = smaller files, "
        "choppier playback). Playback fps adjusts to keep 1 s gif = 1 s sim.",
    )
    parser.add_argument(
        "--render-max-seconds",
        type=float,
        default=None,
        help="Cap each render rollout to N seconds of sim time. Caps both "
        "the gif's playback length AND the wall-clock the callback "
        "spends each interval. Default: no cap (run a full episode).",
    )
    parser.add_argument(
        "--render-grid",
        type=_parse_grid,
        default=(1, 1),
        help="Render a composite gif laid out as a grid of independent "
        "rollouts. Format: 'RxC' (e.g. '2x2' for a 4-cell grid). Each "
        "cell uses --render-eval-seed + cell_index, so progress is "
        "directly comparable across renders. Default: 1x1 (single cell).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_REPO_ROOT / "training_runs",
        help="Parent directory for run outputs.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a checkpoint .zip (e.g. "
        "training_runs/<run_name>/checkpoints/ppo_12000000_steps.zip). "
        "The model's timestep counter is restored and learning continues "
        "in the same TensorBoard run, with --total-timesteps interpreted "
        "as the absolute target across original + resumed runs (so passing "
        "the same flags as the original run will train the remainder up to "
        "that target). Pass --run-name matching the original so checkpoints "
        "and gifs land in the same directories.",
    )
    parser.add_argument(
        "--reset-step-counter",
        action="store_true",
        help="Only valid with --resume. Reset the model's env-step counter "
        "to 0 so the new run's TensorBoard scalars start fresh (SB3 also "
        "auto-suffixes the TB log dir with _2/_3/... so the chart is "
        "decoupled from the prior run). --total-timesteps is then "
        "interpreted as the budget for THIS run only, not an absolute "
        "target. The pretraining step count is preserved as an offset "
        "and logged each rollout as time/lifetime_steps so a 'total "
        "samples seen' view is still available.",
    )
    args = parser.parse_args()

    if args.reset_step_counter and args.resume is None:
        parser.error("--reset-step-counter requires --resume.")

    # Load + validate the YAML config up-front so any schema errors
    # surface immediately, before we set up output dirs / spawn workers.
    config = load_training_config(args.config)
    print(f"[train] loaded config: {args.config}")
    print(f"[train] rewards: {[t.name for t in config.rewards]}")
    print(f"[train] env_kwargs: {config.env_kwargs}")

    # Output directories
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tensorboard"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Persist a copy of the resolved config inside the run directory so
    # the experiment's exact reward shape is recoverable from artefacts
    # alone (no need to remember which version of default_solo.yaml was
    # used). Plain copy — keeps comments etc.
    (output_dir / "config.yaml").write_text(args.config.read_text())

    # Vec env
    vec_env = make_vec_env(
        n_envs=args.n_envs,
        base_seed=args.seed,
        use_subproc=args.use_subproc,
        config=config,
    )

    # PPO with 2x128 MLP for both policy and value heads.
    # `log_std_init` widens the initial action distribution → more
    # exploration during early training. `ent_coef` is the entropy bonus
    # that keeps the policy from collapsing to a low-std deterministic
    # mode before it has found the good actions.
    if args.resume is not None:
        # Resume path: load weights, optimiser state, and the env-step
        # counter from the checkpoint. Override hyperparameters from the
        # CLI so the *current* values take effect (lets you tweak
        # ent_coef / lr mid-run if you need to). policy_kwargs are NOT
        # passed — the network architecture is loaded from the checkpoint.
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
            # CPU is faster than GPU for our MLP size — see PPO()
            # construction below for the rationale. Pin on resume too
            # so a checkpoint saved on GPU doesn't drag training back
            # onto the slower path.
            device="cpu",
            # Explicit verbose=1 — without this the saved value is
            # supposed to carry through, but sometimes doesn't (the
            # stdout tabular output disappears across resume). Pinning
            # here guarantees rollout/, train/, time/ tables show.
            verbose=1,
        )
        loaded_steps = int(model.num_timesteps)
        if args.reset_step_counter:
            # Pretrain offset preserved for lifetime logging only; the
            # model counter restarts at 0 once learn() is called with
            # reset_num_timesteps=True. --total-timesteps is the budget
            # for THIS run, not an absolute lifetime target.
            pretrain_offset = loaded_steps
            remaining_steps = args.total_timesteps
        else:
            pretrain_offset = 0
            remaining_steps = max(0, args.total_timesteps - loaded_steps)
            if remaining_steps == 0:
                print(
                    f"[train] resume target {args.total_timesteps:,} already reached "
                    f"(checkpoint at {loaded_steps:,}). Nothing to do."
                )
                return
    else:
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
            batch_size=args.batch_size,
            n_epochs=10,
            # gamma=0.99,
            # gae_lambda=0.95,
            gamma=0.98,
            gae_lambda=0.92,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            # Force CPU. At our network size (128×128 MLP, 18-d obs)
            # the per-kernel-launch latency on GPU dwarfs the actual
            # compute (<100k FLOPs per forward pass), AND we'd pay for
            # PCIe traffic shipping rollout buffers across the bus.
            # Empirically CPU is ~1.5-3× faster for MLP PPO at this
            # scale. SB3 raises a UserWarning if you don't pin this.
            device="cpu",
            verbose=1,
            tensorboard_log=str(tb_dir),
            seed=args.seed,
        )
        loaded_steps = 0
        remaining_steps = args.total_timesteps
        pretrain_offset = 0

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
    ]
    if not args.disable_reward_breakdown:
        callbacks.append(RewardBreakdownCallback())
    callbacks.append(GoalRateCallback(window=100))
    callbacks.append(EpisodeOutcomeCallback(window=100))
    if args.reset_step_counter:
        callbacks.append(LifetimeStepsCallback(pretrain_offset=pretrain_offset))
    if args.render_every is not None:
        gif_dir = output_dir / "gifs"
        rows, cols = args.render_grid
        callbacks.append(
            GifEvalCallback(
                eval_env_factory=lambda: make_l1_env(
                    seed=args.render_eval_seed,
                    rewards=[
                        type(r)(**_term_kwargs(r, config)) for r in config.rewards
                    ],
                    env_kwargs=config.env_kwargs,
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
    if args.resume is not None:
        print(f"[train] resume from   : {args.resume}")
        print(f"[train] resumed at    : {loaded_steps:,} env steps")
        if args.reset_step_counter:
            print(
                f"[train] step counter  : RESET (TB log dir auto-suffixed; "
                f"pretrain offset {pretrain_offset:,} logged as time/lifetime_steps)"
            )
            print(
                f"[train] this run      : {remaining_steps:,} steps "
                f"(lifetime target ≈ {pretrain_offset + remaining_steps:,})"
            )
        else:
            print(
                f"[train] remaining     : {remaining_steps:,} steps "
                f"to reach target {args.total_timesteps:,}"
            )
    else:
        print(f"[train] total timesteps: {args.total_timesteps:,}")
    print(
        f"[train] n_envs        : {args.n_envs} ({'SubprocVecEnv' if args.use_subproc else 'DummyVecEnv'})"
    )
    print(f"[train] policy net    : 2x128 MLP")
    print(f"[train] config        : {args.config}")
    print(f"[train] reward terms  : {[t.name for t in config.rewards]}")
    print(f"[train] env kwargs    : {config.env_kwargs}")
    print(
        f"[train] exploration   : ent_coef={args.ent_coef}  log_std_init={args.log_std_init}"
    )
    print(f"[train] checkpoints   : every {args.checkpoint_every:,} steps → {ckpt_dir}")
    if args.render_every is not None:
        rows, cols = args.render_grid
        grid_str = f" ({rows}x{cols} grid)" if (rows, cols) != (1, 1) else ""
        print(
            f"[train] gifs          : every {args.render_every:,} steps{grid_str} → {output_dir / 'gifs'}"
        )
    print(f"[train] tensorboard   : {tb_dir}")
    print()

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        tb_log_name=args.run_name,
        # Explicit log_interval=1 — SB3 dumps the rollout/train/time
        # tabular output every `log_interval` rollouts. 1 = every
        # rollout, which matches the non-resume behaviour. Pinning
        # here makes the cadence unambiguous.
        log_interval=1,
        # On resume: by default keep the model's existing num_timesteps
        # counter so checkpoint filenames continue from where they left
        # off (e.g. ppo_13000000_steps.zip after resuming at 12M) and
        # the TensorBoard run picks up its existing log dir instead of
        # starting a fresh `_2` subdir. With --reset-step-counter the
        # counter restarts at 0, SB3 auto-suffixes the TB log dir, and
        # the LifetimeStepsCallback records the pretraining offset.
        reset_num_timesteps=(args.resume is None or args.reset_step_counter),
    )

    final_path = output_dir / "final.zip"
    model.save(final_path)
    print(f"\n[train] training complete. Final model → {final_path}")
    print(f"[train] view metrics  : tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
