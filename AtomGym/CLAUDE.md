# AtomGym

Gymnasium-based RL training stack on top of `AtomSim`. Two operational
training pipelines:

- **Solo PPO** — `AtomSoloEnv` + `train.py`. Single-robot push-to-goal.
- **1v1 self-play** — `AtomTeamEnv` + `train_team.py`. Snapshot-pool
  opponent, fixed reference for benchmarking, weight-transfer warm-start
  from a solo checkpoint.

Both share the same reward composite, the same observation schema (just
sized for n_robots), and the same per-term TensorBoard breakdown. Reward
weights and env shaping params are driven from a YAML config (see
"Training config" below); runtime knobs (`--n-envs`, `--total-timesteps`,
`--learning-rate`, ...) stay as CLI args.

## Observation schema

Per-robot block is **8 dims**: `[px, py, sin θ, cos θ, dx, dy, dθ, time_in_box]`.
The trailing `time_in_box` is the fraction of `goalie_box_terminal_time`
the robot has spent continuously in *its opposing* goalie box this
visit, normalised to [0, 1]. Always present in the obs — when the
goalie-box rule is disabled (`goalie_box_terminal_time=0`, the default)
it just always reads 0, so there's a single observation space across
all rule configurations.

Total obs dim is `4 + 8·n_robots`: solo = 12, team = 20. See
`AtomGym/action_observation.py` for the canonical schema.

## Self-play stack (1v1)

Validated through a 20M-step training run on the 1v1 stack. The pieces:

- `AtomTeamEnv` (`environments/team_env.py`) — 1v1 env. Learner attacks
  +x, opponent driven by a swappable `opponent_policy` callable. Default
  hook ⟹ zero action (stationary body) — see "empty-pool fallback"
  below. Reset places robots on opposite halves of the field.
- `SnapshotPool` (`training/snapshot_pool.py`) — FIFO pool of past
  learner state_dicts. Snapshots stored as **numpy arrays, not torch
  tensors** (see "FD leak" below).
- `OpponentRunner` (`training/opponent_runner.py`) — per-worker bundle
  of {pool replica, CPU shadow policy, RNG}. ε-greedy "latest else
  uniform" sampling at the PPO update boundary; the loaded opponent
  stays fixed for the entire next rollout (AlphaStar-style).
- `ReferenceOpponent` (`training/reference_opponent.py`) — single
  frozen snapshot. The win-rate-vs-reference scalar is the only honest
  progress signal (pool win-rate is ~50% by design).
- `RefEvalCallback` (`training/ref_eval_callback.py`) — runs eval
  episodes vs the reference; promotes a new reference when the win-rate
  over a sliding window exceeds `--promotion-threshold`.
- `PoolSyncCallback` (`training/pool_sync_callback.py`) — at the PPO
  update boundary, snapshot the learner, add to master pool, broadcast
  to workers via `env_method("update_opponent_pool", pool)`.
- `transfer_solo_to_team` (`training/transfer_solo_to_team.py`) —
  warm-start the team policy by zero-padding `(128, 12) → (128, 20)` on
  the first-layer weights. Hidden + output layers transfer 1:1.
- `transfer_extend_obs` (`training/transfer_extend_obs.py`) — same
  shape-extend pattern but within a single env type (e.g. an old
  solo checkpoint with the pre-`time_in_box` obs space → the current
  solo space). Use this whenever the obs schema gains a dim and you
  want to keep training from an existing run.

### Key decisions still load-bearing

- **Observation is fully world-frame**, so the opponent's canonical
  view is built by **slot permutation + x-axis mirror** on the already-
  built learner obs (no second `build_observation` call). The mirror is
  needed because the policy expects "I attack +x" — that requires sign-
  flipping x-coordinate components (ball.px/vx, robot.px/cos θ/dx/ω).
  Action returned by the opponent is in canonical frame too, so
  `action_to_wheel_cmds(mirror=True)` for the opponent. Helper:
  `AtomTeamEnv.opponent_view(learner_obs)`.
- **Empty-pool opponent = stationary body (zero action).** When training
  starts the pool is empty; the opponent emits zero action. Avoids
  poisoning early rollouts with a randomly-weighted opponent. Real
  opponents take over as soon as the first snapshot is added.
- **Only the learner contributes gradients.** Opponent forward passes
  are CPU + `eval` mode under `torch.no_grad()`. Rollouts go into the
  PPO buffer from the learner's perspective only.
- **Snapshot storage is in-memory state_dicts, as numpy.** A few MB per
  snapshot; pool of 20-30 is trivial. Numpy (not torch tensors) — see
  FD leak below.
- **Per-rollout opponent sampling, not per-reset.** Each worker
  resamples ONE opponent at the PPO update boundary and keeps it loaded
  for the entire next rollout. Within-rollout opponent diversity comes
  from N workers each picking their own snapshot.
- **SubprocVec sync via `env_method`.** Workers each own a pool replica
  + CPU shadow policy + `OpponentRunner`. After each PPO update, main
  calls `vec_env.env_method("update_opponent_pool", new_pool)` which
  atomically replaces the replica AND resamples + loads.

### Subtle bugs caught during validation — don't regress

- **Torch FD leak (Errno 24, "too many open files").** Snapshots
  pickled across SubprocVec pipes use torch's `file_descriptor` sharing
  strategy → ~13 tensors × 20-snapshot pool × 16 workers × many syncs
  blew past `ulimit -n=1024` after ~2M steps. Fix: convert state_dicts
  to numpy at the producer (`PoolSyncCallback`, resume bootstrap),
  convert back to tensors at the consumer (`OpponentRunner.update_pool`,
  `ReferenceOpponent.set_snapshot`). Helpers `state_dict_to_numpy`,
  `state_dict_to_tensors` in `_shadow_policy.py`. Numpy pickles as
  plain bytes — no FDs. Regression test:
  `tests/test_pool_sync_callback.py::test_pool_stores_numpy_not_tensors`.
- **GIF eval showed stationary opponent forever.** GIF eval factory
  initially used `make_l2_env()` directly, without binding the
  reference's `predict` as the opponent. Renders showed the learner
  playing against the default zero-action opponent. Fix in
  `train_team.py`: `_make_gif_eval_env` calls
  `env.set_opponent_policy(reference.predict)`. GIFs now match the
  matchup the win-rate gate is tracking.
- **Random-init goal flings credited the policy.** With random ball
  velocity at reset, the ball could fling into a goal in the first
  few steps before any robot touched it; `GoalScoredReward` fired
  ±50 anyway, injecting pure noise into the gradient. Fix:
  `info["ball_touched"]` latch (see "credit-hack guard" below). The
  default-False on missing key is deliberate — a forgotten plumbing
  change makes the goal signal silent (loud failure during training)
  rather than silently corrupting the gradient. Regression tests in
  `tests/test_rewards_terms.py::test_goal_scored_*` and
  `tests/test_solo_env_control_loop.py::test_spurious_goal_*`.
- **Tangential lock at the alignment-gate boundary.** With the old
  `BallAlignmentReward` defaults (`inner_radius=0.044, outer_radius=0.10`),
  the policy got stuck tangentially-perpendicular to the ball at
  40-60 mm — exactly the inner-gate boundary, where distance shaping
  is locally flat under perpendicular motion AND alignment was
  silenced AND ball-progress was 0 (ball stationary). Three dense
  terms saying nothing simultaneously. Fix: drop `inner_radius` to 0
  (alignment active through contact) and widen `outer_radius` to
  0.18. `BallProgressReward` still dominates during active pushing,
  so the "release ball to chase alignment bonus" attractor we
  originally feared doesn't materialise. See heatmap-driven
  diagnosis in `AtomGym/research/ball_alignment/`.
- **Spatial goalie-box penalty was beaten by sparse goal reward.**
  GAE distributes the +R goal reward across many steps, so a
  per-step −k box penalty becomes worth paying once R / n_steps > k.
  In self-play both teams co-evolved into "ignore the box, score
  faster," producing a policy that wouldn't transfer to real-world
  rules. Fix: replace with the time-based `GoalieBoxPenalty` + env
  termination on time-budget exhaustion (see "Goalie-box rule"
  section). The temporal constraint can't be amortised away by goal
  rewards because the discrete violation cost + episode termination
  are independent of how many steps the violator spent in the box.

## Reward shaping

Categories in `AtomGym/rewards/`:

- **Dense progress shaping** — `BallProgressReward`, `DistanceToBallReward`,
  `BallAlignmentReward`. Drive the policy toward "push ball into the
  opposing goal."
- **Sparse terminal** — `GoalScoredReward`. Large ± reward on goal
  events. Gated by `info["ball_touched"]` (see "credit-hack guard"
  below) so spurious random-init flings don't credit the policy.
- **Behaviour penalties** — `StallPenaltyReward`,
  `ObstacleContactPenalty` (impulse-fraction signal),
  `StaticFieldPenalty` (anticipatory potential field around walls).
- **Goalie-box rule** — `GoalieBoxPenalty` + env-side termination on
  exceeding the per-visit time budget. See its own section below.

The composite returns both a scalar total and a per-term breakdown
dict; `RewardBreakdownCallback` feeds each term to TensorBoard via
`record_mean` so per-term contributions are visible per rollout. The
breakdown's keys are each term's `name` attribute, which matches the
YAML reward keys exactly — config diff and TB diff line up 1:1.

### Static field shaping (`StaticFieldPenalty`) — walls-only by default

Sigmoid potential field around field-perimeter walls. Anticipatory:
penalty starts ramping up while the robot is still on the safe side,
default band 30 → 85 mm from the wall (saturated at 30 mm = robot
half-side; zero at 85 mm). PPO sees the boundary before contact and
learns collision-free motion via gradient instead of impact events.

Historically this term ALSO carried a goalie-box source (intrusion-
only sigmoid penalty inside the opposing box). That source was moved
to the dedicated `GoalieBoxPenalty` term — `StaticFieldPenalty` now
defaults to `include_goalie_box=False` (walls only). Set the flag to
`True` to recover the original behaviour for ablation; the box source
geometry is unchanged when enabled.

**Precomputed grid + bilinear lookup, NOT a perf optimisation** — the
analytic per-step cost is already dwarfed by the PPO forward pass. The
grid is an *engineering* win:

- Renderable as a heatmap for inspection —
  `python -m AtomGym.tools.render_static_field` bakes the field,
  overlays walls + boxes + goal mouths, saves a PNG.
- Decouples sigmoid + SDF math from the reward hot path.
- Adding a new static hazard is a one-line edit to `_evaluate_at`
  followed by a re-bake.

**Distribution model**: each SubprocVec worker builds its own grid in
`__init__` (~30-50 ms in pure Python at 5 mm resolution). No main-side
pre-build-and-ship machinery — at our scale the cost is below noise
next to torch + sim_py imports each worker already pays.

**Pusher caveat**: the field is queried at the robot's centre. With a
pusher attached, the true collision distance depends on yaw, so the
shaping kicks in 30 mm later than ideal. The event-based
`ObstacleContactPenalty` still fires on real impacts. Revisit with
orientation-axis or multi-point lookup if eval shows a
pusher-leads-into-walls failure mode.

### Goalie-box rule (`GoalieBoxPenalty` + env termination)

Replaces the spatial goalie-box source from `StaticFieldPenalty`. The
old approach (per-step intrusion penalty) was structurally vulnerable
to the "sparse goal beats dense penalty" failure mode in self-play:
PPO + GAE distribute the +R goal reward across many steps, so a
per-step −k box penalty becomes worth paying. Both teams co-evolved
into a "ignore the box, score faster" equilibrium that wouldn't
transfer to real-world rules.

The current rule is **temporal, not spatial**: each robot may pass
through the opposing box freely up to a configurable budget per visit
(`env.goalie_box_terminal_time`, seconds). Once the budget is
exceeded, the env fires `info["box_violation"]` and terminates the
episode. To give PPO a smooth shaping gradient leading up to the
terminal — instead of a single binary cliff — `GoalieBoxPenalty`
provides:

1. **Time-based ramp** (penalty method): silent below `trigger_time`,
   polynomial ramp `u^p` from trigger to terminal where
   `u = (τ - trigger) / (terminal - trigger)`.
2. **Spatial depth weighting** (interior gradient): ramp scaled by
   `min(depth_into_box, depth_saturation) / depth_saturation`. Together
   the time × depth factors form a potential field whose gradient
   points OUT of the box — a policy that wants to minimise accumulated
   penalty over time naturally learns to head for the boundary.
3. **Sparse violation cost**: discrete penalty fired on the
   terminating step (`info["box_violation_self"]=True`) so the
   integrated cost of "loiter to terminal" strictly exceeds a goal
   reward — closes the "score-then-violate" exploit.

All three knobs (`trigger_time`, `terminal_time`, `power`,
`termination_penalty`, `depth_saturation`) live in the YAML config.
The reward's `terminal_time` MUST equal `env.goalie_box_terminal_time`
since the obs's normalised timer is `min(elapsed / terminal, 1.0)`.

Default in shipped configs: rule **disabled** (`terminal_time = 0`)
so the legacy "pass through freely" behaviour is preserved. Enable
explicitly when running an experiment that wants the box constraint.

Heatmap viz: `python -m AtomGym.tools.render_goalie_box_penalty`
renders 3 panels (time ramp at centroid, spatial @ near-terminal,
joint depth × time).

### Credit-hack guard (`info["ball_touched"]`)

`GoalScoredReward` is silent until *some* robot has touched the ball.
With random initial ball velocity, the ball can fling into a goal in
the first few steps after reset before any robot has had a chance to
influence it. Crediting (or penalising) the policy for these events
is pure noise. The env latches `info["ball_touched"]` True the first
substep ANY robot's contact list contains a `CATEGORY_BALL` entry and
holds it for the rest of the episode; `GoalScoredReward` reads it
from `info` and returns 0 if False — the episode still terminates on
the goal event (game rule) but no sparse signal is delivered.

**In team play the gate covers BOTH robots' contacts**, not just the
learner's. Opponent-driven goals against the learner *should* fire
`scored_against_us` — that's a defensive failure to learn from. Only
goals where neither robot has touched the ball are spurious.

### Ball alignment shaping (`BallAlignmentReward`)

Annular gate × asymmetric front/back alignment. Provides a small
rotational gradient when the ball is near but not touching, where
distance shaping is locally flat under perpendicular motion. Defaults:

- `inner_radius=0.0` — active through contact. Earlier we masked
  contact (inner=0.044) on the theory that "face the ball" shaping
  would suppress emergent dribbling, but empirically the opposite
  failure mode dominated: a "tangential lock" freeze at 40-60 mm
  (right at the old inner-gate boundary) where distance / progress /
  alignment all read 0 simultaneously. Removing the inner gate
  resolves this and `BallProgressReward` continues to dominate during
  active pushing.
- `outer_radius=0.18` — wider than the original 0.10, gives the term
  influence over the approach phase as well as the last-cm regime.
- `back_weight=0.3` — back-aligned earns 30% of front-aligned (mild
  asymmetry: pushers are mechanically advantaged in front).

Heatmap viz: `python -m AtomGym.tools.render_ball_alignment`.

## Training config (YAML)

`AtomGym/training/config.py` defines a YAML schema for reward weights
+ env shaping params. `train.py` and `train_team.py` accept
`--config <path>`; defaults at `AtomGym/configs/default_solo.yaml` and
`default_team.yaml` reproduce the legacy CLI defaults.

```yaml
env:
  max_episode_steps: 400
  goalie_box_depth: 0.12
  goalie_box_y_half: 0.10
  goalie_box_terminal_time: 0.0   # 0 disables the rule
  manipulator: null

rewards:
  ball_progress: { weight: 1.0 }
  ball_alignment: { weight: 0.3 }
  goal_scored: { weight: 50.0 }
  # goalie_box:                   # uncomment + set env.goalie_box_terminal_time
  #   weight: -50.0
  #   trigger_time: 2.0
  #   terminal_time: 3.0          # MUST equal env.goalie_box_terminal_time
  #   ...
```

Each `rewards` sub-key matches a `RewardTerm.name` (also the TB
breakdown key). Loader behaviour:

- **Validator**: `validate_and_construct(cls, params)` introspects
  `cls.__init__` via `inspect.signature` and rejects unknown keys
  (typo guard) + missing required kwargs. Raises `ConfigError`.
- **Omitting a reward key disables that term** — it's not added to
  the composite. The composite is exactly as long as the YAML says.
- **Adding a new reward term**: register `name → class` in
  `REWARD_REGISTRY`. The validator picks up its kwargs from the
  class's `__init__` signature automatically — no schema-file edits.

`async` / `team_async` training scripts haven't been migrated to the
YAML surface yet — they still use the legacy CLI. Migrate them when
needed; the work is mechanical (parallel to `train.py` /
`train_team.py`).

A copy of the resolved YAML is written to `<run_dir>/config.yaml` on
every training launch — diff between two runs is a single readable
file, exactly the property the refactor was meant to deliver.

## What's NOT in scope yet

- **Prioritized fictitious self-play** (PFSP / Elo-weighted sampling).
  Pool sampling is uniform-with-latest-bias; revisit if training
  plateaus.
- **League play** (AlphaStar multi-archetype). Overkill for current
  scale.
- **2v2 / NvN.** Plumbing is N-aware where cheap (`ObsView(n_robots)`,
  `_OBSTACLE_CATEGORIES` already covers robot-robot contacts), but the
  canonical-view helper and self-play machinery are 1v1 only.
- **Behavior cloning warm-start**. Was step 8 in the original roadmap.
  Weight transfer alone produced a learning signal in validation — BC
  is the fallback if a future run cold-starts and fails to learn.
- **Pusher-aware geometry** for `StaticFieldPenalty`. Accepted as a v1
  approximation; only revisit on evidence.
