# AtomGym

Gymnasium-based RL training stack on top of `AtomSim`. Two operational
training pipelines:

- **Solo PPO** — `AtomSoloEnv` + `train.py`. Single-robot push-to-goal.
- **1v1 self-play** — `AtomTeamEnv` + `train_team.py`. Snapshot-pool
  opponent, fixed reference for benchmarking, weight-transfer warm-start
  from a solo checkpoint.

Both share the same reward composite, the same observation schema (just
sized for n_robots), and the same per-term TensorBoard breakdown.

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
  warm-start the team policy by zero-padding `(128, 11) → (128, 18)` on
  the first-layer weights. Hidden + output layers transfer 1:1.

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

## Reward shaping

Three categories in `AtomGym/rewards/`:

- **Dense progress shaping** — `BallProgressReward`, `DistanceToBallReward`,
  `BallAlignmentReward`. Drive the policy toward "push ball into the
  opposing goal."
- **Sparse terminal** — `GoalScoredReward`. Large ± reward on goal events.
- **Behaviour penalties** — `StallPenaltyReward` (don't sit still),
  `ObstacleContactPenalty` (don't bounce off walls; impulse-fraction
  signal), `StaticFieldPenalty` (anticipatory potential field around
  walls + opposing goalie box).

The composite returns both a scalar total and a per-term breakdown
dict; `RewardBreakdownCallback` feeds each term to TensorBoard via
`record_mean` so per-term contributions are visible per rollout.

### Static field shaping (`StaticFieldPenalty`)

Sigmoid potential field around static hazards — designed to give PPO a
smooth proximity gradient near walls and the opposing goalie box,
rather than relying purely on collision events.

**Two source families with deliberately different shaping:**

- **Walls** — anticipatory: penalty starts ramping up while the robot
  is still on the safe side. Default band is 30 → 85 mm from the wall
  (saturated at 30 mm = robot half-side; zero at 85 mm). PPO sees the
  boundary before contact.
- **Goalie box** — intrusion-only: penalty is 0 AT the box boundary
  and ramps up only as the robot enters the box. Default saturation at
  60 mm intrusion (≈ one robot side length). Corners of the field stay
  reachable — a robot tracing the box perimeter is legal. The rule
  modelled is "don't enter the OPPOSING team's goalie box," not "don't
  go near it."

Per-source penalties combined via **max** (no double-counting in
overlap zones).

**Precomputed grid + bilinear lookup, NOT a perf optimisation** — the
analytic per-step cost is already dwarfed by the PPO forward pass. The
grid is an *engineering* win:

- Renderable as a heatmap for inspection — `python -m AtomGym.tools.render_static_field`
  bakes the field, overlays walls + boxes + goal mouths, saves a PNG.
  Iterate on shaping params without spending compute.
- Decouples sigmoid + SDF math from the reward hot path.
- Adding a new static hazard (centre-circle penalty, no-go zones, ...)
  is a one-line edit to `_evaluate_at` followed by a re-bake.

**Distribution model**: each SubprocVec worker builds its own grid in
`__init__` (~30-50 ms in pure Python at 5 mm resolution). No main-side
pre-build-and-ship machinery — at our scale the cost is below noise
next to torch + sim_py imports each worker already pays.

**Pusher caveat**: the field is queried at the robot's centre. With a
pusher attached, the true collision distance depends on yaw, so the
shaping kicks in 30 mm later than ideal in the worst case. The
event-based `ObstacleContactPenalty` still fires on real impacts.
Revisit with orientation-axis or multi-point lookup if eval shows a
pusher-leads-into-walls failure mode.

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
