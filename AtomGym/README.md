# AtomGym

Gymnasium env + PPO training stack on top of `AtomSim`. See `CLAUDE.md`
for full architecture, conventions, and the design rationale behind
the current reward shaping / self-play / config structure.

## Quick start

The legacy `--ball-alignment 0.3 --stall-penalty 0.5 ...` CLI surface
has been replaced by a YAML config (one diff between experiments
instead of a flag soup). Reward weights and env shaping params live in
the YAML; runtime knobs (`--n-envs`, `--total-timesteps`, etc.) stay
on the CLI. Default configs reproduce the legacy behaviour.

### Solo (single robot, push-to-goal)

```bash
.venv/bin/python -m AtomGym.training.train \
    --run-name l1_baseline \
    --total-timesteps 16_000_000 \
    --n-envs 15 \
    --use-subproc \
    --batch-size 1024 --n-steps 1024 \
    --ent-coef 0.01 --log-std-init 0.3 \
    --render-every 2_000_000 --render-grid 3x3 \
    --render-frame-stride 4 --render-max-seconds 8 \
    --checkpoint-every 2_000_000 \
    # config defaults to AtomGym/configs/default_solo.yaml — copy it
    # and edit for an experiment, then point --config at the copy.
```

### 1v1 self-play (snapshot pool + reference promotion)

```bash
.venv/bin/python -m AtomGym.training.train_team \
    --run-name l2_baseline \
    --total-timesteps 16_000_000 \
    --n-envs 15 \
    --use-subproc \
    --batch-size 1024 --n-steps 1024 \
    --ent-coef 0.01 --log-std-init 0.3 \
    --pool-capacity 20 --snapshot-every 500_000 \
    --eval-every 250_000 --eval-episodes-per-cycle 10 \
    --promotion-threshold 0.80 \
    --render-every 2_000_000 --render-grid 3x3 \
    --checkpoint-every 2_000_000
    # config defaults to AtomGym/configs/default_team.yaml.
```

### Resuming a run

Pass `--resume <path-to-checkpoint.zip>` and `--run-name` matching
the original (so checkpoints + gifs land in the same directory). The
checkpoint's timestep counter is restored; `--total-timesteps` is the
absolute target across original + resumed.

```bash
.venv/bin/python -m AtomGym.training.train_team \
    --run-name l2_baseline \
    --total-timesteps 16_000_000 \
    --resume training_runs/l2_baseline/checkpoints/ppo_12000000_steps.zip \
    [other flags as before]
```

### Migrating an existing checkpoint to the new obs space

The observation gained a per-robot `time_in_box` dim (solo: 11→12;
team: 18→20). Existing checkpoints can be migrated by zero-padding
the first-layer columns:

```bash
# Solo → solo (or team → team) — same env type, larger obs space
.venv/bin/python -m AtomGym.training.transfer_extend_obs \
    --checkpoint training_runs/old_l1/final.zip \
    --output     training_runs/new_l1/init.zip \
    --env-type   solo

# Solo → team — the original transfer (now in 12 → 20 form)
.venv/bin/python -m AtomGym.training.transfer_solo_to_team \
    --solo-checkpoint training_runs/l1/final.zip \
    --output          training_runs/l2/team_init.zip
```

Then start the new training run with `--resume <output.zip>`.

## Tools

Render reward heatmaps for visual inspection / parameter tuning:

```bash
python -m AtomGym.tools.render_static_field
python -m AtomGym.tools.render_ball_alignment
python -m AtomGym.tools.render_goalie_box_penalty
```

Outputs land under `AtomGym/research/<term>/` by default.

## Tests

```bash
.venv/bin/pytest AtomGym/tests/
```

Pure-python unit tests (reward terms, config loader, opponent runner,
etc.) and integration tests against `sim_py` (env step loop, contact
detection, goal events). Both run together in the same `pytest`
invocation; the integration tests `pytest.skip` if the AtomSim release
build isn't available.
