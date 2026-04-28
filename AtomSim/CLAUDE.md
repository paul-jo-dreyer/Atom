# AtomSim — Project Context

This document is the entry point for any new Claude Code session on this project. Read it first. It captures the architecture, conventions, key decisions, and the reasoning behind them so you can act consistently across sessions.

## Project goal

Train multiple differential-drive (skid-steer) robots to collaborate on planar tasks — pushing a puck to a goal, collaborative repositioning, eventually 2v2 soccer. The same vehicle dynamics code must serve three uses:

1. **Batched RL simulation** on a desktop CPU (hundreds of parallel environments).
2. **Model predictive control** rolled out on an ESP32 (aspirational — design for it, don't compromise sim quality for it).
3. **Analysis** — linearization about operating points for LQR/MPC design and stability analysis.

The single biggest design pressure: one C++ dynamics core per vehicle type that compiles unchanged for desktop sim and ESP32, with Box2D layered on top in the shared `sim/` for desktop-only contact handling.

## Repository structure

```
AtomSim/
├── vehicle_dynamics/              # One folder per actuated vehicle type
│   └── diff_drive/
│       ├── analysis/              # LaTeX derivations, Julia symbolic verification
│       │   └── diff_drive.tex
│       ├── bindings/              # pybind11 wrapper for this vehicle's core
│       ├── core/                  # C++17 dynamics. ESP32-safe. No allocs, no exceptions.
│       └── embedded/              # PlatformIO ESP32 project + MPC for this vehicle
├── sim/                           # Shared physics: Box2D world, walls, ball, robot wrapper
│   ├── ball.{hpp,cpp}             # Ball dynamics + Box2D contact handling
│   ├── robot.{hpp,cpp}            # Robot wrapper around the vehicle's core
│   ├── world.{hpp,cpp}            # Box2D world + field walls + goal chambers
│   ├── types.hpp                  # WorldConfig, RobotConfig, BallConfig, collision filters
│   ├── bindings/                  # pybind11 module: sim_py
│   ├── tests/                     # doctest integration tests (sim + objects + vehicles)
│   ├── configs/                   # JSON: robots/<n>.json, manipulators/<n>.json, styles/<n>.yaml
│   ├── tools/                     # Polygon designer notebook
│   ├── python/                    # User-facing Python: tools + viz package
│   │   ├── teleop.py              # Real-time teleop (keyboard + gamepad), --record → .npz
│   │   ├── teleop_multi.py        # Multi-robot demo (1-6, blue/orange teams)
│   │   ├── render_episode.py      # CLI: .npz → mp4/gif (headless renderer)
│   │   ├── replay_episode.py      # Live scrub-replay an .npz with timeline + speed control
│   │   ├── random_gif.py          # Minimal random-action demo → gif
│   │   └── viz/                   # Visualization package — see "Visualization" section below
│   └── objects/                   # Per-passive-object dynamics (ball, box, ...)
│       └── ball/
│           ├── analysis/
│           ├── bindings/
│           └── core/              # ESP32-safe rules apply here too
├── tests/                         # Cross-cutting integration tests
└── training/                      # RL scripts: SB3 to start, CleanRL/RLlib if needed
```

The grouping principle: anything specific to one **vehicle** (an actuated body with its own controller, embedded target, and MPC) lives under `vehicle_dynamics/<vehicle>/`. Anything specific to a **passive object** the simulator pushes around (ball, puck, box) lives under `sim/objects/<object>/`. Anything genuinely shared across all vehicles and objects (Box2D world setup, sensor noise wrappers, training scripts) lives at the top level. Adding a second vehicle type (e.g. `vehicle_dynamics/ackermann/`) should not require touching `sim/` or `training/`; adding a second object type (e.g. `sim/objects/box/`) should not require touching any vehicle.

## Layering and what depends on what

- `vehicle_dynamics/<vehicle>/core/` depends on Eigen (header-only, fixed-size matrices). Nothing else.
- `vehicle_dynamics/<vehicle>/bindings/` depends on its sibling `core/` + pybind11.
- `vehicle_dynamics/<vehicle>/embedded/` depends on its sibling `core/` ONLY. If you find yourself wanting `sim/` here, stop.
- `vehicle_dynamics/<vehicle>/analysis/` is independent — LaTeX + Julia for derivations and symbolic Jacobian verification.
- `sim/objects/<object>/core/` follows the same Eigen-only, ESP32-safe rules as vehicle cores. Same constraints, same layout, no `embedded/` (objects have no controller).
- `sim/` (top level) depends on the `core/` of any vehicles AND objects it instantiates + Box2D 3.x.
- `tests/` depends on whatever it's testing — primarily a cross-cutting integration layer.
- `training/` is pure Python — depends on the built Python modules from `vehicle_dynamics/*/bindings/` and `sim/objects/*/bindings/` + Gymnasium, PettingZoo, SB3.

The hard rule: **any `core/` (vehicle or object) never includes Box2D, never allocates on the heap in steady state, never throws, never uses RTTI.** Anything that violates those constraints lives in `sim/` (top level) or above. This is what makes the same code compile for ESP32 — and why the rule extends to object cores too: even if no object's core ever runs on hardware, keeping the constraints uniform avoids per-directory exceptions.

## Per-vehicle internal structure

Within each `vehicle_dynamics/<vehicle>/`:

```
diff_drive/
├── analysis/
│   ├── diff_drive.tex             # Derivation: continuous dynamics, Jacobians, linearization
│   └── jacobian_verify.jl         # Julia: Symbolics.jl re-derivation, numerical comparison
├── bindings/
│   ├── diff_drive_py.cpp          # pybind11 module
│   ├── CMakeLists.txt
│   └── envs/                      # Pure Python: Gymnasium/PettingZoo wrappers per task
├── core/
│   ├── dynamics.hpp               # Templated DiffDriveDynamics<Scalar>
│   ├── integrators.hpp            # RK4, Euler, templated on dynamics + scalar
│   ├── linearize.hpp              # Analytic Jacobians ∂f/∂x, ∂f/∂u
│   ├── types.hpp                  # State/Control aliases, Eigen fixed-size
│   ├── tests/                     # doctest unit tests, run on desktop only
│   └── CMakeLists.txt
└── embedded/
    ├── platformio.ini
    └── src/
        ├── main.cpp               # ESP32 entry point, reuses core/ unchanged
        └── mpc/                   # iLQR or TinyMPC integration
```

## Dynamics model (diff_drive)

State (5D): `x = [px, py, θ, v, ω]` — position, heading, body linear velocity, body angular velocity.

Control (2D): `u = [v_left_cmd, v_right_cmd]` — commanded wheel velocities. Always the hardware-facing parameterization in the core. A `[v, ω]` action wrapper lives in Python for RL convenience.

Continuous-time dynamics:

```
v_cmd_avg = (v_left + v_right) / 2
ω_cmd     = (v_right - v_left) / track_width
ẋ_pos     = v cos θ
ẏ_pos     = v sin θ
θ̇         = ω
v̇         = (v_cmd_avg - v) / τ_motor
ω̇         = (ω_cmd - ω) / τ_motor
```

Integrator: RK4 at `dt = 10 ms`. Box2D runs at the same `dt` when present.

The motor lag is folded into the body velocities (5D state) rather than tracking each wheel velocity separately (which would be 7D). This matches what most skid-steer MPC papers do and keeps the state small for embedded MPC. If wheel-level dynamics ever matter, expand to 7D — but not before.

**Slip is intentionally NOT modeled in core.** The kinematic model plus aggressive domain randomization (effective track width, per-wheel gain, motor τ) is the policy's source of robustness. If policies start exploiting unphysical kinematic behavior, add a velocity-dependent slip term as `ω_actual = ω_cmd · (1 - α|v|)` — but evidence-driven, not preemptive.

## Linearization

Analytic Jacobians `A = ∂f/∂x` and `B = ∂f/∂u` live in `vehicle_dynamics/diff_drive/core/linearize.hpp`. Allocation-free, ESP32-compatible.

### Include conventions

Headers live directly in `vehicle_dynamics/<vehicle>/core/` (or `sim/objects/<object>/core/`), not under a nested `include/<name>/`. Each `core/` CMake target exposes two INTERFACE include dirs — its own directory and its grandparent — so:

- From `sim/`, `bindings/`, or any sibling target: `#include "diff_drive/core/dynamics.hpp"` (or `#include "ball/core/dynamics.hpp"` for objects). Vehicle/object-qualified; siblings under the same grandparent share the include path.
- From inside `core/` or `core/tests/`: `#include "dynamics.hpp"` (unqualified, since they're in the same directory).

In CMake: `target_include_directories(diff_drive_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../..)`. The first path is `core/` itself (for unprefixed includes); the second is the grandparent — `vehicle_dynamics/` for vehicles, `sim/objects/` for objects — so the `<name>/core/...` prefix resolves for consumers. No `/include` suffix anywhere — that nesting is gone.

Verify them three ways:
1. **Autodiff in C++**: instantiate `DiffDriveDynamics<AutoDiffScalar<...>>` and compare in unit tests.
2. **Finite differences**: cheap sanity check in tests.
3. **Symbolic in Julia** (`vehicle_dynamics/diff_drive/analysis/jacobian_verify.jl`, using `Symbolics.jl`): re-derives by symbolic differentiation and compares numerically. Re-run whenever the model changes.

The linear rollout error should scale as `O(‖δx‖²)` near the operating point — write a test that asserts this.

The LaTeX in `analysis/diff_drive.tex` is the source of truth for the derivation. Code matches the LaTeX, not the other way around.

## C++ conventions

- **C++17.** No exceptions in `core/`. Define `EIGEN_NO_EXCEPTIONS` for embedded builds.
- **Templated on scalar type.** `template <typename Scalar> struct DiffDriveDynamics { ... }`. Default to `float` for the embedded build, `double` for desktop tests. Same code, no surprises later.
- **Fixed-size Eigen matrices everywhere.** `Eigen::Matrix<Scalar, 5, 1>`, never `Eigen::VectorXd`. Embedded builds set `EIGEN_NO_MALLOC` so heap usage is a compile error.
- **No virtual functions in `core/`.** Static polymorphism via templates.
- **No STL containers in hot paths.** `std::array` is fine; `std::vector` is not.
- **No `std::cout`, no logging, no I/O in `core/`.** Anything that needs to print lives in `sim/` or above.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for types, `kCamelCase` for constants.
- **Include order**: own header, then project headers, then third-party (Eigen), then std.

Build twice in CI: once for desktop with sanitizers, once cross-compiled for ESP32 (PlatformIO). Both must stay green. The ESP32 build is the canary that catches embedded-hostile code early.

## Python conventions

- **Python 3.11+**, type hints required on public APIs.
- **Gymnasium**, not legacy `gym`. **PettingZoo parallel API** for multi-agent.
- **No business logic in pybind11 layer** — the binding is a thin shim. Wrappers (noise, delay, reward shaping, randomization) are pure Python.
- **One env per file** under `vehicle_dynamics/<vehicle>/bindings/envs/`. Each env exposes a `make()` factory.
- Use `ruff` for lint+format, `pyright` for typecheck.

## Shared simulation layer (`sim/`)

The shared `sim/` owns everything that's not vehicle-specific:

- Box2D world setup, walls, boundary geometry, goal chambers.
- Puck/ball entities, contact filters.
- Sensor abstraction (a vehicle's `core/` produces ground-truth state; `sim/` adds noise, delay, dropout).
- Multi-vehicle coordination (registering N cars in one Box2D world, stepping them together).

Each vehicle's `core/` advances its own state. Each step, `sim/` writes vehicle poses to Box2D as kinematic bodies, steps Box2D, reads contact forces back. Vehicles never know about Box2D directly.

### `kBox2dScale = 10`

`sim/types.hpp` defines `constexpr float kBox2dScale = 10.0f`. **All world-frame coordinates are multiplied by this when crossing INTO Box2D and divided when reading OUT.** It exists because Box2D 3.x bakes in a 5 mm linear slop and a 20 mm vertex-welding threshold (`4 × B2_LINEAR_SLOP`), and our manipulator polygons have features at 5–20 mm — without the scale, `b2ComputeHull` silently drops them as degenerate. User inputs/outputs and every `core/` stay in metres; only the Box2D side is scaled.

### Field geometry + goals

Field walls are static `b2_segmentShape` bodies with `CATEGORY_WALL`, mask excludes ball — the ball passes through the perimeter and is kept in by a soft pull-back force in `Ball::pre_step` (`field_k * penetration` along each violated axis).

When `WorldConfig.goal_y_half > 0` and `goal_extension > 0`, each side wall is split around a gap of half-height `goal_y_half`, and a three-segment "U" chamber of depth `goal_extension` is added behind it under `CATEGORY_GOAL_WALL` (mask DOES include the ball, so the ball physically collides with the goal box). Pull-back also adapts: when the ball's `|py| ≤ goal_y_half` the effective x-bound becomes `xh + goal_extension`, so the ball can sit in the chamber without being yanked back.

**Box2D segments are 2-sided.** A ball that tunnels past the back wall in one step would then have Box2D push it FURTHER out (away from the wall on the wrong side) — the "stuck behind the goal" failure. `Ball::pre_step` therefore includes a hard clamp+bounce on the back wall whenever the ball is in the goal-mouth y-band: if the integrator step exceeds the chamber, the position is pinned at `± (xh + gx) - radius` and the velocity is reflected with restitution. This is a backstop only — the soft pull-back handles the normal-speed case.

Test scenarios that drive a robot/ball straight forward at y=0 will pass through the goal mouth. Add a y-offset (e.g. `y0 = 0.10`) when the test wants a wall impact. See `test_robot.cpp::"Robot (dynamic) physically stops at a wall"` and `test_ball.cpp::"Ball: field pull-back ..."` for the pattern.

## Visualization & episode recording

The viz layer is fully decoupled from the sim. Three core abstractions in `sim/python/viz/`:

```
viz/
├── scene.py         # SceneSpec — pure data (FieldSpec, RobotSpec[], BallSpec[], controls dict)
├── style.py         # StyleConfig — colors, shapes, resolution, team overrides, field markings
├── episode.py       # Episode dataclass + EpisodeRecorder — .npz schema and round-trip
├── recorder.py      # write_video / VideoRecorder (mp4 via ffmpeg, gif via Pillow)
├── input/           # InputDevice protocol; keyboard + gamepad + composite implementations
└── renderers/
    ├── base.py            # Renderer protocol
    ├── _pygame_draw.py    # Shared PygameSceneDrawer — the single source of pixel output
    ├── pygame_live.py     # Window + display.flip(); accepts an `overlay` callback
    └── pygame_headless.py # Off-screen Surface → numpy RGB array; show_hud opt-in
```

The split is deliberate:

- A **`SceneSpec` is what to draw** — pure data, no `sim_py` references. Built either from a live sim (`build_scene(world, robots, balls, ...)`) or reconstructed from a recorded episode (`Episode.scene_at(i)`).
- A **`StyleConfig` is how to draw it** — loaded from a YAML, swappable without touching the renderer.
- A **`Renderer` is where it goes** — live (window) or headless (numpy array). Both delegate to the same `PygameSceneDrawer`, so a video is pixel-identical to what the user sees during teleop.

### Episode `.npz` schema

A single file holds time + per-agent arrays + a JSON metadata blob. State arrays are `(state_dim, T)` — components on axis 0, time on axis 1.

```
time                   shape (T,)        sim time per step
robot_<name>_state     shape (5, T)      [PX, PY, THETA, V, OMEGA]
robot_<name>_action    shape (2, T)      [v_left, v_right] wheel commands
robot_<name>_input     shape (2, T)      [forward, turn] normalised ∈ [-1, 1]   (optional)
ball_<name>_state      shape (4, T)      [PX, PY, VX, VY]
meta                   0-d object        JSON: schema_version, dt, world cfg, agents, coord
```

`meta.agents` is the list-of-dicts manifest (name, type, team, config) — the renderer walks it to reconstruct robot geometry from the .npz alone, with no external file lookup needed. Use `np.load(path, allow_pickle=True)` to read; `Episode.load(path)` wraps that.

### Style YAML — soccer-field markings + team overrides

`sim/configs/styles/default.yaml` is the canonical example. Notable knobs:

- `field.background` is the *surround* (HUD strip, area outside the playfield); `field.field_color` is the green turf rectangle drawn slightly past the field bounds.
- `field.walls` is the perimeter colour (white). Drawing order in the renderer is **surround fill → green turf → markings → walls**, so wall lines sit on top of markings, which sit on top of turf. If you reorder, markings disappear under the turf.
- `markings.*` controls the cosmetic interior lines (halfway line, centre circle, goalie boxes). All overlay-only; no physics interaction.
- `teams.<name>` provides per-team body / manipulator / outline colour overrides applied on top of the default `robot:` style. The control indicator panel uses these colours to fill its bars.

### Control indicator panel

When the renderer has `show_hud=True` and `scene.controls` is populated, the top HUD strip draws per-robot indicator cells:
- **Top row** = blue team (`_TEAM_SIDE = -1`, fans LEFT from centre)
- **Bottom row** = orange team (`+1`, fans RIGHT)
- 1 cell total → dead centre. 1+1 → blue at `−0.5·spacing`, orange at `+0.5·spacing`. Up to 3 per side.
- Vertical bar = `forward` (zero-centred, +up / −down). Horizontal bar = `turn` (zero-centred, **inverted** so +CCW fills LEFT and −CW fills RIGHT — matches the visual direction of the turn).

`scene.controls` is a `dict[str, tuple[float, float]]` (forward, turn). Live teleop passes the live input each frame; `Episode.scene_at` populates it from `robot_inputs` so replays see the recorded bars animate.

### Tools at a glance

| Script | Purpose |
|---|---|
| `teleop.py [--record PATH]` | Drive a single robot live; optionally record an `.npz`. Auto-detects gamepad. |
| `teleop_multi.py [N]` | 1-6 robots in the field for panel-layout review; only the first is driveable. |
| `render_episode.py EP.npz [--out PATH] [--fps N] [--frame-stride N]` | Headless render to mp4 / gif. |
| `replay_episode.py EP.npz` | Live scrub window — play/pause, step, click+drag timeline, speed control. |
| `random_gif.py [PATH]` | Minimal random-action demo → 5 s @ 24 fps gif. |

## Uncertainty injection

Lives in Python wrappers under `training/` or `vehicle_dynamics/<vehicle>/bindings/envs/`, **never in any C++ core**. The MPC running the same core on ESP32 must see a clean, deterministic model.

- **Observation noise**: additive Gaussian per-component, configurable σ.
- **Action noise**: multiplicative gain perturbation + additive noise + occasional dropout (zero with probability p).
- **Action delay**: ring buffer of past commands; per-episode delay sampled from a distribution.
- **Domain randomization per episode**: `τ_motor`, `track_width`, max wheel speed, puck mass, puck friction, surface friction.

## Multi-agent design

PettingZoo parallel API. Centralized training, decentralized execution (CTDE) — each car's policy sees only its own observation plus teammate state, but the critic during training can see global state. MAPPO is the default starting point. Independent PPO is the baseline to beat.

Soccer is the long-horizon goal. Build up:
1. Single car, fixed goal pose (M3).
2. Single car, push puck to goal (M4).
3. Two cars, push puck cooperatively (M5).
4. Two cars vs. two cars, soccer (M7+, its own project).

Reward shaping is potential-based (`r = base + γ·Φ(s') - Φ(s)`) so it doesn't change the optimal policy. Sparse rewards alone won't train; budget real time for shaping and curriculum.

## Milestones (current target)

- **M1** ✅ `diff_drive/core/` dynamics, Jacobians, unit tests, ESP32 cross-compile passes.
- **M2** ✅ `diff_drive/bindings/`, simple matplotlib/pygame viz, no RL yet.
- **M3** ⏳ Gymnasium env in `bindings/envs/`, no contacts, PPO drives car to goal pose.
- **M4** ✅ `sim/` Box2D integration, ball, goals, multi-robot. Visualization + episode recording + scrub-replay all in place. Single-agent push-to-goal RL not started yet (lives at the M3↔M4 boundary).
- **M5** ⏳ PettingZoo multi-agent, two cars cooperative push.
- **M6** Domain randomization + uncertainty wrappers in `training/`.
- **M7** Soccer (open-ended).
- **M8** `diff_drive/embedded/` MPC on ESP32 (later, optional).

Track current milestone in `MILESTONES.md` (separate file). Don't start Mn+1 until Mn's tests pass and the deliverable runs end-to-end.

## Library choices and versions

- **Eigen 3.4+** (header-only).
- **Box2D 3.x** (the C rewrite, not 2.x).
- **doctest** for C++ tests (single header).
- **pybind11 2.11+**.
- **Gymnasium 1.0+**, **PettingZoo 1.24+**.
- **Stable-Baselines3** to start. CleanRL if hacking on algorithms. RLlib if scaling out.
- **PlatformIO** with ESP-IDF framework for ESP32.
- **TinyMPC** as the first MPC solver to try on embedded.
- **Julia 1.10+** with `Symbolics.jl` for analysis only.

Pin versions in `requirements.txt` and the relevant `CMakeLists.txt`. Don't float.

## Adding a new vehicle type

Mirror the `diff_drive/` structure exactly. Same internal layout (`analysis/`, `bindings/`, `core/`, `embedded/`), same conventions, same ESP32-clean rule for `core/`. The shared `sim/` and `training/` layers should accept the new vehicle without modification — if they don't, the abstraction in `sim/` is wrong and needs to be fixed there, not worked around in the new vehicle.

## Things that will trip you up

- **Don't add features to any `core/` for sim convenience.** If `sim/` needs something, add it in `sim/`. The discipline of keeping `core/` embedded-clean pays off later, and erodes fast if you're sloppy now.
- **Don't reach for double precision casually.** Default to `float`; only use `double` where a test demonstrates `float` is insufficient.
- **Don't write your own contact solver.** Box2D exists, is fast, is well-tested. Resist the urge.
- **Don't model slip until you have evidence policies need it.** Domain randomization on the kinematic model handles more than people expect.
- **Don't underestimate reward shaping for multi-agent.** Plan curriculum + potential-based shaping from M5 onward; sparse rewards do not train collaborative behavior in any reasonable wall-clock time.
- **Don't pull in new C++ libraries without checking the ESP32 build.** Keep the canary green.
- **Don't put vehicle-specific logic in `sim/`.** That layer is shared across vehicle types — keep it generic.
- **Don't reorder the renderer draw sequence.** It's surround → turf rect → field markings → wall outlines → balls → robots → HUD → control panel. Markings drawn before the turf will be painted over; walls drawn before markings will be hidden under them.
- **Don't strip `kBox2dScale` "for simplicity".** It looks redundant but it's the only thing keeping `b2ComputeHull` from welding our small manipulator vertices into degeneracy.
- **Don't remove the goal-chamber back-wall clamp in `Ball::pre_step`.** Box2D segments are 2-sided; without the clamp, a fast ball can tunnel past the back wall in one step and Box2D will then push it FURTHER out, not back.

## When in doubt

Ask whether the change you're making would break the ESP32 build. If yes, it goes in `sim/` or above, not `core/`. That single question resolves about 80% of the design decisions on this project.