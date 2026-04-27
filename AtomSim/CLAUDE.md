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
├── dynamics/                      # One folder per vehicle type
│   └── diff_drive/
│       ├── analysis/              # LaTeX derivations, Julia symbolic verification
│       │   └── diff_drive.tex
│       ├── bindings/              # pybind11 wrapper for this vehicle's core
│       ├── core/                  # C++17 dynamics. ESP32-safe. No allocs, no exceptions.
│       └── embedded/              # PlatformIO ESP32 project + MPC for this vehicle
├── sim/                           # Shared physics: Box2D wrapper, walls, puck, sensors
├── tests/                         # Cross-cutting integration tests
└── training/                      # RL scripts: SB3 to start, CleanRL/RLlib if needed
```

The grouping principle: anything specific to one vehicle type lives under `dynamics/<vehicle>/`. Anything shared across vehicles (the world, contact handling, sensor noise wrappers, training scripts) lives at the top level. Adding a second vehicle type (e.g. `dynamics/ackermann/`) should not require touching `sim/` or `training/`.

## Layering and what depends on what

- `dynamics/<vehicle>/core/` depends on Eigen (header-only, fixed-size matrices). Nothing else.
- `dynamics/<vehicle>/bindings/` depends on its sibling `core/` + pybind11.
- `dynamics/<vehicle>/embedded/` depends on its sibling `core/` ONLY. If you find yourself wanting `sim/` here, stop.
- `dynamics/<vehicle>/analysis/` is independent — LaTeX + Julia for derivations and symbolic Jacobian verification.
- `sim/` depends on the `core/` of any vehicles it instantiates + Box2D 3.x.
- `tests/` depends on whatever it's testing — primarily a cross-cutting integration layer.
- `training/` is pure Python — depends on the built Python modules from `dynamics/*/bindings/` + Gymnasium, PettingZoo, SB3.

The hard rule: **`dynamics/<vehicle>/core/` never includes Box2D, never allocates on the heap in steady state, never throws, never uses RTTI.** Anything that violates those constraints lives in `sim/` or above. This is what makes the same code compile for ESP32.

## Per-vehicle internal structure

Within each `dynamics/<vehicle>/`:

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

Analytic Jacobians `A = ∂f/∂x` and `B = ∂f/∂u` live in `dynamics/diff_drive/core/linearize.hpp`. Allocation-free, ESP32-compatible.

### Include conventions

Headers live directly in `dynamics/<vehicle>/core/`, not under a nested `include/<vehicle>/`. The `core/` CMake target exposes two INTERFACE include dirs — its own directory and its parent — so:

- From `sim/`, `bindings/`, or any sibling target: `#include "diff_drive/core/dynamics.hpp"` (vehicle-qualified; multiple vehicles will share the same parent include path).
- From inside `core/` or `core/tests/`: `#include "dynamics.hpp"` (unqualified, since they're in the same directory).

In CMake: `target_include_directories(diff_drive_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../..)`. The first path is `core/` itself (for unprefixed includes); the second is `dynamics/` (so the `diff_drive/core/...` prefix resolves for consumers). No `/include` suffix anywhere — that nesting is gone.

Verify them three ways:
1. **Autodiff in C++**: instantiate `DiffDriveDynamics<AutoDiffScalar<...>>` and compare in unit tests.
2. **Finite differences**: cheap sanity check in tests.
3. **Symbolic in Julia** (`dynamics/diff_drive/analysis/jacobian_verify.jl`, using `Symbolics.jl`): re-derives by symbolic differentiation and compares numerically. Re-run whenever the model changes.

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
- **One env per file** under `dynamics/<vehicle>/bindings/envs/`. Each env exposes a `make()` factory.
- Use `ruff` for lint+format, `pyright` for typecheck.

## Shared simulation layer (`sim/`)

The shared `sim/` owns everything that's not vehicle-specific:

- Box2D world setup, walls, boundary geometry.
- Puck/ball entities, contact filters.
- Sensor abstraction (a vehicle's `core/` produces ground-truth state; `sim/` adds noise, delay, dropout).
- Multi-vehicle coordination (registering N cars in one Box2D world, stepping them together).

Each vehicle's `core/` advances its own state. Each step, `sim/` writes vehicle poses to Box2D as kinematic bodies, steps Box2D, reads contact forces back. Vehicles never know about Box2D directly.

## Uncertainty injection

Lives in Python wrappers under `training/` or `dynamics/<vehicle>/bindings/envs/`, **never in any C++ core**. The MPC running the same core on ESP32 must see a clean, deterministic model.

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

- **M1** — `diff_drive/core/` dynamics, Jacobians, unit tests, ESP32 cross-compile passes.
- **M2** — `diff_drive/bindings/`, simple matplotlib/pygame viz, no RL yet.
- **M3** — Gymnasium env in `bindings/envs/`, no contacts, PPO drives car to goal pose.
- **M4** — `sim/` Box2D integration, puck, single-agent push-to-goal.
- **M5** — PettingZoo multi-agent, two cars cooperative push.
- **M6** — Domain randomization + uncertainty wrappers in `training/`.
- **M7** — Soccer (open-ended).
- **M8** — `diff_drive/embedded/` MPC on ESP32 (later, optional).

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

## When in doubt

Ask whether the change you're making would break the ESP32 build. If yes, it goes in `sim/` or above, not `core/`. That single question resolves about 80% of the design decisions on this project.