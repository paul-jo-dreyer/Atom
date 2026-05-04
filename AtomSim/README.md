# AtomSim

Multi-agent RL simulator for differential-drive robots. See `CLAUDE.md` for full architecture, conventions, and design decisions.

## Layout (high level)

```
AtomSim/
├── vehicle_dynamics/<vehicle>/    # core/, bindings/, embedded/, analysis/ per vehicle
├── sim/objects/<object>/   # core/, bindings/, analysis/ per passive object (ball, box, …)
├── sim/                   # Box2D world (M4+)
├── tests/                 # cross-cutting integration tests (M4+)
└── training/              # RL scripts (M3+)
```

The Python project lives at the **git root** (`pyproject.toml`, `uv.lock`, `.venv/`), one level above this directory.

## Prerequisites

System packages: `cmake >= 3.20`, `ninja-build`, `libeigen3-dev` (3.4+), `g++` with C++17. Optional but recommended: `ccache`, `mold`.

Python env (run from the git root, not from `AtomSim/`):

```bash
uv sync
```

This creates `../.venv/` and installs `pybind11`, `numpy`, `gymnasium`, etc. The CMake presets pick up that venv automatically via `VIRTUAL_ENV=${sourceDir}/../.venv`.

## Build verification

Run these from `AtomSim/` to confirm everything still works end to end after any change.

### 1. Configure + build both presets

```bash
rm -rf build
cmake --preset debug   && cmake --build build/debug
cmake --preset release && cmake --build build/release
```

Each preset should build 2 targets: `diff_drive_tests` (doctest binary) and `diff_drive_py` (pybind11 module). On release you'll see a harmless `mold ... falling back to ld.bfd` warning — GCC LTO plus mold don't co-operate; ignore.

### 2. Run the C++ unit tests

```bash
ctest --preset debug
```

Expect `100% tests passed, 0 tests failed out of 3`:

- `dynamics: zero state with symmetric command accelerates forward`
- `rk4: integrating forward command moves the body`
- `linearize: produces finite A, B at origin`

The debug preset is built with AddressSanitizer + UBSan, so this is the build that catches memory bugs.

### 3. Smoke-test the Python module

Use the **release** build for Python — the debug `.so` is linked against ASan and won't import without `LD_PRELOAD`. Run from the git root:

```bash
cd ..   # to git root
PYTHONPATH=AtomSim/build/release/vehicle_dynamics/diff_drive/bindings \
  .venv/bin/python -c "
import diff_drive_py as dd, numpy as np
dyn = dd.DiffDriveDynamics()
x1 = dd.rk4_step(dyn, np.zeros(5), np.array([1.0, 1.0]), 0.01)
J  = dd.linearize(dyn, np.zeros(5), np.zeros(2))
print('v after 10ms:', x1[3], '(analytic ~0.1813)')
print('A[V,V]:', J.A[3,3], '(expect -20)')
print('B[omega,:]:', J.B[4], '(expect [-100, 100])')
"
```

Expected output:

```
v after 10ms: 0.18126666666666666 (analytic ~0.1813)
A[V,V]: -20.0 (expect -20)
B[omega,:]: [-100.  100.] (expect [-100, 100])
```

The RK4 result matches the analytic motor-lag response `1 − exp(−dt/τ)` with `τ=0.05, dt=0.01`. The linearization at the origin matches the analytic Jacobians (`A[V,V] = −1/τ`, `B[V,:] = 1/(2τ)·[1,1]`, `B[ω,:] = 1/(W·τ)·[−1,1]` with `W=0.2`).

### 4. Verify VSCode / clangd integration

```bash
ls AtomSim/build/debug/compile_commands.json
```

This file is emitted by the debug build (`CMAKE_EXPORT_COMPILE_COMMANDS=ON` in the root `CMakeLists.txt`). The repo `.vscode/settings.json` points clangd at it, so IDE intellisense should work as soon as the debug build has run once.

## Running the sim

All scripts live under `AtomSim/sim/python/` and self-locate the AtomSim root, so you can run them from anywhere. They require the **release** build of `sim_py` plus the `viz` dependency group:

```bash
cmake --preset release && cmake --build build/release   # from AtomSim/
uv sync --group viz                                     # from git root
```

### Real-time teleop

```bash
.venv/bin/python AtomSim/sim/python/teleop.py
```

Controls: **WASD / arrows** drive, **R** reset, **ESC / Q** quit. If a gamepad is plugged in it's used alongside the keyboard — left stick = forward (Y axis), right stick = turn (X axis), button 0 = reset, button 6 = quit. Per-controller axis remapping lives in `viz/input/gamepad.py`.

To record an episode for replay or training data, pass `--record`:

```bash
.venv/bin/python AtomSim/sim/python/teleop.py --record run1.npz
# Or auto-named: --record alone → episode_YYYYMMDD_HHMMSS.npz in cwd.
```

### Replay an episode (live, scrubbable)

```bash
.venv/bin/python AtomSim/sim/python/replay_episode.py run1.npz
```

Controls: **Space** play/pause, **← →** ±1 frame, **Shift+← →** ±10 frames, **Home / End** jump, **R** restart, **[ / ]** ½× / 2× speed, **click + drag** the bottom timeline. The control-bar panel at top shows the **recorded** input bars — you watch the operator's actions animate exactly as they happened.

### Render an episode to mp4 / gif

```bash
.venv/bin/python AtomSim/sim/python/render_episode.py run1.npz --out run1.mp4
# Optional flags: --fps 30 --frame-stride 2 --quality 8 --style PATH
# Format inferred from extension (.mp4 = H.264 via ffmpeg, .gif = Pillow).
```

The headless renderer uses no display server — works on a bare CI box or remote machine.

### Multi-robot demo (1–6 robots)

```bash
.venv/bin/python AtomSim/sim/python/teleop_multi.py 4
```

Splits N robots between blue (top of field) and orange (bottom): 1 = 1 blue, 2 = 1+1, 3 = 2+1, 4 = 2+2, 5 = 3+2, 6 = 3+3. The first robot is driveable; the rest get synthetic animated control signals so all panel cells animate. Useful for inspecting the indicator layout for any team count.

### Random-action gif demo

```bash
.venv/bin/python AtomSim/sim/python/random_gif.py random.gif
```

Five seconds at 24 fps of random `(forward, turn)` ∈ [-1, 1]² inputs converted into wheel commands. Seeded RNG (`default_rng(42)`) for reproducibility. Useful as a smoke test that the full sim → headless render → gif pipeline still works.

## Visual style (YAML-driven)

`AtomSim/sim/configs/styles/default.yaml` controls **everything visual**: resolution, the green field, the white wall and marking colours, robot/ball shape modes (`full | square_only | point` / `circle | point`), per-team colour overrides, the cosmetic interior soccer markings (centre circle, halfway line, goalie boxes), and the mowed-grass stripe overlay (`field.mowed_stripes_n / _delta / _axis` — set `_n: 0` for a flat turf). Copy and tweak the file for new looks (e.g. `analysis.yaml` for points-on-a-grid).

The interior markings are pure overlay — they're drawn between the green turf and the white wall lines, and the ball passes through them with zero physics interaction. Goal physics (the gap in each side wall + the chamber behind it) is controlled separately by `WorldConfig.goal_y_half` and `goal_extension` on the C++ side, exposed through `sim_py.WorldConfig`.

## Episode `.npz` format

A single compressed `.npz` per episode contains:

```
time                   shape (T,)        sim time per step
robot_<name>_state     shape (5, T)      [PX, PY, THETA, V, OMEGA]
robot_<name>_action    shape (2, T)      [v_left, v_right] wheel commands
robot_<name>_input     shape (2, T)      [forward, turn] normalised ∈ [-1, 1]
ball_<name>_state      shape (4, T)      [PX, PY, VX, VY]
meta                   0-d object        JSON: dt, world cfg, agents, coordinate convention
```

State arrays are `(state_dim, T)` so per-component slicing is contiguous. The `meta.agents` manifest lets `Episode.scene_at(i)` fully reconstruct the scene from the .npz alone — no extra config files needed at render time. Read with `np.load(path, allow_pickle=True)` or `viz.Episode.load(path)`.

## Common red flags

| Symptom | Likely cause |
|---|---|
| `Could NOT find Eigen3` | Install `libeigen3-dev` (≥ 3.4). |
| `Could NOT find pybind11` | `uv sync` from the git root; pybind11 is a project dep, not a system package. |
| `ctest` reports 0 tests | doctest's CMake helper didn't load — check the `include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)` line in `vehicle_dynamics/diff_drive/core/tests/CMakeLists.txt`. |
| Python `ImportError: ... undefined symbol: __asan_*` | You're importing the debug `.so`. Use the release build, or `LD_PRELOAD=$(gcc -print-file-name=libasan.so)` the debug one. |
| `Unrecognized "version" field` from cmake | Your cmake is too old for the preset schema. AtomSim targets schema v3 (cmake ≥ 3.21). |

## Include conventions

Headers in `vehicle_dynamics/<vehicle>/core/` (and `sim/objects/<object>/core/`) are flat — no nested `include/<name>/`. Each `core/` CMake target exposes two INTERFACE include dirs (`core/` itself and its grandparent), so:

- From `sim/`, `bindings/`, or any sibling: `#include "diff_drive/core/dynamics.hpp"` (vehicles) or `#include "ball/core/dynamics.hpp"` (objects).
- From inside `core/` or `core/tests/`: `#include "dynamics.hpp"`.

See `CLAUDE.md` "Include conventions" for the rationale.
