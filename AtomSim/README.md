# AtomSim

Multi-agent RL simulator for differential-drive robots. See `CLAUDE.md` for full architecture, conventions, and design decisions.

## Layout (high level)

```
AtomSim/
├── dynamics/<vehicle>/    # core/, bindings/, embedded/, analysis/ per vehicle
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
PYTHONPATH=AtomSim/build/release/dynamics/diff_drive/bindings \
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

## Common red flags

| Symptom | Likely cause |
|---|---|
| `Could NOT find Eigen3` | Install `libeigen3-dev` (≥ 3.4). |
| `Could NOT find pybind11` | `uv sync` from the git root; pybind11 is a project dep, not a system package. |
| `ctest` reports 0 tests | doctest's CMake helper didn't load — check the `include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)` line in `dynamics/diff_drive/core/tests/CMakeLists.txt`. |
| Python `ImportError: ... undefined symbol: __asan_*` | You're importing the debug `.so`. Use the release build, or `LD_PRELOAD=$(gcc -print-file-name=libasan.so)` the debug one. |
| `Unrecognized "version" field` from cmake | Your cmake is too old for the preset schema. AtomSim targets schema v3 (cmake ≥ 3.21). |

## Include conventions

Headers in `dynamics/<vehicle>/core/` are flat — no nested `include/<vehicle>/`. The `core/` CMake target exposes two INTERFACE include dirs (`core/` and `dynamics/`), so:

- From `sim/`, `bindings/`, or any sibling: `#include "diff_drive/core/dynamics.hpp"`.
- From inside `core/` or `core/tests/`: `#include "dynamics.hpp"`.

See `CLAUDE.md` "Include conventions" for the rationale.
