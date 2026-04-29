"""AtomGym — Gymnasium environments + RL training stack for AtomSim.

Pure-Python package. Imports `sim_py` (built from AtomSim) at env-construction
time; the action/observation/reward/curriculum modules themselves are
sim-agnostic and unit-testable without the C++ build.
"""
