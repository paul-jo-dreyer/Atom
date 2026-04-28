"""Episode storage: timestamped trajectories of one or more agents in .npz form.

Schema (single .npz file):

    time                   shape (T,)        sim time per step (seconds)
    robot_<name>_state     shape (5, T)      [PX, PY, THETA, V, OMEGA]
    robot_<name>_action    shape (2, T)      [v_left, v_right] commanded
    robot_<name>_input     shape (2, T)      [forward, turn] normalized ∈ [-1, 1] (optional)
    ball_<name>_state      shape (4, T)      [PX, PY, VX, VY]
    meta                   0-d str           JSON-encoded metadata blob

State arrays are `(state_dim, T)` — components on axis 0, time on axis 1.
Picked over `(T, state_dim)` so per-component slicing (e.g. PX over time) is
contiguous; per-timestep slicing (for rendering) costs a stride but is fast
at our scale.

Meta schema (JSON):

    {
      "schema_version": 1,
      "dt": float,
      "world":  { "field_x_half": float, "field_y_half": float },
      "agents": [
        { "name": str, "type": "diff_drive", "team": str|null,
          "config": { "chassis_side": float, "manipulator_parts": [[ [x,y], ... ], ...] } },
        { "name": str, "type": "ball", "team": null,
          "config": { "radius": float } },
        ...
      ],
      "coordinate_convention": "y_up_meters",
    }

`agents` is the source of truth for which arrays exist and how to draw
them — render code walks this list to reconstruct SceneSpecs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .scene import BallSpec, FieldSpec, RobotSpec, SceneSpec

SCHEMA_VERSION = 1


@dataclass
class Episode:
    time: np.ndarray                                  # (T,)
    robot_states: dict[str, np.ndarray]               # name → (5, T) [PX,PY,THETA,V,OMEGA]
    robot_actions: dict[str, np.ndarray]              # name → (2, T) [vL, vR] wheel cmds
    robot_inputs: dict[str, np.ndarray]               # name → (2, T) [forward, turn] normalized
    ball_states: dict[str, np.ndarray]                # name → (4, T) [PX, PY, VX, VY]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- properties ------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return int(self.time.shape[0])

    @property
    def dt(self) -> float:
        return float(self.meta.get("dt", 0.0))

    # ---- IO --------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "time": np.asarray(self.time, dtype=np.float32),
            "meta": np.array(json.dumps(self.meta), dtype=object),
        }
        for name, arr in self.robot_states.items():
            arrays[f"robot_{name}_state"] = np.asarray(arr, dtype=np.float32)
        for name, arr in self.robot_actions.items():
            arrays[f"robot_{name}_action"] = np.asarray(arr, dtype=np.float32)
        for name, arr in self.robot_inputs.items():
            arrays[f"robot_{name}_input"] = np.asarray(arr, dtype=np.float32)
        for name, arr in self.ball_states.items():
            arrays[f"ball_{name}_state"] = np.asarray(arr, dtype=np.float32)
        # `meta` is a 0-d object array, which np.savez_compressed handles via pickle.
        # Allowed because we control the producer; loaders set allow_pickle=True.
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        path = Path(path)
        with np.load(path, allow_pickle=True) as npz:
            time = np.asarray(npz["time"])
            meta = json.loads(str(npz["meta"]))
            robot_states: dict[str, np.ndarray] = {}
            robot_actions: dict[str, np.ndarray] = {}
            robot_inputs: dict[str, np.ndarray] = {}
            ball_states: dict[str, np.ndarray] = {}
            for key in npz.files:
                if key.startswith("robot_") and key.endswith("_state"):
                    name = key[len("robot_"):-len("_state")]
                    robot_states[name] = np.asarray(npz[key])
                elif key.startswith("robot_") and key.endswith("_action"):
                    name = key[len("robot_"):-len("_action")]
                    robot_actions[name] = np.asarray(npz[key])
                elif key.startswith("robot_") and key.endswith("_input"):
                    name = key[len("robot_"):-len("_input")]
                    robot_inputs[name] = np.asarray(npz[key])
                elif key.startswith("ball_") and key.endswith("_state"):
                    name = key[len("ball_"):-len("_state")]
                    ball_states[name] = np.asarray(npz[key])
        return cls(
            time=time,
            robot_states=robot_states,
            robot_actions=robot_actions,
            robot_inputs=robot_inputs,
            ball_states=ball_states,
            meta=meta,
        )

    # ---- rendering bridge ------------------------------------------------

    def scene_at(self, i: int) -> SceneSpec:
        """Reconstruct a SceneSpec for frame `i`. Walks `meta["agents"]`."""
        if i < 0 or i >= self.num_frames:
            raise IndexError(f"frame {i} out of range [0, {self.num_frames})")

        world = self.meta["world"]
        field_spec = FieldSpec(
            x_half=float(world["field_x_half"]),
            y_half=float(world["field_y_half"]),
        )

        robots: list[RobotSpec] = []
        balls: list[BallSpec] = []
        controls: dict[str, tuple[float, float]] = {}
        for ag in self.meta.get("agents", []):
            name = ag["name"]
            kind = ag["type"]
            if kind == "diff_drive":
                s = self.robot_states[name][:, i]
                cfg = ag.get("config", {})
                manip = cfg.get("manipulator_parts", [])
                robots.append(RobotSpec(
                    name=name,
                    team=ag.get("team"),
                    px=float(s[0]),
                    py=float(s[1]),
                    theta=float(s[2]),
                    chassis_side=float(cfg.get("chassis_side", 0.10)),
                    manipulator_parts=tuple(
                        tuple((float(v[0]), float(v[1])) for v in part) for part in manip
                    ),
                ))
                if name in self.robot_inputs:
                    inp = self.robot_inputs[name][:, i]
                    controls[name] = (float(inp[0]), float(inp[1]))
            elif kind == "ball":
                s = self.ball_states[name][:, i]
                cfg = ag.get("config", {})
                balls.append(BallSpec(
                    name=name,
                    px=float(s[0]),
                    py=float(s[1]),
                    radius=float(cfg.get("radius", 0.025)),
                ))
            else:
                raise ValueError(f"Unknown agent type {kind!r} for {name}")

        return SceneSpec(
            field=field_spec,
            robots=robots,
            balls=balls,
            t=float(self.time[i]),
            controls=controls,
        )


class EpisodeRecorder:
    """Append-only buffer that turns into an Episode on `finalize()`.

    Use during live teleop or rollouts:

        rec = EpisodeRecorder(dt=DT, world=world, agents=[...])
        for ...:
            rec.append(t, robot_states={"robot_blue": np.array([...])},
                          robot_actions={"robot_blue": cmd},
                          ball_states={"ball": np.array([...])})
        ep = rec.finalize()
        ep.save("episode_0001.npz")
    """

    def __init__(
        self,
        dt: float,
        world: dict[str, Any],
        agents: list[dict[str, Any]],
    ) -> None:
        self._meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dt": float(dt),
            "world": world,
            "agents": agents,
            "coordinate_convention": "y_up_meters",
        }
        self._time: list[float] = []
        self._robot_states: dict[str, list[np.ndarray]] = {}
        self._robot_actions: dict[str, list[np.ndarray]] = {}
        self._robot_inputs: dict[str, list[np.ndarray]] = {}
        self._ball_states: dict[str, list[np.ndarray]] = {}

    def append(
        self,
        t: float,
        *,
        robot_states: dict[str, np.ndarray] | None = None,
        robot_actions: dict[str, np.ndarray] | None = None,
        robot_inputs: dict[str, np.ndarray] | None = None,
        ball_states: dict[str, np.ndarray] | None = None,
    ) -> None:
        self._time.append(float(t))
        for name, arr in (robot_states or {}).items():
            self._robot_states.setdefault(name, []).append(np.asarray(arr).copy())
        for name, arr in (robot_actions or {}).items():
            self._robot_actions.setdefault(name, []).append(np.asarray(arr).copy())
        for name, arr in (robot_inputs or {}).items():
            self._robot_inputs.setdefault(name, []).append(np.asarray(arr).copy())
        for name, arr in (ball_states or {}).items():
            self._ball_states.setdefault(name, []).append(np.asarray(arr).copy())

    def finalize(self) -> Episode:
        # Stack each per-agent list into (state_dim, T).
        def stack_t(buf: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
            return {name: np.stack(rows, axis=1) for name, rows in buf.items()}

        return Episode(
            time=np.asarray(self._time, dtype=np.float32),
            robot_states=stack_t(self._robot_states),
            robot_actions=stack_t(self._robot_actions),
            robot_inputs=stack_t(self._robot_inputs),
            ball_states=stack_t(self._ball_states),
            meta=self._meta,
        )

    def __len__(self) -> int:
        return len(self._time)
