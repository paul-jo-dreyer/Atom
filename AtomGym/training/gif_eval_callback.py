"""SB3 callback that periodically writes a gif of the current policy.

Hooks into PPO's training loop, and every `render_every` env steps it:
  1. resets a small grid of dedicated eval envs to fixed seeds (so progress
     across time is comparable — same scenarios, different policy
     iterations);
  2. rolls them all out IN LOCKSTEP with `deterministic=True`, capturing
     each control-step's frame from each env;
  3. tiles those per-cell frames into a single grid frame at every kept
     timestep;
  4. writes the sequence to `<save_dir>/step_<NNNNNNNN>.gif`.

Grid shape is `(rows, cols)`. With (1, 1) the output is a single rollout
with the same look as before; with (2, 2) you get four scenarios (different
seeds) tiled side-by-side. Each cell carries its own per-cell HUD (its own
running cumulative reward) since they all start at the same global step.

Cells that finish (terminated or truncated) earlier than the others freeze
on their last frame for the remainder of the composite — the policy looks
"parked" in those cells while the others continue, which reads naturally.

The gif's playback fps is tied to the env's `control_dt` and the chosen
`frame_stride` so the gif plays back at real-time sim speed regardless
of how many frames you keep.

Failures are caught and logged — a viz hiccup never kills training.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# AtomSim's viz package isn't pip-installed; it lives at
# AtomSim/sim/python/viz. Add that to sys.path so we can import it.
def _ensure_atomsim_viz_on_path() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "AtomSim" / "sim" / "python"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Could not locate AtomSim/sim/python (needed for the viz package)."
    )


_ATOMSIM_VIZ_PARENT = _ensure_atomsim_viz_on_path()

from viz.recorder import VideoRecorder  # noqa: E402
from viz.renderers import PygameHeadlessRenderer  # noqa: E402
from viz.scene import build_scene  # noqa: E402
from viz.style import load_style  # noqa: E402

from AtomGym.environments import AtomSoloEnv, AtomTeamEnv  # noqa: E402


_DEFAULT_STYLE_YAML = (
    _ATOMSIM_VIZ_PARENT.parent / "configs" / "styles" / "default.yaml"
)


def _composite_grid(
    cells: list[np.ndarray], rows: int, cols: int
) -> np.ndarray:
    """Tile per-cell frames row-major into a single image. Cells must all
    share the same (H, W, 3) shape."""
    assert len(cells) == rows * cols
    row_arrays = [
        np.concatenate(cells[r * cols : (r + 1) * cols], axis=1)
        for r in range(rows)
    ]
    return np.concatenate(row_arrays, axis=0)


class GifEvalCallback(BaseCallback):
    """Periodically render the current policy as a (possibly grid-composited) .gif.

    Parameters
    ----------
    eval_env_factory
        A no-arg callable returning a fresh env (AtomSoloEnv or
        AtomTeamEnv). One env per grid cell is constructed at training
        start; subsequent renders reset each one to a fixed seed. The
        callback duck-types on `env.opponent` to decide whether to
        render the second robot.
    render_every
        Render cadence in total env steps. Counted against
        `self.num_timesteps`, which SB3 increments by `n_envs` per env.step.
    save_dir
        Directory where gifs land. Created if missing.
    eval_seed
        Base seed. Cell `i` is reset with seed `eval_seed + i`, so a 2×2
        grid evaluates 4 distinct scenarios. Same scenarios every render
        ⟹ visual progress over training is directly comparable.
    grid_rows, grid_cols
        Composite layout. (1, 1) ⟹ single-cell rollout (no grid). The
        callback owns `grid_rows × grid_cols` env instances, all
        constructed via `eval_env_factory`.
    frame_stride
        Keep every Nth control-step frame in the gif. Higher = smaller
        files, choppier playback. Default 1 keeps every frame.
    max_seconds
        Hard cap on sim time per rollout, in seconds. None ⟹ no cap (run
        until every cell terminates/truncates, or env.max_episode_steps).
    style_yaml_path
        Path to an AtomSim style YAML. Defaults to AtomSim's
        `configs/styles/default.yaml`.
    verbose
        Standard SB3 verbosity. >= 1 prints a one-line summary per render.
    """

    def __init__(
        self,
        *,
        eval_env_factory: Callable[[], AtomSoloEnv | AtomTeamEnv],
        render_every: int,
        save_dir: Path,
        eval_seed: int = 999,
        grid_rows: int = 1,
        grid_cols: int = 1,
        frame_stride: int = 1,
        max_seconds: float | None = None,
        style_yaml_path: Path | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if render_every <= 0:
            raise ValueError(f"render_every must be > 0, got {render_every}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if max_seconds is not None and max_seconds <= 0.0:
            raise ValueError(f"max_seconds must be > 0 or None, got {max_seconds}")
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError(
                f"grid dimensions must be >= 1, got {grid_rows}x{grid_cols}"
            )
        self._eval_env_factory = eval_env_factory
        self._render_every = int(render_every)
        self._save_dir = Path(save_dir)
        self._eval_seed = int(eval_seed)
        self._grid_rows = int(grid_rows)
        self._grid_cols = int(grid_cols)
        self._n_cells = self._grid_rows * self._grid_cols
        self._frame_stride = int(frame_stride)
        self._max_seconds = float(max_seconds) if max_seconds is not None else None
        self._style_yaml_path = (
            Path(style_yaml_path) if style_yaml_path is not None else _DEFAULT_STYLE_YAML
        )
        self._next_render_at: int = self._render_every

        # Initialised lazily in `_init_callback` (after model + env are bound).
        self._eval_envs: list[AtomSoloEnv | AtomTeamEnv] = []
        self._renderer: PygameHeadlessRenderer | None = None

    # ---- SB3 lifecycle hooks --------------------------------------------

    def _init_callback(self) -> None:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        # Build N envs, one per grid cell. Their internal _rng is seeded
        # at construction; we then re-seed via reset(seed=...) on each render
        # so every render starts from a known state.
        self._eval_envs = [self._eval_env_factory() for _ in range(self._n_cells)]
        # Single shared renderer — all envs share field bounds, so one
        # PygameHeadlessRenderer can serve all cells.
        env0 = self._eval_envs[0]
        style = load_style(self._style_yaml_path)
        self._renderer = PygameHeadlessRenderer(
            style,
            field_x_half=env0.field_x_half,
            field_y_half=env0.field_y_half,
            show_hud=True,
        )
        if self.verbose >= 1:
            grid = f"{self._grid_rows}x{self._grid_cols}"
            print(
                f"[gif_eval] rendering every {self._render_every:,} steps "
                f"({grid} grid, seeds {self._eval_seed}..{self._eval_seed + self._n_cells - 1}) "
                f"→ {self._save_dir}"
            )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_render_at:
            try:
                self._render_grid_episode()
            except Exception as e:
                # Never let a viz hiccup take down training. Print a
                # traceback so the user can fix it later.
                print(f"[gif_eval] render failed at step {self.num_timesteps:,}: {e}")
                traceback.print_exc()
            # Schedule the next render aligned to the next interval boundary.
            n = (self.num_timesteps // self._render_every) + 1
            self._next_render_at = n * self._render_every
        return True

    # ---- rollout + render -----------------------------------------------

    def _render_grid_episode(self) -> None:
        assert self._renderer is not None and self._eval_envs

        # Reset each cell with a distinct fixed seed.
        obs_per_cell: list[np.ndarray] = []
        cumulative_R_per_cell: list[float] = [0.0] * self._n_cells
        done_per_cell: list[bool] = [False] * self._n_cells
        last_action_per_cell: list[np.ndarray] = [
            np.zeros(2, dtype=np.float32) for _ in range(self._n_cells)
        ]
        last_frame_per_cell: list[np.ndarray | None] = [None] * self._n_cells
        for i, env in enumerate(self._eval_envs):
            obs, _ = env.reset(seed=self._eval_seed + i)
            obs_per_cell.append(obs)

        # Stream frames directly to disk via VideoRecorder rather than
        # accumulating a list[ndarray] in memory. Each composite frame is
        # ~render_w × render_h × n_cells × 3 bytes (e.g. 3×3 × 1280×720 ≈
        # 25 MB), so a multi-second rollout buffered in RAM can spike to
        # >1 GB on the main process — enough to OOM the box at high n_envs
        # with SubprocVecEnv. imageio's gif writer streams (writes each
        # frame's bytes to fp on add_image), so this is O(1) in rollout
        # length.
        env0 = self._eval_envs[0]
        playback_fps = (1.0 / env0.control_dt) / self._frame_stride
        out = self._save_dir / f"step_{self.num_timesteps:08d}.gif"
        recorder: VideoRecorder | None = None  # lazy: create on first frame
        n_frames_written = 0
        sub_idx = 0
        sim_t = 0.0
        time_capped = False

        # Lockstep loop: advance every not-yet-done cell by one control step
        # per outer iteration. Cells that have ended freeze on their last
        # rendered frame.
        while not all(done_per_cell):
            for i, env in enumerate(self._eval_envs):
                if done_per_cell[i]:
                    continue
                action, _state = self.model.predict(
                    obs_per_cell[i], deterministic=True
                )
                obs, reward, term, trunc, _info = env.step(action)
                obs_per_cell[i] = obs
                cumulative_R_per_cell[i] += float(reward)
                last_action_per_cell[i] = np.asarray(action, dtype=np.float32)
                if term or trunc:
                    done_per_cell[i] = True

            # All running cells are now at the same control step. Pick any
            # not-done env's t for the global timestamp; if all are done we
            # don't need t for anything (loop exits).
            running = [i for i in range(self._n_cells) if not done_per_cell[i]]
            if running:
                sim_t = self._eval_envs[running[0]].t
            sub_idx += 1

            # Frame stride: only render every Nth control step.
            if sub_idx % self._frame_stride != 0:
                if self._max_seconds is not None and sim_t >= self._max_seconds:
                    time_capped = True
                    break
                continue

            # Render each cell. Done cells reuse their last frozen frame.
            cell_frames: list[np.ndarray] = []
            for i, env in enumerate(self._eval_envs):
                if done_per_cell[i] and last_frame_per_cell[i] is not None:
                    cell_frames.append(last_frame_per_cell[i])
                    continue

                # Duck-type on `env.opponent` so this callback works for
                # both AtomSoloEnv (no opponent) and AtomTeamEnv (1v1).
                robots: list[Any] = [("blue", env.robot)]
                teams: dict[str, str] = {"blue": "blue"}
                a = last_action_per_cell[i]
                controls: dict[str, Any] = {"blue": (float(a[0]), float(a[1]))}
                if hasattr(env, "opponent") and env.opponent is not None:
                    # "orange" is defined as a team override in the style
                    # YAML (matched saturation/lightness of "blue"); a
                    # team key not in the YAML falls back to the default
                    # robot style — a light grey-blue, which is what we
                    # had before this fix.
                    robots.append(("orange", env.opponent))
                    teams["orange"] = "orange"
                    opp_a = getattr(env, "last_opponent_action", None)
                    if opp_a is None:
                        controls["orange"] = (0.0, 0.0)
                    else:
                        controls["orange"] = (float(opp_a[0]), float(opp_a[1]))
                scene = build_scene(
                    env.world,
                    robots,
                    [("main", env.ball)],
                    t=sim_t,
                    teams=teams,
                )
                scene.controls = controls
                hud_lines = [
                    f"step {self.num_timesteps:,}  cell {i}  seed {self._eval_seed + i}",
                    f"t={sim_t:5.2f}s   R={cumulative_R_per_cell[i]:+8.2f}",
                ]
                frame = self._renderer.render(scene, hud_lines=hud_lines)
                cell_frames.append(frame)
                last_frame_per_cell[i] = frame

            composite = _composite_grid(
                cell_frames, self._grid_rows, self._grid_cols
            )
            if recorder is None:
                recorder = VideoRecorder(out, fps=playback_fps)
            recorder.add_frame(composite)
            n_frames_written += 1

            if self._max_seconds is not None and sim_t >= self._max_seconds:
                time_capped = True
                break

        if recorder is None:
            return  # nothing to write
        recorder.close()

        if self.verbose >= 1:
            n_scored = sum(
                1 for env in self._eval_envs
                if env.ball.state[0] - env.ball_radius > env.field_x_half
            )
            mean_R = float(np.mean(cumulative_R_per_cell))
            outcome = "time_capped" if time_capped else "all_ended"
            print(
                f"[gif_eval] step {self.num_timesteps:>9,}: {out.name}  "
                f"({n_frames_written} frames, t={sim_t:.2f}s, "
                f"mean_R={mean_R:+.2f}, scored={n_scored}/{self._n_cells}, "
                f"{outcome})"
            )
