"""StaticFieldPenalty — sigmoid potential field around walls + goalie box.

Why this exists
---------------
Penalty is shaped to give PPO a smooth proximity gradient near static
hazards (the field walls and the opposing goalie box) so the policy
learns to slow / steer away *before* contact, instead of only seeing a
discrete signal at the moment of impact. Smooth gradients also let the
policy learn the boundary fluidly rather than discovering it through
collision events.

Anatomy of the shaping
----------------------
Two source families, with deliberately different shaping curves:

**Walls** — anticipatory: penalty starts ramping up while the robot is
still on the safe side, so PPO sees the boundary before contact.

       penalty = 1 when d <= unavoidable_dist  (close enough that the
                                                robot's body would
                                                collide regardless of
                                                orientation)
       penalty ramps via logistic in (unavoidable_dist, safe_dist)
       penalty = 0 when d >= safe_dist          (well clear)

   Defaults: `unavoidable_dist=0.030` and `safe_dist=0.085` for a 60 mm
   square robot — 30 mm = robot half-extent (collision unavoidable when
   the centre is this close, regardless of yaw); 85 mm gives ~55 mm of
   shaping band before the body must commit.

**Goalie box** — intrusion-only: penalty is zero AT the box boundary
and ramps up only as the robot intrudes deeper inside. The corners of
the field (which sit just outside the box) stay reachable without
penalty, so the policy can still play balls into the corner. The rule
this models is "don't enter the opposing goalie box," not "don't go
near it" — a robot tracing the perimeter is legal.

       penalty = 0 when intrusion <= 0           (outside or on edge)
       penalty ramps via logistic in (0, goalie_box_full_depth)
       penalty = 1 when intrusion >= goalie_box_full_depth

   Default `goalie_box_full_depth=0.06` — one robot side length. The
   policy hits saturation roughly when its body is fully inside the box.

Per-source penalties are combined via **max** — the dominant hazard
shapes behaviour locally. A point inside the goalie box AND near a wall
gets penalty = 1 from both sources; max keeps it at 1.

Why a precomputed grid
----------------------
The static field is small (sub-megabyte at 5 mm resolution over a
0.75 × 0.45 m field) and the analytic per-step computation (a handful
of distance + sigmoid evaluations) is dwarfed by the PPO forward pass.
The grid isn't a perf win — it's an *engineering* win:

  * The shaping field is a thing you can render as a heatmap and inspect.
  * Reward hot path becomes a 4-cell bilinear lookup, decoupled from the
    sigmoid math.
  * Adding a new static hazard (centre-circle penalty, no-go zones) is
    a one-line edit to `_evaluate_at` followed by a re-bake — runtime
    code is unchanged.

Bilinear interpolation gives C0 continuity in value; the gradient is
piecewise constant per cell with jumps at cell boundaries. PPO doesn't
differentiate through reward, so this is fine. At 5 mm cells the
shaping band is sampled ~10× — fine enough that the staircase artefacts
are below noise.

Pusher caveat
-------------
The field is queried at the robot's centre. With a pusher attached, the
true collision distance depends on yaw. We accept this approximation in
v1 — the event-based contact penalty still fires on real impacts, and
the shaping kicks in 30 mm later than ideal in the worst case. Revisit
with an orientation-axis or multi-point lookup if eval shows a
pusher-leads-into-walls failure mode.
"""

from __future__ import annotations

import math

import numpy as np

from AtomGym.rewards._base_reward import RewardContext, RewardTerm


class StaticFieldPenalty(RewardTerm):
    """Sigmoid potential field penalty over walls + opposing goalie box.

    The grid is built once at construction from the supplied geometry
    and sigmoid parameters; per-step lookup is a bilinear interpolation.
    Use with a POSITIVE weight: the term itself returns values in [0, 1],
    so the penalty contribution to total reward is `weight * lookup`.
    Use a positive weight if your composite already negates penalty
    terms; otherwise set weight negative directly.
    """

    name = "static_field"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        field_x_half: float = 0.375,
        field_y_half: float = 0.225,
        goalie_box_depth: float = 0.12,
        goalie_box_y_half: float = 0.10,
        goalie_box_full_depth: float = 0.03,
        safe_dist: float = 0.06,
        unavoidable_dist: float = 0.035,
        grid_resolution: float = 0.005,
        penalize_own_box: bool = False,
        include_goalie_box: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        weight
            Multiplier on the per-step value (which lives in [0, 1]).
        field_x_half, field_y_half
            Half-extents of the playing field in metres. Defaults match
            the sim's WorldConfig (0.75 × 0.45 m).
        goalie_box_depth
            Depth of the goalie box, measured from the goal line into
            the field, in metres. The box sits flush against the goal
            line.
        goalie_box_y_half
            Half-width of the goalie box (extent in ±y), in metres.
        goalie_box_full_depth
            Intrusion depth at which the goalie-box penalty saturates
            to 1. The penalty is 0 at the box boundary and ramps via
            logistic to 1 over this depth. Default 0.06 m ≈ one robot
            side length, so saturation lines up roughly with "robot
            body fully inside the box." If `goalie_box_depth` is
            shorter than this, the penalty inside the box never reaches
            1 on its own — the wall sigmoid still saturates near the
            back wall, so the max-overlay total still ends at 1.
        safe_dist
            Distance from a hazard at which the penalty is zero. Outside
            this distance the term is silent.
        unavoidable_dist
            Distance at which the penalty saturates to 1. For a square
            robot this should be ~half-side-length, since closer than
            this the body would collide regardless of yaw.
        grid_resolution
            Cell size of the precomputed grid in metres. The shaping
            band has width `safe_dist - unavoidable_dist`; pick a
            resolution that samples it at least ~10× to avoid visible
            staircase artefacts at cell boundaries.
        penalize_own_box
            If True, the own goalie box (at -x) is also a forbidden
            zone. Default False — the rule we're modelling is "don't
            enter the OPPOSING team's goalie box."
        include_goalie_box
            If True, the spatial goalie-box source is added to the
            shaping field. **Default False** since the time-based
            `GoalieBoxPenalty` (paired with the env's box-violation
            termination rule) now handles box-rule shaping. This term
            is then walls-only — anticipatory wall avoidance and
            nothing else. Set True to keep the original
            "wall + spatial goalie-box" composite, e.g. for ablation.
        """
        super().__init__(weight=weight)
        if not (0.0 < unavoidable_dist < safe_dist):
            raise ValueError(
                f"need 0 < unavoidable_dist ({unavoidable_dist}) < "
                f"safe_dist ({safe_dist})"
            )
        if grid_resolution <= 0.0:
            raise ValueError(f"grid_resolution must be > 0, got {grid_resolution}")
        if field_x_half <= 0.0 or field_y_half <= 0.0:
            raise ValueError(
                f"field half-extents must be > 0, got ({field_x_half}, {field_y_half})"
            )
        if goalie_box_depth <= 0.0 or goalie_box_y_half <= 0.0:
            raise ValueError(
                f"goalie box dims must be > 0, got "
                f"depth={goalie_box_depth}, y_half={goalie_box_y_half}"
            )
        if goalie_box_depth >= 2.0 * field_x_half:
            raise ValueError(
                f"goalie_box_depth ({goalie_box_depth}) must be < "
                f"field width ({2 * field_x_half})"
            )
        if goalie_box_full_depth <= 0.0:
            raise ValueError(
                f"goalie_box_full_depth must be > 0, got {goalie_box_full_depth}"
            )

        self.field_x_half = float(field_x_half)
        self.field_y_half = float(field_y_half)
        self.goalie_box_depth = float(goalie_box_depth)
        self.goalie_box_y_half = float(goalie_box_y_half)
        self.goalie_box_full_depth = float(goalie_box_full_depth)
        self.safe_dist = float(safe_dist)
        self.unavoidable_dist = float(unavoidable_dist)
        self.grid_resolution = float(grid_resolution)
        self.penalize_own_box = bool(penalize_own_box)
        self.include_goalie_box = bool(include_goalie_box)

        # Wall sigmoid params: hit ~0.99 at d=unavoidable, ~0.01 at
        # d=safe. ln(99) / half_band gives the steepness that lands
        # those endpoints at 1% from saturation; we then hard-clamp
        # outside the band so the sigmoid never returns < 0 or > 1.
        self._sigmoid_midpoint = 0.5 * (self.safe_dist + self.unavoidable_dist)
        half_band = 0.5 * (self.safe_dist - self.unavoidable_dist)
        self._sigmoid_k = math.log(99.0) / half_band

        # Intrusion sigmoid params (for goalie box). Same logistic shape
        # but reframed: zero at intrusion=0 (boundary), one at
        # intrusion=goalie_box_full_depth. Midpoint and steepness are
        # set so the endpoints land at ~1% / ~99%, hard-clamped outside.
        self._intrusion_midpoint = 0.5 * self.goalie_box_full_depth
        self._intrusion_k = math.log(99.0) / self._intrusion_midpoint

        # Build the grid. Cells are aligned so cell (0, 0) sits at
        # (-field_x_half, -field_y_half) and cell (nx-1, ny-1) sits at
        # (+field_x_half, +field_y_half). A point at world (x, y) maps
        # to fractional grid coord ((x + x_half)/dx, (y + y_half)/dx).
        self._x_min = -self.field_x_half
        self._y_min = -self.field_y_half
        self._dx = self.grid_resolution
        self._nx = int(round(2.0 * self.field_x_half / self._dx)) + 1
        self._ny = int(round(2.0 * self.field_y_half / self._dx)) + 1
        self._grid = self._build_grid()

    # ---- core sigmoid ---------------------------------------------------

    def _sigmoid(self, d: float) -> float:
        """Wall shaping. Map a (signed) distance from a hazard surface
        to a [0, 1] penalty, with hard clamps outside the shaping band.

        Inputs follow the convention "positive = safe side"; negative
        values are inside the forbidden region and saturate to 1.
        """
        if d <= self.unavoidable_dist:
            return 1.0
        if d >= self.safe_dist:
            return 0.0
        return 1.0 / (1.0 + math.exp(self._sigmoid_k * (d - self._sigmoid_midpoint)))

    def _intrusion_sigmoid(self, intrusion: float) -> float:
        """Goalie-box shaping. Penalty is 0 at the boundary
        (intrusion <= 0), ramps via logistic to 1 over
        `goalie_box_full_depth`, with hard clamps outside the band.

        `intrusion` is positive when the robot is inside the box and
        negative when it is outside; outside ⟹ zero penalty.
        """
        if intrusion <= 0.0:
            return 0.0
        if intrusion >= self.goalie_box_full_depth:
            return 1.0
        return 1.0 / (
            1.0 + math.exp(-self._intrusion_k * (intrusion - self._intrusion_midpoint))
        )

    # ---- analytic source evaluation (used to build the grid) -----------

    def _evaluate_at(self, x: float, y: float) -> float:
        """Penalty at world point (x, y), as the max over all sources.
        Used only at grid-build time."""
        # Walls: distance from each wall, positive inside the field.
        d_left = x + self.field_x_half
        d_right = self.field_x_half - x
        d_bottom = y + self.field_y_half
        d_top = self.field_y_half - y
        wall_penalty = max(
            self._sigmoid(d_left),
            self._sigmoid(d_right),
            self._sigmoid(d_bottom),
            self._sigmoid(d_top),
        )

        if self.include_goalie_box:
            opp_penalty = self._goalie_box_penalty(x, y, side=+1.0)
            own_penalty = (
                self._goalie_box_penalty(x, y, side=-1.0)
                if self.penalize_own_box else 0.0
            )
        else:
            opp_penalty = 0.0
            own_penalty = 0.0

        return max(wall_penalty, opp_penalty, own_penalty)

    def _goalie_box_penalty(self, x: float, y: float, side: float) -> float:
        """Penalty for intrusion into the goalie box on `side`
        (+1 ⟹ +x box, -1 ⟹ -x box). Zero outside the box, ramps inward
        — see `_intrusion_sigmoid`.

        The box's signed-distance-field (SDF) is computed (negative
        inside, positive outside); intrusion = -SDF is fed into the
        intrusion sigmoid.
        """
        if side > 0:
            x_box_min = self.field_x_half - self.goalie_box_depth
            x_box_max = self.field_x_half
        else:
            x_box_min = -self.field_x_half
            x_box_max = -self.field_x_half + self.goalie_box_depth
        y_box_min = -self.goalie_box_y_half
        y_box_max = self.goalie_box_y_half

        # Axis-aligned-box SDF: positive outside, zero on edge,
        # negative inside.
        cx = 0.5 * (x_box_min + x_box_max)
        cy = 0.5 * (y_box_min + y_box_max)
        hx = 0.5 * (x_box_max - x_box_min)
        hy = 0.5 * (y_box_max - y_box_min)
        ex = abs(x - cx) - hx
        ey = abs(y - cy) - hy
        outside = math.sqrt(max(ex, 0.0) ** 2 + max(ey, 0.0) ** 2)
        inside = min(max(ex, ey), 0.0)  # ≤ 0 when inside
        sdf = outside + inside

        return self._intrusion_sigmoid(-sdf)

    # ---- grid build -----------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        """Vectorised precompute of the static penalty over the field."""
        xs = self._x_min + np.arange(self._nx, dtype=np.float64) * self._dx
        ys = self._y_min + np.arange(self._ny, dtype=np.float64) * self._dx
        grid = np.empty((self._nx, self._ny), dtype=np.float32)
        # Tight loop is fine — done once per env construction, not per
        # step. Vectorising the sigmoid+SDF would be faster but obscures
        # the math and is not on any critical path.
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid[i, j] = self._evaluate_at(float(x), float(y))
        return grid

    # ---- runtime lookup -------------------------------------------------

    def _world_to_grid(self, x: float, y: float) -> tuple[float, float]:
        """Continuous world coord (m) → fractional grid coord."""
        gi = (x - self._x_min) / self._dx
        gj = (y - self._y_min) / self._dx
        return gi, gj

    def _bilinear_lookup(self, gi: float, gj: float) -> float:
        """Bilinear interpolation at fractional grid coord. Out-of-grid
        points clamp to the nearest in-grid cell — robots shouldn't be
        outside the field, but if they tunnel through (bug or DR edge
        case) we return the boundary value rather than crashing."""
        nx, ny = self._grid.shape
        gi = max(0.0, min(gi, nx - 1.0))
        gj = max(0.0, min(gj, ny - 1.0))
        i0 = int(math.floor(gi))
        j0 = int(math.floor(gj))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        ti = gi - i0
        tj = gj - j0
        v00 = self._grid[i0, j0]
        v10 = self._grid[i1, j0]
        v01 = self._grid[i0, j1]
        v11 = self._grid[i1, j1]
        return float(
            (1.0 - ti) * (1.0 - tj) * v00
            + ti * (1.0 - tj) * v10
            + (1.0 - ti) * tj * v01
            + ti * tj * v11
        )

    def lookup(self, x: float, y: float) -> float:
        """World-coord → penalty. Public so the heatmap inspector and
        unit tests can hit the same code path the env uses."""
        gi, gj = self._world_to_grid(x, y)
        return self._bilinear_lookup(gi, gj)

    # ---- RewardTerm protocol -------------------------------------------

    def __call__(self, ctx: RewardContext) -> float:
        v = ctx.obs_view
        rx = v.self_px(ctx.obs) * ctx.field_x_half
        ry = v.self_py(ctx.obs) * ctx.field_y_half
        return self.lookup(rx, ry)

    # ---- introspection --------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """The precomputed (nx, ny) penalty grid. Read-only intent."""
        return self._grid

    @property
    def grid_origin(self) -> tuple[float, float]:
        """World coord of grid cell (0, 0) — i.e. (-x_half, -y_half)."""
        return self._x_min, self._y_min
