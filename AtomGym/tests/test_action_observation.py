"""Unit tests for the action / observation encoding.

Pure-Python: doesn't import sim_py, no C++ build required.
"""

from __future__ import annotations

import numpy as np
import pytest

from AtomGym.action_observation import (
    ActionView,
    BALL_BLOCK_DIM,
    OMEGA_MAX_DEFAULT,
    ObsView,
    ROBOT_BLOCK_DIM,
    TRACK_WIDTH_DEFAULT,
    V_BALL_MAX,
    V_MAX_DEFAULT,
    action_to_wheel_cmds,
    build_observation,
    obs_dim,
    sincos_from_theta,
    theta_from_sincos,
)


FIELD_X_HALF = 0.375
FIELD_Y_HALF = 0.225


# ---------------------------------------------------------------------------
# obs_dim / ObsView shape + slice properties
# ---------------------------------------------------------------------------


def test_obs_dim_solo() -> None:
    # 1 robot total → 4 ball + 7 self = 11
    assert obs_dim(1) == 11


def test_obs_dim_1v1() -> None:
    # 2 robots total → 4 + 7 + 7 = 18
    assert obs_dim(2) == 18


def test_obs_dim_2v2() -> None:
    # 4 robots total → 4 + 7·4 = 32
    assert obs_dim(4) == 32


def test_obsview_total_dim_matches_obs_dim() -> None:
    assert ObsView(3).total_dim == obs_dim(3)


def test_obsview_slice_indices() -> None:
    view = ObsView(n_robots=3)  # self + 2 others
    assert view.total_dim == 4 + 7 * 3
    assert view.ball_slice == slice(0, 4)
    assert view.self_slice == slice(4, 11)
    assert view.other_slice(0) == slice(11, 18)
    assert view.other_slice(1) == slice(18, 25)


def test_obsview_other_slice_out_of_range() -> None:
    view = ObsView(n_robots=2)  # only 1 other
    with pytest.raises(IndexError):
        view.other_slice(1)
    with pytest.raises(IndexError):
        view.other_slice(-1)


def test_obsview_rejects_zero_robots() -> None:
    with pytest.raises(ValueError):
        ObsView(n_robots=0)


# ---------------------------------------------------------------------------
# build_observation — shape, normalization, clipping
# ---------------------------------------------------------------------------


def _zero_robot() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _zero_ball() -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def test_build_observation_shape_solo() -> None:
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=_zero_robot(),
    )
    assert obs.shape == (11,)
    assert obs.dtype == np.float32


def test_build_observation_shape_with_others() -> None:
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=_zero_robot(),
        others_states_5d=[_zero_robot(), _zero_robot()],
    )
    assert obs.shape == (4 + 7 * 3,)


def test_build_observation_zeros_at_origin() -> None:
    """Robot at origin, theta=0 → sin=0, cos=1, all positions and velocities 0."""
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=_zero_robot(),
    )
    view = ObsView(n_robots=1)
    np.testing.assert_allclose(view.self_(obs), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_build_observation_position_normalization() -> None:
    # Robot at the field corner → x_norm = +1, y_norm = +1
    state = np.array([FIELD_X_HALF, FIELD_Y_HALF, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
    )
    view = ObsView(1)
    assert view.self_px(obs) == pytest.approx(1.0)
    assert view.self_py(obs) == pytest.approx(1.0)


def test_build_observation_position_clipping() -> None:
    # Robot 2× past the field bound → clipped to ±1.
    state = np.array([2 * FIELD_X_HALF, -2 * FIELD_Y_HALF, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
    )
    view = ObsView(1)
    assert view.self_px(obs) == pytest.approx(1.0)
    assert view.self_py(obs) == pytest.approx(-1.0)


def test_build_observation_velocity_to_world_frame() -> None:
    # Robot at theta=π/2 with body-v=0.225 (V_MAX) → world dx=0, dy=+0.225.
    state = np.array([0.0, 0.0, np.pi / 2, V_MAX_DEFAULT, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
    )
    view = ObsView(1)
    assert view.self_sin_th(obs) == pytest.approx(1.0, abs=1e-6)
    assert view.self_cos_th(obs) == pytest.approx(0.0, abs=1e-6)
    # dx_world ≈ 0, dy_world ≈ +V_MAX → norm = +1
    assert view.self_dx(obs) == pytest.approx(0.0, abs=1e-6)
    assert view.self_dy(obs) == pytest.approx(1.0, abs=1e-6)


def test_build_observation_yaw_rate_normalization() -> None:
    state = np.array([0.0, 0.0, 0.0, 0.0, OMEGA_MAX_DEFAULT], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
    )
    assert ObsView(1).self_dth(obs) == pytest.approx(1.0)


def test_build_observation_ball_velocity_clipping() -> None:
    # Ball flying at 2× V_BALL_MAX → clipped to +1
    ball = np.array([0.0, 0.0, 2 * V_BALL_MAX, -2 * V_BALL_MAX], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=ball,
        self_state_5d=_zero_robot(),
    )
    view = ObsView(1)
    assert view.ball_vx(obs) == pytest.approx(1.0)
    assert view.ball_vy(obs) == pytest.approx(-1.0)


def test_build_observation_others_block_appears_after_self() -> None:
    """Others come after self; encoding identical to self_."""
    other = np.array([FIELD_X_HALF, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=_zero_robot(),
        others_states_5d=[other],
    )
    view = ObsView(n_robots=2)
    # px norm = +1 (at field edge), the rest zeros / cos=1
    assert view.other_px(obs, 0) == pytest.approx(1.0)
    assert view.other_cos_th(obs, 0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mirror semantics
# ---------------------------------------------------------------------------


def test_mirror_flips_x_position() -> None:
    state = np.array([0.10, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
        mirror=True,
    )
    view = ObsView(1)
    # x: +0.10 → -0.10 → norm = -0.10/0.375
    assert view.self_px(obs) == pytest.approx(-0.10 / FIELD_X_HALF)
    # y unchanged
    assert view.self_py(obs) == pytest.approx(0.05 / FIELD_Y_HALF)


def test_mirror_flips_cos_theta_keeps_sin_theta() -> None:
    # Robot at theta = π/4 (pointing into +x +y quadrant).
    # In mirrored view, robot should appear to face (-x, +y) quadrant ⟹
    # sin unchanged, cos sign-flipped.
    state = np.array([0.0, 0.0, np.pi / 4, 0.0, 0.0], dtype=np.float32)
    direct = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
        mirror=False,
    )
    mirrored = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
        mirror=True,
    )
    view = ObsView(1)
    assert view.self_sin_th(mirrored) == pytest.approx(view.self_sin_th(direct))
    assert view.self_cos_th(mirrored) == pytest.approx(-view.self_cos_th(direct))


def test_mirror_flips_yaw_rate() -> None:
    state = np.array([0.0, 0.0, 0.0, 0.0, +OMEGA_MAX_DEFAULT], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=state,
        mirror=True,
    )
    assert ObsView(1).self_dth(obs) == pytest.approx(-1.0)


def test_mirror_ball_velocity_x_flips() -> None:
    ball = np.array([0.10, 0.05, +1.0, -2.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=ball,
        self_state_5d=_zero_robot(),
        mirror=True,
    )
    view = ObsView(1)
    assert view.ball_px(obs) == pytest.approx(-0.10 / FIELD_X_HALF)
    assert view.ball_py(obs) == pytest.approx(0.05 / FIELD_Y_HALF)
    assert view.ball_vx(obs) == pytest.approx(-1.0 / V_BALL_MAX)
    assert view.ball_vy(obs) == pytest.approx(-2.0 / V_BALL_MAX)


# ---------------------------------------------------------------------------
# action_to_wheel_cmds — anti-windup, mirroring, boundary cases
# ---------------------------------------------------------------------------


def test_action_pure_forward() -> None:
    vL, vR = action_to_wheel_cmds(1.0, 0.0)
    assert vL == pytest.approx(V_MAX_DEFAULT)
    assert vR == pytest.approx(V_MAX_DEFAULT)


def test_action_pure_reverse() -> None:
    vL, vR = action_to_wheel_cmds(-1.0, 0.0)
    assert vL == pytest.approx(-V_MAX_DEFAULT)
    assert vR == pytest.approx(-V_MAX_DEFAULT)


def test_action_pure_ccw_spin() -> None:
    """Ω=+1 alone → wheels at ±v_max (right > left for CCW)."""
    vL, vR = action_to_wheel_cmds(0.0, 1.0)
    assert vL == pytest.approx(-V_MAX_DEFAULT)
    assert vR == pytest.approx(+V_MAX_DEFAULT)


def test_action_pure_cw_spin() -> None:
    vL, vR = action_to_wheel_cmds(0.0, -1.0)
    assert vL == pytest.approx(+V_MAX_DEFAULT)
    assert vR == pytest.approx(-V_MAX_DEFAULT)


def test_action_anti_windup_at_full_combined() -> None:
    """V=1, Ω=1: requested vR = 2 v_max. Anti-windup scales BOTH so max-abs = v_max,
    preserving the V/Ω ratio. Result: vL=0, vR=v_max."""
    vL, vR = action_to_wheel_cmds(1.0, 1.0)
    assert max(abs(vL), abs(vR)) == pytest.approx(V_MAX_DEFAULT)
    assert vL == pytest.approx(0.0, abs=1e-6)
    assert vR == pytest.approx(V_MAX_DEFAULT)


def test_action_anti_windup_preserves_ratio() -> None:
    """Half forward + full CCW: requested vL = 0.5·v_max - v_max = -0.5 v_max,
    vR = 0.5·v_max + v_max = 1.5 v_max. Ratio = -1/3 (vL/vR).
    After anti-windup: vR = v_max, vL = -v_max/3, ratio preserved."""
    vL, vR = action_to_wheel_cmds(0.5, 1.0)
    assert vR == pytest.approx(V_MAX_DEFAULT)
    assert vL == pytest.approx(-V_MAX_DEFAULT / 3.0)


def test_action_clips_input_to_unit_interval() -> None:
    vL, vR = action_to_wheel_cmds(2.0, 0.0)  # over-saturated input
    assert vL == pytest.approx(V_MAX_DEFAULT)
    assert vR == pytest.approx(V_MAX_DEFAULT)


def test_action_mirror_negates_omega() -> None:
    """Mirror flag should invert the yaw component but leave V untouched."""
    vL, vR = action_to_wheel_cmds(0.5, +0.5)
    vL_m, vR_m = action_to_wheel_cmds(0.5, +0.5, mirror=True)
    # Mirroring should produce the same as Ω=-0.5, V=0.5
    vL_eq, vR_eq = action_to_wheel_cmds(0.5, -0.5)
    assert vL_m == pytest.approx(vL_eq)
    assert vR_m == pytest.approx(vR_eq)
    # And different from the non-mirrored
    assert vL_m != pytest.approx(vL)


def test_action_zero_input_yields_zero_wheels() -> None:
    vL, vR = action_to_wheel_cmds(0.0, 0.0)
    assert vL == 0.0
    assert vR == 0.0


def test_action_custom_track_width() -> None:
    """OMEGA_MAX scales as 2 v_max / track_width — wider track means slower
    max yaw for the same Ω=1 input."""
    vL, vR = action_to_wheel_cmds(0.0, 1.0, track_width=0.10)
    # Pure spin still saturates wheel speeds; only the corresponding ω in rad/s differs.
    assert vL == pytest.approx(-V_MAX_DEFAULT)
    assert vR == pytest.approx(+V_MAX_DEFAULT)


# ---------------------------------------------------------------------------
# ObsView named accessors — convenience reads + recovered theta
# ---------------------------------------------------------------------------


def test_obsview_self_xy_returns_view() -> None:
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=np.array([FIELD_X_HALF, -FIELD_Y_HALF, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    view = ObsView(1)
    xy = view.self_xy(obs)
    assert xy.shape == (2,)
    np.testing.assert_allclose(xy, [1.0, -1.0])


def test_obsview_self_theta_round_trip() -> None:
    """self_theta uses atan2(sin, cos); should recover the original heading."""
    for theta in [0.0, np.pi / 4, np.pi / 2, -np.pi / 3, np.pi - 0.01]:
        state = np.array([0.0, 0.0, theta, 0.0, 0.0], dtype=np.float32)
        obs = build_observation(
            field_x_half=FIELD_X_HALF,
            field_y_half=FIELD_Y_HALF,
            ball_state=_zero_ball(),
            self_state_5d=state,
        )
        recovered = ObsView(1).self_theta(obs)
        assert recovered == pytest.approx(theta, abs=1e-5)


def test_obsview_other_indexing_with_two_others() -> None:
    a = np.array([+0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([-0.20, +FIELD_Y_HALF, np.pi / 2, 0.0, 0.0], dtype=np.float32)
    obs = build_observation(
        field_x_half=FIELD_X_HALF,
        field_y_half=FIELD_Y_HALF,
        ball_state=_zero_ball(),
        self_state_5d=_zero_robot(),
        others_states_5d=[a, b],
    )
    view = ObsView(n_robots=3)
    assert view.other_px(obs, 0) == pytest.approx(0.10 / FIELD_X_HALF)
    assert view.other_py(obs, 1) == pytest.approx(1.0)
    assert view.other_cos_th(obs, 1) == pytest.approx(0.0, abs=1e-6)
    assert view.other_sin_th(obs, 1) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Heading <-> sin/cos free helpers
# ---------------------------------------------------------------------------


def test_sincos_from_theta_known_values() -> None:
    s, c = sincos_from_theta(0.0)
    assert s == pytest.approx(0.0)
    assert c == pytest.approx(1.0)
    s, c = sincos_from_theta(np.pi / 2)
    assert s == pytest.approx(1.0)
    assert c == pytest.approx(0.0, abs=1e-15)


def test_theta_from_sincos_inverts_sincos_from_theta() -> None:
    for theta in [0.0, np.pi / 6, np.pi / 2, 2.5, -np.pi / 3, np.pi - 0.01, -np.pi + 0.01]:
        s, c = sincos_from_theta(theta)
        assert theta_from_sincos(s, c) == pytest.approx(theta, abs=1e-6)


def test_theta_from_sincos_returns_in_principal_range() -> None:
    """atan2 returns angles in (-π, π]; verify the boundary behavior."""
    # +π and -π map to the same direction; atan2 conventionally returns +π
    s, c = sincos_from_theta(np.pi)
    assert abs(theta_from_sincos(s, c)) == pytest.approx(np.pi, abs=1e-6)


def test_theta_helpers_match_obs_encoding() -> None:
    """The free helpers must agree with what build_observation / ObsView
    produce — they're the single source of truth for the encoding."""
    for theta in [0.0, 0.7, -1.2, np.pi / 3]:
        state = np.array([0.0, 0.0, theta, 0.0, 0.0], dtype=np.float32)
        obs = build_observation(
            field_x_half=FIELD_X_HALF,
            field_y_half=FIELD_Y_HALF,
            ball_state=_zero_ball(),
            self_state_5d=state,
        )
        view = ObsView(1)
        s_expected, c_expected = sincos_from_theta(theta)
        assert view.self_sin_th(obs) == pytest.approx(s_expected, abs=1e-6)
        assert view.self_cos_th(obs) == pytest.approx(c_expected, abs=1e-6)
        assert view.self_theta(obs) == pytest.approx(theta, abs=1e-6)


# ---------------------------------------------------------------------------
# View-vs-copy mutation semantics — explicitly verified
# ---------------------------------------------------------------------------


def test_obsview_scalar_read_does_not_alias() -> None:
    """Reading a scalar through `ball_px` returns an immutable Python float;
    rebinding the local can't possibly affect the source array."""
    obs = np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 7, dtype=np.float32)
    view = ObsView(1)
    val = view.ball_px(obs)
    val = 999.0  # rebind only — Python floats are immutable
    assert obs[0] == pytest.approx(0.1)


def test_obsview_block_read_writes_propagate() -> None:
    """Reading a block returns a numpy view; writes through it MUST mutate
    the source array. (Documented behavior — flagged in the docstring.)"""
    obs = np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 7, dtype=np.float32)
    view = ObsView(1)
    block = view.ball(obs)
    block[0] = 0.99
    assert obs[0] == pytest.approx(0.99)


def test_obsview_block_copy_isolates() -> None:
    """If the caller wants isolation, .copy() does it."""
    obs = np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 7, dtype=np.float32)
    view = ObsView(1)
    isolated = view.ball(obs).copy()
    isolated[0] = 0.99
    assert obs[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# ActionView
# ---------------------------------------------------------------------------


def test_actionview_total_dim() -> None:
    assert ActionView().total_dim == 2


def test_actionview_index_properties() -> None:
    av = ActionView()
    assert av.v_idx == 0
    assert av.omega_idx == 1


def test_actionview_field_reads() -> None:
    av = ActionView()
    a = np.array([0.7, -0.3], dtype=np.float32)
    assert av.v(a) == pytest.approx(0.7)
    assert av.omega(a) == pytest.approx(-0.3)
    assert av.as_tuple(a) == (pytest.approx(0.7), pytest.approx(-0.3))


def test_actionview_scalar_read_does_not_alias() -> None:
    a = np.array([0.7, -0.3], dtype=np.float32)
    av = ActionView()
    v = av.v(a)
    v = 999.0
    assert a[0] == pytest.approx(0.7)
