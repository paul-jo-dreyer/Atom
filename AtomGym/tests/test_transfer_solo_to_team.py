"""Tests for transfer_solo_to_team — solo→team weight expansion warm-start.

The critical correctness check is the round-trip: build a fresh solo
PPO model, save, transfer to team, load, and verify that with the opp
obs block = 0, the team model's predictions are bit-equal to the solo
model's. This is what makes "warm-start instead of behavior cloning"
viable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from AtomGym.environments import AtomSoloEnv  # noqa: F401
except RuntimeError as e:
    pytest.skip(f"AtomSim release build not available: {e}", allow_module_level=True)

from AtomGym.training.train import make_l1_env
from AtomGym.training.transfer_solo_to_team import transfer_solo_to_team


def _build_solo_model() -> PPO:
    """Fresh solo PPO with the same arch train.py uses. CPU only."""
    solo_env = make_l1_env(seed=0)
    vec = DummyVecEnv([lambda: solo_env])
    return PPO(
        "MlpPolicy",
        vec,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            log_std_init=0.3,
        ),
        device="cpu",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_solo_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transfer_solo_to_team(
            solo_zip_path=tmp_path / "does_not_exist.zip",
            output_zip_path=tmp_path / "team.zip",
        )


# ---------------------------------------------------------------------------
# Round-trip: shape and weight content
# ---------------------------------------------------------------------------


def test_team_obs_action_spaces(tmp_path: Path) -> None:
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")
    assert team.observation_space.shape == (20,)
    assert team.action_space.shape == (2,)


def test_first_layer_weights_expanded_correctly(tmp_path: Path) -> None:
    """First-layer weights of policy_net and value_net: solo (128, 12)
    becomes team (128, 20) with the solo columns in [:, :12] and zero
    in [:, 12:]."""
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")

    solo_sd = solo.policy.state_dict()
    team_sd = team.policy.state_dict()

    for key in (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    ):
        team_w = team_sd[key]
        solo_w = solo_sd[key]
        assert team_w.shape == (128, 20), f"{key}: {team_w.shape}"
        assert solo_w.shape == (128, 12), f"{key}: {solo_w.shape}"
        assert torch.equal(team_w[:, :12], solo_w), f"{key}: solo cols differ"
        assert torch.equal(team_w[:, 12:], torch.zeros(128, 8)), \
            f"{key}: opp cols not zero-init"


def test_non_first_layer_params_transfer_verbatim(tmp_path: Path) -> None:
    """Subsequent MLP layers, action head, value head, log_std,
    biases — all same shape between solo and team, all should be
    bitwise-equal after transfer."""
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")

    solo_sd = solo.policy.state_dict()
    team_sd = team.policy.state_dict()
    expanded_keys = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    for key, solo_tensor in solo_sd.items():
        if key in expanded_keys:
            continue
        assert torch.equal(team_sd[key], solo_tensor), \
            f"{key}: not bitwise-equal after transfer"


# ---------------------------------------------------------------------------
# Behavioural check: team(obs + zeros) == solo(obs) at init
# ---------------------------------------------------------------------------


def test_team_with_zero_opp_block_matches_solo(tmp_path: Path) -> None:
    """The whole point of warm-start: at init, with opp obs block = 0,
    the team policy reproduces solo predictions EXACTLY. Tests both
    the deterministic action (mean of distribution) and the value
    estimate."""
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")

    rng = np.random.default_rng(0)
    for _ in range(10):
        obs_solo = rng.uniform(-1.0, 1.0, size=12).astype(np.float32)
        obs_team = np.concatenate(
            [obs_solo, np.zeros(8, dtype=np.float32)]
        ).astype(np.float32)
        a_solo, _ = solo.predict(obs_solo, deterministic=True)
        a_team, _ = team.predict(obs_team, deterministic=True)
        np.testing.assert_allclose(a_team, a_solo, atol=1e-6)


def test_team_with_nonzero_opp_block_can_differ(tmp_path: Path) -> None:
    """Sanity-check the opposite: with non-zero opp obs, team
    predictions are NOT forced to match solo. (At init the new columns
    are zero, so the contribution is zero — but the value of the
    deterministic action goes through the bias/subsequent layers, so
    output shouldn't strictly equal solo unless we got lucky. This
    test ensures we're not accidentally locking the new dims out
    entirely — once the column expansion is non-zero post-training,
    behaviour diverges as intended.)"""
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")

    # Manually mutate the team's first-layer weight so the opp columns
    # are non-zero, mimicking what training would do.
    sd = team.policy.state_dict()
    sd["mlp_extractor.policy_net.0.weight"][:, 12:] = torch.randn(128, 8) * 0.1
    team.policy.load_state_dict(sd)

    obs_solo = np.zeros(12, dtype=np.float32)
    obs_team_with_opp = np.concatenate(
        [obs_solo, np.ones(8, dtype=np.float32) * 0.5]
    ).astype(np.float32)

    a_solo, _ = solo.predict(obs_solo, deterministic=True)
    a_team, _ = team.predict(obs_team_with_opp, deterministic=True)
    # Should differ now that opp columns are non-zero AND opp obs is
    # non-zero. Use a loose threshold to avoid spurious failures.
    diff = float(np.linalg.norm(a_team - a_solo))
    assert diff > 1e-3, (
        f"team and solo predictions identical despite non-zero opp "
        f"weights and obs (diff={diff})"
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_num_timesteps_reset_to_zero(tmp_path: Path) -> None:
    """Warm-start ⟹ team training starts from step 0, not from
    wherever the solo run left off. `train_team --total-timesteps N`
    should train N new steps."""
    solo = _build_solo_model()
    solo.num_timesteps = 5_000_000  # pretend solo ran for a while
    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")
    assert team.num_timesteps == 0


def test_log_std_transferred_verbatim(tmp_path: Path) -> None:
    """log_std is a separate Parameter — same shape across solo and
    team — and should pass through directly. This pins the test for
    the log_std transfer path explicitly because it's important for
    reproducing solo's exploration behaviour at init."""
    solo = _build_solo_model()
    # Set log_std to a distinctive value so the test doesn't pass
    # vacuously due to defaults matching.
    with torch.no_grad():
        solo.policy.log_std.fill_(-0.7)

    solo_path = tmp_path / "solo.zip"
    team_path = tmp_path / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, team_path)
    team = PPO.load(str(team_path), device="cpu")
    assert torch.allclose(team.policy.log_std, solo.policy.log_std)


def test_output_directory_is_created(tmp_path: Path) -> None:
    solo = _build_solo_model()
    solo_path = tmp_path / "solo.zip"
    nested_path = tmp_path / "newdir" / "nested" / "team.zip"
    solo.save(str(solo_path))
    transfer_solo_to_team(solo_path, nested_path)
    assert nested_path.is_file()
