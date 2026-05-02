"""Warm-start: transfer a solo PPO checkpoint into a team-shaped one.

The solo policy lives in an 11-d obs space (`[ball(4) | self(7)]`); the
team policy lives in an 18-d obs space (`[ball(4) | self(7) | opp(7)]`).
The first 11 dims of the team obs correspond *exactly* to the solo obs
layout — same per-block ordering, same normalisation. So the warm-start
is just a single column expansion on each MLP branch's first layer:

  * `mlp_extractor.policy_net.0.weight`: copy the solo (128, 11) into
    columns 0..10 of the new (128, 18); zero-init columns 11..17.
  * `mlp_extractor.value_net.0.weight`: same.
  * Every other parameter (subsequent layers, action head, value head,
    log_std, biases) has identical shape and transfers verbatim.

At init, with the opp obs block = 0, the team policy produces
*identical* outputs to the solo policy — verified by the round-trip
test in test_transfer_solo_to_team.py. Gradient descent then learns to
use the new columns as opponent information becomes informative.

The output `.zip` is a fresh PPO checkpoint:
  * `num_timesteps` is reset to 0 (this is a warm-start, not a
    continuation — `train_team --total-timesteps N` will train N steps).
  * Optimizer state is fresh (Adam moments tied to old shapes wouldn't
    be valid for the expanded weights).
  * Saved with the team-shaped vec env, so `train_team --resume <path>`
    loads it cleanly.

Usage
-----

    .venv/bin/python -m AtomGym.training.transfer_solo_to_team \
        --solo-checkpoint training_runs/l1_baseline/final.zip \
        --output training_runs/l2_baseline/team_init.zip \
        --log-std-init 0.3

Then start team training with:

    .venv/bin/python -m AtomGym.training.train_team \
        --run-name l2_baseline \
        --resume training_runs/l2_baseline/team_init.zip \
        ... (other flags) ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Make `AtomGym` importable regardless of how this script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def _build_team_vec_env() -> DummyVecEnv:
    """Construct a single AtomTeamEnv inside a DummyVecEnv. Used only
    for shape resolution + PPO save format — never stepped here."""
    from AtomGym.training.train_team import make_l2_env  # local: AtomSim is heavy
    return DummyVecEnv([lambda: make_l2_env()])


def transfer_solo_to_team(
    solo_zip_path: Path,
    output_zip_path: Path,
    log_std_init: float = 0.3,
) -> None:
    """Load `solo_zip_path`, expand into a team-shaped PPO model with
    weight transfer, save to `output_zip_path`.

    `log_std_init` is the initial log-std for the team model's action
    distribution. The solo model's `log_std` parameter is transferred
    directly (same shape), which OVERWRITES this default — use the
    same value as the solo run to avoid surprise. The kwarg is here
    in case you want a different starting std for team training; pass
    it through to PPO's policy_kwargs and it'll be replaced on transfer
    anyway, so the value matters only as a placeholder.
    """
    if not solo_zip_path.is_file():
        raise FileNotFoundError(f"solo checkpoint not found: {solo_zip_path}")

    # 1. Load the solo model. `device='cpu'` keeps the transfer self-
    # contained — no CUDA needed for a one-shot weight ops script.
    print(f"[transfer] loading solo model: {solo_zip_path}")
    solo_model = PPO.load(str(solo_zip_path), device="cpu")
    solo_obs_dim = int(np.prod(solo_model.observation_space.shape))
    solo_act_dim = int(np.prod(solo_model.action_space.shape))
    print(f"[transfer] solo obs_dim={solo_obs_dim}, act_dim={solo_act_dim}")

    # 2. Build a fresh team model with the team-shaped vec env.
    print("[transfer] constructing fresh team model")
    team_vec_env = _build_team_vec_env()
    team_obs_dim = int(np.prod(team_vec_env.observation_space.shape))
    team_act_dim = int(np.prod(team_vec_env.action_space.shape))
    print(f"[transfer] team obs_dim={team_obs_dim}, act_dim={team_act_dim}")

    if solo_act_dim != team_act_dim:
        raise ValueError(
            f"action_dim mismatch: solo {solo_act_dim} vs team {team_act_dim}. "
            f"Weight transfer assumes same action space."
        )
    if solo_obs_dim > team_obs_dim:
        raise ValueError(
            f"solo obs_dim ({solo_obs_dim}) > team obs_dim ({team_obs_dim}). "
            f"Cannot warm-start a smaller obs space from a larger one."
        )

    # `policy_kwargs` must match what `train_team.py` uses for the
    # learner (otherwise loading via `PPO.load` later would fail on
    # arch mismatch). Hardcoded here to keep it simple — if you change
    # the team learner arch in train_team.py, change it here too.
    policy_kwargs: dict[str, Any] = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=float(log_std_init),
    )
    team_model = PPO(
        "MlpPolicy",
        team_vec_env,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )

    # 3. Transfer state_dicts.
    solo_sd = solo_model.policy.state_dict()
    team_sd = team_model.policy.state_dict()
    new_sd: dict[str, torch.Tensor] = {}
    n_copied = 0
    n_expanded = 0
    for key, team_tensor in team_sd.items():
        if key not in solo_sd:
            raise ValueError(
                f"key {key!r} present in team policy but not in solo. "
                f"Architecture mismatch — both should have the same MLP "
                f"layout. Did you change net_arch in train_team.py?"
            )
        solo_tensor = solo_sd[key]
        if solo_tensor.shape == team_tensor.shape:
            new_sd[key] = solo_tensor.clone()
            n_copied += 1
            continue
        # Mismatched shape — only expected for first-layer weights of
        # the policy and value MLPs, where input dim differs.
        if not (
            team_tensor.dim() == 2
            and solo_tensor.dim() == 2
            and solo_tensor.shape[0] == team_tensor.shape[0]
            and solo_tensor.shape[1] < team_tensor.shape[1]
        ):
            raise ValueError(
                f"unsupported shape transition for {key!r}: "
                f"solo {tuple(solo_tensor.shape)} → team {tuple(team_tensor.shape)}. "
                f"Expected (out, in_solo) → (out, in_team) with in_solo < in_team."
            )
        # Zero-init then copy the solo columns into the leading slice.
        # Behaviour at init: if the trailing (opp) obs columns are zero,
        # the team policy reproduces solo predictions exactly.
        expanded = torch.zeros_like(team_tensor)
        expanded[:, : solo_tensor.shape[1]] = solo_tensor
        new_sd[key] = expanded
        n_expanded += 1
        print(
            f"[transfer]   expanded {key}: "
            f"{tuple(solo_tensor.shape)} → {tuple(team_tensor.shape)}"
        )

    print(f"[transfer] {n_copied} keys copied, {n_expanded} keys expanded")

    # 4. Load the merged state_dict into the team model.
    team_model.policy.load_state_dict(new_sd)

    # 5. Reset num_timesteps so the warm-start is treated as iteration=0.
    team_model.num_timesteps = 0

    # 6. Save.
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    team_model.save(str(output_zip_path))
    print(f"[transfer] saved team checkpoint: {output_zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start: transfer a solo PPO checkpoint into a team-shaped one."
    )
    parser.add_argument(
        "--solo-checkpoint",
        type=Path,
        required=True,
        help="Path to a solo PPO .zip (e.g. training_runs/l1_baseline/final.zip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the team-shaped .zip "
        "(e.g. training_runs/l2_baseline/team_init.zip).",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=0.3,
        help="Initial log-std placeholder for the team model construction. "
        "Replaced by the solo model's log_std on transfer (same shape).",
    )
    args = parser.parse_args()
    transfer_solo_to_team(
        solo_zip_path=args.solo_checkpoint,
        output_zip_path=args.output,
        log_std_init=args.log_std_init,
    )


if __name__ == "__main__":
    main()
