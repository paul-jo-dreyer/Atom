"""Generic obs-dim extension: load a PPO checkpoint with the OLD obs
space, build a fresh model with the NEW obs space, and zero-pad the
first-layer weights so the new dims start as no-ops.

Use this when the observation schema gains new dimensions (e.g. the
`time_in_box` dim added in 2026-05). At init the new obs columns are
zero-init'd in the policy/value first-layer weight matrix; if the env
also writes 0 to those obs slots (which `build_observation` does by
default), the extended model produces identical outputs to the old
model. Gradient descent then learns to use the new columns as the env
starts feeding non-zero values in (e.g. when the goalie-box rule is
turned on and the env starts ticking the timer).

The mechanism is identical to `transfer_solo_to_team`: same shape-
expand-and-zero-pad logic on `mlp_extractor.policy_net.0.weight` and
`mlp_extractor.value_net.0.weight`. The only difference is the env
type: this script builds an env of the SAME type as the source
checkpoint (solo→solo or team→team), just with the new obs space.

Usage
-----

    .venv/bin/python -m AtomGym.training.transfer_extend_obs \\
        --checkpoint training_runs/old_run/final.zip \\
        --output training_runs/new_run/init.zip \\
        --env-type team
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def _build_solo_vec_env() -> DummyVecEnv:
    from AtomGym.environments import AtomSoloEnv
    return DummyVecEnv([lambda: AtomSoloEnv()])


def _build_team_vec_env() -> DummyVecEnv:
    from AtomGym.environments import AtomTeamEnv
    return DummyVecEnv([lambda: AtomTeamEnv()])


def transfer_extend_obs(
    src_zip_path: Path,
    output_zip_path: Path,
    env_type: str,
    log_std_init: float = 0.3,
) -> None:
    """Load `src_zip_path`, expand into a same-env-type model with the
    current obs space, save to `output_zip_path`."""
    if not src_zip_path.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {src_zip_path}")

    print(f"[extend-obs] loading source: {src_zip_path}")
    src_model = PPO.load(str(src_zip_path), device="cpu")
    src_obs_dim = int(np.prod(src_model.observation_space.shape))
    src_act_dim = int(np.prod(src_model.action_space.shape))
    print(f"[extend-obs] source obs_dim={src_obs_dim}, act_dim={src_act_dim}")

    if env_type == "solo":
        new_vec = _build_solo_vec_env()
    elif env_type == "team":
        new_vec = _build_team_vec_env()
    else:
        raise ValueError(f"env_type must be 'solo' or 'team', got {env_type!r}")

    new_obs_dim = int(np.prod(new_vec.observation_space.shape))
    new_act_dim = int(np.prod(new_vec.action_space.shape))
    print(f"[extend-obs] new {env_type} obs_dim={new_obs_dim}, act_dim={new_act_dim}")

    if src_act_dim != new_act_dim:
        raise ValueError(
            f"action dim mismatch: src {src_act_dim} vs new {new_act_dim}"
        )
    if src_obs_dim > new_obs_dim:
        raise ValueError(
            f"src obs_dim ({src_obs_dim}) > new obs_dim ({new_obs_dim}). "
            f"This script extends, it doesn't truncate."
        )
    if src_obs_dim == new_obs_dim:
        print("[extend-obs] obs_dim unchanged — re-saving the source as is")
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        src_model.save(str(output_zip_path))
        return

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        log_std_init=float(log_std_init),
    )
    new_model = PPO(
        "MlpPolicy",
        new_vec,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )

    src_sd = src_model.policy.state_dict()
    new_sd = new_model.policy.state_dict()
    merged: dict[str, torch.Tensor] = {}
    n_copied = 0
    n_expanded = 0
    for key, new_tensor in new_sd.items():
        if key not in src_sd:
            raise ValueError(
                f"key {key!r} present in new policy but not in source. "
                f"Architecture mismatch."
            )
        src_tensor = src_sd[key]
        if src_tensor.shape == new_tensor.shape:
            merged[key] = src_tensor.clone()
            n_copied += 1
            continue
        if not (
            src_tensor.dim() == 2
            and new_tensor.dim() == 2
            and src_tensor.shape[0] == new_tensor.shape[0]
            and src_tensor.shape[1] < new_tensor.shape[1]
        ):
            raise ValueError(
                f"unsupported shape transition for {key!r}: "
                f"src {tuple(src_tensor.shape)} → new {tuple(new_tensor.shape)}"
            )
        expanded = torch.zeros_like(new_tensor)
        expanded[:, : src_tensor.shape[1]] = src_tensor
        merged[key] = expanded
        n_expanded += 1
        print(
            f"[extend-obs]   expanded {key}: "
            f"{tuple(src_tensor.shape)} → {tuple(new_tensor.shape)}"
        )

    print(f"[extend-obs] {n_copied} keys copied, {n_expanded} keys expanded")
    new_model.policy.load_state_dict(merged)
    new_model.num_timesteps = 0
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(str(output_zip_path))
    print(f"[extend-obs] saved: {output_zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--env-type",
        choices=["solo", "team"],
        required=True,
        help="Target env type — must match the source checkpoint's env type. "
        "(If you want to migrate solo → team, use transfer_solo_to_team instead.)",
    )
    parser.add_argument("--log-std-init", type=float, default=0.3)
    args = parser.parse_args()
    transfer_extend_obs(
        src_zip_path=args.checkpoint,
        output_zip_path=args.output,
        env_type=args.env_type,
        log_std_init=args.log_std_init,
    )


if __name__ == "__main__":
    main()
