"""Internal helpers for opponent-side CPU shadow policies.

Both `OpponentRunner` (snapshot-pool-driven) and `ReferenceOpponent`
(single fixed-snapshot benchmark) construct a CPU `ActorCriticPolicy`
that mirrors the learner's architecture, then load state_dicts into
it for inference. This module owns that build + predict pair so the
two consumers can't drift on arch matching.

CPU is intentional: keeping torch+CUDA off SubprocVec workers avoids a
multi-GB memory hit per worker (this was the OOM trigger we hit
earlier). Inference for a 128×128 MLP on 18-d obs is microseconds on
CPU — comfortably faster than the env step it serves.

Numpy ↔ tensor state-dict conversion
------------------------------------
Pool snapshots are stored as **numpy arrays**, not torch tensors. The
reason is `torch`'s multiprocessing pickling: when a torch tensor is
shipped through a `multiprocessing.Pipe`, torch's `file_descriptor`
sharing strategy creates an OS file descriptor for each tensor. Send
~13 tensors × ~20 pool entries × N workers across many sync cycles
and you blow past the default `ulimit -n = 1024` and crash with
`Errno 24`. Converting to numpy at the producer side and back to
tensors at the consumer side sidesteps this — numpy pickles via
regular bytes, no FDs in the pipeline. The cost is one extra memcpy
per state-dict transfer (~80 KB for our model), which is negligible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy


def build_shadow_policy(
    observation_space: spaces.Box,
    action_space: spaces.Box,
    policy_kwargs: dict[str, Any] | None = None,
) -> ActorCriticPolicy:
    """Build a CPU `ActorCriticPolicy` configured to match the learner's
    architecture, in eval mode (no dropout/BN side-effects, no grad).

    `lr_schedule` is a required positional arg on the SB3 ctor but is
    unused here — this module never trains. We pass a constant-zero
    schedule.
    """
    policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 0.0,
        **(policy_kwargs or {}),
    ).to("cpu")
    policy.set_training_mode(False)
    return policy


def shadow_predict(
    policy: ActorCriticPolicy,
    obs: np.ndarray,
    *,
    deterministic: bool = False,
) -> np.ndarray:
    """No-grad CPU forward → numpy action of shape `(action_dim,)`.

    Stochastic by default — matches the convention used during PPO
    rollout collection. Deterministic mode returns the action
    distribution mean (no RNG involvement)."""
    with torch.no_grad():
        action, _ = policy.predict(obs, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def state_dict_to_numpy(state_dict: dict) -> dict:
    """Convert a torch state_dict (or already-numpy dict) into a dict
    of independent numpy arrays. Idempotent.

    Used at the producer side (PoolSyncCallback) before adding to the
    pool — keeps torch tensors out of the SubprocVec pickling chain
    so we don't leak file descriptors. The `.copy()` ensures the
    returned arrays don't share memory with the source tensor's
    storage (so the source tensor can be GC'd safely).
    """
    out: dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy().copy()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            # Non-tensor / non-ndarray values (rare for policy
            # state_dicts) — pass through unchanged.
            out[k] = v
    return out


def state_dict_to_tensors(state_dict: dict) -> dict:
    """Inverse of `state_dict_to_numpy`. Converts numpy arrays in a
    pool snapshot's state_dict back to torch tensors for
    `module.load_state_dict()`. Tensors share memory with the source
    numpy arrays (cheap), but `load_state_dict` copies into the
    module's parameters internally, so the temporary tensors can be
    discarded safely after."""
    out: dict = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            out[k] = v
        else:
            out[k] = v
    return out
