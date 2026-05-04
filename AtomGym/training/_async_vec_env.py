"""SB3-compatible adapter around `gymnasium.vector.AsyncVectorEnv`
with `shared_memory=True`.

Why this exists
---------------
SB3's built-in `SubprocVecEnv` uses pickle-over-pipe IPC for every step's
observation array. That has a fixed per-step overhead — when the env
itself is fast (sim_py is C++ Box2D, sub-millisecond per step), IPC ends
up being a meaningful fraction of total step time. Symptom: rollout
phase shows lower-than-expected per-core utilization because workers
spend significant time blocked on pipe send/recv.

Gymnasium's `AsyncVectorEnv(env_fns, shared_memory=True)` pre-allocates
obs/action buffers as `multiprocessing.shared_memory` blocks. Workers
write directly into shared pages without pickle. Same multiprocessing
semantics as SB3's SubprocVecEnv, but with per-step latency dominated
by sim time rather than IPC.

This module bridges gymnasium's vector API (which differs from SB3's
in several places) into something SB3's `PPO` can drop in unchanged:

  * gymnasium reset returns `(obs, info)`; SB3 expects `obs`.
  * gymnasium step returns `(obs, rew, term, trunc, info)`; SB3 expects
    `(obs, rew, done, info_list)`. We OR terminated|truncated for done.
  * gymnasium info is dict-of-arrays with `_key` masks per gymnasium 1.x
    spec; SB3 wants list-of-dicts. We translate per-env.
  * gymnasium auto-resets on episode end and stashes the pre-reset obs
    under `info['final_observation']` with mask `info['_final_observation']`;
    SB3 reads `info[i]['terminal_observation']` for GAE bootstrap. We
    rename per-env.
  * gymnasium uses `.call()` for arbitrary method invocation across
    workers; SB3 uses `env_method()`. We wrap.

What this does NOT do
---------------------
- `set_attr` / `get_attr` are minimally implemented (delegated to
  gymnasium's equivalents) but not extensively tested. PPO's training
  loop doesn't exercise them in our setup; the self-play stack uses
  `env_method('update_opponent_pool', pool)` which IS supported.
- `env_is_wrapped` returns `False` for everything. SB3 uses this for
  `Monitor` detection — if Monitor wrapping needs to be detected at
  runtime, we'd need to plumb that through. For us it isn't.
- Render / `get_images` are not supported. We don't render through the
  vec env in training (GIF callback creates its own envs).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence, Type

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)


def _unvectorize_info_for_env(vec_info: dict[str, Any], i: int) -> dict[str, Any]:
    """Extract env `i`'s per-env info dict from a gymnasium vectorized
    info dict. Recursive — gymnasium nests vectorized dicts inside
    vectorized dicts (e.g. `info['episode'] = {'r': array(num_envs),
    '_r': mask, 'l': array(num_envs), '_l': mask, ...}`)."""
    out: dict[str, Any] = {}
    for key, values in vec_info.items():
        if key.startswith("_"):
            continue  # mask key, paired with the unmasked one we'll handle
        mask_key = f"_{key}"
        mask = vec_info.get(mask_key)
        # Skip if this env didn't populate this key this step.
        if mask is not None and not mask[i]:
            continue
        # Rename gymnasium's `final_observation` to SB3's
        # `terminal_observation` for GAE bootstrap.
        out_key = "terminal_observation" if key == "final_observation" else key
        if isinstance(values, dict):
            # Vectorized dict — recurse to extract env i's per-env view.
            out[out_key] = _unvectorize_info_for_env(values, i)
        else:
            try:
                out[out_key] = values[i]
            except (IndexError, TypeError, KeyError):
                # Non-vectorized scalar/object; assign as-is.
                out[out_key] = values
    return out


def _gymnasium_info_to_sb3_list(
    info_dict: dict[str, Any], num_envs: int
) -> list[dict[str, Any]]:
    """Translate gymnasium 1.x's dict-of-arrays info (with `_key` masks
    and recursively-nested dicts) into SB3's list-of-dicts.

    SB3 reads `info[i]['episode']` for rollout/ep_rew_mean logging and
    `info[i]['terminal_observation']` for GAE bootstrap on episode end.
    Both need the un-vectorized form."""
    return [_unvectorize_info_for_env(info_dict, i) for i in range(num_envs)]


def _normalize_indices(
    indices: None | int | Iterable[int], num_envs: int
) -> list[int]:
    if indices is None:
        return list(range(num_envs))
    if isinstance(indices, int):
        return [indices]
    return list(indices)


class GymAsyncVecEnv(VecEnv):
    """SB3 `VecEnv` over `gymnasium.vector.AsyncVectorEnv`. Shared memory
    by default — that's the whole point of the adapter."""

    def __init__(
        self,
        env_fns: Sequence,
        shared_memory: bool = True,
    ) -> None:
        if len(env_fns) < 1:
            raise ValueError("env_fns must contain at least one factory")

        # Probe spaces from a one-shot env. Cheap; this is what
        # gymnasium does internally too.
        probe = env_fns[0]()
        observation_space = probe.observation_space
        action_space = probe.action_space
        probe.close()

        super().__init__(len(env_fns), observation_space, action_space)
        self._async_env = AsyncVectorEnv(
            list(env_fns), shared_memory=shared_memory
        )
        self._actions: np.ndarray | None = None
        self._reset_seeds: int | list[int] | None = None

    # ---- core step / reset ------------------------------------------

    def reset(self) -> VecEnvObs:
        obs, _info = self._async_env.reset(seed=self._reset_seeds)
        # Use seeds only on the first reset; subsequent resets should
        # NOT replay the same trajectories.
        self._reset_seeds = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions
        self._async_env.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terminated, truncated, info = self._async_env.step_wait()
        dones = np.logical_or(terminated, truncated)
        info_list = _gymnasium_info_to_sb3_list(info, self.num_envs)
        return obs, rewards.astype(np.float32), dones, info_list

    def close(self) -> None:
        self._async_env.close()

    # ---- attribute / method access across workers --------------------

    def get_attr(
        self, attr_name: str, indices: None | int | Iterable[int] = None
    ) -> list[Any]:
        idx = _normalize_indices(indices, self.num_envs)
        # Gymnasium's get_attr returns a tuple over ALL envs. Subset.
        all_vals = self._async_env.get_attr(attr_name)
        return [all_vals[i] for i in idx]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: None | int | Iterable[int] = None,
    ) -> None:
        idx = _normalize_indices(indices, self.num_envs)
        if len(idx) != self.num_envs:
            # Gymnasium AsyncVectorEnv.set_attr broadcasts to all envs.
            # Per-env subset isn't supported without re-reading + writing
            # back, which we don't need for our workload.
            raise NotImplementedError(
                "Per-env-subset set_attr is not supported by GymAsyncVecEnv"
            )
        # Gymnasium expects a sequence of values, one per env.
        self._async_env.set_attr(attr_name, [value] * self.num_envs)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: None | int | Iterable[int] = None,
        **method_kwargs,
    ) -> list[Any]:
        idx = _normalize_indices(indices, self.num_envs)
        # Gymnasium broadcasts the call across ALL envs and returns a
        # tuple of results. Subset.
        results = self._async_env.call(
            method_name, *method_args, **method_kwargs
        )
        return [results[i] for i in idx]

    def env_is_wrapped(
        self,
        wrapper_class: Type[gym.Wrapper],
        indices: None | int | Iterable[int] = None,
    ) -> list[bool]:
        # SB3 uses this primarily to detect Monitor for bookkeeping.
        # Gymnasium's AsyncVectorEnv doesn't expose a clean way to
        # introspect each worker's wrapper stack without an env_method
        # round-trip. Returning False is conservative — SB3 falls back
        # to its own bookkeeping path when this returns False.
        idx = _normalize_indices(indices, self.num_envs)
        return [False] * len(idx)

    # ---- seeding / rendering -----------------------------------------

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        # Defer to gymnasium: pass via the next reset() call.
        if seed is None:
            self._reset_seeds = None
            return [None] * self.num_envs
        # Distinct seeds per env, deterministic from the base seed.
        seeds = [seed + i for i in range(self.num_envs)]
        self._reset_seeds = seeds
        return seeds

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [None] * self.num_envs
