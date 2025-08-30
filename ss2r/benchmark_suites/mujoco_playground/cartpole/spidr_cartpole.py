from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper
from ml_collections import config_dict
from mujoco_playground._src.dm_control_suite import cartpole

from ss2r.benchmark_suites.wrappers import BraxDomainRandomizationVmapWrapper, _get_obs


class VisionSPiDRCartpole(Wrapper):
    def __init__(
        self,
        env,
        randomization_fn: Any,
        lambda_: float = 5e4,
        config: config_dict.ConfigDict = cartpole.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(env)
        config["vision"] = False
        base_cartpole = cartpole.Balance(True, False, config, config_overrides)
        self.perturbed_env = BraxDomainRandomizationVmapWrapper(
            base_cartpole, randomization_fn, augment_state=False
        )
        self.num_perturbed_envs = 8
        self.lambda_ = lambda_
        self.alpha = 0.0

    def reset(self, rng):
        state = self.env.reset(rng)
        disagreement = jnp.zeros_like(state.reward)
        state.info["disagreement"] = disagreement
        state.metrics["disagreement"] = disagreement
        return state

    def step(self, state, action):
        nstate = self.env.step(state, action)
        spidr_state = state.replace(
            obs=self.perturbed_env.env._get_obs(state.data, state.info)
        )
        v_state, v_action = self._tile(spidr_state), self._tile(action)
        perturbed_nstate = self.perturbed_env.step(v_state, v_action)
        next_obs = _get_obs(perturbed_nstate)
        disagreement = self._compute_disagreement(next_obs)
        nstate.info["disagreement"] = disagreement
        nstate.metrics["disagreement"] = disagreement
        return nstate

    def _compute_disagreement(self, next_obs: jax.Array) -> jax.Array:
        variance = jnp.nanvar(next_obs, axis=0).mean(-1)
        variance = jnp.where(jnp.isnan(variance), 0.0, variance)
        return jnp.clip(variance, a_max=1000.0) * self.lambda_ + self.alpha

    def _tile(self, tree):
        def tile(x):
            x = jnp.asarray(x)
            return jnp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim)

        return jax.tree_map(tile, tree)
