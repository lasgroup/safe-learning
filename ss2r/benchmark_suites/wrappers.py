from typing import Callable, Mapping, Optional, Tuple

import jax
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training
from jax import numpy as jp
from mujoco_playground import State as MjxState


class DomainRandomizationVmapBase(Wrapper):
    """Base class for domain randomization wrappers."""

    def __init__(self, env, randomization_fn, *, augment_state=True):
        super().__init__(env)
        self.augment_state = augment_state
        (
            self._randomized_models,
            self._in_axes,
            self.domain_parameters,
        ) = self._init_randomization(randomization_fn)
        dummy = self.env.reset(jax.random.PRNGKey(0))
        self.strip_privileged_state = isinstance(dummy.obs, jax.Array)

    def _init_randomization(self, randomization_fn):
        """To be implemented by subclasses to handle model-specific randomization."""
        raise NotImplementedError

    def _env_fn(self, model):
        """To be implemented by subclasses to return an environment with the given model."""
        raise NotImplementedError

    def reset(self, rng: jax.Array):
        def reset_fn(model, rng):
            env = self._env_fn(model)
            return env.reset(rng)

        state = jax.vmap(reset_fn, in_axes=[self._in_axes, 0])(
            self._randomized_models, rng
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def step(self, state, action: jax.Array):
        def step_fn(model, s, a):
            env = self._env_fn(model)
            return env.step(s, a)

        if self.augment_state and self.strip_privileged_state:
            state = state.replace(obs=state.obs["state"])

        state = jax.vmap(step_fn, in_axes=[self._in_axes, 0, 0])(
            self._randomized_models, state, action
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def _add_privileged_state(self, state):
        """Adds privileged state to the observation if augmentation is enabled."""
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs={
                    "state": state.obs,
                    "privileged_state": jp.concatenate(
                        [state.obs, self.domain_parameters], -1
                    ),
                }
            )
        else:
            state = state.replace(
                obs={
                    "state": state.obs["state"],
                    "privileged_state": jp.concatenate(
                        [state.obs["privileged_state"], self.domain_parameters], -1
                    ),
                }
            )
        return state

    @property
    def observation_size(self):
        """Compute observation size based on the augmentation setting."""
        if not self.augment_state:
            return self.env.observation_size

        if isinstance(self.env.observation_size, int):
            return {
                "state": (self.env.observation_size,),
                "privileged_state": (
                    self.env.observation_size + self.domain_parameters.shape[1],
                ),
            }
        else:
            return {
                "state": (self.env.observation_size["state"],),
                "privileged_state": (
                    self.env.observation_size["privileged_state"]
                    + self.domain_parameters.shape[1],
                ),
            }


class DomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.sys)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped.sys = model
        return env


class CostEpisodeWrapper(brax_training.EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            maybe_cost = nstate.info.get("cost", None)
            maybe_eval_reward = nstate.info.get("eval_reward", None)
            return nstate, (nstate.reward, maybe_cost, maybe_eval_reward)

        state, (rewards, maybe_costs, maybe_eval_rewards) = jax.lax.scan(
            f, state, (), self.action_repeat
        )
        state = state.replace(reward=jp.sum(rewards, axis=0))
        if maybe_costs is not None:
            state.info["cost"] = jp.sum(maybe_costs, axis=0)
        if maybe_eval_rewards is not None:
            state.info["eval_reward"] = jp.sum(maybe_eval_rewards, axis=0)
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)


class NonEpisodicWrapper(Wrapper):
    def __init__(self, env: Env, action_repeat: int):
        super().__init__(env)
        self.action_repeat = action_repeat

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["average_reward"] = jp.zeros(rng.shape[:-1])
        metrics = {
            "average_reward": state.info["average_reward"],
        }
        state.info["truncation"] = jp.zeros(rng.shape[:-1])
        state.metrics.update(metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            maybe_cost = nstate.info.get("cost", None)
            maybe_eval_reward = nstate.info.get("eval_reward", None)
            return nstate, (nstate.reward, maybe_cost, maybe_eval_reward)

        state, (rewards, maybe_costs, maybe_eval_rewards) = jax.lax.scan(
            f, state, (), self.action_repeat
        )
        sum_rewards = jp.sum(rewards, axis=0)
        state = state.replace(reward=sum_rewards, done=jp.zeros_like(state.done))
        if maybe_costs is not None:
            state.info["cost"] = jp.sum(maybe_costs, axis=0)
        if maybe_eval_rewards is not None:
            state.info["eval_reward"] = jp.sum(maybe_eval_rewards, axis=0)
        steps = state.info["steps"] + self.action_repeat
        sum_rewards /= self.action_repeat
        average_reward = (
            state.info["average_reward"]
            + (sum_rewards - state.info["average_reward"]) * self.action_repeat / steps
        )
        state.info["steps"] = steps
        state.info["average_reward"] = average_reward
        state.metrics["average_reward"] = average_reward
        return state


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System, jax.Array]]
    ] = None,
    hard_resets: bool = False,
    *,
    augment_state: bool = True,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(
            env, randomization_fn, augment_state=augment_state
        )
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    if hard_resets:
        env = HardAutoResetWrapper(env)
    else:
        env = brax_training.AutoResetWrapper(env)
    return env


def _get_obs(state):
    if isinstance(state.obs, jax.Array):
        return state.obs
    else:
        assert isinstance(state.obs, Mapping)
        return state.obs["state"]


class SPiDR(Wrapper):
    def __init__(self, env, randomzation_fn, num_perturbed_envs, lambda_, alpha):
        super().__init__(env)
        if hasattr(env, "sys"):
            self.perturbed_env = DomainRandomizationVmapWrapper(
                env, randomzation_fn, augment_state=False
            )
        elif hasattr(env, "mjx_model"):
            self.perturbed_env = BraxDomainRandomizationVmapWrapper(
                env, randomzation_fn, augment_state=False
            )
        else:
            raise ValueError("Should be either mujoco playground or brax env")
        self.num_perturbed_envs = num_perturbed_envs
        self.lambda_ = lambda_
        self.alpha = alpha

    def reset(self, rng: jax.Array) -> State:
        # No need to randomize the initial state. Otherwise, even without
        # domain randomization, the initial states will be different, having
        # a non-zero disagreement.
        state = self.env.reset(rng)
        cost = jp.zeros_like(state.reward)
        state.info["state_propagation"] = {}
        state.info["state_propagation"]["next_obs"] = self._tile(_get_obs(state))
        state.info["state_propagation"]["cost"] = self._tile(cost)
        disagreement = self._compute_disagreement(
            state.info["state_propagation"]["next_obs"]
        )
        state.info["disagreement"] = disagreement
        state.metrics["disagreement"] = disagreement
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        v_state, v_action = self._tile(state), self._tile(action)
        perturbed_nstate = self.perturbed_env.step(v_state, v_action)
        next_obs = _get_obs(perturbed_nstate)
        disagreement = self._compute_disagreement(next_obs)
        nstate.info["state_propagation"]["next_obs"] = next_obs
        nstate.info["state_propagation"]["cost"] = perturbed_nstate.info.get(
            "cost", jp.zeros_like(perturbed_nstate.reward)
        )
        nstate.info["disagreement"] = disagreement
        nstate.metrics["disagreement"] = disagreement
        return nstate

    def _compute_disagreement(self, next_obs: jax.Array) -> jax.Array:
        variance = jp.nanvar(next_obs, axis=0).mean(-1)
        variance = jp.where(jp.isnan(variance), 0.0, variance)
        return jp.clip(variance, a_max=1000.0) * self.lambda_ + self.alpha

    def _tile(self, tree):
        def tile(x):
            x = jp.asarray(x)
            return jp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim)

        return jax.tree_map(tile, tree)


class BraxDomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.mjx_model)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped._mjx_model = model
        return env


class HardAutoResetWrapper(Wrapper):
    """Automatically reset Brax envs that are done.

    Resample only when >=1 environment is actually done. Still resamples for all
    """

    def reset(self, rng: jax.Array) -> State | MjxState:
        rng, sample_rng = jax.vmap(jax.random.split, out_axes=1)(rng)
        state = self.env.reset(sample_rng)
        state.info["reset_rng"] = rng
        return state

    def step(self, state: State | MjxState, action: jax.Array) -> State | MjxState:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        if hasattr(state, "pipeline_state"):
            data_name = "pipeline_state"
        elif hasattr(state, "data"):
            data_name = "data"
        else:
            raise NotImplementedError

        def safe_reset(rng):
            nstate = self.reset(rng)
            return nstate.obs, getattr(state, data_name)

        obs, new_data = jax.lax.cond(
            state.done.any(),
            safe_reset,
            lambda rng: (state.obs, getattr(state, data_name)),
            state.info["reset_rng"],
        )
        return state.replace(**{data_name: new_data, "obs": obs})


class Saute(Wrapper):
    def __init__(
        self, env, discounting, budget, penalty, terminate, termination_probability
    ):
        super().__init__(env)
        # Assumes that this is the budget for the undiscounted
        # episode.
        self.budget = budget
        self.discounting = discounting
        self.terminate = terminate
        self.penalty = penalty
        self.termination_probability = termination_probability

    @property
    def observation_size(self):
        observation_size = self.env.observation_size
        if isinstance(observation_size, dict):
            observation_size = {k: v + 1 for k, v in observation_size.items()}
        else:
            observation_size += 1
        return observation_size

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["saute_state"] = jp.ones(())
        state.info["eval_reward"] = state.reward
        state.info["prng"] = jax.random.split(rng, 2)[0]
        if isinstance(state.obs, jax.Array):
            state = state.replace(obs=jp.hstack([state.obs, state.info["saute_state"]]))
        else:
            obs = {
                k: jp.hstack([v, state.info["saute_state"]])
                for k, v in state.obs.items()
            }
            state = state.replace(obs=obs)
        state.metrics["saute_unsafe"] = jp.zeros_like(state.reward)
        state.metrics["saute_reward"] = state.reward
        state.metrics["saute_terminate"] = jp.zeros_like(state.reward)
        return state

    def step(self, state, action):
        saute_state = state.info["saute_state"]
        ones = jp.ones_like(saute_state)
        saute_state = jp.where(
            state.info.get("truncation", jp.zeros_like(state.done)), ones, saute_state
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jp.zeros_like(nstate.reward))
        cost += nstate.info.get("disagreement", 0.0)
        saute_state -= cost / self.budget
        saute_reward = jp.where(saute_state <= 0.0, -self.penalty, nstate.reward)
        terminate = jp.where(
            ((saute_state <= 0.0) & self.terminate) | nstate.done.astype(jp.bool),
            True,
            False,
        )
        rng = state.info["prng"]
        rng, sample_rng = jax.random.split(rng)
        state.info["prng"] = rng
        terminate = jp.where(
            terminate,
            jax.random.bernoulli(sample_rng, self.termination_probability).astype(
                jp.bool
            ),
            jp.zeros_like(terminate),
        )
        saute_state = jp.where(terminate, ones, saute_state)
        nstate.info["saute_state"] = saute_state
        nstate.info["eval_reward"] = nstate.reward
        nstate.metrics["saute_reward"] = saute_reward
        nstate.metrics["saute_unsafe"] = (saute_state <= 0.0).astype(jp.float32)
        nstate.metrics["saute_terminate"] = terminate.astype(jp.float32)
        if isinstance(nstate.obs, jax.Array):
            obs = jp.hstack([nstate.obs, saute_state])
        else:
            obs = {k: jp.hstack([v, saute_state]) for k, v in nstate.obs.items()}
        nstate = nstate.replace(
            obs=obs,
            done=terminate.astype(jp.float32),
            reward=saute_reward,
        )
        return nstate


class GoToGoalObservationWrapper(Wrapper):
    """Wrapper that filters GoToGoal observations to keep only dimensions with significant variance."""

    def __init__(self, env):
        """Initialize the wrapper with indices to keep based on std > 1e-6."""
        super().__init__(env)
        # Keep only indices where std > 1e-6
        # These are indices 0-31 and 48-54 based on the provided std values
        self.keep_indices = jp.array(
            list(range(0, 32)) + list(range(48, 55)),
            dtype=jp.int32,
        )

    def _filter_obs(self, obs):
        """Filter observation to keep only the specified indices."""
        if isinstance(obs, jax.Array):
            return jp.take(obs, self.keep_indices, axis=-1)
        else:
            filtered_obs = {}
            for k, v in obs.items():
                if k == "state":
                    filtered_obs[k] = jp.take(v, self.keep_indices, axis=-1)
                elif k == "privileged_state":
                    base_obs_size = 55
                    if len(v.shape) > 1:
                        filtered_obs[k] = jp.concatenate(
                            [
                                jp.take(
                                    v[..., :base_obs_size], self.keep_indices, axis=-1
                                ),
                                v[..., base_obs_size:],
                            ],
                            axis=-1,
                        )
                    else:
                        filtered_obs[k] = jp.concatenate(
                            [
                                jp.take(v[:base_obs_size], self.keep_indices),
                                v[base_obs_size:],
                            ]
                        )
                else:
                    filtered_obs[k] = v
            return filtered_obs

    def reset(self, rng):
        """Reset the environment and filter the observation."""
        state = self.env.reset(rng)
        filtered_obs = self._filter_obs(state.obs)
        return state.replace(obs=filtered_obs)

    def step(self, state, action):
        """Step the environment and filter the observation."""
        next_state = self.env.step(state, action)
        filtered_obs = self._filter_obs(next_state.obs)
        return next_state.replace(obs=filtered_obs)

    @property
    def observation_size(self):
        """Return the size of the filtered observation."""
        orig_size = self.env.observation_size
        if isinstance(orig_size, int):
            return len(self.keep_indices)
        else:
            sizes = {}
            for k, v in orig_size.items():
                if k == "state":
                    sizes[k] = len(self.keep_indices)
                elif k == "privileged_state":
                    base_size = 55
                    extra_size = v - base_size if v > base_size else 0
                    sizes[k] = len(self.keep_indices) + extra_size
                else:
                    sizes[k] = v
            return sizes


class WalkerObservationWrapper(Wrapper):
    """Wrapper that filters Walker observations to keep only specified indices."""

    def __init__(self, env):
        """Initialize the wrapper with pre-determined indices to keep."""
        super().__init__(env)
        self.keep_indices = jp.array(
            [0]
            + [i for j in range(2, 85, 3) for i in (j, j + 1) if i < 85]
            + list(range(85, 94)),
            dtype=jp.int32,
        )

    def _filter_obs(self, obs):
        """Filter observation to keep only the specified indices."""
        if isinstance(obs, jax.Array):
            return jp.take(obs, self.keep_indices, axis=-1)
        else:
            filtered_obs = {}
            for k, v in obs.items():
                if k == "state":
                    filtered_obs[k] = jp.take(v, self.keep_indices, axis=-1)
                elif k == "privileged_state":
                    base_obs_size = 94
                    if len(v.shape) > 1:
                        filtered_obs[k] = jp.concatenate(
                            [
                                jp.take(
                                    v[..., :base_obs_size], self.keep_indices, axis=-1
                                ),
                                v[..., base_obs_size:],
                            ],
                            axis=-1,
                        )
                    else:
                        filtered_obs[k] = jp.concatenate(
                            [
                                jp.take(v[:base_obs_size], self.keep_indices),
                                v[base_obs_size:],
                            ]
                        )
                else:
                    filtered_obs[k] = v
            return filtered_obs

    def reset(self, rng):
        """Reset the environment and filter the observation."""
        state = self.env.reset(rng)
        filtered_obs = self._filter_obs(state.obs)
        return state.replace(obs=filtered_obs)

    def step(self, state, action):
        """Step the environment and filter the observation."""
        next_state = self.env.step(state, action)
        filtered_obs = self._filter_obs(next_state.obs)
        return next_state.replace(obs=filtered_obs)

    @property
    def observation_size(self):
        """Return the size of the filtered observation."""
        orig_size = self.env.observation_size
        if isinstance(orig_size, int):
            return len(self.keep_indices)
        else:
            sizes = {}
            for k, v in orig_size.items():
                if k == "state":
                    sizes[k] = len(self.keep_indices)
                elif k == "privileged_state":
                    base_size = 94
                    extra_size = v - base_size if v > base_size else 0
                    sizes[k] = len(self.keep_indices) + extra_size
                else:
                    sizes[k] = v
            return sizes
