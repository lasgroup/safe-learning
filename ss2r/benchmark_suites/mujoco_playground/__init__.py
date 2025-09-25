from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground import wrapper as mujoco_playground_wrapper
from mujoco_playground._src import mjx_env

from ss2r.benchmark_suites import wrappers
from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def _dig(env):
    if env == env.unwrapped:
        raise ValueError("Not wrapped")
    if isinstance(env, wrappers.BraxDomainRandomizationVmapWrapper):
        return env
    else:
        return _dig(env.env)


def render(env, policy, steps, rng, camera=None, num_envs=5):
    state = env.reset(rng)
    if num_envs is not None:
        state = jax.tree_map(lambda x: x[:5], state)
    orig_model = env._mjx_model
    if hasattr(env, "_randomized_models") and num_envs is not None:
        render_env = _dig(env)
        model = jax.tree_map(
            lambda x, ax: jnp.take(x, jnp.arange(num_envs), axis=ax)
            if ax is not None
            else x,
            env._randomized_models,
            env._in_axes,
        )
        render_env._randomized_models = model
    else:
        render_env = env
    _, trajectory = rollout(render_env, policy, steps, rng[0], state)
    env._mjx_model = orig_model
    videos = []
    for i in range(5):
        ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        video = env.render(ep_trajectory, camera=camera)
        videos.append(video)
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    hard_resets: bool = False,
    nonepisodic: bool = False,
    *,
    augment_state: bool = False,
) -> mujoco_playground_wrapper.Wrapper:
    """Common wrapper pattern for all brax training agents.

    Args:
      env: environment to be wrapped
      vision: whether the environment will be vision based
      num_vision_envs: number of environments the renderer should generate,
        should equal the number of batched envs
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized model
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if vision:
        env = mujoco_playground_wrapper.MadronaWrapper(
            env, num_vision_envs, randomization_fn
        )
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = wrappers.BraxDomainRandomizationVmapWrapper(
            env, randomization_fn, augment_state=augment_state
        )
    if nonepisodic:
        env = wrappers.NonEpisodicWrapper(env, action_repeat)
    else:
        env = wrappers.CostEpisodeWrapper(env, episode_length, action_repeat)
        if hard_resets:
            env = wrappers.HardAutoResetWrapper(env)
        else:
            env = mujoco_playground_wrapper.BraxAutoResetWrapper(env)
    return env
