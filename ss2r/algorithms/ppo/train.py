# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from brax import envs
from brax.training import pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint
from brax.training.types import Params, PRNGKey
from ml_collections import config_dict

from ss2r.algorithms.penalizers import Penalizer
from ss2r.algorithms.ppo import _PMAP_AXIS_NAME, Metrics, TrainingState
from ss2r.algorithms.ppo import losses as ppo_losses
from ss2r.algorithms.ppo import networks as ppo_networks
from ss2r.algorithms.ppo import training_step as ppo_training_step
from ss2r.algorithms.ppo.wrappers import TrackOnlineCosts
from ss2r.rl.evaluation import ConstraintsEvaluator
from ss2r.rl.utils import restore_state


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    update_step_factory=ppo_training_step.update_fn,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    safety_gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        ppo_networks.SafePPONetworks
    ] = ppo_networks.make_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    safety_budget: float = float("inf"),
    penalizer: Penalizer | None = None,
    penalizer_params: Params | None = None,
    safe: bool = False,
    use_disagreement: bool = False,
    normalize_budget: bool = True,
):
    assert batch_size * num_minibatches % num_envs == 0
    if not safe:
        penalizer = None
        penalizer_params = None
    original_safety_budget = safety_budget
    if normalize_budget:
        safety_budget = (safety_budget / episode_length) / (1.0 - safety_discounting)
    xt = time.time()
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key, 2)
    del global_key
    assert num_envs % device_count == 0
    env = environment
    env = TrackOnlineCosts(env)
    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs,
        (local_devices_to_use, -1) + key_envs.shape[1:],
    )
    env_state = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    value_optimizer = optax.adam(learning_rate=learning_rate)
    cost_value_optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is None:
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        cost_value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
    policy_loss, value_loss, cost_value_loss = ppo_losses.make_losses(
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        safety_discounting=safety_discounting,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        gae_lambda=gae_lambda,
        safety_gae_lambda=safety_gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        safety_budget=safety_budget,
        use_disagreement=use_disagreement,
    )
    training_step = update_step_factory(
        policy_loss,
        value_loss,
        cost_value_loss,
        policy_optimizer,
        value_optimizer,
        cost_value_optimizer,
        env,
        unroll_length,
        num_minibatches,
        make_policy,
        penalizer,
        num_updates_per_batch,
        batch_size,
        num_envs,
        env_step_per_training_step,
        safe,
        use_disagreement,
    )

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime  # type: ignore
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.SafePPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=ppo_network.cost_value_network.init(key_value),
    )  # type: ignore
    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    policy_optimizer_state = policy_optimizer.init(init_params.policy)
    value_optimizer_state = value_optimizer.init(init_params.value)
    cost_value_optimizer_state = cost_value_optimizer.init(init_params.cost_value)
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=(
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape),
        env_steps=0,
        penalizer_params=penalizer_params,
    )  # type: ignore

    if restore_checkpoint_path is not None:
        loaded_params = checkpoint.load(restore_checkpoint_path)
        restored_normalizer = restore_state(
            loaded_params[0], training_state.normalizer_params
        )
        restored_network_params = training_state.params
        restored_penalizer_params = training_state.penalizer_params
        restored_optimizer_state = training_state.optimizer_state

        if len(loaded_params) >= 2:
            try:
                restored_network_params = restore_state(
                    loaded_params[1], training_state.params
                )
            except Exception:
                if len(loaded_params) >= 3:
                    restored_network_params = training_state.params.replace(  # type: ignore
                        policy=restore_state(
                            loaded_params[1], training_state.params.policy
                        ),
                        value=restore_state(
                            loaded_params[2], training_state.params.value
                        ),
                    )
        if len(loaded_params) >= 3:
            try:
                restored_penalizer_params = restore_state(
                    loaded_params[2], training_state.penalizer_params
                )
            except Exception:
                pass
        if len(loaded_params) >= 4:
            try:
                restored_optimizer_state = restore_state(
                    loaded_params[3], training_state.optimizer_state
                )
            except Exception:
                pass

        training_state = training_state.replace(  # type: ignore
            normalizer_params=restored_normalizer,
            params=restored_network_params,
            penalizer_params=restored_penalizer_params,
            optimizer_state=restored_optimizer_state,
        )  # type: ignore

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )
    evaluator = ConstraintsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        budget=original_safety_budget,
        num_episodes=num_eval_episodes,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics: Metrics = {}
    training_walltime = 0.0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))
            key_env, tmp_key = jax.random.split(key_env)
            key_envs = jax.random.split(tmp_key, num_envs // process_count)
            key_envs = jnp.reshape(
                key_envs,
                (local_devices_to_use, -1) + key_envs.shape[1:],
            )
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                        training_state.params.value,
                    )
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            params = _unpmap((training_state.normalizer_params, training_state.params))
            policy_params_fn(current_step, make_policy, params)
            if checkpoint_logdir:
                checkpoint_params = _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params,
                        training_state.penalizer_params,
                        training_state.optimizer_state,
                    )
                )
                dummy_ckpt_config = config_dict.ConfigDict()
                checkpoint.save(
                    checkpoint_logdir,
                    current_step,
                    checkpoint_params,
                    dummy_ckpt_config,
                )

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
