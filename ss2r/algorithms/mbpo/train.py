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

"""Model-Based Policy Optimization.

See: https://arxiv.org/abs/1906.08253
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.training import replay_buffers
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint
from brax.training.types import PRNGKey
from ml_collections import config_dict

import ss2r.algorithms.mbpo.losses as mbpo_losses
import ss2r.algorithms.mbpo.networks as mbpo_networks
from ss2r.algorithms.mbpo import safety_filters
from ss2r.algorithms.mbpo.model_env import create_model_env
from ss2r.algorithms.mbpo.on_policy_training_step import make_training_step
from ss2r.algorithms.mbpo.types import TrainingState
from ss2r.algorithms.penalizers import Params, Penalizer
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.data import collect_single_step
from ss2r.algorithms.sac.q_transforms import QTransformation, SACBase, SACCost
from ss2r.algorithms.sac.types import (
    CollectDataFn,
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
)
from ss2r.rl.evaluation import ConstraintsEvaluator, InterventionConstraintsEvaluator
from ss2r.rl.utils import restore_state


def get_dict_normalizer_params(params, ts_normalizer_params):
    mean = {
        "state": params[0].mean,
        "cumulative_cost": ts_normalizer_params.mean["cumulative_cost"],
    }
    std = {
        "state": params[0].std,
        "cumulative_cost": ts_normalizer_params.std["cumulative_cost"],
    }
    summed_var = {
        "state": params[0].summed_variance,
        "cumulative_cost": ts_normalizer_params.summed_variance["cumulative_cost"],
    }
    count = params[0].count
    ts_normalizer_params = ts_normalizer_params.replace(
        mean=mean,
        std=std,
        count=count,
        summed_variance=summed_var,
    )
    return ts_normalizer_params


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    mbpo_network: mbpo_networks.MBPONetworks,
    init_alpha: float,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    qr_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation,
    model_optimizer: optax.GradientTransformation,
    model_ensemble_size: int,
    penalizer_params: Params | None,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_qr, key_model = jax.random.split(key, 3)
    log_alpha = jnp.asarray(jnp.log(init_alpha), dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    policy_params = mbpo_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    qr_params = mbpo_network.qr_network.init(key_qr)
    qr_optimizer_state = qr_optimizer.init(qr_params)
    init_model_ensemble = jax.vmap(mbpo_network.model_network.init)
    model_keys = jax.random.split(key_model, model_ensemble_size)
    model_params = init_model_ensemble(model_keys)
    model_optimizer_state = model_optimizer.init(model_params)
    if mbpo_network.qc_network is not None:
        backup_qc_params = mbpo_network.qc_network.init(key_qr)
        assert qc_optimizer is not None
        backup_qc_optimizer_state = qc_optimizer.init(backup_qc_params)
        backup_qr_params = qr_params
    else:
        backup_qc_params = None
        backup_qc_optimizer_state = None
        backup_qr_params = None
    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
    normalizer_params = running_statistics.init_state(obs_shape)
    training_state = TrainingState(
        behavior_policy_optimizer_state=policy_optimizer_state,
        behavior_policy_params=policy_params,
        backup_policy_params=policy_params,
        behavior_qr_optimizer_state=qr_optimizer_state,
        behavior_qr_params=qr_params,
        backup_qr_params=backup_qr_params,
        behavior_qc_optimizer_state=backup_qc_optimizer_state,
        behavior_qc_params=backup_qc_params,
        behavior_target_qr_params=qr_params,
        behavior_target_qc_params=backup_qc_params,
        backup_qc_params=backup_qc_params,
        backup_qc_optimizer_state=backup_qc_optimizer_state,
        backup_target_qc_params=backup_qc_params,
        model_params=model_params,
        model_optimizer_state=model_optimizer_state,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        penalizer_params=penalizer_params,
    )  #  type: ignore
    return training_state


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    get_experience_fn: CollectDataFn = collect_single_step,
    make_training_step_fn=make_training_step,
    learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-4,
    model_learning_rate: float = 1e-4,
    alpha_learning_rate: float = 3e-4,
    init_alpha: float = 1.0,
    min_alpha: float = 0.0,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    sac_batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    model_grad_updates_per_step: int = 1,
    critic_grad_updates_per_step: int = 1,
    num_critic_updates_per_actor_update: int = 1,
    model_to_real_data_ratio: float = 1.0,
    unroll_length: int = 1,
    num_model_rollouts: int = 400,
    deterministic_eval: bool = False,
    network_factory: mbpo_networks.NetworkFactory[
        mbpo_networks.MBPONetworks,
    ] = mbpo_networks.make_mbpo_networks,
    n_critics: int = 2,
    n_heads: int = 1,
    model_ensemble_size: int = 1,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    safe: bool = False,
    safety_budget: float = float("inf"),
    penalizer: Penalizer | None = None,
    penalizer_params: Params | None = None,
    reward_q_transform: QTransformation = SACBase(),
    cost_q_transform: QTransformation = SACCost(),
    use_bro: bool = True,
    normalize_budget: bool = True,
    reset_on_eval: bool = True,
    store_buffer: bool = False,
    optimism: float = 0.0,
    pessimism: float = 0.0,
    model_propagation: str = "nominal",
    use_termination: bool = True,
    safety_filter: str | None = None,
    advantage_threshold: float = 0.2,
    offline: bool = False,
    learn_from_scratch: bool = False,
    load_auxiliaries: bool = False,
    load_normalizer: bool = True,
    target_entropy: float | None = None,
    pure_exploration_steps: int | None = None,
):
    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )
    episodic_safety_budget = safety_budget
    budget_scaling_fn = lambda x: x
    if safety_discounting != 1.0 and normalize_budget:
        budget_scaling_fn = (
            lambda x: x * episode_length * (1.0 - safety_discounting) / action_repeat
        )
    logging.info(f"Episode safety budget: {budget_scaling_fn(safety_budget)}")
    if max_replay_size is None:
        max_replay_size = num_timesteps
    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    env_steps_per_experience_call = env_steps_per_actor_step
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_experience_call = -(-min_replay_size // num_envs)
    if get_experience_fn != collect_single_step:
        # Using episodic or hardware (which is episodic)
        env_steps_per_experience_call *= episode_length
        num_prefill_experience_call = -(-num_prefill_experience_call // episode_length)
    num_prefill_env_steps = num_prefill_experience_call * env_steps_per_experience_call
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_experience_call)
    )
    env = environment
    if wrap_env_fn is not None:
        env = wrap_env_fn(env)
    rng = jax.random.PRNGKey(seed)
    obs_size = env.observation_size
    action_size = env.action_size
    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = functools.partial(
            running_statistics.normalize, max_abs_value=5.0
        )

    mbpo_network = network_factory(
        observation_size=obs_size["state"]
        if isinstance(obs_size, Mapping)
        else obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        safe=safe,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    alpha_optimizer = optax.adam(learning_rate=alpha_learning_rate)
    make_optimizer = lambda lr, grad_clip_norm: optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=lr),
    )
    policy_optimizer = make_optimizer(learning_rate, 1.0)
    qr_optimizer = make_optimizer(critic_learning_rate, 1.0)
    qc_optimizer = make_optimizer(critic_learning_rate, 1.0)
    model_optimizer = make_optimizer(model_learning_rate, 1.0)
    if isinstance(obs_size, Mapping):
        dummy_obs = {k: jnp.zeros(v) for k, v in obs_size.items()}
    else:
        dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    extras = {
        "state_extras": {
            "truncation": jnp.zeros(()),
        },
        "policy_extras": {},
    }
    if safe:
        extras["state_extras"]["cost"] = jnp.zeros(())  # type: ignore
    if safety_filter is not None:
        extras["policy_extras"] = {
            "intervention": jnp.zeros(()),
            "policy_distance": jnp.zeros(()),
            "safety_gap": jnp.zeros(()),
            "cumulative_cost": jnp.zeros(()),
            "expected_total_cost": jnp.zeros(()),
            "q_c": jnp.zeros(()),
        }

    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras=extras,
    )
    dummy_transition = float16(dummy_transition)
    global_key, local_key = jax.random.split(rng)
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        mbpo_network=mbpo_network,
        init_alpha=init_alpha,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        qr_optimizer=qr_optimizer,
        qc_optimizer=qc_optimizer,
        model_optimizer=model_optimizer,
        model_ensemble_size=model_ensemble_size,
        penalizer_params=penalizer_params,
    )
    del global_key
    local_key, model_rb_key, actor_critic_rb_key, env_key, eval_key = jax.random.split(
        local_key, 5
    )
    model_replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * model_grad_updates_per_step,
    )
    sac_replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=sac_batch_size * critic_grad_updates_per_step,
    )
    model_buffer_state = model_replay_buffer.init(model_rb_key)
    sac_buffer_state = sac_replay_buffer.init(actor_critic_rb_key)
    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        ts_normalizer_params = training_state.normalizer_params
        if load_normalizer:
            if isinstance(ts_normalizer_params.mean, dict) and not isinstance(
                params[0].mean, dict
            ):
                ts_normalizer_params = get_dict_normalizer_params(
                    params, ts_normalizer_params
                )
            else:
                ts_normalizer_params = params[0]
        if offline:
            model_buffer_state = replay_buffers.ReplayBufferState(**params[-1])
            training_state = training_state.replace(  # type: ignore
                normalizer_params=ts_normalizer_params
            )
        elif learn_from_scratch:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=ts_normalizer_params,
                backup_policy_params=params[1],
                backup_qr_params=params[3],
                backup_qc_params=params[4] if safe else None,
                backup_target_qc_params=params[4] if safe else None,
            )
        else:
            training_state = training_state.replace(  # type: ignore
                normalizer_params=ts_normalizer_params,
                behavior_policy_params=params[1],
                backup_policy_params=params[1],
                behavior_qr_params=params[3],
                behavior_target_qr_params=params[3],
                backup_qr_params=params[3],
                behavior_qc_params=params[4] if safe else None,
                behavior_target_qc_params=params[4] if safe else None,
                backup_qc_params=params[4] if safe else None,
                backup_target_qc_params=params[4] if safe else None,
            )
        if load_auxiliaries:
            policy_optimizer_state = restore_state(
                params[6][1]["inner_state"]
                if isinstance(params[6][1], dict)
                else params[6],
                training_state.behavior_policy_optimizer_state,
            )
            alpha_optimizer_state = restore_state(
                params[7], training_state.alpha_optimizer_state
            )
            qr_optimizer_state = restore_state(
                params[8][1]["inner_state"]
                if isinstance(params[8][1], dict)
                else params[8],
                training_state.behavior_qr_optimizer_state,
            )
            if not safe:
                qc_optimizer_state = None
            else:
                qc_optimizer_state = restore_state(
                    params[9][1]["inner_state"]
                    if isinstance(params[9][1], dict)
                    else params[9],
                    training_state.backup_qc_optimizer_state,
                )
            training_state = training_state.replace(  # type: ignore
                behavior_policy_optimizer_state=policy_optimizer_state,
                alpha_optimizer_state=alpha_optimizer_state,
                behavior_qr_optimizer_state=qr_optimizer_state,
                behavior_qc_optimizer_state=qc_optimizer_state,
                backup_qc_optimizer_state=qc_optimizer_state,
                alpha_params=params[5],
            )
    make_planning_policy = mbpo_networks.make_inference_fn(mbpo_network)
    make_rollout_policy, get_rollout_policy_params = safety_filters.make(
        safety_filter if safe else None,
        mbpo_network,
        training_state,
        advantage_threshold if safety_filter == "advantage" else safety_budget,
        budget_scaling_fn,
    )
    alpha_loss, critic_loss, actor_loss, model_loss = mbpo_losses.make_losses(
        mbpo_network=mbpo_network,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        discounting=discounting,
        safety_discounting=safety_discounting,
        action_size=action_size,
        use_bro=use_bro,
        normalize_fn=normalize_fn,
        target_entropy=target_entropy,
    )
    alpha_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=None
        )
    )
    critic_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qr_optimizer, pmap_axis_name=None
        )
    )
    if safe:
        cost_critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qc_optimizer, pmap_axis_name=None
        )
    else:
        cost_critic_update = None
    model_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            model_loss, model_optimizer, pmap_axis_name=None
        )
    )
    actor_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=None, has_aux=True
        )
    )
    extra_fields = ("truncation",)
    if safe:
        extra_fields += ("cost",)  # type: ignore

    make_model_env = functools.partial(
        create_model_env,
        mbpo_network=mbpo_network,
        action_size=action_size,
        observation_size=obs_size,
        ensemble_selection=model_propagation,
        safety_budget=safety_budget
        if safety_filter == "sooper"
        else advantage_threshold,
        cost_discount=safety_discounting,
        scaling_fn=budget_scaling_fn,
        use_termination=penalizer is not None and use_termination,
        safety_filter=safety_filter,
        initial_normalizer_params=training_state.normalizer_params,
    )
    training_step = make_training_step_fn(
        env,
        make_planning_policy,
        make_rollout_policy,
        get_rollout_policy_params,
        make_model_env,
        model_replay_buffer,
        sac_replay_buffer,
        alpha_update,
        critic_update,
        cost_critic_update,
        model_update,
        actor_update,
        safe,
        min_alpha,
        reward_q_transform,
        cost_q_transform,
        model_grad_updates_per_step,
        critic_grad_updates_per_step,
        extra_fields,
        get_experience_fn,
        env_steps_per_experience_call,
        tau,
        num_critic_updates_per_actor_update,
        unroll_length,
        num_model_rollouts,
        optimism,
        pessimism,
        model_to_real_data_ratio,
        budget_scaling_fn,
        use_termination,
        penalizer,
        safety_budget,
        safety_filter,
        offline,
        pure_exploration_steps,
    )

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience_fn(
                env,
                make_rollout_policy,
                get_rollout_policy_params(training_state),
                training_state.normalizer_params,
                model_replay_buffer,
                env_state,
                buffer_state,
                key,
                extra_fields,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_experience_call,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_experience_call,
        )[0]

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        model_buffer_state: ReplayBufferState,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, envs.State, ReplayBufferState, ReplayBufferState, Metrics
    ]:
        def f(carry, unused_t):
            ts, es, mbs, acbs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, mbs, metrics = training_step(ts, es, mbs, acbs, k)
            return (ts, es, mbs, acbs, new_key), metrics

        (
            (
                training_state,
                env_state,
                model_buffer_state,
                sac_buffer_state,
                key,
            ),
            metrics,
        ) = jax.lax.scan(
            f,
            (
                training_state,
                env_state,
                model_buffer_state,
                sac_buffer_state,
                key,
            ),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return (
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            metrics,
        )

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        model_buffer_state: ReplayBufferState,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, envs.State, ReplayBufferState, ReplayBufferState, Metrics
    ]:
        nonlocal training_walltime  # type: ignore
        t = time.time()
        (
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            metrics,
        ) = training_epoch(
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            key,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            env_steps_per_experience_call * num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Training state init
    # Env init
    env_keys = jax.random.split(env_key, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(env_keys)

    # Replay buffer init

    if not eval_env:
        eval_env = environment
    Evaluator = (
        InterventionConstraintsEvaluator
        if safety_filter is not None
        else ConstraintsEvaluator
    )
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_rollout_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        budget=episodic_safety_budget,
        num_episodes=num_eval_episodes,
    )

    # Run initial eval
    metrics = {}
    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            (
                training_state.normalizer_params,
                get_rollout_policy_params(training_state),
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    if not offline:
        training_state, env_state, model_buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, model_buffer_state, prefill_key
        )
    else:
        training_state = training_state.replace(  # type: ignore
            env_steps=training_state.env_steps
            + num_prefill_experience_call * env_steps_per_experience_call,
        )
    replay_size = jnp.sum(model_replay_buffer.size(model_buffer_state))
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)
        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        (
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            epoch_key,
        )
        if reset_on_eval:
            reset_keys = jax.random.split(epoch_key, num_envs)
            env_state = reset_fn(reset_keys)
        current_step = int(training_state.env_steps)

        # Eval and logging
        if checkpoint_logdir:
            # Save current policy.
            params = (
                training_state.normalizer_params,
                training_state.behavior_policy_params,
                training_state.penalizer_params,
                training_state.behavior_qr_params,
                training_state.behavior_qc_params,
                training_state.alpha_params,
                training_state.behavior_policy_optimizer_state,
                training_state.alpha_optimizer_state,
                training_state.behavior_qr_optimizer_state,
                training_state.behavior_qc_optimizer_state,
            )
            if store_buffer:
                params += (model_buffer_state,)
            dummy_ckpt_config = config_dict.ConfigDict()
            checkpoint.save(checkpoint_logdir, current_step, params, dummy_ckpt_config)

        # Run evals.
        metrics = evaluator.run_evaluation(
            (
                training_state.normalizer_params,
                get_rollout_policy_params(training_state),
            ),
            training_metrics,
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps
    params = (
        training_state.normalizer_params,
        training_state.behavior_policy_params,
        training_state.penalizer_params,
        training_state.behavior_qr_params,
        training_state.behavior_qc_params,
        training_state.alpha_params,
        training_state.behavior_policy_optimizer_state,
        training_state.alpha_optimizer_state,
        training_state.behavior_qr_optimizer_state,
        training_state.behavior_qc_optimizer_state,
    )
    if store_buffer:
        params += (model_buffer_state,)
    logging.info("total steps: %s", total_steps)
    return make_rollout_policy, params, metrics
