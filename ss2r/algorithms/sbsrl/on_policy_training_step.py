import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import VmapWrapper
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.types import Policy, PRNGKey

from ss2r.algorithms.sac.q_transforms import SACBase, SACCost
from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
    float32,
)
from ss2r.algorithms.sbsrl.model_env import ModelBasedEnv
from ss2r.algorithms.sbsrl.types import TrainingState, TrainingStepFn


def make_on_policy_training_step(
    env,
    make_policy,
    make_model_env,
    model_replay_buffer,
    sac_replay_buffer,
    alpha_update,
    critic_update,
    cost_critic_update,
    backup_critic_update,
    backup_cost_critic_update,
    model_update,
    actor_update,
    safe,
    uncertainty_constraint,
    min_alpha,
    reward_q_transform,
    cost_q_transform,
    model_grad_updates_per_step,
    critic_grad_updates_per_step,
    extra_fields,
    extra_fields_model,
    get_experience_fn,
    env_steps_per_experience_call,
    tau,
    num_critic_updates_per_actor_update,
    unroll_length,
    num_model_rollouts,
    penalizer,
    safety_budget,
    reward_pessimism,
    cost_pessimism,
    model_to_real_data_ratio,
    offline,
    save_sooper_backup,
    ensemble_size,
    sac_batch_size,
    normalize_fn,
) -> TrainingStepFn:
    def split_transitions_ensemble(
        transitions: Transition, ensemble_axis: int = 1
    ) -> Transition:
        def _reshape_leaf(x, name):
            if isinstance(x, dict):
                return {k: _reshape_leaf(v, k) for k, v in x.items()}
            x = jnp.asarray(x)
            if name in ("observation", "action"):
                expanded = jnp.expand_dims(x, axis=0)
                target_shape = (ensemble_size,) + x.shape
                return jnp.broadcast_to(expanded, target_shape)
            else:
                perm = list(range(x.ndim))
                perm.pop(ensemble_axis)
                perm = [ensemble_axis] + perm
                return jnp.transpose(x, axes=perm)

        trans_per_ens = transitions._replace(
            **{
                f: _reshape_leaf(getattr(transitions, f), f)
                for f in transitions._fields
            }
        )

        # add index of ensemble prediction as an extra field
        idx = jnp.arange(ensemble_size, dtype=jnp.int32)[:, None, None]
        idx = jnp.broadcast_to(idx, (ensemble_size, sac_batch_size, 1))
        new_extras = {
            **trans_per_ens.extras,
            "state_extras": {
                **trans_per_ens.extras.get("state_extras", {}),
                "idx": idx,
            },
        }
        trans_per_ens = trans_per_ens._replace(extras=new_extras)
        return trans_per_ens

    def compress_transitions_ensemble(
        transitions: Transition, ensemble_axis: int = 1
    ) -> Transition:
        def _reduce_leaf(x: Any, name: str):
            if isinstance(x, dict):
                return {k: _reduce_leaf(v, k) for k, v in x.items()}
            x_arr = jnp.asarray(x)
            if name in ("observation", "action"):
                return x_arr
            return jnp.mean(x_arr, axis=ensemble_axis)

        replacements = {
            f: _reduce_leaf(getattr(transitions, f), f) for f in transitions._fields
        }
        trans_compressed = transitions._replace(**replacements)
        return trans_compressed

    def sgd_step(
        carry: Tuple[TrainingState, ReplayBufferState, PRNGKey, int],
        unused_t,
        planning_env: ModelBasedEnv,
    ) -> Tuple[Tuple[TrainingState, ReplayBufferState, PRNGKey, int], Metrics]:
        training_state, sac_buffer_state, key, count = carry
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
        new_buffer_state, transitions = sac_replay_buffer.sample(sac_buffer_state)
        transitions = float32(transitions)
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) + min_alpha

        if offline:
            transitions = relabel_offline_transitions(planning_env, transitions)

        # reshape transitions with leading ensemble size
        trans_per_ens = split_transitions_ensemble(transitions, ensemble_axis=1)

        # reward critic update for each ensemble prediction
        _, *ens_keys_reward = jax.random.split(key_critic, ensemble_size + 1)
        ens_keys_reward = jnp.stack(ens_keys_reward)
        critic_loss, behavior_qr_params, behavior_qr_optimizer_state = critic_update(
            training_state.behavior_qr_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_target_qr_params,
            alpha,
            trans_per_ens,
            ens_keys_reward,
            reward_q_transform,
            False,
            False,
            optimizer_state=training_state.behavior_qr_optimizer_state,
            params=training_state.behavior_qr_params,
        )
        if save_sooper_backup:
            compressed_transitions = compress_transitions_ensemble(
                transitions, ensemble_axis=1
            )
            (
                backup_critic_loss,
                backup_qr_params,
                backup_qr_optimizer_state,
            ) = backup_critic_update(
                training_state.backup_qr_params,
                training_state.behavior_policy_params,  # TODO: Is it correct to use a common policy for backup and behavior?
                training_state.normalizer_params,
                training_state.backup_target_qr_params,
                alpha,
                compressed_transitions,
                key_critic,
                SACBase(),
                optimizer_state=training_state.backup_qr_optimizer_state,
                params=training_state.backup_qr_params,
            )
        else:
            backup_qr_params = training_state.backup_qr_params
            backup_qr_optimizer_state = training_state.backup_qr_optimizer_state

        if safe or uncertainty_constraint:
            cost_metrics = {}
            if penalizer is not None:
                # cost critic update for each ensemble prediction
                key, key_cost = jax.random.split(key)
                _, *ens_keys_cost = jax.random.split(key_cost, ensemble_size + 1)
                ens_keys_cost = jnp.stack(ens_keys_cost)
                (
                    cost_loss,
                    behavior_qc_params,
                    behavior_qc_optimizer_state,
                ) = cost_critic_update(
                    training_state.behavior_qc_params,
                    training_state.behavior_policy_params,
                    training_state.normalizer_params,
                    training_state.behavior_target_qc_params,
                    alpha,
                    trans_per_ens,
                    ens_keys_cost,
                    cost_q_transform,
                    safe,
                    uncertainty_constraint,
                    optimizer_state=training_state.behavior_qc_optimizer_state,
                    params=training_state.behavior_qc_params,
                )
                cost_metrics["behavior_cost_critic_loss"] = cost_loss
            else:
                behavior_qc_params = training_state.behavior_qc_params
                behavior_qc_optimizer_state = training_state.behavior_qc_optimizer_state
            if save_sooper_backup:
                (
                    backup_cost_critic_loss,
                    backup_qc_params,
                    backup_qc_optimizer_state,
                ) = backup_cost_critic_update(
                    training_state.backup_qc_params,
                    training_state.behavior_policy_params,
                    training_state.normalizer_params,
                    training_state.backup_target_qc_params,
                    alpha,
                    compressed_transitions,
                    key_critic,
                    SACCost(),
                    True,
                    optimizer_state=training_state.backup_qc_optimizer_state,
                    params=training_state.backup_qc_params,
                )
                cost_metrics["backup_cost_critic_loss"] = backup_cost_critic_loss
            else:
                backup_qc_params = training_state.backup_qc_params
                backup_qc_optimizer_state = training_state.backup_qc_optimizer_state
        else:
            cost_metrics = {}
            backup_qc_params = training_state.backup_qc_params
            backup_qc_optimizer_state = training_state.backup_qc_optimizer_state
            behavior_qc_params = training_state.behavior_qc_params
            behavior_qc_optimizer_state = training_state.behavior_qc_optimizer_state

        (actor_loss, aux), new_policy_params, new_policy_optimizer_state = actor_update(
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_qr_params,
            training_state.behavior_qc_params,
            training_state.model_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            penalizer,
            training_state.penalizer_params,
            optimizer_state=training_state.behavior_policy_optimizer_state,
            params=training_state.behavior_policy_params,
        )

        should_update_actor = count % num_critic_updates_per_actor_update == 0
        update_if_needed = lambda x, y: jnp.where(should_update_actor, x, y)
        policy_params = jax.tree_map(
            update_if_needed, new_policy_params, training_state.behavior_policy_params
        )
        policy_optimizer_state = jax.tree_map(
            update_if_needed,
            new_policy_optimizer_state,
            training_state.behavior_policy_optimizer_state,
        )
        new_penalizer_params = training_state.penalizer_params
        if "penalizer_params" in aux:
            new_penalizer_params = aux.pop("penalizer_params")
        penalizer_params = jax.tree_map(
            update_if_needed,
            new_penalizer_params,
            training_state.penalizer_params,
        )

        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_behavior_target_qr_params = polyak(
            training_state.behavior_target_qr_params, behavior_qr_params
        )
        if safe or uncertainty_constraint:
            new_backup_target_qc_params = polyak(
                training_state.backup_target_qc_params, backup_qc_params
            )
            if penalizer is not None:
                new_behavior_target_qc_params = polyak(
                    training_state.behavior_target_qc_params, behavior_qc_params
                )
            else:
                new_behavior_target_qc_params = training_state.behavior_target_qc_params
        else:
            new_behavior_target_qc_params = training_state.behavior_target_qc_params

        if save_sooper_backup:
            new_backup_target_qr_params = polyak(
                training_state.backup_target_qr_params, backup_qr_params
            )
            new_backup_target_qc_params = polyak(
                training_state.backup_target_qc_params, backup_qc_params
            )
        else:
            new_backup_target_qr_params = training_state.backup_target_qr_params
            new_backup_target_qc_params = training_state.backup_target_qc_params

        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            "critic_loss": critic_loss,
            "fraction_done": 1.0 - transitions.discount.mean(),
            **cost_metrics,
            **aux,
        }
        # Note that we store the qc of the backup also as the
        # qc of behavior, this is so that later it
        # will get save correctly to the checkpoint. #TODO: I changed this, ok?
        new_training_state = training_state.replace(  # type: ignore
            behavior_qr_optimizer_state=behavior_qr_optimizer_state,
            behavior_qr_params=behavior_qr_params,
            behavior_target_qr_params=new_behavior_target_qr_params,
            behavior_target_qc_params=new_behavior_target_qc_params,
            backup_qc_optimizer_state=backup_qc_optimizer_state,
            backup_qr_optimizer_state=backup_qr_optimizer_state,
            backup_qc_params=backup_qc_params,
            behavior_qc_optimizer_state=behavior_qc_optimizer_state,
            behavior_qc_params=behavior_qc_params,
            backup_target_qc_params=new_backup_target_qc_params,
            backup_target_qr_params=new_backup_target_qr_params,
            gradient_steps=training_state.gradient_steps + 1,
            behavior_policy_optimizer_state=policy_optimizer_state,
            behavior_policy_params=policy_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            penalizer_params=penalizer_params,
        )
        return (new_training_state, new_buffer_state, key, count + 1), metrics

    def model_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        # TODO (yarden): can remove this
        key, _ = jax.random.split(key)
        transitions = float32(transitions)
        model_loss, model_params, model_optimizer_state = model_update(
            training_state.model_params,
            training_state.normalizer_params,
            transitions,
            optimizer_state=training_state.model_optimizer_state,  # type: ignore
            params=training_state.model_params,
        )
        new_training_state = training_state.replace(  # type: ignore
            model_optimizer_state=model_optimizer_state,
            model_params=model_params,
        )
        metrics = {"model_loss": model_loss}
        return (new_training_state, key), metrics

    def run_experience_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        """Runs the non-jittable experience collection step."""
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience_fn(
            env,
            make_policy,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            model_replay_buffer,
            env_state,
            buffer_state,
            experience_key,
            extra_fields,
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_experience_call,
        )
        return training_state, env_state, buffer_state, training_key

    def generate_model_data(
        training_state: TrainingState,
        planning_env: ModelBasedEnv,
        policy: Policy,
        sac_replay_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        keys = jax.random.split(key, num_model_rollouts + 2)
        key = keys[0]
        key_generate_unroll = keys[1]
        rollout_keys = keys[2:]

        if unroll_length != 1:
            raise ValueError("Unrolls with more than one step not supported")
        state = planning_env.reset(rollout_keys)  # one-step rollout
        _, transitions = acting.actor_step(
            planning_env,
            state,
            policy,
            key_generate_unroll,
            extra_fields=extra_fields_model,
        )
        disagreement = jnp.expand_dims(
            transitions.extras["state_extras"]["disagreement"], axis=1
        )  # (B,1)
        new_disagreement_normalizer_params = running_statistics.update(
            training_state.disagreement_normalizer_params, disagreement
        )
        training_state = training_state.replace(  # type: ignore
            disagreement_normalizer_params=new_disagreement_normalizer_params
        )
        disagreement = normalize_fn(
            disagreement, training_state.disagreement_normalizer_params
        )
        transitions.extras["state_extras"]["disagreement"] = jnp.full(
            transitions.reward.shape, disagreement
        )  # (B,ensemble_size)
        sac_replay_buffer_state = sac_replay_buffer.insert(
            sac_replay_buffer_state, float16(transitions)
        )
        disagreement_metrics = {"normalized_disagreement": disagreement.mean()}
        return training_state, sac_replay_buffer_state, disagreement_metrics

    def relabel_real_transitions(
        training_state: TrainingState,
        planning_env: ModelBasedEnv,
        transitions: Transition,
    ):
        pred_fn = planning_env.model_network.apply
        model_params = planning_env.model_params
        normalizer_params = planning_env.normalizer_params
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, *_ = vmap_pred_fn(
            normalizer_params, model_params, transitions.observation, transitions.action
        )
        disagreement = (
            next_obs_pred.std(axis=0).mean(-1)
            if isinstance(next_obs_pred, jax.Array)
            else next_obs_pred["state"].std(axis=0).mean(-1)
        )  # (B,)
        disagreement = jnp.expand_dims(disagreement, axis=1)  # (B, 1)
        new_disagreement_normalizer_params = running_statistics.update(
            training_state.disagreement_normalizer_params, disagreement
        )
        training_state = training_state.replace(  # type: ignore
            disagreement_normalizer_params=new_disagreement_normalizer_params
        )
        disagreement = normalize_fn(
            disagreement, training_state.disagreement_normalizer_params
        )

        new_reward = jnp.tile(transitions.reward[:, None], (1, ensemble_size))
        new_discount = jnp.tile(transitions.discount[:, None], (1, ensemble_size))
        new_next_observation = jnp.tile(
            transitions.next_observation[:, None, :], (1, ensemble_size, 1)
        )
        transitions.extras["state_extras"]["truncation"] = jnp.tile(
            transitions.extras["state_extras"]["truncation"][:, None],
            (1, ensemble_size),
        )
        transitions.extras["state_extras"]["disagreement"] = jnp.tile(
            disagreement, (1, ensemble_size)
        )
        if safe:
            transitions.extras["state_extras"]["cost"] = jnp.tile(
                transitions.extras["state_extras"]["cost"][:, None], (1, ensemble_size)
            )

        return training_state, Transition(
            observation=transitions.observation,
            next_observation=new_next_observation,
            action=transitions.action,
            reward=new_reward,
            discount=new_discount,
            extras=transitions.extras,
        )

    def relabel_offline_transitions(
        planning_env: ModelBasedEnv,
        transitions: Transition,
    ):
        pred_fn = planning_env.model_network.apply
        model_params = planning_env.model_params
        normalizer_params = planning_env.normalizer_params
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward, cost = vmap_pred_fn(
            normalizer_params, model_params, transitions.observation, transitions.action
        )
        disagreement = (
            next_obs_pred.std(axis=0).mean(-1)
            if isinstance(next_obs_pred, jax.Array)
            else next_obs_pred["state"].std(axis=0).mean(-1)
        )  # (B,)
        transitions.extras["state_extras"]["cost"] += (
            jnp.expand_dims(disagreement, 1) * cost_pessimism
        )
        new_reward = (
            transitions.reward - jnp.expand_dims(disagreement, 1) * reward_pessimism
        )
        return Transition(
            observation=transitions.observation,
            next_observation=transitions.next_observation,
            action=transitions.action,
            reward=new_reward,
            discount=transitions.discount,
            extras=transitions.extras,
        )

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        model_buffer_state: ReplayBufferState,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, envs.State, ReplayBufferState, ReplayBufferState, Metrics
    ]:
        """Splits training into experience collection and a jitted training step."""
        # Keep the original buffer state, so that model-generated data is discarded
        initial_sac_buffer_state = sac_buffer_state
        if not offline:
            (
                training_state,
                env_state,
                model_buffer_state,
                training_key,
            ) = run_experience_step(training_state, env_state, model_buffer_state, key)
        else:
            training_state = training_state.replace(  # type: ignore
                env_steps=training_state.env_steps + env_steps_per_experience_call,
            )
            training_key = key
        model_buffer_state, transitions = model_replay_buffer.sample(model_buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        tmp_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (model_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), model_metrics = jax.lax.scan(
            model_sgd_step, (training_state, training_key), tmp_transitions
        )
        planning_env = make_model_env(
            training_state=training_state,
            transitions=transitions,
        )
        planning_env = VmapWrapper(planning_env)
        policy = make_policy(
            (training_state.normalizer_params, training_state.behavior_policy_params)
        )
        # Rollout trajectories from the sampled transitions
        training_state, sac_buffer_state, disagreement_metrics = generate_model_data(
            training_state, planning_env, policy, sac_buffer_state, training_key
        )
        # Add some real transitions
        if model_to_real_data_ratio < 1:
            training_state, transitions = relabel_real_transitions(
                training_state, planning_env, transitions
            )
            transitions = float16(transitions)
            sac_buffer_state = sac_replay_buffer.insert(sac_buffer_state, transitions)

        # Train SAC
        sgd_with_env = functools.partial(sgd_step, planning_env=planning_env)
        (training_state, sac_buffer_state, *_), metrics = jax.lax.scan(
            sgd_with_env,
            (training_state, sac_buffer_state, training_key, 0),
            (),
            length=critic_grad_updates_per_step,
        )

        metrics["buffer_current_size"] = model_replay_buffer.size(model_buffer_state)
        metrics |= env_state.metrics
        metrics |= disagreement_metrics
        metrics |= model_metrics
        return (
            training_state,
            env_state,
            model_buffer_state,
            initial_sac_buffer_state,
            metrics,
        )

    return training_step
