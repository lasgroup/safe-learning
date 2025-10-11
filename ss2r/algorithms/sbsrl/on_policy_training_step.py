from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import VmapWrapper
from brax.training import acting
from brax.training.types import Policy, PRNGKey

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
    scaling_fn,
    use_termination,
    penalizer,
    safety_budget,
    safety_filter,
    offline,
    pure_exploration_steps,
) -> TrainingStepFn:
    def split_transitions_ensemble(
        transitions: Transition, ensemble_axis: int = 1
    ) -> Transition:  # TODO: add types
        ref = (
            transitions.next_observation
            if hasattr(transitions, "next_observation")
            else jax.tree_util.tree_leaves(transitions)[0]
        )
        ref = jnp.asarray(ref)
        if ref.ndim <= ensemble_axis:
            raise ValueError(
                f"Reference leaf has ndim {ref.ndim} <= ensemble_axis {ensemble_axis}"
            )

        E = ref.shape[ensemble_axis]

        def _per_ens_leaf(x):
            x = jnp.asarray(x)
            if x.ndim > ensemble_axis and x.shape[ensemble_axis] == E:
                perm = list(range(x.ndim))
                perm.pop(ensemble_axis)
                perm = [ensemble_axis] + perm
                return jnp.transpose(x, axes=perm)
            else:  # no ensemble axis
                expanded = jnp.expand_dims(x, axis=0)  # (1, U, B, ...)
                target_shape = (E,) + x.shape
                return jnp.broadcast_to(expanded, target_shape)

        trans_per_ens = jax.tree_util.tree_map(_per_ens_leaf, transitions)

        sample_leaf = jax.tree_util.tree_leaves(trans_per_ens)[0]
        slshape = sample_leaf.shape
        B = int(slshape[1])
        idx = jnp.arange(E, dtype=jnp.int32)[:, None, None]  # (E,1,1)
        idx = jnp.broadcast_to(idx, (E, B, 1))  # (E,B,1)

        def _update_extras_map(extras):
            extras_dict = dict(extras)
            state_extras = dict(extras_dict.get("state_extras", {}))
            state_extras["idx"] = idx
            extras_dict["state_extras"] = state_extras
            return extras_dict

        orig_extras = trans_per_ens.extras
        new_extras = _update_extras_map(orig_extras)

        new_trans = trans_per_ens._replace(extras=new_extras)
        return new_trans

    def critic_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        transitions = float32(transitions)
        alpha = jnp.exp(training_state.alpha_params) + min_alpha

        trans_per_ens = split_transitions_ensemble(transitions, ensemble_axis=1)

        sample_leaf = jax.tree_util.tree_leaves(trans_per_ens)[0]
        E = int(jnp.asarray(sample_leaf).shape[0])
        _unused, *ens_keys = jax.random.split(key_critic, E + 1)
        ens_keys = jnp.stack(ens_keys)  # shape (E, 2)

        def _scan_body(carry, elems):
            """
            carry: (params, opt_state)
            elems: (trans_single, key_i)
            """
            params, opt_state = carry
            trans_single, key_i = elems

            loss_i, new_params, new_opt_state = critic_update(
                params,
                training_state.behavior_policy_params,
                training_state.normalizer_params,
                training_state.behavior_target_qr_params,
                alpha,
                trans_single,
                key_i,
                reward_q_transform,
                optimizer_state=opt_state,
                params=params,
            )
            carry_out = (new_params, new_opt_state)
            return carry_out, loss_i

        elems = (trans_per_ens, ens_keys)
        (
            (behavior_qr_params, behavior_qr_optimizer_state),
            ensemble_losses,
        ) = jax.lax.scan(
            _scan_body,
            (
                training_state.behavior_qr_params,
                training_state.behavior_qr_optimizer_state,
            ),
            elems,
            length=E,
        )
        critic_loss = jnp.mean(ensemble_losses)

        if safe:
            raise Exception("not yet implemented for safety")
        else:
            cost_metrics: Dict[str, Any] = {}
            backup_qc_params = training_state.backup_qc_params
            backup_qc_optimizer_state = training_state.backup_qc_optimizer_state
            behavior_qc_params = training_state.behavior_qc_params
            behavior_qc_optimizer_state = training_state.behavior_qc_optimizer_state

        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_behavior_target_qr_params = polyak(
            training_state.behavior_target_qr_params, behavior_qr_params
        )
        if safe:
            raise Exception("not yet implemented for safety")
        else:
            new_backup_target_qc_params = training_state.backup_target_qc_params
            new_behavior_target_qc_params = training_state.behavior_target_qc_params
        metrics = {
            "critic_loss": critic_loss,
            "fraction_done": 1.0 - transitions.discount.mean(),
            **cost_metrics,
        }
        new_training_state = training_state.replace(  # type: ignore
            behavior_qr_optimizer_state=behavior_qr_optimizer_state,
            behavior_qc_optimizer_state=behavior_qc_optimizer_state,
            behavior_qr_params=behavior_qr_params,
            behavior_qc_params=behavior_qc_params,
            behavior_target_qr_params=new_behavior_target_qr_params,
            behavior_target_qc_params=new_behavior_target_qc_params,
            backup_qc_optimizer_state=backup_qc_optimizer_state,
            backup_qc_params=backup_qc_params,
            backup_target_qc_params=new_backup_target_qc_params,
            gradient_steps=training_state.gradient_steps + E,
        )
        return (new_training_state, key), metrics

    def actor_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_actor = jax.random.split(key, 3)
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
        (actor_loss, aux), policy_params, policy_optimizer_state = actor_update(
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_qr_params,
            training_state.behavior_qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            penalizer,
            training_state.penalizer_params,
            optimizer_state=training_state.behavior_policy_optimizer_state,
            params=training_state.behavior_policy_params,
        )
        if aux:
            new_penalizer_params = aux.pop("penalizer_params")
            additional_metrics = {
                **aux,
            }
        else:
            new_penalizer_params = training_state.penalizer_params
            additional_metrics = {}
        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            **additional_metrics,
        }
        new_training_state = training_state.replace(  # type: ignore
            behavior_policy_optimizer_state=policy_optimizer_state,
            behavior_policy_params=policy_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            penalizer_params=new_penalizer_params,
        )
        return (new_training_state, key), metrics

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
            make_rollout_policy,
            get_rollout_policy_params(training_state),
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
        planning_env: ModelBasedEnv,
        policy: Policy,
        sac_replay_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> ReplayBufferState:
        keys = jax.random.split(key, num_model_rollouts + 2)
        key = keys[0]
        key_generate_unroll = keys[1]
        rollout_keys = keys[2:]

        if unroll_length != 1:
            raise ValueError("Unrolls with more than one step not supported")
        state = planning_env.reset(rollout_keys)  # one-step rollout
        _, transitions = acting.actor_step(
            planning_env, state, policy, key_generate_unroll, extra_fields=extra_fields
        )
        # transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions) #don't need it anymore as we don't have unroll_length dimension
        sac_replay_buffer_state = sac_replay_buffer.insert(
            sac_replay_buffer_state, float16(transitions)
        )

        return sac_replay_buffer_state

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
        policy = make_planning_policy(
            (training_state.normalizer_params, training_state.behavior_policy_params)
        )
        # Rollout trajectories from the sampled transitions
        sac_buffer_state = generate_model_data(
            planning_env, policy, sac_buffer_state, training_key
        )
        # Train SAC with model data
        sac_buffer_state, model_transitions = sac_replay_buffer.sample(sac_buffer_state)
        transitions = model_transitions
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (critic_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), critic_metrics = jax.lax.scan(
            critic_sgd_step, (training_state, training_key), transitions
        )
        num_actor_updates = -(
            -critic_grad_updates_per_step // num_critic_updates_per_actor_update
        )
        assert num_actor_updates > 0, "Actor updates is non-positive"
        transitions = jax.tree_util.tree_map(
            lambda x: x[:num_actor_updates], transitions
        )
        (training_state, _), actor_metrics = jax.lax.scan(
            actor_sgd_step,
            (training_state, training_key),
            transitions,
            length=num_actor_updates,
        )
        metrics = {**model_metrics, **critic_metrics, **actor_metrics}
        metrics["buffer_current_size"] = model_replay_buffer.size(model_buffer_state)
        metrics |= env_state.metrics
        return (
            training_state,
            env_state,
            model_buffer_state,
            initial_sac_buffer_state,
            metrics,
        )

    return training_step
