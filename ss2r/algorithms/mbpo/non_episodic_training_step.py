from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.types import PRNGKey

from ss2r.algorithms.mbpo.types import TrainingState, TrainingStepFn
from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    Transition,
    float32,
)


def make_non_episodic_training_step(
    env,
    make_rollout_policy,
    get_rollout_policy_params,
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
    safety_budget,
    qc_netwrok,
) -> TrainingStepFn:
    def critic_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        transitions = float32(transitions)
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        critic_loss, behavior_qr_params, behavior_qr_optimizer_state = critic_update(
            training_state.behavior_qr_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_target_qr_params,
            alpha,
            transitions,
            key_critic,
            reward_q_transform,
            optimizer_state=training_state.behavior_qr_optimizer_state,
            params=training_state.behavior_qr_params,
        )
        if safe:
            # If the cost is greater than 1, we haven't reached our
            # goal safe set yet.
            cost = transitions.extras["state_extras"]["cost"]
            new_discount = jnp.where(
                cost > 0.0,
                jnp.ones_like(transitions.discount),
                jnp.zeros_like(transitions.discount),
            )
            backup_transitions = transitions._replace(discount=new_discount)
            cost_metrics = {}
            (
                backup_cost_critic_loss,
                backup_qc_params,
                backup_qc_optimizer_state,
            ) = cost_critic_update(
                training_state.backup_qc_params,
                training_state.backup_policy_params,
                training_state.normalizer_params,
                training_state.backup_target_qc_params,
                alpha,
                backup_transitions,
                key_critic,
                cost_q_transform,
                True,
                optimizer_state=training_state.backup_qc_optimizer_state,
                params=training_state.backup_qc_params,
            )
            time_to_recovery = qc_netwrok.apply(
                training_state.normalizer_params,
                backup_qc_params,
                transitions.observation,
                transitions.action,
            ).mean()
            cost_metrics["backup_cost_critic_loss"] = backup_cost_critic_loss
            cost_metrics["time_to_recovery"] = time_to_recovery
        else:
            cost_metrics = {}
            backup_qc_params = training_state.backup_qc_params
            backup_qc_optimizer_state = training_state.backup_qc_optimizer_state
        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_behavior_target_qr_params = polyak(
            training_state.behavior_target_qr_params, behavior_qr_params
        )
        if safe:
            new_backup_target_qc_params = polyak(
                training_state.backup_target_qc_params, backup_qc_params
            )
        else:
            new_backup_target_qc_params = training_state.backup_target_qc_params
        metrics = {
            "critic_loss": critic_loss,
            "fraction_done": 1.0 - transitions.discount.mean(),
            **cost_metrics,
        }
        # Note that we store the qc of the backup also as the
        # qc of behavior, this is so that later it
        # will get save correctly to the checkpoint.
        new_training_state = training_state.replace(  # type: ignore
            behavior_qr_optimizer_state=behavior_qr_optimizer_state,
            behavior_qr_params=behavior_qr_params,
            behavior_target_qr_params=new_behavior_target_qr_params,
            backup_qc_optimizer_state=backup_qc_optimizer_state,
            backup_qc_params=backup_qc_params,
            behavior_qc_optimizer_state=backup_qc_optimizer_state,
            behavior_qc_params=backup_qc_params,
            backup_target_qc_params=new_backup_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
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
        (actor_loss, _), policy_params, policy_optimizer_state = actor_update(
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_qr_params,
            training_state.behavior_qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            None,
            None,
            optimizer_state=training_state.behavior_policy_optimizer_state,
            params=training_state.behavior_policy_params,
        )
        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
        }
        new_training_state = training_state.replace(  # type: ignore
            behavior_policy_optimizer_state=policy_optimizer_state,
            behavior_policy_params=policy_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
        )
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
            sac_replay_buffer,
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
        (
            training_state,
            env_state,
            sac_buffer_state,
            training_key,
        ) = run_experience_step(training_state, env_state, sac_buffer_state, key)
        # Train SAC
        sac_buffer_state, sac_transitions = sac_replay_buffer.sample(sac_buffer_state)
        # FIXME (yarden): this is not the original plan.
        # As long as there are no interventions, agent should continue.
        discount = 1.0 - sac_transitions.extras["policy_extras"]["intervention"]
        sac_transitions = sac_transitions._replace(discount=discount)
        critic_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (critic_grad_updates_per_step, -1) + x.shape[1:]),
            sac_transitions,
        )
        (training_state, _), critic_metrics = jax.lax.scan(
            critic_sgd_step, (training_state, training_key), critic_transitions
        )
        num_actor_updates = -(
            -critic_grad_updates_per_step // num_critic_updates_per_actor_update
        )
        assert num_actor_updates > 0, "Actor updates is non-positive"
        actor_transitions = jax.tree_util.tree_map(
            lambda x: x[:num_actor_updates], sac_transitions
        )
        (training_state, _), actor_metrics = jax.lax.scan(
            actor_sgd_step,
            (training_state, training_key),
            actor_transitions,
            length=num_actor_updates,
        )
        metrics = {**critic_metrics, **actor_metrics}
        metrics["buffer_current_size"] = sac_replay_buffer.size(sac_buffer_state)
        metrics |= env_state.metrics
        return (
            training_state,
            env_state,
            model_buffer_state,
            sac_buffer_state,
            metrics,
        )

    return training_step
