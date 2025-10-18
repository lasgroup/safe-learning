from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from brax.training.types import Transition


def get_reward_q_transform(cfg):
    pessimistic_q = True
    if cfg.agent.use_bro:
        pessimistic_q = cfg.agent.pessimistic_q
    return SACBaseEnsemble(pessimistic_q=pessimistic_q)


def get_cost_q_transform(cfg):
    if (
        "cost_robustness" not in cfg.agent
        or cfg.agent.cost_robustness is None
        or cfg.agent.cost_robustness.name == "neutral"
    ):
        return SACCostEnsemble()
    else:
        raise ValueError("Unknown robustness")


class QTransformation(Protocol):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        alpha: jax.Array | None = None,
        scale: float = 1.0,
        key: jax.Array | None = None,
        safe: bool = False,
        uncertainty_constraint: bool = False,
    ):
        ...


class SACBaseEnsemble(QTransformation):
    def __init__(self, pessimistic_q: bool = True) -> None:
        self.pessimistic_q = pessimistic_q

    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        alpha: jax.Array | None = None,
        scale: float = 1.0,
        key: jax.Array | None = None,
        safe: bool = False,
        uncertainty_constraint: bool = False,
    ):
        next_action, next_log_prob = policy(transitions.next_observation)
        next_q = q_fn(
            transitions.next_observation,
            next_action,
            transitions.extras["state_extras"]["idx"],
        )
        if not self.pessimistic_q:
            next_v = next_q.mean(axis=-1)
        else:
            next_v = next_q.min(axis=-1)
        next_v -= alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * scale + transitions.discount * gamma * next_v
        )
        return target_q


class SACCostEnsemble(QTransformation):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        alpha: jax.Array | None = None,
        scale: float = 1.0,
        key: jax.Array | None = None,
        safe: bool = False,
        uncertainty_constraint: bool = False,
        n_critics: int = 2,
    ):
        next_action, _ = policy(transitions.next_observation)
        next_q = q_fn(
            transitions.next_observation,
            next_action,
            transitions.extras["state_extras"]["idx"],
        )

        qc_head_size = int(safe) + int(uncertainty_constraint)
        next_q = next_q.reshape(-1, n_critics, qc_head_size)
        next_v = next_q.mean(axis=1)
        discount = jnp.expand_dims(transitions.discount, -1)
        stage_value = []
        if safe:
            cost = transitions.extras["state_extras"]["cost"]
            stage_value.append(cost)
        if uncertainty_constraint:
            disagreement = transitions.extras["state_extras"]["disagreement"]
            stage_value.append(disagreement)
        stage_value_vec = jnp.stack(stage_value, axis=-1)
        target_q = jax.lax.stop_gradient(
            stage_value_vec * scale + discount * gamma * next_v
        )
        return target_q
