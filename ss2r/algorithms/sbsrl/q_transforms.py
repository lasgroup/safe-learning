from typing import Callable, Protocol

import jax
from brax.training.types import Transition


def get_reward_q_transform(cfg):
    pessimistic_q = True
    if cfg.agent.use_bro:
        pessimistic_q = cfg.agent.pessimistic_q
    return SACBaseEnsemble(pessimistic_q=pessimistic_q)


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
    ):
        next_action, next_log_prob = policy(transitions.next_observation)  # TODO:
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
