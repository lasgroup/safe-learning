from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.types import Policy, PRNGKey, Transition


def safe_actor_step(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[envs.State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    intervetion = policy_extras.get("intervention", jnp.zeros_like(nstate.reward))
    if "intervention" in nstate.info:
        nstate.info["intervention"] = intervetion
        nstate.info["policy_distance"] = policy_extras.get(
            "policy_distance", jnp.zeros_like(nstate.reward)
        )
        nstate.info["safety_gap"] = policy_extras.get(
            "safety_gap", jnp.zeros_like(nstate.reward)
        )
        nstate.info["expected_total_cost"] = policy_extras.get(
            "expected_total_cost", jnp.zeros_like(nstate.reward)
        )
        nstate.info["cumulative_cost"] = policy_extras.get(
            "cumulative_cost", jnp.zeros_like(nstate.reward)
        )
        nstate.info["q_c"] = policy_extras.get("q_c", jnp.zeros_like(nstate.reward))
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


def generate_safe_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[envs.State, Transition]:
    """Collect trajectories of given unroll_length."""

    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)

        nstate, transition = safe_actor_step(
            env, state, policy, current_key, extra_fields=extra_fields
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, data
