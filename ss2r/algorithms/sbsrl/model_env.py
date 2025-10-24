import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base

from ss2r.algorithms.sac.types import float32


class ModelBasedEnv(envs.Env):
    """Environment wrapper that uses a learned model for predictions."""

    def __init__(
        self,
        transitions,
        observation_size,
        action_size,
        sbsrl_network,
        training_state,
        ensemble_selection="mean",
        safety_budget=float("inf"),
        cost_discount=1.0,
        scaling_fn=lambda x: x,
        use_termination=True,
        safety_filter="sooper",
        initial_normalizer_params=None,
    ):
        super().__init__()
        self.model_network = sbsrl_network.model_network
        self.model_params = training_state.model_params
        self.qc_network = sbsrl_network.qc_network
        self.backup_qc_params = training_state.backup_qc_params
        self.qr_network = sbsrl_network.qr_network
        self.backup_qr_params = training_state.backup_qr_params
        self.policy_network = sbsrl_network.policy_network
        self.backup_policy_params = training_state.backup_policy_params
        self.normalizer_params = training_state.normalizer_params
        self.ensemble_selection = ensemble_selection
        self.safety_budget = safety_budget
        self._observation_size = observation_size
        self._action_size = action_size
        self.transitions = transitions
        self.cost_discount = cost_discount
        self.scaling_fn = scaling_fn
        self.use_termination = use_termination
        self.safety_filter = safety_filter
        self.initial_normalizer_params = (
            initial_normalizer_params if initial_normalizer_params is not None else {}
        )

    def reset(self, rng: jax.Array) -> base.State:
        sample_key, model_key = jax.random.split(rng)
        indcs = jax.random.randint(sample_key, (), 0, self.transitions.reward.shape[0])
        transitions = float32(
            jax.tree_util.tree_map(lambda x: x[indcs], self.transitions)
        )
        state = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=jnp.zeros_like(transitions.reward),
            info={
                "truncation": jnp.zeros_like(transitions.reward),
                "cost": transitions.extras["state_extras"].get(
                    "cost", jnp.zeros_like(transitions.reward)
                ),
                "disagreement": transitions.extras["state_extras"].get(
                    "disagreement", jnp.zeros_like(transitions.reward)
                ),
                "key": model_key,
            },
        )
        return state

    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Predict next state, reward, and cost using the model
        key = state.info["key"]
        sample_key, key = jax.random.split(key)
        next_obs, reward, cost = _propagate_ensemble(
            self.model_network.apply,
            self.normalizer_params,
            self.model_params,
            state.obs,
            action,
            self.ensemble_selection,
            sample_key,
        )
        state.info["key"] = key
        done = jnp.zeros_like(reward, dtype=jnp.float32)
        truncation = jnp.zeros_like(reward, dtype=jnp.float32)
        state.info["cost"] = cost
        state.info["truncation"] = truncation
        disagreement = (
            next_obs.std(axis=0).mean(-1)
            if isinstance(next_obs, jax.Array)
            else next_obs["state"].std(axis=0).mean(-1)
        )
        state.info["disagreement"] = disagreement

        state = state.replace(
            obs=next_obs,
            reward=reward,
            done=done if self.use_termination else jnp.zeros_like(done),
            info=state.info,
        )
        return state

    @property
    def observation_size(self) -> int:
        """Return the size of the observation space."""
        return self._observation_size

    @property
    def action_size(self) -> int:
        """Return the size of the action space."""
        return self._action_size

    @property
    def backend(self) -> str:
        """Return the backend used by the environment."""
        return "model_based"


def create_model_env(
    transitions,
    sbsrl_network,
    training_state,
    observation_size,
    action_size,
    ensemble_selection="all",
    safety_budget=float("inf"),
    cost_discount=1.0,
    scaling_fn=lambda x: x,
    use_termination=True,
    safety_filter="sooper",
    initial_normalizer_params=None,
) -> ModelBasedEnv:
    """Factory function to create a model-based environment."""
    return ModelBasedEnv(
        transitions=transitions,
        sbsrl_network=sbsrl_network,
        training_state=training_state,
        observation_size=observation_size,
        action_size=action_size,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget,
        cost_discount=cost_discount,
        scaling_fn=scaling_fn,
        use_termination=use_termination,
        safety_filter=safety_filter,
        initial_normalizer_params=initial_normalizer_params,
    )


def _propagate_ensemble(
    pred_fn,
    normalizer_params,
    model_params,
    obs,
    action,
    ensemble_selection,
    key,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate the ensemble predictions based on the selection method."""
    # Calculate the nominal predictions
    if ensemble_selection == "nominal":
        # Get the average model parameters
        avg_model_params = jax.tree_util.tree_map(
            lambda p: jnp.mean(p, axis=0), model_params
        )
        next_obs, reward, cost = pred_fn(
            normalizer_params, avg_model_params, obs, action
        )
    elif ensemble_selection == "random":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        # Randomly select one of the ensemble predictions
        idx = jax.random.randint(key, (1,), 0, reward_pred.shape[0])[0]
        next_obs = jax.tree_map(lambda x: x[idx], next_obs_pred)
        reward = reward_pred[idx]
        cost = cost_pred[idx]
    elif ensemble_selection == "mean":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        next_obs = jax.tree_map(lambda x: jnp.mean(x, axis=0), next_obs_pred)
        reward = jnp.mean(reward_pred, axis=0)
        cost = jnp.mean(cost_pred, axis=0)
    elif ensemble_selection == "all":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs, reward, cost = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
    else:
        raise ValueError(f"Unknown ensemble selection: {ensemble_selection}")
    return next_obs, reward, cost
