from typing import Mapping, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.mbpo.networks import MBPONetworks, make_world_model_ensemble
from ss2r.algorithms.sac.networks import (
    MLP,
    ActivationFn,
    BroNet,
    SafeSACNetworks,
)
from ss2r.algorithms.sac.vision_networks import Encoder


def make_vision_world_model_ensemble(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (512, 512),
    activation: networks.ActivationFn = linen.swish,
    obs_key: str = "state",
) -> networks.FeedForwardNetwork:
    # Convert obs_size to integer if it's a shape tuple
    if isinstance(obs_size, (tuple, list)):
        obs_size = obs_size[0] if len(obs_size) == 1 else sum(obs_size)
    obs_size = int(obs_size)

    class MModule(linen.Module):
        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            net = networks.MLP
            single_output_dim = obs_size + 2  # +2 for reward and cost
            output_dim = single_output_dim
            hidden_dims = list(hidden_layer_sizes) + [output_dim]
            out = net(
                layer_sizes=hidden_dims,
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            return out

    model = MModule(obs_size=obs_size)

    def apply(preprocessor_params, params, obs, actions):
        obs = preprocess_observations_fn(obs, preprocessor_params)
        obs_state = obs if isinstance(obs, jax.Array) else obs[obs_key]
        raw_output = model.apply(params, obs_state, actions)
        # Std devs also need to match the shape (B, E, feature_dim)
        diff_obs_raw, reward, cost = (
            raw_output[..., :obs_size],
            raw_output[..., obs_size],
            raw_output[..., obs_size + 1],
        )
        if isinstance(obs, Mapping):
            next_next_obs = {
                "state": diff_obs_raw + obs_state,
                "cumulative_cost": obs["cumulative_cost"],
            }
        else:
            next_next_obs = diff_obs_raw + obs_state
        obs = postprocess_observations_fn(next_next_obs, preprocessor_params)
        return obs, reward, cost

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    net = networks.FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), apply=apply
    )
    return net


def make_policy_vision_network(
    observation_size: Mapping[str, Tuple[int, ...]],
    output_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    state_obs_key: str = "",
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
):
    class Policy(linen.Module):
        @linen.compact
        def __call__(self, obs):
            # Create dummy encoder so that it's easier to load the
            # checkpoint
            hidden = Encoder(name="SharedEncoder")(
                {"pixels/view_0": np.zeros((1, 64, 64, 3))}
            )
            hidden = linen.Dense(encoder_hidden_dim)(hidden)
            linen.LayerNorm()(hidden)
            if isinstance(obs, Mapping):
                hidden = obs["state"]
            else:
                hidden = obs
            if tanh:
                hidden = jnn.tanh(hidden)
            outs = networks.MLP(
                layer_sizes=list(hidden_layer_sizes) + [output_size],
                activation=activation,
            )(hidden)
            return outs

    pi_module = Policy()

    def apply(processor_params, params, obs):
        if state_obs_key:
            state_obs = preprocess_observations_fn(
                obs[state_obs_key],
                networks.normalizer_select(processor_params, state_obs_key),
            )
            obs = {**obs, state_obs_key: state_obs}
        return pi_module.apply(params, obs)

    if isinstance(observation_size, Mapping):
        observation_size = observation_size["state"]  # type: ignore
    dummy_obs = jnp.zeros((1, observation_size))
    return networks.FeedForwardNetwork(
        init=lambda key: pi_module.init(key, dummy_obs), apply=apply
    )


def make_q_vision_network(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    n_critics: int = 2,
    state_obs_key: str = "",
    use_bro: bool = True,
    n_heads: int = 1,
    head_size: int = 1,
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
):
    class QModule(linen.Module):
        n_critics: int

        @linen.compact
        def __call__(self, obs, actions):
            # Create dummy encoder so that it's easier to load the
            # checkpoint
            hidden = Encoder(name="SharedEncoder")(
                {"pixels/view_0": np.zeros((1, 64, 64, 3))}
            )
            hidden = linen.Dense(encoder_hidden_dim)(hidden)
            linen.LayerNorm()(hidden)
            if isinstance(obs, Mapping):
                hidden = obs["state"]
            else:
                hidden = obs
            if tanh:
                hidden = jnn.tanh(hidden)
            hidden = jnp.concatenate([hidden, actions], axis=-1)
            res = []
            net = BroNet if use_bro else MLP
            for _ in range(self.n_critics):
                q = net(  # type: ignore
                    layer_sizes=list(hidden_layer_sizes) + [head_size],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                    num_heads=n_heads,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, params, obs, actions):
        if state_obs_key:
            state_obs = preprocess_observations_fn(
                obs[state_obs_key],
                networks.normalizer_select(processor_params, state_obs_key),
            )
            obs = {**obs, state_obs_key: state_obs}
        return q_module.apply(params, obs, actions)

    if isinstance(observation_size, Mapping):
        observation_size = observation_size["state"]  # type: ignore
    dummy_obs = jnp.zeros((1, observation_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_mbpo_vision_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    activation: ActivationFn = linen.swish,
    state_obs_key: str = "",
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    model_hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
    *,
    safe: bool = False,
) -> SafeSACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = make_policy_vision_network(
        observation_size=observation_size,
        output_size=parametric_action_distribution.param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        state_obs_key=state_obs_key,
        encoder_hidden_dim=encoder_hidden_dim,
        tanh=tanh,
    )
    qr_network = make_q_vision_network(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
        encoder_hidden_dim=encoder_hidden_dim,
        tanh=tanh,
    )
    if safe:
        qc_network = make_q_vision_network(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            use_bro=use_bro,
            n_critics=n_critics,
            n_heads=n_heads,
            encoder_hidden_dim=encoder_hidden_dim,
            tanh=tanh,
        )
        old_apply = qc_network.apply
        qc_network.apply = lambda *args, **kwargs: jnn.softplus(
            old_apply(*args, **kwargs)
        )
    else:
        qc_network = None
    model_network = make_world_model_ensemble(
        encoder_hidden_dim,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        postprocess_observations_fn=postprocess_observations_fn,
        hidden_layer_sizes=model_hidden_layer_sizes,
        activation=activation,
    )
    return MBPONetworks(
        policy_network=policy_network,
        qr_network=qr_network,
        qc_network=qc_network,
        model_network=model_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
