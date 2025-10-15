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

"""SBSRL networks."""

from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

import brax.training.agents.sac.networks as sac_networks
import flax
import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.sac.networks import MLP, BroNet

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

make_inference_fn = sac_networks.make_inference_fn
NetworkType = TypeVar("NetworkType", covariant=True)


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        *,
        n_critics: int = 2,
        n_heads: int = 1,
        safe: bool = False,
        use_bro: bool = True,
        ensemble_size: int = 10,
        embedding_dim: int = 4,
    ) -> NetworkType:
        pass


@flax.struct.dataclass
class SBSRLNetworks:
    policy_network: networks.FeedForwardNetwork
    qr_network: networks.FeedForwardNetwork
    qc_network: networks.FeedForwardNetwork | None
    model_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_world_model_ensemble(
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
        obs_state = obs if isinstance(obs, jnp.ndarray) else obs[obs_key]
        raw_output = model.apply(params, obs_state, actions)
        # Std devs also need to match the shape (B, E, feature_dim)
        diff_obs_raw, reward, cost = (
            raw_output[..., :obs_size],
            raw_output[..., obs_size],
            raw_output[..., obs_size + 1],
        )
        if isinstance(obs, dict):
            next_next_obs = {
                "state": diff_obs_raw + obs_state,
            }
            if "cumulative_cost" in obs:
                next_next_obs["cumulative_cost"] = obs["cumulative_cost"]
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


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


def make_q_network_ensemble(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    obs_key: str = "state",
    use_bro: bool = True,
    n_heads: int = 1,
    head_size: int = 1,
    ensemble_size: int = 10,
    embedding_dim: int = 4,
) -> networks.FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray, idx: jnp.ndarray):
            embed = linen.Embed(
                num_embeddings=ensemble_size,
                features=embedding_dim,
                name="ensemble_embed",
            )
            idx_emb = embed(idx)
            if idx_emb.ndim > obs.ndim:
                idx_emb = jnp.reshape(idx_emb, obs.shape[:-1] + (embedding_dim,))

            hidden = jnp.concatenate([obs, actions, idx_emb], axis=-1)

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

    def apply(processor_params, q_params, obs, actions, idx):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        idx = jnp.asarray(idx, dtype=jnp.int32)
        return q_module.apply(q_params, obs, actions, idx)

    obs_size = _get_obs_state_size(obs_size, obs_key)
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    dummy_idx = jnp.zeros((1,), dtype=jnp.int32)
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action, dummy_idx),
        apply=apply,
    )


def make_sbsrl_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    model_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    value_obs_key: str = "state",
    policy_obs_key: str = "state",
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
    safe: bool = False,
    ensemble_size: int = 10,
    embedding_dim: int = 4,
) -> SBSRLNetworks:
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    qr_network = make_q_network_ensemble(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
        ensemble_size=ensemble_size,
        embedding_dim=embedding_dim,
    )
    if safe:
        qc_network = make_q_network_ensemble(
            observation_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            obs_key=value_obs_key,
            use_bro=use_bro,
            n_critics=n_critics,
            n_heads=n_heads,
            ensemble_size=ensemble_size,
            embedding_dim=embedding_dim,
        )
        old_apply = qc_network.apply
        qc_network.apply = lambda *args, **kwargs: jnn.softplus(
            old_apply(*args, **kwargs)
        )
    else:
        qc_network = None
    model_network = make_world_model_ensemble(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        postprocess_observations_fn=postprocess_observations_fn,
        hidden_layer_sizes=model_hidden_layer_sizes,
        activation=activation,
    )
    return SBSRLNetworks(
        policy_network=policy_network,
        qr_network=qr_network,
        qc_network=qc_network,
        model_network=model_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
