import functools
from typing import Any, Callable, Optional, Tuple, TypeAlias

import flax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import Params

from ss2r.algorithms.penalizers import get_penalizer
from ss2r.algorithms.ppo import losses as ppo_losses

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: tuple[optax.OptState, optax.OptState, Optional[optax.OptState]]
    params: ppo_losses.SafePPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    penalizer_params: Params
    env_steps: jnp.ndarray


Metrics: TypeAlias = types.Metrics


TrainingStep: TypeAlias = Callable[
    [
        Tuple[TrainingState, envs.State, types.PRNGKey, int],
    ],
    Tuple[Tuple[TrainingState, envs.State, types.PRNGKey], Metrics],
]

TrainingStepFactory: Any

InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]


def get_train_fn(cfg, checkpoint_path, restore_checkpoint_path):
    import jax.nn as jnn

    from ss2r.algorithms.ppo import networks as ppo_networks
    from ss2r.algorithms.ppo import train as ppo

    agent_cfg = dict(cfg.agent)
    training_cfg = {
        k: v
        for k, v in cfg.training.items()
        if k
        not in [
            "render_episodes",
            "train_domain_randomization",
            "eval_domain_randomization",
            "render",
            "store_checkpoint",
            "wandb_id",
            "hard_resets",
            "nonepisodic",
            "action_delay",
        ]
    }
    policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
    value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
    activation = getattr(jnn, agent_cfg.pop("activation"))
    value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
    policy_obs_key = "privileged_state" if cfg.training.policy_privileged else "state"
    del training_cfg["value_privileged"]
    del training_cfg["policy_privileged"]
    del agent_cfg["name"]
    if "cost_robustness" in agent_cfg:
        del agent_cfg["cost_robustness"]
    if "reward_robustness" in agent_cfg:
        del agent_cfg["reward_robustness"]
    if "penalizer" in agent_cfg:
        del agent_cfg["penalizer"]
    if "propagation" in agent_cfg:
        del agent_cfg["propagation"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        value_obs_key=value_obs_key,
        policy_obs_key=policy_obs_key,
    )
    penalizer, penalizer_params = get_penalizer(cfg)
    train_fn = functools.partial(
        ppo.train,
        **agent_cfg,
        **training_cfg,
        network_factory=network_factory,
        restore_checkpoint_path=restore_checkpoint_path,
        penalizer=penalizer,
        penalizer_params=penalizer_params,
    )
    return train_fn
