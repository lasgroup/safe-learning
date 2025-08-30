import functools

from brax.training.replay_buffers import UniformSamplingQueue

import ss2r.algorithms.sac.networks as sac_networks
import ss2r.algorithms.sac.vision_networks as sac_vision_networks
from ss2r.algorithms.penalizers import get_penalizer
from ss2r.algorithms.sac.data import get_collection_fn
from ss2r.algorithms.sac.pytree_uniform_sampling_queue import PytreeUniformSamplingQueue
from ss2r.algorithms.sac.q_transforms import (
    get_cost_q_transform,
    get_reward_q_transform,
)
from ss2r.algorithms.sac.rae import RAEReplayBuffer


def _get_replay_buffer(cfg):
    if "replay_buffer" not in cfg.agent:
        return UniformSamplingQueue
    elif cfg.agent.replay_buffer.name == "pytree":
        return PytreeUniformSamplingQueue
    elif cfg.agent.replay_buffer.name == "rae":
        return functools.partial(
            RAEReplayBuffer,
            wandb_ids=cfg.agent.replay_buffer.wandb_ids,
            wandb_entity=cfg.wandb.entity,
            mix=cfg.agent.replay_buffer.mix,
        )


def get_train_fn(cfg, checkpoint_path, restore_checkpoint_path):
    import jax.nn as jnn

    import ss2r.algorithms.sac.train as sac

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
            "value_privileged",
            "policy_privileged",
            "wandb_id",
            "hard_resets",
        ]
    }
    policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
    value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
    activation = getattr(jnn, agent_cfg.pop("activation"))
    del agent_cfg["name"]
    if "cost_robustness" in agent_cfg:
        del agent_cfg["cost_robustness"]
    if "reward_robustness" in agent_cfg:
        del agent_cfg["reward_robustness"]
    if "penalizer" in agent_cfg:
        del agent_cfg["penalizer"]
    if "propagation" in agent_cfg:
        del agent_cfg["propagation"]
    if "data_collection" in agent_cfg:
        del agent_cfg["data_collection"]
    if "replay_buffer" in agent_cfg:
        del agent_cfg["replay_buffer"]
    if "use_vision" in agent_cfg and agent_cfg["use_vision"]:
        network_factory = functools.partial(
            sac_vision_networks.make_sac_vision_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            encoder_hidden_dim=agent_cfg["encoder_hidden_dim"],
            tanh=agent_cfg["tanh"],
        )
        del (
            agent_cfg["use_vision"],
            agent_cfg["encoder_hidden_dim"],
            agent_cfg["tanh"],
        )
        if "lambda_" in agent_cfg:
            del agent_cfg["lambda_"]
        if "state_obs_key" in agent_cfg:
            network_factory = functools.partial(
                network_factory,
                state_obs_key=agent_cfg["state_obs_key"],
            )
            del agent_cfg["state_obs_key"]
    else:
        value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
        policy_obs_key = (
            "privileged_state" if cfg.training.policy_privileged else "state"
        )
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            value_obs_key=value_obs_key,
            policy_obs_key=policy_obs_key,
        )
    penalizer, penalizer_params = get_penalizer(cfg)
    cost_q_transform = get_cost_q_transform(cfg)
    reward_q_transform = get_reward_q_transform(cfg)
    data_collection = get_collection_fn(cfg)
    replay_buffer_factory = _get_replay_buffer(cfg)
    train_fn = functools.partial(
        sac.train,
        **agent_cfg,
        **training_cfg,
        network_factory=network_factory,
        checkpoint_logdir=checkpoint_path,
        cost_q_transform=cost_q_transform,
        reward_q_transform=reward_q_transform,
        penalizer=penalizer,
        penalizer_params=penalizer_params,
        get_experience_fn=data_collection,
        replay_buffer_factory=replay_buffer_factory,
        restore_checkpoint_path=restore_checkpoint_path,
    )
    return train_fn
