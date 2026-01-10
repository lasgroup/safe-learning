import functools

import ss2r.algorithms.mbpo.networks as mbpo_networks
import ss2r.algorithms.mbpo.vision_networks as mbpo_vision_networks
from ss2r.algorithms.mbpo import non_episodic_training_step, on_policy_training_step
from ss2r.algorithms.penalizers import get_penalizer
from ss2r.algorithms.sac.data import get_collection_fn
from ss2r.algorithms.sac.q_transforms import (
    get_cost_q_transform,
    get_reward_q_transform,
)


def get_training_step_fn(cfg):
    if cfg.agent.training_step_fn == "on_policy":
        return on_policy_training_step.make_on_policy_training_step
    if cfg.agent.training_step_fn == "nonepisodic":
        return non_episodic_training_step.make_non_episodic_training_step
    else:
        raise ValueError(f"Unknown training_step_fn: {cfg.agent.training_step_fn}")


def get_train_fn(cfg, checkpoint_path, restore_checkpoint_path):
    import jax.nn as jnn

    import ss2r.algorithms.mbpo.train as mbpo

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
            "nonepisodic",
            "action_delay",
        ]
    }
    policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
    value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
    model_hidden_layer_sizes = agent_cfg.pop("model_hidden_layer_sizes")
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
    if "training_step_fn" in agent_cfg:
        del agent_cfg["training_step_fn"]
    if "pessimistic_q" in agent_cfg:
        del agent_cfg["pessimistic_q"]
    if "use_vision" in agent_cfg and agent_cfg["use_vision"]:
        network_factory = functools.partial(
            mbpo_vision_networks.make_mbpo_vision_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            encoder_hidden_dim=50,
            tanh=True,
        )
        del agent_cfg["use_vision"]
    else:
        value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
        policy_obs_key = (
            "privileged_state" if cfg.training.policy_privileged else "state"
        )
        network_factory = functools.partial(
            mbpo_networks.make_mbpo_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            model_hidden_layer_sizes=model_hidden_layer_sizes,
            activation=activation,
            value_obs_key=value_obs_key,
            policy_obs_key=policy_obs_key,
        )
    penalizer, penalizer_params = get_penalizer(cfg)
    reward_q_transform = get_reward_q_transform(cfg)
    cost_q_transform = get_cost_q_transform(cfg)
    data_collection = get_collection_fn(cfg)
    training_step_fn = get_training_step_fn(cfg)
    train_fn = functools.partial(
        mbpo.train,
        **agent_cfg,
        **training_cfg,
        network_factory=network_factory,
        checkpoint_logdir=checkpoint_path,
        reward_q_transform=reward_q_transform,
        cost_q_transform=cost_q_transform,
        penalizer=penalizer,
        penalizer_params=penalizer_params,
        get_experience_fn=data_collection,
        make_training_step_fn=training_step_fn,
        restore_checkpoint_path=restore_checkpoint_path,
    )
    return train_fn
