"""Train PPO on PandaPickCubeCartesianExtended from pixels via SS2R env factory.

This script follows the Madrona-MJX tutorial flow:
1) Build a vision environment with domain randomization.
2) Train a vision PPO policy.
3) Print reward metrics during training.

The environment is always created with:
`ss2r.benchmark_suites.make_mujoco_playground_envs`.
"""

import functools
from datetime import datetime
from pathlib import Path
from typing import Any

import brax.training.agents.ppo.train as brax_ppo_train
from absl import app, flags
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from flax import linen
from ml_collections import config_dict
from mujoco_playground.config import manipulation_params

from ss2r import benchmark_suites
from ss2r.benchmark_suites.mujoco_playground.pick_cartesian import (
    pick_cartesian as pick_cartesian_task,
)

_ENV_NAME = "PandaPickCubeCartesianExtended"
_BASE_CONFIG_ENV_NAME = "PandaPickCubeCartesian"

_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of parallel environments")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 7_000_000, "Total environment steps for PPO training"
)
_SEED = flags.DEFINE_integer("seed", 0, "PRNG seed")
_SAVE_CHECKPOINT_PATH = flags.DEFINE_string(
    "save_checkpoint_path",
    "franka_experiments/checkpoints/pick_cartesian_vision_ppo",
    "Directory to save Brax PPO checkpoints. Set empty string to disable.",
)
_TRAIN_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "train_domain_randomization", True, "Enable domain randomization for training"
)
_EVAL_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "eval_domain_randomization", True, "Enable domain randomization for evaluation"
)
_RENDER_WIDTH = flags.DEFINE_integer("render_width", 64, "Render width")
_RENDER_HEIGHT = flags.DEFINE_integer("render_height", 64, "Render height")
_USE_RASTERIZER = flags.DEFINE_boolean(
    "use_rasterizer", False, "Use rasterizer backend for rendering"
)


def _make_ppo_networks_vision_ckpt_compatible(
    observation_size,
    action_size,
    preprocess_observations_fn,
    policy_hidden_layer_sizes=(256, 256),
    value_hidden_layer_sizes=(256, 256),
    normalise_channels=False,
    policy_obs_key="",
    value_obs_key="",
    activation_name: str = "relu",
):
    # FIXME: Brax checkpoint.network_config rejects non-default `activation` kwargs.
    # Route activation through `activation_name` so checkpoints can be saved while
    # still building the ReLU network used in this experiment.
    activation = getattr(linen, activation_name, None)
    if activation is None:
        raise ValueError(f"Unsupported activation_name={activation_name!r}")
    return ppo_networks_vision.make_ppo_networks_vision(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        normalise_channels=normalise_channels,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
    )


def _set_nested(cfg: config_dict.ConfigDict, key: str, value: Any) -> None:
    keys = key.split(".")
    node = cfg
    for part in keys[:-1]:
        node = node[part]
    node[keys[-1]] = value


def _build_env_and_cfg():
    num_envs = _NUM_ENVS.value
    env_cfg = pick_cartesian_task.default_config()
    episode_length = int(4 / env_cfg.ctrl_dt)

    overrides = {
        "use_ball": False,
        "use_x": False,
        "episode_length": episode_length,
        "vision": True,
        "obs_noise.brightness": [0.75, 2.0],
        "vision_config.use_rasterizer": _USE_RASTERIZER.value,
        "vision_config.render_batch_size": num_envs,
        "vision_config.render_width": _RENDER_WIDTH.value,
        "vision_config.render_height": _RENDER_HEIGHT.value,
        "box_init_range": 0.1,
        "action_history_length": 5,
        "success_threshold": 0.03,
    }
    for k, v in overrides.items():
        _set_nested(env_cfg, k, v)

    cfg = config_dict.ConfigDict()
    cfg.environment = config_dict.ConfigDict()
    cfg.environment.domain_name = "mujoco_playground"
    cfg.environment.task_name = _ENV_NAME
    cfg.environment.task_params = env_cfg.to_dict()
    cfg.environment.train_params = config_dict.ConfigDict()
    cfg.environment.eval_params = config_dict.ConfigDict()

    cfg.agent = config_dict.ConfigDict()
    cfg.agent.use_vision = True

    cfg.training = config_dict.ConfigDict()
    cfg.training.seed = _SEED.value
    cfg.training.num_envs = num_envs
    cfg.training.num_eval_envs = num_envs
    cfg.training.train_domain_randomization = _TRAIN_DOMAIN_RANDOMIZATION.value
    cfg.training.eval_domain_randomization = _EVAL_DOMAIN_RANDOMIZATION.value
    cfg.training.episode_length = episode_length
    cfg.training.action_repeat = 1
    cfg.training.hard_resets = False
    cfg.training.nonepisodic = False
    cfg.training.action_delay = config_dict.ConfigDict(
        {"enable": False, "max_delay": 0}
    )

    train_env, _ = benchmark_suites.make_mujoco_playground_envs(
        cfg, lambda env: env, lambda env: env
    )
    return train_env, episode_length


def main(argv):
    del argv

    print(f"Using Brax PPO trainer from: {brax_ppo_train.__file__}")
    save_checkpoint_path = None
    if _SAVE_CHECKPOINT_PATH.value:
        save_checkpoint_path = str(
            Path(_SAVE_CHECKPOINT_PATH.value).expanduser().resolve()
        )
        Path(save_checkpoint_path).mkdir(parents=True, exist_ok=True)
    print(
        "Checkpoint saving: "
        f"{save_checkpoint_path if save_checkpoint_path else 'disabled'}"
    )

    train_env, episode_length = _build_env_and_cfg()
    num_envs = _NUM_ENVS.value

    network_factory = functools.partial(
        _make_ppo_networks_vision_ckpt_compatible,
        policy_hidden_layer_sizes=[256, 256],
        value_hidden_layer_sizes=[256, 256],
        activation_name="relu",
        normalise_channels=True,
    )

    ppo_params = manipulation_params.brax_vision_ppo_config(_BASE_CONFIG_ENV_NAME)
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
    ppo_params.num_envs = num_envs
    ppo_params.num_eval_envs = num_envs
    ppo_params.episode_length = episode_length
    ppo_params.action_repeat = 1
    del ppo_params.network_factory
    ppo_params.network_factory = network_factory

    times = [datetime.now()]

    def progress(num_steps, metrics):
        if "eval/episode_reward" in metrics:
            print(
                f"{num_steps}: eval/episode_reward={metrics['eval/episode_reward']:.3f} "
                f"+- {metrics.get('eval/episode_reward_std', 0.0):.3f}"
            )
        times.append(datetime.now())

    train_kwargs = dict(ppo_params)
    train_kwargs.update(
        {
            "augment_pixels": True,
            "wrap_env": False,
            "madrona_backend": True,
            "progress_fn": progress,
            "seed": _SEED.value,
            "save_checkpoint_path": save_checkpoint_path,
        }
    )
    train_fn = functools.partial(brax_ppo_train.train, **train_kwargs)

    _ = train_fn(environment=train_env, eval_env=None)
    if len(times) > 1:
        print(f"time to jit: {times[1] - times[0]}")
        print(f"time to train: {times[-1] - times[1]}")


if __name__ == "__main__":
    app.run(main)
