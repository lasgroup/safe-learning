import functools
import logging
from pathlib import Path

import hydra
import jax
from brax import envs
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import mbpo, ppo, sac, sbsrl
from ss2r.common.logging import TrainingLogger
from ss2r.common.wandb import get_state_path, get_wandb_checkpoint

_LOG = logging.getLogger(__name__)


def locate_last_checkpoint() -> Path | None:
    ckpt_dir = Path(get_state_path())
    # Get all directories or files that match the 12-digit pattern
    checkpoints = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 12
    ]
    if not checkpoints:
        return None  # No checkpoints found
    # Sort by step number (converted from the directory name)
    latest_ckpt = max(checkpoints, key=lambda p: int(p.name))
    return latest_ckpt


def _validate_madrona_args(
    train_env: envs.Env,
    eval_env: envs.Env,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    num_render_envs: int,
):
    """Validates arguments for Madrona-MJX."""
    if train_env != eval_env:
        raise ValueError("Madrona-MJX requires a fixed environment")
    if num_eval_envs != num_envs != num_render_envs:
        raise ValueError("Madrona-MJX requires a fixed batch size")
    if action_repeat != 1:
        raise ValueError(
            "Implement action_repeat using PipelineEnv's _n_frames to avoid"
            " unnecessary rendering!"
        )


def get_train_fn(cfg):
    if cfg.training.wandb_id:
        restore_checkpoint_path = get_wandb_checkpoint(
            cfg.training.wandb_id, cfg.wandb.entity
        )
    else:
        restore_checkpoint_path = None
    if cfg.agent.name == "sac":
        train_fn = sac.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "ppo":
        train_fn = ppo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "mbpo":
        train_fn = mbpo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "sbsrl":
        train_fn = sbsrl.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


class Counter:
    def __init__(self):
        self.count = 0


def report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_fn = get_train_fn(cfg)
    train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(cfg)
    use_vision = "use_vision" in cfg.agent and cfg.agent.use_vision
    train_env, eval_env = benchmark_suites.make(
        cfg, train_env_wrap_fn, eval_env_wrap_fn
    )
    if use_vision:
        _validate_madrona_args(
            train_env,
            eval_env,
            cfg.training.num_envs,
            cfg.training.num_eval_envs,
            cfg.training.action_repeat,
            cfg.environment.task_params.vision_config.render_batch_size,
        )
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=train_env,
            eval_env=eval_env,
            progress_fn=functools.partial(report, logger, steps),
        )
        if cfg.training.render:
            rng = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), cfg.training.num_eval_envs
            )
            # TODO (yarden): write a function that handles this
            # more cleanly.
            if len(params) != 2:
                policy_params = params[:2]
            else:
                policy_params = params
            if cfg.agent.name == "mbpo" and cfg.agent.safety_filter is not None:
                budget = (
                    (
                        cfg.training.safety_budget
                        / (1 - cfg.agent.safety_discounting)
                        / cfg.training.episode_length
                        * cfg.training.action_repeat
                    )
                    if cfg.agent.safety_discounting != 1.0
                    else cfg.training.safety_budget
                )
                policy_params = (
                    params[0],
                    (
                        params[1],
                        params[3],
                        budget,
                    ),
                )
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(policy_params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            fps = (
                1 / cfg.environment.task_params.ctrl_dt
                if "task_params" in cfg.environment
                and "ctrl_dt" in cfg.environment.task_params
                else 30.0
            )
            logger.log_video(
                video,
                steps.count,
                "eval/video",
                fps,
            )
        if cfg.training.store_checkpoint:
            artifacts = locate_last_checkpoint()
            if artifacts:
                logger.log_artifact(artifacts, "model", "checkpoint")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
