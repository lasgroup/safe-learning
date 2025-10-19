import copy
import functools

import jax
import jax.numpy as jnp
from brax import envs
from mujoco_playground import dm_control_suite, locomotion, manipulation
from mujoco_playground._src.manipulation.franka_emika_panda.randomize_vision import (
    domain_randomize as franka_vision_randomize,
)

from ss2r.algorithms.mbpo.wrappers import TrackOnlineCostsInObservation, VisionWrapper
from ss2r.benchmark_suites import brax, mujoco_playground, safety_gym
from ss2r.benchmark_suites.brax.ant import ant
from ss2r.benchmark_suites.brax.cartpole import cartpole
from ss2r.benchmark_suites.brax.humanoid import humanoid
from ss2r.benchmark_suites.mujoco_playground.cartpole import cartpole as dm_cartpole
from ss2r.benchmark_suites.mujoco_playground.cartpole.spidr_cartpole import (
    VisionSPiDRCartpole,
)
from ss2r.benchmark_suites.mujoco_playground.go1_joystick import go1_joystick
from ss2r.benchmark_suites.mujoco_playground.go2_joystick import (
    getup,
    handstand,
    joystick,
)
from ss2r.benchmark_suites.mujoco_playground.humanoid import humanoid as dm_humanoid
from ss2r.benchmark_suites.mujoco_playground.humanoid import (
    nonepisodic as nonepisodic_humanoid,
)
from ss2r.benchmark_suites.mujoco_playground.pick_cartesian import pick_cartesian
from ss2r.benchmark_suites.mujoco_playground.quadruped import quadruped
from ss2r.benchmark_suites.mujoco_playground.walker import walker
from ss2r.benchmark_suites.rccar import rccar
from ss2r.benchmark_suites.safety_gym import go_to_goal
from ss2r.benchmark_suites.utils import get_domain_name, get_task_config
from ss2r.benchmark_suites.wrappers import (
    GoToGoalObservationWrapper,
    Saute,
    SPiDR,
    WalkerObservationWrapper,
    wrap,
)

locomotion.register_environment(
    "Go2JoystickFlatTerrain",
    functools.partial(joystick.Joystick, task="flat_terrain"),
    joystick.default_config,
)
locomotion.register_environment(
    "Go2JoystickRoughTerrain",
    functools.partial(joystick.Joystick, task="rough_terrain"),
    joystick.default_config,
)
locomotion.register_environment("Go2Getup", getup.Getup, getup.default_config)
locomotion.register_environment(
    "Go2Handstand", handstand.Handstand, handstand.default_config
)
locomotion.register_environment(
    "Go2Footstand", handstand.Footstand, handstand.default_config
)
manipulation.register_environment(
    "PandaPickCubeCartesianExtended",
    pick_cartesian.PandaPickCubeCartesian,
    pick_cartesian.default_config(),
)
dm_control_suite.register_environment(
    "NonEpisodicHumanoidStand",
    functools.partial(nonepisodic_humanoid.NonEpisodicHumanoid, move_speed=0.0),
    dm_control_suite.humanoid.default_config,
)
dm_control_suite.register_environment(
    "NonEpisodicHumanoidWalk",
    functools.partial(
        nonepisodic_humanoid.NonEpisodicHumanoid,
        move_speed=nonepisodic_humanoid.humanoid.WALK_SPEED,
    ),
    dm_control_suite.humanoid.default_config,
)
dm_control_suite.register_environment(
    "NonEpisodicHumanoidRun",
    functools.partial(
        nonepisodic_humanoid.NonEpisodicHumanoid,
        move_speed=nonepisodic_humanoid.humanoid.RUN_SPEED,
    ),
    dm_control_suite.humanoid.default_config,
)


def get_wrap_env_fn(cfg):
    if (
        cfg.environment.task_name == "SafeWalkerWalk"
        or cfg.environment.task_name == "SafeWalkerRun"
    ):

        def wrap_fn(env):
            env = WalkerObservationWrapper(env)
            return env

        out = wrap_fn, wrap_fn
    elif cfg.environment.task_name == "go_to_goal":

        def wrap_fn(env):
            env = GoToGoalObservationWrapper(env)
            return env

        out = wrap_fn, wrap_fn
    else:
        out = lambda env: env, lambda env: env
    if "propagation" not in cfg.agent:
        out = out[0], out[1]
    elif cfg.agent.propagation.name == "spidr":

        def fn(env):
            key = jax.random.PRNGKey(cfg.training.seed)
            env = out[0](env)
            env = SPiDR(
                env,
                prepare_randomization_fn(
                    key,
                    cfg.agent.propagation.num_envs,
                    cfg.environment.train_params,
                    cfg.environment.task_name,
                ),
                cfg.agent.propagation.num_envs,
                cfg.agent.propagation.lambda_,
                cfg.agent.propagation.alpha,
            )
            return env

        return fn, lambda env: out[1](env)
    else:
        raise ValueError("Propagation method not provided.")
    if "penalizer" in cfg.agent and cfg.agent.penalizer.name == "saute":

        def saute_train(env):
            env = out[0](env)
            env = Saute(
                env,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                cfg.agent.penalizer.penalty,
                cfg.agent.penalizer.terminate,
                cfg.agent.penalizer.termination_probability,
            )
            return env

        def saute_eval(env):
            env = out[1](env)
            env = Saute(
                env,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                0.0,
                False,
            )
            return env

        return saute_train, saute_eval

    if cfg.agent.name == "mbpo":

        def safe_mbpo_train(env):
            env = out[0](env)
            use_safety_filter = (
                cfg.training.safe and cfg.agent.safety_filter is not None
            )
            if "use_vision" in cfg.agent and cfg.agent.use_vision:
                env = VisionWrapper(
                    env, cfg.training.wandb_id, cfg.wandb.entity, use_safety_filter
                )
            if use_safety_filter:
                env = TrackOnlineCostsInObservation(env)
            return env

        def safe_mbpo_eval(env):
            env = out[1](env)
            use_safety_filter = (
                cfg.training.safe and cfg.agent.safety_filter is not None
            )
            if "use_vision" in cfg.agent and cfg.agent.use_vision:
                env = VisionWrapper(
                    env, cfg.training.wandb_id, cfg.wandb.entity, use_safety_filter
                )
            if use_safety_filter:
                env = TrackOnlineCostsInObservation(env)
            return env

        return safe_mbpo_train, safe_mbpo_eval
    return out


def make(cfg, train_wrap_env_fn=lambda env: env, eval_wrap_env_fn=lambda env: env):
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        return make_brax_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "rccar":
        return make_rccar_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "mujoco_playground":
        return make_mujoco_playground_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "safety_gym":
        return make_safety_gym_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn)
    elif domain_name == "cartpole_spidr_vision":
        return make_spidr_cartpole_vision(cfg, train_wrap_env_fn, eval_wrap_env_fn)


def prepare_randomization_fn(key, num_envs, cfg, task_name):
    randomize_fn = lambda sys, rng: randomization_fns[task_name](sys, rng, cfg)
    v_randomization_fn = functools.partial(
        randomize_fn, rng=jax.random.split(key, num_envs)
    )
    vf_randomization_fn = lambda sys: v_randomization_fn(sys)  # type: ignore
    return vf_randomization_fn


def make_rccar_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn, use_vision=False):
    if "use_vision" in cfg.agent and cfg.agent.use_vision:
        raise ValueError("RCCar does not support vision.")
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_params")
    eval_car_params = task_cfg.pop("eval_params")
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))

    action_delay, observation_delay = (
        task_cfg.pop("action_delay"),
        task_cfg.pop("observation_delay"),
    )
    sliding_window = task_cfg.pop("sliding_window")
    # Create train environment with built-in features
    train_env = rccar.RCCar(
        train_car_params["nominal"],
        action_delay=action_delay,
        observation_delay=observation_delay,
        sliding_window=sliding_window,
        **task_cfg,
    )
    train_env = train_wrap_env_fn(train_env)

    train_randomization_fn = (
        prepare_randomization_fn(
            train_key,
            cfg.training.num_envs,
            train_car_params,
            cfg.environment.task_name,
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
        augment_state=False,
    )

    # Create eval environment with built-in features
    eval_env = rccar.RCCar(
        eval_car_params["nominal"],
        action_delay=action_delay,
        observation_delay=observation_delay,
        sliding_window=sliding_window,
        **task_cfg,
    )
    eval_env = eval_wrap_env_fn(eval_env)
    eval_randomization_fn = (
        prepare_randomization_fn(
            eval_key,
            cfg.training.num_eval_envs,
            eval_car_params,
            cfg.environment.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=False,
    )
    return train_env, eval_env


def make_brax_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    if "use_vision" in cfg.agent and cfg.agent.use_vision:
        raise ValueError("RCCar does not support vision.")
    task_cfg = get_task_config(cfg)
    train_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    train_env = train_wrap_env_fn(train_env)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg.train_params, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap(
        train_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
        augment_state=False,
        hard_resets=cfg.training.hard_resets,
    )
    eval_env = envs.get_environment(
        task_cfg.task_name, backend=cfg.environment.backend, **task_cfg.task_params
    )
    eval_env = eval_wrap_env_fn(eval_env)
    eval_randomization_fn = prepare_randomization_fn(
        eval_key, cfg.training.num_eval_envs, task_cfg.eval_params, task_cfg.task_name
    )
    eval_env = wrap(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn
        if cfg.training.eval_domain_randomization
        else None,
        augment_state=False,
    )
    return train_env, eval_env


def _preinitialize_vision_env(task_name, task_params, registry):
    # https://github.com/shacklettbp/madrona_mjx/issues/39
    new_params = copy.deepcopy(task_params)
    new_params["vision"] = False
    train_env = registry.load(task_name, config=new_params)
    dummy_state = train_env.reset(jax.random.PRNGKey(0))
    train_env.step(dummy_state, jnp.zeros(train_env.action_size))


def make_spidr_cartpole_vision(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    from ml_collections import config_dict
    from mujoco_playground import registry

    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training

    task_cfg = get_task_config(cfg)
    task_params = config_dict.ConfigDict(task_cfg.task_params)
    _preinitialize_vision_env(task_cfg.task_name, task_params, registry)
    train_key, spidr_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    spidr_train_randomization_fn = prepare_randomization_fn(
        spidr_key,
        8,
        task_cfg.train_params,
        task_cfg.task_name,
    )
    train_env = registry.load(task_cfg.task_name, config=task_params)
    train_env = VisionSPiDRCartpole(
        train_env, spidr_train_randomization_fn, cfg.agent.lambda_, config=task_params
    )
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key,
            cfg.training.num_envs,
            task_cfg.train_params,
            task_cfg.task_name,
        )
        if cfg.training.train_domain_randomization
        else None
    )
    if cfg.training.safe:
        limit = task_params.slider_position_bound
        train_env = dm_cartpole.ConstraintWrapper(train_env, limit)
    randomization_fn = lambda model: train_randomization_fn(model)[:2]
    train_env = wrap_for_brax_training(
        train_env,
        randomization_fn=randomization_fn,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        augment_state=False,
        hard_resets=cfg.training.hard_resets,
        vision=True,
        num_vision_envs=cfg.training.num_envs,
    )
    return train_env, train_env


def make_mujoco_playground_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    from ml_collections import config_dict
    from mujoco_playground import registry

    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training

    task_cfg = get_task_config(cfg)
    task_params = config_dict.ConfigDict(task_cfg.task_params)
    vision = "use_vision" in cfg.agent and cfg.agent.use_vision
    if vision:
        _preinitialize_vision_env(task_cfg.task_name, task_params, registry)
    train_env = registry.load(task_cfg.task_name, config=task_params)
    train_env = train_wrap_env_fn(train_env)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    if vision and cfg.training.train_domain_randomization:
        train_randomization_fn = functools.partial(
            randomization_fns[task_cfg.task_name], num_worlds=cfg.training.num_envs
        )
    else:
        train_randomization_fn = (
            prepare_randomization_fn(
                train_key,
                cfg.training.num_envs,
                task_cfg.train_params,
                task_cfg.task_name,
            )
            if cfg.training.train_domain_randomization
            else None
        )
    train_env = wrap_for_brax_training(
        train_env,
        randomization_fn=train_randomization_fn,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        augment_state=False,
        hard_resets=cfg.training.hard_resets,
        vision=vision,
        num_vision_envs=cfg.training.num_envs,
        nonepisodic=cfg.training.nonepisodic,
    )
    if vision:
        return train_env, train_env
    eval_env = registry.load(task_cfg.task_name, config=task_params)
    eval_env = eval_wrap_env_fn(eval_env)
    eval_randomization_fn = (
        prepare_randomization_fn(
            eval_key,
            cfg.training.num_eval_envs,
            task_cfg.eval_params,
            task_cfg.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=False,
    )
    return train_env, eval_env


def make_safety_gym_envs(cfg, train_wrap_env_fn, eval_wrap_env_fn):
    from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training
    from ss2r.benchmark_suites.safety_gym import go_to_goal

    task_cfg = get_task_config(cfg)
    train_env = go_to_goal.GoToGoal(**task_cfg.task_params)
    train_env = train_wrap_env_fn(train_env)
    eval_env = go_to_goal.GoToGoal(**task_cfg.task_params)
    eval_env = eval_wrap_env_fn(eval_env)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    train_randomization_fn = (
        prepare_randomization_fn(
            train_key, cfg.training.num_envs, task_cfg.train_params, task_cfg.task_name
        )
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = wrap_for_brax_training(
        train_env,
        randomization_fn=train_randomization_fn,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        augment_state=False,
        hard_resets=cfg.training.hard_resets,
    )
    eval_randomization_fn = (
        prepare_randomization_fn(
            eval_key,
            cfg.training.num_eval_envs,
            task_cfg.eval_params,
            task_cfg.task_name,
        )
        if cfg.training.eval_domain_randomization
        else None
    )
    eval_env = wrap_for_brax_training(
        eval_env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=eval_randomization_fn,
        augment_state=False,
    )
    return train_env, eval_env


randomization_fns = {
    "cartpole": cartpole.domain_randomization,
    "cartpole_safe": cartpole.domain_randomization,
    "rccar": rccar.domain_randomization,
    "humanoid": humanoid.domain_randomization,
    "humanoid_safe": humanoid.domain_randomization,
    "Go1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "Go1JoystickRoughTerrain": go1_joystick.domain_randomization,
    "SafeJointGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeFlipGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "SafeJointTorqueGo1JoystickFlatTerrain": go1_joystick.domain_randomization,
    "Go2JoystickFlatTerrain": go1_joystick.domain_randomization,
    "Go2JoystickRoughTerrain": go1_joystick.domain_randomization,
    "ant": ant.domain_randomization,
    "ant_safe": ant.domain_randomization,
    "QuadrupedWalk": quadruped.domain_randomization,
    "QuadrupedRun": quadruped.domain_randomization,
    "SafeQuadrupedWalk": quadruped.domain_randomization,
    "SafeQuadrupedRun": quadruped.domain_randomization,
    "WalkerWalk": walker.domain_randomization,
    "WalkerRun": walker.domain_randomization,
    "SafeWalkerWalk": walker.domain_randomization,
    "SafeWalkerRun": walker.domain_randomization,
    "SafeCartpoleSwingup": dm_cartpole.domain_randomization,
    "SafeCartpoleSwingupSparse": dm_cartpole.domain_randomization,
    "SafeCartpoleBalanceSparse": dm_cartpole.domain_randomization,
    "SafeCartpoleBalance": dm_cartpole.domain_randomization,
    "CartpoleSwingup": dm_cartpole.domain_randomization,
    "CartpoleSwingupSparse": dm_cartpole.domain_randomization,
    "CartpoleBalanceSparse": dm_cartpole.domain_randomization,
    "CartpoleBalance": dm_cartpole.domain_randomization,
    "HumanoidWalk": dm_humanoid.domain_randomization,
    "NonEpisodicHumanoidStand": mujoco_playground.render,
    "NonEpisodicHumanoidWalk": mujoco_playground.render,
    "NonEpisodicHumanoidRun": mujoco_playground.render,
    "SafeHumanoidWalk": dm_humanoid.domain_randomization,
    "AlohaPegInsertionDistill": manipulation.get_domain_randomizer(
        "AlohaPegInsertionDistill"
    ),
    "AlohaSinglePegInsertion": manipulation.get_domain_randomizer(
        "AlohaSinglePegInsertion"
    ),
    "go_to_goal": go_to_goal.domain_randomization,
    "PandaPickCubeCartesian": franka_vision_randomize,
    "PandaPickCubeCartesianExtended": pick_cartesian.domain_randomize,
}

render_fns = {
    "cartpole": brax.render,
    "cartpole_safe": brax.render,
    "humanoid": functools.partial(brax.render, camera="track"),
    "humanoid_safe": functools.partial(brax.render, camera="track"),
    "ant": functools.partial(brax.render, camera="track"),
    "ant_safe": functools.partial(brax.render, camera="track"),
    "rccar": rccar.render,
    "Go1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "Go1JoystickRoughTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeJointGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeFlipGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeJointTorqueGo1JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "Go2JoystickFlatTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "Go2JoystickRoughTerrain": functools.partial(
        mujoco_playground.render, camera="track"
    ),
    "SafeCartpoleSwingup": functools.partial(mujoco_playground.render),
    "SafeCartpoleSwingupSparse": functools.partial(mujoco_playground.render),
    "SafeCartpoleBalanceSparse": functools.partial(mujoco_playground.render),
    "SafeCartpoleBalance": functools.partial(mujoco_playground.render),
    "CartpoleSwingup": functools.partial(mujoco_playground.render),
    "CartpoleSwingupSparse": functools.partial(mujoco_playground.render),
    "CartpoleBalanceSparse": functools.partial(mujoco_playground.render),
    "CartpoleBalance": functools.partial(mujoco_playground.render),
    "QuadrupedWalk": functools.partial(mujoco_playground.render, camera="x"),
    "QuadrupedRun": functools.partial(mujoco_playground.render, camera="x"),
    "SafeQuadrupedWalk": functools.partial(mujoco_playground.render, camera="x"),
    "SafeQuadrupedRun": functools.partial(mujoco_playground.render, camera="x"),
    "WalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "WalkerRun": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerWalk": functools.partial(mujoco_playground.render, camera="side"),
    "SafeWalkerRun": functools.partial(mujoco_playground.render, camera="side"),
    "HumanoidWalk": mujoco_playground.render,
    "NonEpisodicHumanoidStand": mujoco_playground.render,
    "NonEpisodicHumanoidWalk": mujoco_playground.render,
    "NonEpisodicHumanoidRun": mujoco_playground.render,
    "SafeHumanoidWalk": mujoco_playground.render,
    "AlohaSinglePegInsertion": mujoco_playground.render,
    "AlohaPegInsertionDistill": mujoco_playground.render,
    "PandaPickCubeCartesian": functools.partial(
        mujoco_playground.render, num_envs=None, camera="front"
    ),
    "PandaPickCubeCartesianExtended": functools.partial(
        mujoco_playground.render, num_envs=None, camera="front"
    ),
    "go_to_goal": functools.partial(safety_gym.render, camera="fixedfar"),
}
