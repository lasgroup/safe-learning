from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid


def normalize_angle(angle, lower_bound=-jp.pi, upper_bound=jp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,
        sim_dt=0.005,  # 0.0025 in DM Control
        episode_length=1000,
        action_repeat=1,
        vision=False,
        ground_start_probability=0.0,
    )


class NonEpisodicHumanoid(humanoid.Humanoid):
    """Humanoid getup task adapted from Go1 getup."""

    def __init__(
        self,
        move_speed: float = 0.0,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            move_speed=move_speed, config=config, config_overrides=config_overrides
        )

        self._xml_path = humanoid._XML_PATH.as_posix()
        mj_spec = mujoco.MjSpec.from_file(
            filename=str(humanoid._XML_PATH), assets=humanoid.common.get_assets()
        )
        self._mj_model = self._modify_model(mj_spec)
        self._mjx_model = mjx.put_model(self._mj_model)
        self.ground_start_probability = config.ground_start_probability
        self._post_init()

    def _modify_model(self, mj_spec: mujoco.MjSpec) -> mujoco.MjModel:
        mj_spec.add_sensor(
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            name="head_touch",
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="head",
        )
        mj_spec.add_sensor(
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            name="torso_touch",
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="torso",
        )
        return mj_spec.compile()

    def _post_init(self):
        super()._post_init()
        self._settle_steps = int(1.0 / self.sim_dt)
        joint_names = [
            "abdomen_z",
            "abdomen_y",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ]
        joint_ids = jp.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)
                for name in joint_names
            ]
        )
        self.joint_ranges = jp.array(
            tuple(zip(*[self.mj_model.jnt_range[id_] for id_ in joint_ids]))
        )
        self.qpos_ids = jp.asarray(
            [self.mj_model.jnt_qposadr[id_] for id_ in joint_ids]
        )
        self._init_q = mjx_env.init(self.mjx_model).qpos

    def _get_random_qpos(self, key: jax.Array) -> jax.Array:
        key1, key2 = jax.random.split(key, 2)
        qpos = jp.zeros(self.mjx_model.nq)
        qpos = qpos.at[2].set(humanoid._STAND_HEIGHT)  # drop height
        quat = jax.random.normal(key1, (4,))
        quat /= jp.linalg.norm(quat) + 1e-6
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[self.qpos_ids].set(
            jax.random.uniform(
                key2,
                len(self.qpos_ids),
                minval=normalize_angle(self.joint_ranges[0]),
                maxval=normalize_angle(self.joint_ranges[1]),
            )
        )
        return qpos

    def reset(self, rng: jax.Array) -> mjx_env.State:
        key, key1, key2 = jax.random.split(rng, 3)
        start_on_ground = jax.random.bernoulli(key1, self.ground_start_probability)

        def ground_init():
            qpos = self._get_random_qpos(key2)
            qvel = jp.zeros(self.mjx_model.nv)
            qvel = qvel.at[0:6].set(
                jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
            )
            data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)
            data = mjx_env.step(
                self.mjx_model, data, jp.zeros(self.mjx_model.nu), self._settle_steps
            )
            data = data.replace(time=0.0)
            return data

        def normal_init():
            data = mjx_env.init(self.mjx_model)
            return data

        data = jax.lax.cond(
            start_on_ground,
            ground_init,
            normal_init,
        )
        info = {"rng": rng, "cost": jp.zeros(())}
        metrics = {
            "reward/standing": jp.zeros(()),
            "reward/upright": jp.zeros(()),
            "reward/stand": jp.zeros(()),
            "reward/small_control": jp.zeros(()),
            "reward/move": jp.zeros(()),
            "reward/on_ground": jp.zeros(()),
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        outs = super().step(state, action)
        head_height = self._head_height(outs.data)
        standing = head_height > humanoid._STAND_HEIGHT
        torso_height = self._torso_upright(outs.data)
        upright = torso_height > 0.9
        outs.info["cost"] = jp.where(
            standing | upright, jp.zeros_like(outs.reward), jp.ones_like(outs.reward)
        )
        on_ground = (head_height < 0.1) | (torso_height < 0.1)
        outs.metrics["reward/on_ground"] = on_ground.astype(jp.float32)
        torso_force = (
            jp.linalg.norm(
                mjx_env.get_sensor_data(self.mj_model, outs.data, "torso_touch")
            )
            > 4000.0
        )
        head_force = (
            jp.linalg.norm(
                mjx_env.get_sensor_data(self.mj_model, outs.data, "head_touch")
            )
            > 4000.0
        )
        terminate = on_ground & (torso_force | head_force)
        terminate = jp.isnan(terminate) | terminate
        nans = jp.isnan(outs.data.qpos).any() | jp.isnan(outs.data.qvel).any()
        done = terminate | nans
        outs = outs.replace(done=done.astype(jp.float32))
        return outs
