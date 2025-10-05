from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid


def normalize_angle(angle, lower_bound=-jp.pi, upper_bound=jp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


class HumanoidGetup(humanoid.Humanoid):
    """Humanoid getup task adapted from Go1 getup."""

    def __init__(
        self,
        config: config_dict.ConfigDict = humanoid.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            move_speed=0.0, config=config, config_overrides=config_overrides
        )

    def _post_init(self):
        super()._post_init()
        self._settle_steps = int(0.75 / self.sim_dt)
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
        self.joint_ranges = zip(*[self.mj_model.jnt_range[id_] for id_ in joint_ids])
        self._init_q = mjx_env.init(self.mjx_model).qpos

    def _get_random_qpos(self, key: jax.Array) -> jax.Array:
        key1, key2 = jax.random.split(key, 2)
        qpos = jp.zeros(self.mjx_model.nq)
        # TODO (yarden): make sure that this entry is indeed the height of the robot
        qpos = qpos.at[2].set(humanoid._STAND_HEIGHT)  # drop height
        quat = jax.random.normal(key1, (4,))
        quat /= jp.linalg.norm(quat) + 1e-6
        # TODO (yarden): make sure that this is the orientation of
        # the humanoid
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[7:].set(
            jax.random.uniform(
                key2,
                (self.mjx_model.nq - 7,),
                minval=normalize_angle(self.joint_ranges[0]),
                maxval=normalize_angle(self.joint_ranges[1]),
            )
        )
        return qpos

    def reset(self, rng: jax.Array) -> mjx_env.State:
        key, key1, key2 = jax.random.split(rng, 3)
        qpos = jp.where(
            jax.random.bernoulli(key1, 0.6),
            self._get_random_qpos(key2),
            self._init_q,
        )
        qvel = jp.zeros(self.mjx_model.nv)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
        data = mjx_env.init(self.mj_model, qpos=qpos, qvel=qvel)
        data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
        data = data.replace(time=0.0)
        info = {
            "rng": rng,
        }
        metrics = {
            k: jp.zeros(())
            for k in [
                "reward/upright",
                "reward/stand",
            ]
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)
