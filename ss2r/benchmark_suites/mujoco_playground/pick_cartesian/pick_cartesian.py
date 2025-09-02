# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""A simple task with demonstrating sim2real transfer for pixels observations.
Pick up a cube to a fixed location using a cartesian controller."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import collision, mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import (
    panda_kinematics,
    pick,
    pick_cartesian,
    randomize_vision,
)


def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
    """
    Intensity-weigh the colors.
    This expects the input to have the channels in the last dim.
    """
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def domain_randomize(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
    """Tile the necessary axes for the Madrona BatchRenderer."""
    mj_model = pick_cartesian.PandaPickCubeCartesian().mj_model
    floor_geom_id = mj_model.geom("floor").id
    box_geom_id = mj_model.geom("box").id
    # strip_geom_id = mj_model.geom("init_space").id

    in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
    in_axes = in_axes.tree_replace(
        {
            "geom_rgba": 0,
            "geom_matid": 0,
            "cam_pos": 0,
            "cam_quat": 0,
            "light_pos": 0,
            "light_dir": 0,
            "light_directional": 0,
            "light_castshadow": 0,
        }
    )
    rng = jax.random.key(0)

    # Simpler logic implementing via Numpy.
    np.random.seed(0)
    light_positions = [
        randomize_vision.sample_light_position() for _ in range(num_worlds)
    ]
    light_positions = jp.array(light_positions)

    @jax.vmap
    def rand(rng: jax.Array, light_position: jax.Array):
        """Generate randomized model fields."""
        _, key = jax.random.split(rng, 2)

        #### Apearance ####
        # Sample a random color for the box
        key_box, key_strip, key_floor, key = jax.random.split(key, 4)
        # rgba = jp.array(
        #     [jax.random.uniform(key_box, (), minval=0.5, maxval=1.0), 0.0, 0.0, 1.0]
        # )
        # geom_rgba = mjx_model.geom_rgba.at[box_geom_id].set(rgba)
        geom_rgba = mjx_model.geom_rgba.copy()

        # strip_white = jax.random.uniform(key_strip, (), minval=0.8, maxval=1.0)
        # geom_rgba = geom_rgba.at[strip_geom_id].set(
        #     jp.array([strip_white, strip_white, strip_white, 1.0])
        # )  # type: ignore

        # Sample a shade of gray
        gray_scale = jax.random.uniform(key_floor, (), minval=0.0, maxval=0.25)
        geom_rgba = geom_rgba.at[floor_geom_id].set(
            jp.array([gray_scale, gray_scale, gray_scale, 1.0])
        )

        mat_offset, num_geoms = 5, geom_rgba.shape[0]
        key_matid, key = jax.random.split(key)
        geom_matid = (
            jax.random.randint(key_matid, shape=(num_geoms,), minval=0, maxval=10)
            + mat_offset
        )
        geom_matid = geom_matid.at[box_geom_id].set(
            -1
        )  # Use the above randomized colors
        geom_matid = geom_matid.at[floor_geom_id].set(-2)
        # geom_matid = geom_matid.at[strip_geom_id].set(-2)

        #### Cameras ####
        key_pos, key_ori, key = jax.random.split(key, 3)
        cam_offset = jax.random.uniform(key_pos, (3,), minval=-0.05, maxval=0.05)
        assert (
            len(mjx_model.cam_pos) == 1
        ), f"Expected single camera, got {len(mjx_model.cam_pos)}"
        cam_pos = mjx_model.cam_pos.at[0].set(mjx_model.cam_pos[0] + cam_offset)
        cam_quat = mjx_model.cam_quat.at[0].set(
            randomize_vision.perturb_orientation(key_ori, mjx_model.cam_quat[0], 10)
        )

        #### Lighting ####
        nlight = mjx_model.light_pos.shape[0]
        assert (
            nlight == 1
        ), f"Sim2Real was trained with a single light source, got {nlight}"
        key_lsha, key_ldir, key = jax.random.split(key, 3)

        # Direction
        shine_at = jp.array([0.661, -0.001, 0.179])  # Gripper starting position
        nom_dir = (shine_at - light_position) / jp.linalg.norm(
            shine_at - light_position
        )
        light_dir = mjx_model.light_dir.at[0].set(
            randomize_vision.perturb_orientation(key_ldir, nom_dir, 20)
        )

        # Whether to cast shadows
        light_castshadow = jax.random.bernoulli(key_lsha, 0.75, shape=(nlight,)).astype(
            jp.float32
        )

        # No need to randomize into specular lighting
        light_directional = jp.ones((nlight,))

        return (
            geom_rgba,
            geom_matid,
            cam_pos,
            cam_quat,
            light_dir,
            light_directional,
            light_castshadow,
        )

    (
        geom_rgba,
        geom_matid,
        cam_pos,
        cam_quat,
        light_dir,
        light_directional,
        light_castshadow,
    ) = rand(jax.random.split(rng, num_worlds), light_positions)

    mjx_model = mjx_model.tree_replace(
        {
            "geom_rgba": geom_rgba,
            "geom_matid": geom_matid,
            "cam_pos": cam_pos,
            "cam_quat": cam_quat,
            "light_pos": light_positions,
            "light_dir": light_dir,
            "light_directional": light_directional,
            "light_castshadow": light_castshadow,
        }
    )

    return mjx_model, in_axes


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        gpu_id=0,
        render_batch_size=1024,
        render_width=64,
        render_height=64,
        use_rasterizer=False,
        enabled_geom_groups=[0, 1, 2],
    )


def default_config():
    config = config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=200,
        action_repeat=1,
        # Size of cartesian increment.
        action_scale=0.005,
        reward_config=config_dict.create(
            reward_scales=config_dict.create(
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=8.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Do not collide cube with gripper
                no_box_collision=0.05,
                # Destabilizes training in cartesian action space.
                robot_target_qpos=0.0,
            ),
            action_rate=-0.0005,
            no_soln_reward=-0.01,
            lifted_reward=0.5,
            success_reward=2.0,
        ),
        vision=False,
        vision_config=default_vision_config(),
        obs_noise=config_dict.create(brightness=[1.0, 1.0]),
        box_init_range=0.05,
        success_threshold=0.05,
        action_history_length=1,
    )
    return config


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jp.clip(img * scale, 0, 1)


_ROOT_PATH = epath.Path(__file__).parent
_MENAGERIE_FRANKA_DIR = "franka_emika_panda"


def get_assets():
    assets = {}
    path = _ROOT_PATH / "manipulation" / "franka_emika_panda" / "xmls"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, _ROOT_PATH / "assets", "*.png")
    path = mjx_env.MENAGERIE_PATH / _MENAGERIE_FRANKA_DIR
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


class PandaPickCubeCartesian(pick.PandaPickCube):
    """Environment for training the Franka Panda robot to pick up a cube in
    Cartesian space."""

    def __init__(  # pylint: disable=non-parent-init-called,super-init-not-called
        self,
        config=default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        self._vision = config.vision

        xml_path = (
            epath.Path(__file__).parent
            / "assets"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )
        self._xml_path = xml_path.as_posix()

        mj_model = self.modify_model(
            mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=get_assets())
        )
        mj_model.opt.timestep = config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        # Set gripper in sight of camera
        self._post_init(obj_name="box", keyframe="low_home")
        self._box_geom = self._mj_model.geom("box").id

        if self._vision:
            try:
                # pylint: disable=import-outside-toplevel
                from madrona_mjx.renderer import (
                    BatchRenderer,
                )
            except ImportError:
                warnings.warn(
                    "Madrona MJX not installed. Cannot use vision with"
                    " PandaPickCubeCartesian."
                )
                return
            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=self._config.vision_config.gpu_id,
                num_worlds=self._config.vision_config.render_batch_size,
                batch_render_view_width=self._config.vision_config.render_width,
                batch_render_view_height=self._config.vision_config.render_height,
                enabled_geom_groups=np.asarray(
                    self._config.vision_config.enabled_geom_groups
                ),
                enabled_cameras=None,  # Use all cameras.
                add_cam_debug_geo=False,
                use_rasterizer=self._config.vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def _post_init(self, obj_name, keyframe):
        super()._post_init(obj_name, keyframe)
        self._guide_q = self._mj_model.keyframe("picked").qpos
        self._guide_ctrl = self._mj_model.keyframe("picked").ctrl
        # Use forward kinematics to init cartesian control
        self._start_tip_transform = panda_kinematics.compute_franka_fk(
            self._init_ctrl[:7]
        )
        self._sample_orientation = False

    def modify_model(self, mj_model: mujoco.MjModel):
        # Expand floor size to non-zero so Madrona can render it
        mj_model.geom_size[mj_model.geom("floor").id, :2] = [5.0, 5.0]

        # Make the finger pads white for increased visibility
        mesh_id = mj_model.mesh("finger_1").id
        geoms = [
            idx
            for idx, data_id in enumerate(mj_model.geom_dataid)
            if data_id == mesh_id
        ]
        mj_model.geom_matid[geoms] = mj_model.mat("off_white").id
        if self._config.use_ball:
            mj_model.geom("box").matid = jp.array([-1])
            mj_model.geom("box").rgba = jp.array([1, 0, 0, 0.2])
            mj_model.geom("box").type = mujoco.mjtGeom.mjGEOM_SPHERE
            mj_model.geom("box").size = jp.array([0.03])
            mj_model.opt.impratio = 5
            mj_model.opt.noslip_iterations = 3
            mj_model.body("box").inertia *= 150
        return mj_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Resets the environment to an initial state."""
        x_plane = self._start_tip_transform[0, 3] - 0.03  # Account for finite gain

        # intialize box position
        rng, rng_box = jax.random.split(rng)
        r_range = self._config.box_init_range
        box_pos = jp.array(
            [
                x_plane,
                jax.random.uniform(rng_box, (), minval=-r_range, maxval=r_range),
                0.03 + 1e-5,
            ]
        )

        # Fixed target position to simplify pixels-only training.
        target_pos = jp.array([x_plane, 0.0, 0.20])

        # initialize pipeline state
        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(box_pos)
        )
        data = mjx_env.init(
            self._mjx_model,
            init_q,
            jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
        )

        target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        data = data.replace(
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat)
        )
        if not self._vision:
            # mocap target should not appear in the pixels observation.
            data = data.replace(
                mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos)
            )

        # initialize env state and info
        metrics = {
            "out_of_bounds": jp.array(0.0),
            **{
                f"reward/{k}": 0.0
                for k in self._config.reward_config.reward_scales.keys()
            },
            "reward/lifted": jp.array(0.0),
            "reward/success": jp.array(0.0),
        }

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "reached_box": jp.array(0.0, dtype=float),
            "prev_reward": jp.array(0.0, dtype=float),
            "current_pos": self._start_tip_transform[:3, 3],
            "newly_reset": jp.array(False, dtype=bool),
            "prev_action": jp.zeros(3),
            "_steps": jp.array(0, dtype=int),
            "action_history": jp.zeros(
                (self._config.action_history_length,)
            ),  # Gripper only
        }

        reward, done = jp.zeros(2)

        obs = self._get_obs(data, info)
        obs = jp.concat([obs, jp.zeros(1), jp.zeros(3)], axis=0)
        if self._vision:
            rng_brightness, rng = jax.random.split(rng)
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._config.obs_noise.brightness[0],
                maxval=self._config.obs_noise.brightness[1],
            )
            info.update({"brightness": brightness})

            render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
            info.update({"render_token": render_token})

            obs = _rgba_to_grayscale(jp.asarray(rgb[0], dtype=jp.float32)) / 255.0
            obs = adjust_brightness(obs, brightness)[..., None]
            gripper_pos = data.site_xpos[self._gripper_site]
            fingers = data.qpos[7:9] / 0.04
            proprioceptive = jp.concatenate(
                [gripper_pos, fingers, info["action_history"].copy()]
            )
            obs = {"pixels/view_0": obs, "state": proprioceptive}

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Runs one timestep of the environment's dynamics."""
        action_history = jp.roll(state.info["action_history"], 1).at[0].set(action[2])
        state.info["action_history"] = action_history
        # Add action delay
        state.info["rng"], key = jax.random.split(state.info["rng"])
        action_idx = jax.random.randint(
            key, (), minval=0, maxval=self._config.action_history_length
        )
        action = action.at[2].set(state.info["action_history"][action_idx])

        state.info["newly_reset"] = state.info["_steps"] == 0

        newly_reset = state.info["newly_reset"]
        state.info["prev_reward"] = jp.where(
            newly_reset, 0.0, state.info["prev_reward"]
        )
        state.info["current_pos"] = jp.where(
            newly_reset, self._start_tip_transform[:3, 3], state.info["current_pos"]
        )
        state.info["reached_box"] = jp.where(
            newly_reset, 0.0, state.info["reached_box"]
        )
        state.info["prev_action"] = jp.where(
            newly_reset, jp.zeros(3), state.info["prev_action"]
        )

        # Ocassionally aid exploration.
        state.info["rng"], key_swap = jax.random.split(state.info["rng"])
        to_sample = newly_reset * jax.random.bernoulli(key_swap, 0.05)
        swapped_data = state.data.replace(
            qpos=self._guide_q, ctrl=self._guide_ctrl
        )  # help hit the terminal sparse reward.
        data = jax.tree_util.tree_map(
            lambda x, y: (1 - to_sample) * x + to_sample * y,
            state.data,
            swapped_data,
        )

        # Cartesian control
        increment = jp.zeros(4)
        if not self._config.use_x:
            increment = increment.at[1:].set(action)  # set y, z and gripper commands.
        else:
            increment = action
        ctrl, new_tip_position, no_soln = self._move_tip(
            state.info["current_pos"],
            self._start_tip_transform[:3, :3],
            data.ctrl,
            increment,
        )
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        state.info.update({"current_pos": new_tip_position})

        # Simulator step
        data = mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps)

        # Dense rewards
        raw_rewards = self._get_reward(data, state.info)
        rewards = {
            k: v * self._config.reward_config.reward_scales[k]
            for k, v in raw_rewards.items()
        }

        # Penalize collision with box.
        hand_box = collision.geoms_colliding(data, self._box_geom, self._hand_geom)
        raw_rewards["no_box_collision"] = jp.where(hand_box, 0.0, 1.0)

        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        if not self._vision:
            # Vision policy cannot access the required state-based observations.
            da = jp.linalg.norm(action - state.info["prev_action"])
            state.info["prev_action"] = action
            total_reward += self._config.reward_config.action_rate * da
            total_reward += no_soln * self._config.reward_config.no_soln_reward

        # Sparse rewards
        box_pos = data.xpos[self._obj_body]
        lifted = (box_pos[2] > 0.05) * self._config.reward_config.lifted_reward
        total_reward += lifted
        success = self._get_success(data, state.info)
        total_reward += success * self._config.reward_config.success_reward

        # Reward progress
        reward = jp.maximum(
            total_reward - state.info["prev_reward"], jp.zeros_like(total_reward)
        )
        state.info["prev_reward"] = jp.maximum(total_reward, state.info["prev_reward"])
        reward = jp.where(newly_reset, 0.0, reward)  # Prevent first-step artifact

        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        state.metrics.update(out_of_bounds=out_of_bounds.astype(float))
        state.metrics.update({f"reward/{k}": v for k, v in raw_rewards.items()})
        state.metrics.update(
            {
                "reward/lifted": lifted.astype(float),
                "reward/success": success.astype(float),
            }
        )

        done = (
            out_of_bounds
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | success
        )

        # Ensure exact sync between newly_reset and the autoresetwrapper.
        state.info["_steps"] += self._config.action_repeat
        state.info["_steps"] = jp.where(
            done | (state.info["_steps"] >= self._config.episode_length),
            0,
            state.info["_steps"],
        )

        obs = self._get_obs(data, state.info)
        obs = jp.concat([obs, no_soln.reshape(1), action], axis=0)
        if self._vision:
            _, rgb, _ = self.renderer.render(state.info["render_token"], data)
            obs = _rgba_to_grayscale(jp.asarray(rgb[0], dtype=jp.float32)) / 255.0
            obs = adjust_brightness(obs, state.info["brightness"])[..., None]
            gripper_pos = data.site_xpos[self._gripper_site]
            fingers = data.qpos[7:9] / 0.04
            proprioceptive = jp.concatenate(
                [gripper_pos, fingers, action_history.copy()]
            )
            obs = {"pixels/view_0": obs, "state": proprioceptive}

        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=done.astype(float),
            info=state.info,
        )

    def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]
        if (
            self._vision
        ):  # Randomized camera positions cannot see location along y line.
            box_pos, target_pos = box_pos[2], target_pos[2]
        return jp.linalg.norm(box_pos - target_pos) < self._config.success_threshold

    def _move_tip(
        self,
        current_tip_pos: jax.Array,
        current_tip_rot: jax.Array,
        current_ctrl: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Calculate new tip position from cartesian increment."""
        # Discrete gripper action where a < 0 := closed
        close_gripper = jp.where(action[3] < 0, 1.0, 0.0)

        scaled_pos = action[:3] * self._config.action_scale
        new_tip_pos = current_tip_pos.at[:3].add(scaled_pos)

        new_ctrl = current_ctrl

        new_tip_pos = new_tip_pos.at[0].set(jp.clip(new_tip_pos[0], 0.25, 0.77))
        new_tip_pos = new_tip_pos.at[1].set(jp.clip(new_tip_pos[1], -0.32, 0.32))
        new_tip_pos = new_tip_pos.at[2].set(jp.clip(new_tip_pos[2], 0.02, 0.5))

        new_tip_mat = jp.identity(4)
        new_tip_mat = new_tip_mat.at[:3, :3].set(current_tip_rot)
        new_tip_mat = new_tip_mat.at[:3, 3].set(new_tip_pos)

        out_jp = panda_kinematics.compute_franka_ik(
            new_tip_mat, current_ctrl[6], current_ctrl[:7]
        )
        no_soln = jp.any(jp.isnan(out_jp))
        out_jp = jp.where(no_soln, current_ctrl[:7], out_jp)
        no_soln = jp.logical_or(no_soln, jp.any(jp.isnan(out_jp)))
        new_tip_pos = jp.where(jp.any(jp.isnan(out_jp)), current_tip_pos, new_tip_pos)

        new_ctrl = new_ctrl.at[:7].set(out_jp)
        jaw_action = jp.where(close_gripper, -1.0, 1.0)
        claw_delta = jaw_action * 0.02  # up to 2 cm movement per ctrl.
        new_ctrl = new_ctrl.at[7].set(new_ctrl[7] + claw_delta)

        return new_ctrl, new_tip_pos, no_soln

    @property
    def action_size(self) -> int:
        return 3

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
