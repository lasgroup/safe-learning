import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from brax.envs.base import Env, State
from omegaconf import OmegaConf

from ss2r.benchmark_suites.rccar.hardware import HardwareDynamics
from ss2r.benchmark_suites.rccar.model import CarParams, RaceCarDynamics
from ss2r.rl.utils import rollout

X_LIM = (-0.45, 3.0)
Y_LIM = (-2.1, 1.5)


def domain_randomization(sys, rng, cfg):
    def sample_from_bounds(value, key):
        """
        Sample from a JAX uniform distribution if the value is a list of two elements.
        """
        if isinstance(value, list) and len(value) == 2:
            lower, upper = value
            # Sample from jax.random.uniform with the given key
            return jax.random.uniform(key, shape=(), minval=lower, maxval=upper)
        return value

    @jax.vmap
    def randomize(rng):
        bounds = cfg["bounds"]
        bounds = CarParams(**bounds)
        # Define a custom tree structure that treats lists as leaves
        treedef = jtu.tree_structure(bounds, is_leaf=lambda x: isinstance(x, list))
        # Generate random keys only for the relevant leaves (i.e., lists with 2 elements)
        keys = jax.random.split(rng, num=treedef.num_leaves)
        # Rebuild the tree with the keys, only where there are valid leaves
        keys = jtu.tree_unflatten(treedef, keys)
        # Map over the tree, generating random values where needed
        sys = jtu.tree_map(
            sample_from_bounds, bounds, keys, is_leaf=lambda x: isinstance(x, list)
        )
        return sys, jax.flatten_util.ravel_pytree(sys)[0]

    cfg = OmegaConf.to_container(cfg)
    in_axes = jax.tree_map(lambda _: 0, sys)
    sys, params = randomize(rng)
    return sys, in_axes, params


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = (
        state[..., 0:1],
        state[..., 3 + int(encode_angle) : 4 + int(encode_angle)],
    )
    y_pos, y_vel = (
        state[..., 1:2],
        state[:, 4 + int(encode_angle) : 5 + int(encode_angle)],
    )
    theta = state[..., 2 : 3 + int(encode_angle)] - jnp.pi / 2
    new_state = jnp.concatenate(
        [y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle) :]],
        axis=-1,
    )
    assert state.shape == new_state.shape
    return new_state


def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) ant cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = jnp.concatenate(
        [
            state[..., :angle_idx],
            jnp.sin(theta),
            jnp.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = jnp.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


def cost_fn(xy, obstacles, *, scale_factor=1.0) -> jax.Array:
    total = 0.0
    for obstacle in obstacles:
        position, radius = jnp.asarray(obstacle[:2]), obstacle[2]
        distance = jnp.linalg.norm(xy - position)
        total += jnp.where(distance >= radius * scale_factor, 0.0, 1.0)
    return total


def in_arena(xy, scale=1.0):
    x, y = xy[..., 0], xy[..., 1]
    in_bounds = lambda x, lower, upper: jnp.where(
        (x >= lower * scale) & (x <= upper * scale), True, False
    )
    in_x = in_bounds(x, *X_LIM)
    in_y = in_bounds(y, *Y_LIM)
    return jnp.asarray(in_x & in_y, dtype=jnp.float32)


class RCCar(Env):
    def __init__(
        self,
        car_model_params: dict,
        dt: float = 1 / 30.0,
        obstacles: list[tuple[float, float, float]] = [(0.75, -0.75, 0.2)],
        sample_init_pose: bool = True,
        control_penalty_scale: float = 0.0,
        last_action_penalty_scale: float = 0.0,
        action_delay: int = 0,  # Parameter for action delay
        observation_delay: int = 0,  # Parameter for observation delay
        sliding_window: int = 0,  # New parameter for frame stacking (0 means no stacking)
        *,
        hardware: HardwareDynamics | None = None,
    ):
        self.goal = jnp.array([0.0, 0.0, 0.0])
        self.obstacles = obstacles
        self.init_pose = jnp.array([1.42, -1.04, jnp.pi])
        self.sample_init_pose = sample_init_pose
        self.control_penalty_scale = control_penalty_scale
        self.last_action_penalty_scale = last_action_penalty_scale
        self.action_delay = action_delay
        self.observation_delay = observation_delay
        self.sliding_window = sliding_window  # Store the sliding window size
        self.angle_idx = 2
        self.dim_action = (2,)
        self.encode_angle = True
        self.dim_state = (7,) if self.encode_angle else (6,)
        self.dynamics_model: RaceCarDynamics | HardwareDynamics = (
            RaceCarDynamics(dt=dt) if hardware is None else hardware
        )
        self.sys = CarParams(**car_model_params)

    def _obs(self, state: jnp.array) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        # add observation noise
        if self.encode_angle:
            obs = encode_angles(state, self.angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def _init_delay_buffers(self, obs):
        zero_action = jnp.zeros(self.action_size)
        action_buffer = jnp.tile(zero_action[None], (self.action_delay + 1, 1))
        obs_buffer = jnp.tile(obs[None], (self.observation_delay + 1, 1))
        return obs_buffer, action_buffer

    def _init_stack_buffers(self, obs):
        zero_action = jnp.zeros(self.action_size)
        action_stack = jnp.tile(zero_action[None], (self.sliding_window, 1))
        obs_stack = jnp.tile(obs[None], (self.sliding_window, 1))
        return obs_stack, action_stack

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to a random initial state close to the initial pose"""
        key_pos, key_vel, key_obs = jax.random.split(rng, 3)
        if isinstance(self.dynamics_model, HardwareDynamics):
            for _ in range(2):
                self.dynamics_model.step(
                    jnp.zeros(self.dim_state), jnp.zeros(self.dim_action), self.sys
                )
                init_state = self.dynamics_model.mocap_state()
            init_pos = init_state[:2]
        else:

            def sample_init_pos(ins):
                _, key = ins
                key, nkey = jax.random.split(key, 2)
                x_key, y_key = jax.random.split(key, 2)
                init_x = jax.random.uniform(x_key, shape=(1,), minval=1.0, maxval=3.0)
                init_y = jax.random.uniform(y_key, shape=(1,), minval=-2.3, maxval=1.5)
                init_pos = jnp.concatenate([init_x, init_y])
                return init_pos, nkey

            if self.sample_init_pose:
                init_pos, key_pos = jax.lax.while_loop(
                    lambda ins: (
                        cost_fn(ins[0], self.obstacles, scale_factor=1.5) > 0.0
                    )
                    | ((ins[1] == key_pos).all()),
                    sample_init_pos,
                    (self.init_pose[:2], key_pos),
                )
                init_theta = self.init_pose[2:] + jax.random.uniform(
                    key_pos, shape=(1,), minval=-jnp.pi, maxval=jnp.pi
                )
            else:
                init_pos = self.init_pose[:2] + jax.random.uniform(
                    key_pos, shape=(2,), minval=-0.10, maxval=0.10
                )
                init_theta = self.init_pose[2:] + jax.random.uniform(
                    key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
                )
            init_vel = jnp.zeros((3,)) + jnp.array(
                [0.005, 0.005, 0.02]
            ) * jax.random.normal(key_vel, shape=(3,))
            init_state = jnp.concatenate([init_pos, init_theta, init_vel])
        init_obs = self._obs(init_state)
        obs_buffer, action_buffer = self._init_delay_buffers(jnp.zeros_like(init_obs))
        if self.observation_delay == 0:
            obs_buffer = None
        if self.action_delay == 0:
            action_buffer = None
        if self.sliding_window > 0:
            obs_stack, action_stack = self._init_stack_buffers(jnp.zeros_like(init_obs))
            stacked_obs = self._get_stacked_obs(obs_stack, action_stack)
        else:
            obs_stack = None
            action_stack = None
            stacked_obs = init_obs

        return State(
            pipeline_state=(init_state, key_pos, jnp.linalg.norm(init_pos)),
            obs=stacked_obs,  # Use stacked observation if sliding window is enabled
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            info={
                "cost": jnp.array(0.0),
                "last_act": jnp.zeros((self.action_size)),  # Last action taken
                "action_buffer": action_buffer,  # Buffer for action delay
                "obs_buffer": obs_buffer,  # Buffer for observation delay
                "obs_stack": obs_stack,  # Buffer for observation history (sliding window)
                "action_stack": action_stack,  # Buffer for action history (sliding window)
            },
            metrics={
                "get_close_reward": jnp.array(0.0),
                "goal_achieved": jnp.array(0.0),
                "action_magnitude": jnp.array(0.0),
                "action_jitter": jnp.array(0.0),
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        def where_done(done, x, y):
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[1].set(self.sys.max_throttle * action[1])

        if self.sliding_window > 0:
            action_stack = state.info["action_stack"]
            obs_stack = state.info["obs_stack"]
        if self.action_delay > 0:
            new_action_buffer = jnp.roll(state.info["action_buffer"], shift=-1, axis=0)
            new_action_buffer = new_action_buffer.at[-1].set(action)
            delayed_action = new_action_buffer[0]  # Use the oldest action in buffer
        else:
            new_action_buffer = None
            delayed_action = action

        # Environment step
        dynamics_state = state.pipeline_state[0]
        next_dynamics_state, step_info = self.dynamics_model.step(
            dynamics_state,
            delayed_action,
            self.sys,
        )
        key = state.pipeline_state[1]
        nkey, key = jax.random.split(key, 2)
        goal_dist = jnp.linalg.norm(next_dynamics_state[:2])
        prev_goal_dist = state.pipeline_state[2]
        get_close_reward = prev_goal_dist - goal_dist
        goal_achieved = jnp.less_equal(goal_dist, 0.35).astype(jnp.float32)
        # TODO (yarden): take only the second component of action that relates
        # to linear velocity
        action_magnitude_cost = -(
            jnp.linalg.norm(delayed_action) * self.control_penalty_scale
        )
        last_act = state.info["last_act"]
        action_jitter_cost = -(
            jnp.sum(jnp.square(delayed_action - last_act))
            * self.last_action_penalty_scale
        )
        reward = (
            get_close_reward
            + action_magnitude_cost
            + action_jitter_cost
            + goal_achieved
        )
        cost = cost_fn(dynamics_state[..., :2], self.obstacles)
        if not isinstance(self.dynamics_model, HardwareDynamics):
            negative_vel = -dynamics_state[..., 3:5]
            next_dynamics_state = jnp.where(
                cost == 0.0,
                next_dynamics_state,
                dynamics_state.at[..., 3:5].set(negative_vel * 0.05),
            )
            next_dynamics_state = jnp.where(
                cost == 0.0,
                next_dynamics_state,
                next_dynamics_state.at[..., :2].set(
                    dynamics_state[..., :2] + negative_vel * 0.1
                ),
            )
        next_obs = self._obs(next_dynamics_state)
        vx, vy = dynamics_state[..., 3:5]
        energy = 0.5 * self.sys.m * (vx**2 + vy**2)
        done = 1.0 - in_arena(next_obs[..., :2], 1.3)

        if self.observation_delay > 0:
            new_obs_buffer = jnp.roll(state.info["obs_buffer"], shift=-1, axis=0)
            new_obs_buffer = new_obs_buffer.at[-1].set(next_obs)
            delayed_obs = new_obs_buffer[0]  # Use the oldest observation in buffer
            init_obs_buffer, init_action_buffer = self._init_delay_buffers(
                jnp.zeros_like(next_obs)
            )
            new_obs_buffer = where_done(done, init_obs_buffer, new_obs_buffer)
            new_action_buffer = where_done(done, init_action_buffer, new_action_buffer)
        else:
            new_obs_buffer = None
            delayed_obs = next_obs

        # Handle sliding window (frame stacking) if enabled
        if self.sliding_window > 0:
            new_obs_stack = jnp.roll(obs_stack, shift=-1, axis=0)
            new_obs_stack = new_obs_stack.at[-1].set(delayed_obs)

            # Update action stack
            new_action_stack = jnp.roll(action_stack, shift=-1, axis=0)
            new_action_stack = new_action_stack.at[-1].set(delayed_action)

            stacked_obs = self._get_stacked_obs(new_obs_stack, new_action_stack)

            init_obs_stack, init_action_stack = self._init_stack_buffers(
                jnp.zeros_like(next_obs)
            )
            new_obs_stack = where_done(done, init_obs_stack, new_obs_stack)
            new_action_stack = where_done(done, init_action_stack, new_action_stack)

            info = {
                **state.info,
                "cost": jnp.where(cost > 0.0, energy, 0.0),
                **step_info,
                "last_act": delayed_action,
                "action_buffer": new_action_buffer,
                "obs_buffer": new_obs_buffer,
                "obs_stack": new_obs_stack,
                "action_stack": new_action_stack,
            }
            final_obs = stacked_obs
        else:
            # Without sliding window, just use delayed observation
            info = {
                **state.info,
                "cost": jnp.where(cost > 0.0, energy, 0.0),
                **step_info,
                "last_act": delayed_action,
                "action_buffer": new_action_buffer,
                "obs_buffer": new_obs_buffer,
            }
            final_obs = delayed_obs
        state.metrics["get_close_reward"] = get_close_reward
        state.metrics["goal_achieved"] = goal_achieved
        state.metrics["action_magnitude"] = action_magnitude_cost
        state.metrics["action_jitter"] = action_jitter_cost
        return State(
            pipeline_state=(next_dynamics_state, nkey, goal_dist),
            obs=final_obs,
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=info,
        )

    @property
    def observation_size(self) -> int:
        base_size = 7 if self.encode_angle else 6
        if self.sliding_window > 0:
            # If sliding window is enabled, multiply by window size and add action history
            return self.sliding_window * (base_size + self.action_size)
        return base_size

    @property
    def action_size(self) -> int:
        # [steering, throttle]
        return 2

    def backend(self) -> str:
        return "positional"

    def _get_stacked_obs(
        self, obs_stack: jax.Array, action_stack: jax.Array
    ) -> jax.Array:
        """Combine the observation and action stacks into a single observation."""
        flat_obs = obs_stack.reshape(-1)
        flat_actions = action_stack.reshape(-1)
        return jnp.concatenate([flat_obs, flat_actions])


def render(env, policy, steps, rng):
    import numpy as np

    state = env.reset(rng)
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    trajectory = jax.tree_map(lambda x: x[:, 0], trajectory.obs)
    if env.encode_angle:
        trajectory = decode_angles(trajectory, 2)
    images = [
        draw_scene(trajectory, timestep, env.obstacles) for timestep in range(steps)
    ]
    return np.asanyarray(images).transpose(0, 3, 1, 2)


def draw_scene(obs, timestep, obstacles):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    # Create a figure and axis
    fig = Figure(figsize=(2.5, 2.5), dpi=300)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    pos_domain_size = 3.5
    ax.set_xlim(-pos_domain_size, pos_domain_size)
    ax.set_ylim(-pos_domain_size, pos_domain_size)
    target_center = (0, 0)
    colors = ["red", "white", "blue"]
    radii = [0.3, 0.2, 0.1]

    for radius, color in zip(radii, colors):
        circle = Circle(target_center, radius, color=color, ec="black", lw=0.5)
        ax.add_patch(circle)
    # Rotate coordinates if required
    rotated_trajectory = rotate_coordinates(obs, encode_angle=False)
    # Plot the car's position and velocity at the specified timestep
    x, y = rotated_trajectory[timestep, 0], rotated_trajectory[timestep, 1]
    vx, vy = rotated_trajectory[timestep, 3], rotated_trajectory[timestep, 4]
    car_width, car_length = 0.07, 0.2
    car = Rectangle(
        (x - car_length / 2, y - car_width / 2),
        car_length,
        car_width,
        angle=rotated_trajectory[timestep, 2] * 180 / np.pi,
        color="green",
        alpha=0.7,
        ec="black",
        rotation_point="center",
    )
    ax.add_patch(car)
    for obstacle in obstacles:
        position, radius = obstacle[:2], obstacle[2]
        obstacle_position = jnp.array([position[1], -position[0]])
        obstacle = Circle(
            obstacle_position, radius, color="gray", alpha=0.5, ec="black", lw=1.5
        )
        ax.add_patch(obstacle)
    ax.quiver(
        x,
        y,
        vx,
        vy,
        color="blue",
        scale=10,
        headlength=3,
        headaxislength=3,
        headwidth=3,
        linewidth=0.5,
    )
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    # Render figure to canvas and retrieve RGB array
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").copy()
    image = image.reshape(*reversed(canvas.get_width_height()), 3)
    return image
