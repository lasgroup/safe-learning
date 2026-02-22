import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from flax import linen

from ss2r.algorithms.ppo import franka_ppo_to_onnx


def test_policy_to_onnx_export():
    obs_shape = (64, 64, 1)
    action_size = 3
    ppo_network = ppo_networks_vision.make_ppo_networks_vision(
        observation_size={"pixels/view_0": obs_shape, "state": (10,)},
        action_size=action_size,
        policy_hidden_layer_sizes=(256, 256),
        value_hidden_layer_sizes=(256, 256),
        activation=linen.relu,
        normalise_channels=True,
    )
    policy_params = ppo_network.policy_network.init(jax.random.PRNGKey(0))
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    try:
        model_proto = franka_ppo_to_onnx.convert_policy_to_onnx(
            make_inference_fn=make_inference_fn,
            params=(None, policy_params, None),
            observation_shapes={"pixels/view_0": obs_shape},
            pixel_obs_keys=("pixels/view_0",),
            state_obs_key="",
            normalise_channels=True,
        )
    except ImportError as exc:
        print(f"Skipping export: {exc}")
        return
    print(f"ONNX bytes: {len(model_proto.SerializeToString())}")


if __name__ == "__main__":
    test_policy_to_onnx_export()
