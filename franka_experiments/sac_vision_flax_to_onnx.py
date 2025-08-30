import jax
import jax.nn as jnn
from brax.training.acme import running_statistics
from hydra import compose, initialize

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac import franka_sac_to_onnx
from ss2r.algorithms.sac.vision_networks import make_sac_vision_networks


def get_cfg():
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "writers=[stderr]",
                "+experiment=franka_online",
            ],
        )
        return cfg


def test_policy_to_onnx_export():
    # Define dummy config
    cfg = get_cfg()
    act_size = 3
    obs_shape = (64, 64, 1)  # shape for each image input
    # Dummy observation with a single pixel input
    activation = getattr(jnn, cfg.agent.activation)
    sac_network = make_sac_vision_networks(
        observation_size={"pixels/view_0": obs_shape, "state": (3,)},
        action_size=act_size,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
        activation=activation,
        tanh=cfg.agent.tanh,
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = sac_network.policy_network.init(jax.random.PRNGKey(0))
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    franka_sac_to_onnx.make_franka_policy(make_inference_fn, (None, params), cfg)


if __name__ == "__main__":
    for _ in range(3):
        test_policy_to_onnx_export()
