import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from flax import linen

from ss2r.algorithms.ppo import franka_ppo_to_onnx

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _make_ort_session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def test_policy_to_onnx_export(num_tests: int = 5, atol: float = 1e-4):
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

    if ort is None:
        print("Skipping ONNX runtime parity check: onnxruntime is not installed.")
        return

    session = ort.InferenceSession(
        model_proto.SerializeToString(),
        sess_options=_make_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    jax_policy = make_inference_fn((None, policy_params, None), deterministic=True)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for i in range(num_tests):
        obs = {"pixels/view_0": rng.standard_normal((1, *obs_shape), dtype=np.float32)}
        onnx_action = session.run(["continuous_actions"], obs)[0][0]
        jax_action = np.asarray(
            jax_policy(
                {"pixels/view_0": jax.numpy.asarray(obs["pixels/view_0"])},
                jax.random.PRNGKey(i),
            )[0][0]
        )
        err = float(np.max(np.abs(onnx_action - jax_action)))
        max_err = max(max_err, err)
        print(f"sample={i} max_abs_err={err:.6e}")
        print(f"  jax : {jax_action}")
        print(f"  onnx: {onnx_action}")

    print(f"overall max_abs_err={max_err:.6e} (atol={atol:.1e})")
    if max_err > atol:
        raise AssertionError(
            f"ONNX/JAX mismatch: max_abs_err={max_err:.6e} > atol={atol}"
        )


if __name__ == "__main__":
    test_policy_to_onnx_export()
