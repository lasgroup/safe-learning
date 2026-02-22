import argparse
from pathlib import Path
from typing import Mapping

import jax
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from flax import linen

from ss2r.algorithms.ppo import franka_ppo_to_onnx

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _resolve_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    config_name = "ppo_network_config.json"
    if (path / config_name).exists():
        return path

    step_dirs = [p for p in path.iterdir() if p.is_dir() and (p / config_name).exists()]
    if not step_dirs:
        raise FileNotFoundError(
            f"No checkpoint step with {config_name} found under: {path}"
        )

    numeric = [p for p in step_dirs if p.name.isdigit()]
    if numeric:
        return max(numeric, key=lambda p: int(p.name))
    return sorted(step_dirs)[-1]


def _shape_tuple(raw: object) -> tuple[int, ...]:
    if isinstance(raw, tuple):
        return tuple(int(x) for x in raw)
    if isinstance(raw, list):
        return tuple(int(x) for x in raw)
    if isinstance(raw, Mapping) and "shape" in raw:
        return _shape_tuple(raw["shape"])
    raise TypeError(f"Unsupported shape payload: {raw}")


def _extract_policy_params(params) -> Mapping[str, object]:
    policy_params = params[1] if isinstance(params, (tuple, list)) else params
    if isinstance(policy_params, Mapping) and "params" in policy_params:
        policy_params = policy_params["params"]
    if not isinstance(policy_params, Mapping):
        raise TypeError("Could not extract policy parameter mapping.")
    return policy_params


def _extract_pixel_obs_shapes(
    observation_size_cfg: Mapping[str, object],
    policy_params: Mapping[str, object],
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    pixel_obs_keys = tuple(k for k in observation_size_cfg if k.startswith("pixels/"))
    if not pixel_obs_keys:
        raise ValueError(
            "No pixel observation keys found in checkpoint observation_size."
        )

    obs_shapes: dict[str, tuple[int, ...]] = {}
    for i, key in enumerate(pixel_obs_keys):
        try:
            shape = _shape_tuple(observation_size_cfg[key])
        except Exception:
            shape = ()

        if len(shape) == 3:
            obs_shapes[key] = shape
            continue

        channels = int(
            np.asarray(policy_params[f"CNN_{i}"]["Conv_0"]["kernel"]).shape[2]  # type: ignore[index]
        )
        obs_shapes[key] = (64, 64, channels)

    return pixel_obs_keys, obs_shapes


def _extract_state_shape(
    observation_size_cfg: Mapping[str, object],
    params,
) -> tuple[int, ...]:
    if isinstance(params, (tuple, list)) and params:
        running_stats = params[0]
        mean = getattr(running_stats, "mean", None)
        if isinstance(mean, Mapping) and "state" in mean:
            shape = tuple(int(d) for d in np.asarray(mean["state"]).shape)  # type: ignore
            if shape:
                return shape

    if "state" in observation_size_cfg:
        try:
            shape = _shape_tuple(observation_size_cfg["state"])
            if shape:
                return shape
        except Exception:
            pass

    return (1,)


def _make_ort_session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def _onnx_input_shapes(session: "ort.InferenceSession") -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for inp in session.get_inputs():
        raw_shape = list(inp.shape)[1:]
        if any((d is None or isinstance(d, str)) for d in raw_shape):
            raise ValueError(
                f"Cannot infer static shape for ONNX input {inp.name}: {inp.shape}."
            )
        shapes[inp.name] = tuple(int(d) for d in raw_shape)
    return shapes


def _build_random_obs(
    rng: np.random.Generator,
    obs_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    return {
        k: rng.standard_normal((1, *shape), dtype=np.float32)
        for k, shape in obs_shapes.items()
    }


def convert_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str | None = None,
    num_tests: int = 0,
    atol: float = 1e-4,
) -> Path:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path)
    print(f"Using PPO checkpoint: {ckpt_dir}")

    params = ppo_checkpoint.load(ckpt_dir)
    config = ppo_checkpoint.load_config(ckpt_dir)
    config_dict = config.to_dict()

    observation_size_cfg = config_dict.get("observation_size")
    if not isinstance(observation_size_cfg, dict):
        raise TypeError("Checkpoint observation_size must be a mapping for vision PPO.")

    policy_params = _extract_policy_params(params)
    pixel_obs_keys, network_obs_shapes = _extract_pixel_obs_shapes(
        observation_size_cfg, policy_params
    )

    network_factory_kwargs = config_dict.get("network_factory_kwargs", {})
    if not isinstance(network_factory_kwargs, dict):
        network_factory_kwargs = {}
    network_factory_kwargs = dict(network_factory_kwargs)

    activation_name = str(network_factory_kwargs.pop("activation_name", ""))
    activation = getattr(linen, activation_name, None) if activation_name else None
    if activation_name and activation is None:
        raise ValueError(
            f"Unsupported activation_name in checkpoint: {activation_name!r}"
        )

    # Force pixel-only policy reconstruction.
    network_factory_kwargs["policy_obs_key"] = ""
    network_factory_kwargs["value_obs_key"] = ""

    normalize = (
        running_statistics.normalize
        if bool(config_dict.get("normalize_observations", False))
        else (lambda x, y: x)
    )

    ppo_network_kwargs = dict(network_factory_kwargs)
    if activation is not None:
        ppo_network_kwargs["activation"] = activation

    ppo_network = ppo_networks_vision.make_ppo_networks_vision(
        observation_size=network_obs_shapes,
        action_size=int(config_dict["action_size"]),
        preprocess_observations_fn=normalize,
        **ppo_network_kwargs,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    export_obs_shapes = dict(network_obs_shapes)
    export_obs_shapes["state"] = _extract_state_shape(observation_size_cfg, params)

    model_proto = franka_ppo_to_onnx.convert_policy_to_onnx(
        make_inference_fn=make_inference_fn,
        params=params,
        observation_shapes=export_obs_shapes,
        pixel_obs_keys=pixel_obs_keys,
        state_obs_key="",
        normalise_channels=True,
    )

    if output_path is None:
        output_file = ckpt_dir / "policy.onnx"
    else:
        output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(model_proto.SerializeToString())
    print(f"Wrote ONNX model: {output_file}")

    if num_tests <= 0:
        return output_file
    if ort is None:
        print("onnxruntime not installed, skipping parity checks.")
        return output_file

    session = ort.InferenceSession(
        model_proto.SerializeToString(),
        sess_options=_make_ort_session_options(),
        providers=["CPUExecutionProvider"],
    )
    onnx_shapes = _onnx_input_shapes(session)
    jax_policy = make_inference_fn(params, deterministic=True)
    jax_input_keys = set(network_obs_shapes)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for i in range(num_tests):
        obs = _build_random_obs(rng, onnx_shapes)
        onnx_action = session.run(["continuous_actions"], obs)[0][0]
        jax_obs = {
            k: jax.numpy.asarray(v) for k, v in obs.items() if k in jax_input_keys
        }
        jax_action = np.asarray(jax_policy(jax_obs, jax.random.PRNGKey(i))[0][0])
        err = float(np.max(np.abs(onnx_action - jax_action)))
        max_err = max(max_err, err)
        print(f"sample={i} max_abs_err={err:.6e}")

    print(f"overall max_abs_err={max_err:.6e} (atol={atol:.1e})")
    if max_err > atol:
        raise AssertionError(
            f"ONNX/JAX mismatch: max_abs_err={max_err:.6e} > atol={atol}"
        )

    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PPO vision policy checkpoint to ONNX."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Checkpoint step directory or parent directory containing step subdirs.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output ONNX file path (default: <checkpoint_dir>/policy.onnx).",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=0,
        help="Number of ONNX-vs-JAX parity tests to run.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for parity checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        num_tests=args.num_tests,
        atol=args.atol,
    )
