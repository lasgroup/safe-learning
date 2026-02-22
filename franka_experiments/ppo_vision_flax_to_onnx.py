from __future__ import annotations

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


def _extract_policy_params(params):
    if isinstance(params, (tuple, list)):
        if len(params) < 2:
            raise ValueError("Expected params as (normalizer, policy, value) tuple.")
        policy_params = params[1]
    else:
        policy_params = params
    if isinstance(policy_params, Mapping) and "params" in policy_params:
        policy_params = policy_params["params"]
    if not isinstance(policy_params, Mapping):
        raise TypeError("Could not extract policy parameter mapping.")
    return policy_params


def _sorted_cnn_names(policy_params: Mapping[str, object]) -> list[str]:
    cnn_names = [k for k in policy_params.keys() if k.startswith("CNN_")]
    cnn_names.sort(key=lambda x: int(x.split("_")[-1]))
    return cnn_names


def _extract_running_stats_shapes(params) -> dict[str, tuple[int, ...]]:
    if not isinstance(params, (tuple, list)) or not params:
        return {}
    running_stats = params[0]
    mean = getattr(running_stats, "mean", None)
    if not isinstance(mean, Mapping):
        return {}

    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in mean.items():
        arr = np.asarray(value)
        shapes[str(key)] = tuple(int(d) for d in arr.shape)
    return shapes


def _resolve_checkpoint_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    config_name = "ppo_network_config.json"
    if (path / config_name).exists():
        return path

    candidates = [
        p for p in path.iterdir() if p.is_dir() and (p / config_name).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint step with {config_name} found under: {path}"
        )

    numeric = [p for p in candidates if p.name.isdigit()]
    if numeric:
        return max(numeric, key=lambda p: int(p.name))
    return sorted(candidates)[-1]


def _as_obs_shapes(
    observation_size: Mapping[str, object]
) -> dict[str, tuple[int, ...]]:
    def _shape_to_tuple(raw_shape: object, key: str) -> tuple[int, ...]:
        if isinstance(raw_shape, tuple):
            return tuple(int(x) for x in raw_shape)
        if isinstance(raw_shape, list):
            return tuple(int(x) for x in raw_shape)
        if isinstance(raw_shape, Mapping) and "shape" in raw_shape:
            return _shape_to_tuple(raw_shape["shape"], key)
        raise TypeError(f"Unsupported serialized shape for key {key}: {raw_shape}")

    shapes: dict[str, tuple[int, ...]] = {}
    for k, shape in observation_size.items():
        shapes[k] = _shape_to_tuple(shape, k)
    return shapes


def _make_ort_session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def _build_random_obs(
    rng: np.random.Generator, obs_shapes: Mapping[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    return {
        k: rng.standard_normal((1, *shape), dtype=np.float32)
        for k, shape in obs_shapes.items()
    }


def _sanitize_observation_size_for_network(
    raw_observation_size: Mapping[str, object],
    params,
    pixel_obs_keys: tuple[str, ...],
    state_obs_keys: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    obs_shapes = _as_obs_shapes(raw_observation_size)
    running_stats_shapes = _extract_running_stats_shapes(params)
    sanitized: dict[str, tuple[int, ...]] = {}
    for key, shape in obs_shapes.items():
        if key in pixel_obs_keys:
            sanitized[key] = shape
            continue
        running_shape = running_stats_shapes.get(key)
        if running_shape is not None and (shape == (1,) or shape != running_shape):
            sanitized[key] = running_shape
        else:
            sanitized[key] = shape

    for key in state_obs_keys:
        if not key:
            continue
        if key not in sanitized and key in running_stats_shapes:
            sanitized[key] = running_stats_shapes[key]

    policy_params = _extract_policy_params(params)
    cnn_names = _sorted_cnn_names(policy_params)

    if len(cnn_names) != len(pixel_obs_keys):
        raise ValueError(
            f"Checkpoint CNN count ({len(cnn_names)}) does not match pixel keys "
            f"({len(pixel_obs_keys)}): {pixel_obs_keys}"
        )

    for i, key in enumerate(pixel_obs_keys):
        shape = sanitized.get(key)
        if shape is not None and len(shape) == 3:
            continue
        in_channels = int(
            np.asarray(policy_params[cnn_names[i]]["Conv_0"]["kernel"]).shape[2]  # type: ignore[index]
        )
        sanitized[key] = (64, 64, in_channels)
    return sanitized


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

    observation_size = config_dict.get("observation_size")
    if not isinstance(observation_size, dict):
        raise TypeError("Checkpoint observation_size must be a mapping for vision PPO.")
    pixel_obs_keys = tuple(k for k in observation_size if k.startswith("pixels/"))
    if not pixel_obs_keys:
        raise ValueError(
            "No pixel observation keys found in checkpoint observation_size."
        )

    network_factory_kwargs = config_dict.get("network_factory_kwargs", {})
    if not isinstance(network_factory_kwargs, dict):
        network_factory_kwargs = {}
    network_factory_kwargs = dict(network_factory_kwargs)
    activation_name = network_factory_kwargs.pop("activation_name", "")
    activation = None
    if activation_name:
        activation = getattr(linen, activation_name, None)
        if activation is None:
            raise ValueError(
                f"Unsupported activation_name in checkpoint: {activation_name!r}"
            )

    state_obs_key = str(network_factory_kwargs.get("policy_obs_key", ""))
    value_state_obs_key = str(network_factory_kwargs.get("value_obs_key", ""))
    required_obs_keys = set(pixel_obs_keys)
    if state_obs_key:
        required_obs_keys.add(state_obs_key)
    if value_state_obs_key:
        required_obs_keys.add(value_state_obs_key)
    filtered_observation_size = {
        key: observation_size[key]
        for key in required_obs_keys
        if key in observation_size
    }
    missing_required_keys = sorted(required_obs_keys - set(filtered_observation_size))
    if missing_required_keys:
        raise ValueError(
            "Checkpoint observation_size is missing required keys: "
            + ", ".join(missing_required_keys)
        )

    normalize = (
        running_statistics.normalize
        if bool(config_dict.get("normalize_observations", False))
        else (lambda x, y: x)
    )
    observation_size_for_network = _sanitize_observation_size_for_network(
        filtered_observation_size,
        params,
        pixel_obs_keys,
        (state_obs_key, value_state_obs_key),
    )
    ppo_network_kwargs = dict(network_factory_kwargs)
    if activation is not None:
        ppo_network_kwargs["activation"] = activation
    ppo_network = ppo_networks_vision.make_ppo_networks_vision(
        observation_size=observation_size_for_network,
        action_size=int(config_dict["action_size"]),
        preprocess_observations_fn=normalize,
        **ppo_network_kwargs,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    obs_shapes = observation_size_for_network
    model_input_shapes: dict[str, tuple[int, ...]] = {}
    needs_pixel_shape_inference = False
    for key in pixel_obs_keys:
        shape = obs_shapes.get(key)
        if shape is None or len(shape) != 3:
            needs_pixel_shape_inference = True
            continue
        model_input_shapes[key] = shape
    if state_obs_key:
        state_shape = obs_shapes.get(state_obs_key)
        if state_shape is not None:
            model_input_shapes[state_obs_key] = state_shape
        elif not needs_pixel_shape_inference:
            raise ValueError(
                f"state_obs_key='{state_obs_key}' missing from checkpoint observation_size"
            )

    observation_shapes_for_export: dict[str, tuple[int, ...]] | None
    if needs_pixel_shape_inference:
        print(
            "Checkpoint pixel observation shapes are incomplete; "
            "falling back to exporter shape inference."
        )
        observation_shapes_for_export = None
    else:
        observation_shapes_for_export = model_input_shapes

    model_proto = franka_ppo_to_onnx.convert_policy_to_onnx(
        make_inference_fn=make_inference_fn,
        params=params,
        observation_shapes=observation_shapes_for_export,
        pixel_obs_keys=pixel_obs_keys,
        state_obs_key=state_obs_key,
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
    jax_policy = make_inference_fn(params, deterministic=True)

    if observation_shapes_for_export is None:
        model_input_shapes_for_test: dict[str, tuple[int, ...]] = {}
        for inp in session.get_inputs():
            raw_shape = list(inp.shape)[1:]
            if any((d is None or isinstance(d, str)) for d in raw_shape):
                raise ValueError(
                    f"Cannot infer static shape for ONNX input {inp.name}: {inp.shape}. "
                    "Run without --num_tests or provide a checkpoint with full "
                    "observation shapes."
                )
            model_input_shapes_for_test[inp.name] = tuple(int(d) for d in raw_shape)
    else:
        model_input_shapes_for_test = model_input_shapes

    rng = np.random.default_rng(0)
    max_err = 0.0
    for i in range(num_tests):
        obs = _build_random_obs(rng, model_input_shapes_for_test)
        onnx_action = session.run(["continuous_actions"], obs)[0][0]
        jax_obs = {k: jax.numpy.asarray(v) for k, v in obs.items()}
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
