from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

try:
    import tensorflow as tf
    import tf2onnx
    from tensorflow.keras import layers  # type: ignore
except ImportError:
    tf = None
    layers = None
    tf2onnx = None
    logging.warning("TensorFlow is not installed. Skipping conversion to ONNX.")


logger = logging.getLogger(__name__)


def _get_path(tree: Any, *path: str, default: Any = None) -> Any:
    cur = tree
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _extract_policy_params(params: Any) -> Mapping[str, Any]:
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


def _sorted_hidden_keys(mlp_params: Mapping[str, Any]) -> list[str]:
    hidden_keys = [k for k in mlp_params.keys() if k.startswith("hidden_")]
    hidden_keys.sort(key=lambda x: int(x.split("_")[-1]))
    return hidden_keys


def _infer_action_size(policy_params: Mapping[str, Any]) -> int:
    mlp_params = policy_params["MLP_0"]
    hidden_keys = _sorted_hidden_keys(mlp_params)
    if not hidden_keys:
        raise ValueError("MLP_0 has no hidden_* layers.")
    last_bias = np.asarray(mlp_params[hidden_keys[-1]]["bias"])
    if last_bias.shape[-1] % 2 != 0:  # type: ignore
        raise ValueError(
            f"Expected even policy logits dimension, got {last_bias.shape[-1]}."  # type: ignore
        )
    return int(last_bias.shape[-1] // 2)  # type: ignore


def _infer_hidden_layers(policy_params: Mapping[str, Any]) -> tuple[int, ...]:
    mlp_params = policy_params["MLP_0"]
    hidden_keys = _sorted_hidden_keys(mlp_params)
    if len(hidden_keys) < 2:
        return ()
    return tuple(
        int(np.asarray(mlp_params[k]["bias"]).shape[0])  # type: ignore
        for k in hidden_keys[:-1]  # type: ignore
    )


def _resolve_render_hw(
    cfg: Any, default_h: int = 64, default_w: int = 64
) -> tuple[int, int]:
    h = _get_path(cfg, "environment", "task_params", "vision_config", "render_height")
    w = _get_path(cfg, "environment", "task_params", "vision_config", "render_width")
    if h is None or w is None:
        return default_h, default_w
    return int(h), int(w)


def _resolve_hidden_layers(cfg: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    from_cfg = _get_path(cfg, "agent", "policy_hidden_layer_sizes")
    if from_cfg is None:
        return fallback
    return tuple(int(x) for x in from_cfg)


def _resolve_tf_activation(cfg: Any):
    activation_name = _get_path(cfg, "agent", "activation")
    if isinstance(activation_name, str):
        activation = getattr(tf.nn, activation_name, None)
        if activation is not None:
            return activation
    # Matches train_pick_cartesian_vision_ppo.py
    return tf.nn.relu


if tf is not None and layers is not None:

    class VisionPPOPolicy(tf.keras.Model):
        """TensorFlow equivalent of Brax VisionMLP policy head for PPO."""

        def __init__(
            self,
            action_size: int,
            pixel_obs_keys: Sequence[str],
            hidden_layer_sizes: Sequence[int],
            activation: Any,
            normalise_channels: bool = True,
            state_obs_key: str = "",
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.action_size = action_size
            self.pixel_obs_keys = tuple(pixel_obs_keys)
            self.normalise_channels = normalise_channels
            del state_obs_key  # state is intentionally not used by this policy.

            self.cnn_blocks = []
            for i, _ in enumerate(self.pixel_obs_keys):
                cnn = tf.keras.Sequential(name=f"CNN_{i}")
                cnn.add(
                    layers.Conv2D(
                        filters=32,
                        kernel_size=(8, 8),
                        strides=(4, 4),
                        padding="same",
                        activation=tf.nn.relu,
                        use_bias=False,
                        name="Conv_0",
                    )
                )
                cnn.add(
                    layers.Conv2D(
                        filters=64,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding="same",
                        activation=tf.nn.relu,
                        use_bias=False,
                        name="Conv_1",
                    )
                )
                cnn.add(
                    layers.Conv2D(
                        filters=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        activation=tf.nn.relu,
                        use_bias=False,
                        name="Conv_2",
                    )
                )
                self.cnn_blocks.append(cnn)

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            layer_sizes = list(hidden_layer_sizes) + [action_size * 2]
            for i, size in enumerate(layer_sizes):
                self.mlp_block.add(
                    layers.Dense(
                        units=size,
                        activation=activation if i < len(layer_sizes) - 1 else None,
                        name=f"hidden_{i}",
                    )
                )

        @staticmethod
        def _normalise_channels(x: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
            mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
            var = tf.reduce_mean(tf.square(x - mean), axis=(1, 2), keepdims=True)
            return (x - mean) * tf.math.rsqrt(var + eps)

        def call(self, obs: Mapping[str, tf.Tensor]) -> tf.Tensor:
            cnn_outs = []
            for i, key in enumerate(self.pixel_obs_keys):
                hidden = obs[key]
                if self.normalise_channels:
                    hidden = self._normalise_channels(hidden)
                hidden = self.cnn_blocks[i](hidden)
                hidden = tf.reduce_mean(hidden, axis=(1, 2))
                cnn_outs.append(hidden)

            logits = self.mlp_block(tf.concat(cnn_outs, axis=-1))
            loc, _ = tf.split(logits, 2, axis=-1)
            action = tf.tanh(loc)
            return action

else:

    class VisionPPOPolicy:  # type: ignore[no-redef]
        pass


def transfer_weights(
    policy_params: Mapping[str, Any], tf_model: tf.keras.Model
) -> None:
    """Copies Flax PPO vision policy weights into TF model."""
    cnn_names = sorted(
        [k for k in policy_params.keys() if k.startswith("CNN_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    for i, cnn_name in enumerate(cnn_names):
        tf_cnn = tf_model.get_layer(f"CNN_{i}")
        for conv_name in ("Conv_0", "Conv_1", "Conv_2"):
            kernel = np.asarray(policy_params[cnn_name][conv_name]["kernel"])
            tf_layer = tf_cnn.get_layer(conv_name)
            tf_layer.set_weights([kernel])

    tf_mlp = tf_model.get_layer("MLP_0")
    hidden_keys = _sorted_hidden_keys(policy_params["MLP_0"])
    for hidden_name in hidden_keys:
        layer_params = policy_params["MLP_0"][hidden_name]
        kernel = np.asarray(layer_params["kernel"])
        bias = np.asarray(layer_params["bias"])
        tf_layer = tf_mlp.get_layer(hidden_name)
        tf_layer.set_weights([kernel, bias])


def convert_policy_to_onnx(
    make_inference_fn: Any,
    params: Any,
    cfg: Any | None = None,
    observation_shapes: Mapping[str, tuple[int, ...]] | None = None,
    pixel_obs_keys: Sequence[str] | None = None,
    state_obs_key: str = "",
    normalise_channels: bool = True,
):
    """Converts PPO vision policy params to ONNX via TensorFlow."""
    if tf is None or tf2onnx is None or layers is None:
        raise ImportError("TensorFlow/tf2onnx is required for ONNX export.")

    policy_params = _extract_policy_params(params)
    action_size = _infer_action_size(policy_params)
    hidden_layers = _resolve_hidden_layers(cfg, _infer_hidden_layers(policy_params))
    activation = _resolve_tf_activation(cfg)

    cnn_names = sorted(
        [k for k in policy_params.keys() if k.startswith("CNN_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not cnn_names:
        raise ValueError("No CNN_* modules found in policy params.")

    if pixel_obs_keys is None:
        pixel_obs_keys = tuple(f"pixels/view_{i}" for i in range(len(cnn_names)))
    if len(pixel_obs_keys) != len(cnn_names):
        raise ValueError(
            f"Expected {len(cnn_names)} pixel keys, got {len(pixel_obs_keys)}."
        )

    if observation_shapes is None:
        observation_shapes = {}
    else:
        observation_shapes = dict(observation_shapes)

    h, w = _resolve_render_hw(cfg)
    used_observation_shapes: dict[str, tuple[int, ...]] = {}
    for i, key in enumerate(pixel_obs_keys):
        if key in observation_shapes:
            used_observation_shapes[key] = tuple(observation_shapes[key])
            continue
        channels = int(
            np.asarray(policy_params[f"CNN_{i}"]["Conv_0"]["kernel"]).shape[2]  # type: ignore
        )
        used_observation_shapes[key] = (h, w, channels)
    del state_obs_key  # state is intentionally never passed to the policy network.

    tf_policy_network = VisionPPOPolicy(
        action_size=action_size,
        pixel_obs_keys=pixel_obs_keys,
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        normalise_channels=normalise_channels,
    )

    model_input_shapes = dict(used_observation_shapes)
    if "state" in observation_shapes:
        model_input_shapes["state"] = tuple(observation_shapes["state"])
    dummy_obs = {
        k: np.ones((1, *shape), dtype=np.float32)
        for k, shape in model_input_shapes.items()
    }
    tf_policy_network(dummy_obs).numpy()
    transfer_weights(policy_params, tf_policy_network)

    inference_fn = make_inference_fn(params, deterministic=True)
    jax_obs = {
        k: jax.numpy.asarray(v)
        for k, v in dummy_obs.items()
        if k in used_observation_shapes
    }
    jax_pred = np.asarray(inference_fn(jax_obs, jax.random.PRNGKey(0))[0][0])
    tf_pred = np.asarray(tf_policy_network(dummy_obs).numpy()[0])
    max_abs_err = float(np.max(np.abs(jax_pred - tf_pred)))
    logger.info("PPO vision ONNX export sanity max_abs_err=%.6e", max_abs_err)

    tf_policy_network.output_names = ["continuous_actions"]
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=[
            {
                k: tf.TensorSpec([1, *shape], tf.float32, name=k)
                for k, shape in model_input_shapes.items()
            }
        ],
        opset=11,
    )
    return model_proto


def make_franka_policy(
    make_policy_fn: Any, params: Any, cfg: Any | None = None
) -> bytes:
    """Builds and serializes an ONNX PPO vision policy for Franka pick tasks."""
    policy_params = _extract_policy_params(params)
    cnn_names = sorted(
        [k for k in policy_params.keys() if k.startswith("CNN_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    h, w = _resolve_render_hw(cfg)
    obs_shapes = {}
    obs_keys = []
    for i, cnn_name in enumerate(cnn_names):
        channels = int(np.asarray(policy_params[cnn_name]["Conv_0"]["kernel"]).shape[2])  # type: ignore
        key = f"pixels/view_{i}"
        obs_keys.append(key)
        obs_shapes[key] = (h, w, channels)

    model_proto = convert_policy_to_onnx(
        make_policy_fn,
        params,
        cfg=cfg,
        observation_shapes=obs_shapes,
        pixel_obs_keys=obs_keys,
        state_obs_key="",
        normalise_channels=True,
    )
    return model_proto.SerializeToString()
