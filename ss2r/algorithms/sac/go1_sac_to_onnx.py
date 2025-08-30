import numpy as np
from brax.training.types import Transition

try:
    import tensorflow as tf
    import tf2onnx
    from tensorflow.keras import layers  # type: ignore
except ImportError:
    tf = None
    layers = None
    import logging

    logging.warning("TensorFlow is not installed. Skipping conversion to ONNX.")


class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.swish,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        mean_std=None,
        deterministic=True,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm
        self.deterministic = deterministic

        if mean_std is not None:
            self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
            self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        else:
            self.mean = None
            self.std = None

        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=self.activation,
                kernel_initializer=self.kernel_init,
                name=f"hidden_{i}",
                use_bias=self.bias,
            )
            self.mlp_block.add(dense_layer)
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        if not self.activate_final and self.mlp_block.layers:
            if (
                hasattr(self.mlp_block.layers[-1], "activation")
                and self.mlp_block.layers[-1].activation is not None
            ):
                self.mlp_block.layers[-1].activation = None

        self.submodules = [self.mlp_block]

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.mean is not None and self.std is not None:
            inputs = (inputs - self.mean) / self.std
        logits = self.mlp_block(inputs)
        loc, std = tf.split(logits, 2, axis=-1)
        if self.deterministic:
            return tf.tanh(loc)
        std = tf.math.softplus(std) + 0.001
        sample = tf.random.normal(tf.shape(loc)) * std + loc
        return tf.tanh(sample)


def make_policy_network(
    param_size,
    mean_std,
    hidden_layer_sizes=[256, 256],
    activation=tf.nn.relu,
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        mean_std=mean_std,
    )


def transfer_weights(jax_params, tf_model):
    """
    Transfer weights from a JAX parameter dictionary to the TensorFlow model.

    Parameters:
    - jax_params: dict
      Nested dictionary with structure {block_name: {layer_name: {params}}}.
      For example:
      {
        'CNN_0': {
          'Conv_0': {'kernel': np.ndarray},
          'Conv_1': {'kernel': np.ndarray},
          'Conv_2': {'kernel': np.ndarray},
        },
        'MLP_0': {
          'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
        }
      }

    - tf_model: tf.keras.Model
      An instance of the adapted VisionMLP model containing named submodules and layers.
    """
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
        except ValueError:
            print(f"Layer {layer_name} not found in TensorFlow model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            kernel = np.array(layer_params["kernel"])
            bias = np.array(layer_params["bias"])
            tf_layer.set_weights([kernel, bias])
        else:
            print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")


def convert_policy_to_onnx(params, cfg, act_size, obs_size):
    """
    Converts a JAX policy to ONNX format using a TensorFlow intermediate model.

    Args:
        jax_params (dict): Parameters of the JAX policy model.
        policy_cfg (OmegaConf): Configuration object with network architecture info.
        act_size (int): Size of the action space.
        obs_shape (int): Flattened size of the observation input.
        output_path (str): Path to save the ONNX model.
        mean_std (Tuple[tf.Tensor, tf.Tensor] or None): Optional normalization tensors.

    Returns:
        onnx_prediction (np.ndarray): Output from the ONNX model for sanity check.
        onnx_model (InferenceSession): Loaded ONNX runtime model.
    """
    # Define the TF policy network
    # TODO (yarden): generalize to more activation functions
    mean = params[0].mean["state"]
    std = params[0].std["state"]
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )
    example_input = tf.zeros((1, obs_size))
    tf_policy_network(example_input).numpy()[0]
    # Transfer JAX weights to TF model
    transfer_weights(params[1]["params"], tf_policy_network)
    # Export to ONNX
    tf_policy_network.output_names = ["continuous_actions"]
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=[
            tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
        ],
        opset=11,
    )
    return model_proto


def make_go1_policy(make_policy_fn, params, cfg):
    del make_policy_fn
    proto_model = convert_policy_to_onnx(params, cfg, 12, 48)
    return proto_model.SerializeToString()


def go1_postprocess_data(raw_data, extra_fields):
    observation, action, reward, next_observation, done, info = raw_data
    state_extras = {x: info[x] for x in extra_fields}
    policy_extras = {}
    transitions = Transition(
        observation=observation,
        action=action,
        reward=reward,
        discount=1 - done,
        next_observation=next_observation,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )
    return transitions
