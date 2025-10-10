from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from brax.training import gradients


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(*args, optimizer_state, params=None):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(
            grads, optimizer_state, params
        )
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f


def ensemble_gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str] = None,
    has_aux: bool = False,
):
    """
    Ensemble-aware gradient update wrapper:
      f(*args, optimizer_state, params=None) -> (value_mean, new_params, new_optimizer_state)
    """
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(*args, optimizer_state, params=None):
        transitions = args[5]
        key = args[6]

        trans_per_ens = split_transitions_ensemble(transitions, ensemble_axis=1)
        shapes = jax.tree_map(lambda x: jnp.asarray(x).shape, trans_per_ens)
        print("Transitions shapes0:", shapes)
        sample_leaf = jax.tree_util.tree_leaves(trans_per_ens)[0]
        E = sample_leaf.shape[0]

        _key, *ens_keys = jax.random.split(key, E + 1)
        ens_keys = jnp.stack(ens_keys)

        n_args = len(args)
        in_axes = [None] * n_args
        in_axes[5] = 0
        in_axes[6] = 0

        # vmap wrapper: returns (E, value) and (E, grads)
        vmap_fn = jax.vmap(
            lambda *vmapped_args: loss_and_pgrad_fn(*vmapped_args),
            in_axes=tuple(in_axes),
        )
        vmapped_args = list(args)
        vmapped_args[5] = trans_per_ens
        vmapped_args[6] = ens_keys

        value_per_ens, grads_per_ens = vmap_fn(*vmapped_args)

        loss_mean = jnp.mean(value_per_ens)
        grads_mean = jax.tree_util.tree_map(
            lambda g: jnp.mean(g, axis=0), grads_per_ens
        )

        params_update, optimizer_state = optimizer.update(
            grads_mean, optimizer_state, params
        )
        new_params = optax.apply_updates(args[0], params_update)

        return loss_mean, new_params, optimizer_state

    return f


def split_transitions_ensemble(transitions: Any, ensemble_axis: int = 2) -> Any:
    ref = (
        transitions.next_observation
        if hasattr(transitions, "next_observation")
        else jax.tree_util.tree_leaves(transitions)[0]
    )
    ref = jnp.asarray(ref)
    if ref.ndim <= ensemble_axis:
        raise ValueError(
            f"Reference leaf has ndim {ref.ndim} <= ensemble_axis {ensemble_axis}"
        )

    E = ref.shape[ensemble_axis]

    def _per_ens_leaf(x):
        x = jnp.asarray(x)
        if x.ndim > ensemble_axis and x.shape[ensemble_axis] == E:
            perm = list(range(x.ndim))
            perm.pop(ensemble_axis)
            perm = [ensemble_axis] + perm
            return jnp.transpose(x, axes=perm)
        else:  # no ensemble axis
            expanded = jnp.expand_dims(x, axis=0)  # (1, U, B, ...)
            target_shape = (E,) + x.shape
            return jnp.broadcast_to(expanded, target_shape)

    trans_per_ens = jax.tree_util.tree_map(_per_ens_leaf, transitions)

    sample_leaf = jax.tree_util.tree_leaves(trans_per_ens)[0]
    slshape = sample_leaf.shape
    B = int(slshape[1])
    idx = jnp.arange(E, dtype=jnp.int32)[:, None, None]  # (E,1,1)
    idx = jnp.broadcast_to(idx, (E, B, 1))  # (E,B,1)

    def _update_extras_map(extras):
        extras_dict = dict(extras)
        state_extras = dict(extras_dict.get("state_extras", {}))
        state_extras["idx"] = idx
        extras_dict["state_extras"] = state_extras
        return extras_dict

    orig_extras = trans_per_ens.extras
    new_extras = _update_extras_map(orig_extras)

    new_trans = trans_per_ens._replace(extras=new_extras)
    return new_trans
