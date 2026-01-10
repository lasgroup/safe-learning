"""Utilities for randomization."""

import jax
import jax.numpy as jnp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16
NUM_ACTUATED_DOFS = 29


def domain_randomization(model: mjx.Model, rng: jax.Array, cfg):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor / foot friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(
            key, minval=cfg.floor_friction[0], maxval=cfg.floor_friction[1]
        )
        pair_friction = model.pair_friction.at[
            FLOOR_GEOM_ID : FLOOR_GEOM_ID + 2,
            FLOOR_GEOM_ID : FLOOR_GEOM_ID + 2,
        ].set(friction)

        # Scale static friction: *U(cfg.scale_friction).
        rng, key = jax.random.split(rng)
        friction_scale = jax.random.uniform(
            key,
            shape=(NUM_ACTUATED_DOFS,),
            minval=cfg.scale_friction[0],
            maxval=cfg.scale_friction[1],
        )
        frictionloss = model.dof_frictionloss[6:] * friction_scale
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature_scale = jax.random.uniform(
            key,
            shape=(NUM_ACTUATED_DOFS,),
            minval=cfg.scale_armature[0],
            maxval=cfg.scale_armature[1],
        )
        armature = model.dof_armature[6:] * armature_scale
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key,
            shape=(model.nbody,),
            minval=cfg.scale_link_mass[0],
            maxval=cfg.scale_link_mass[1],
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-1.0, 1.0).
        rng, key = jax.random.split(rng)
        dmass_torso = jax.random.uniform(
            key, minval=cfg.add_torso_mass[0], maxval=cfg.add_torso_mass[1]
        )
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + dmass_torso
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0_jitter = jax.random.uniform(
            key,
            shape=(NUM_ACTUATED_DOFS,),
            minval=cfg.jitter_qpos0[0],
            maxval=cfg.jitter_qpos0[1],
        )
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(qpos0[7:] + qpos0_jitter)

        samples = jnp.hstack(
            [
                jnp.array([friction]),
                friction_scale,
                armature_scale,
                dmass,
                jnp.array([dmass_torso]),
                qpos0_jitter,
            ]
        )

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            samples,
        )

    (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
        samples,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
        }
    )

    model = model.tree_replace(
        {
            "pair_friction": pair_friction,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
        }
    )

    return model, in_axes, samples
