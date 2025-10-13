from typing import Any, NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import optax

Params = TypeVar("Params")


class Penalizer(Protocol):
    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: Params,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        ...


class CRPOParams(NamedTuple):
    burnin: int


class CRPO:
    def __init__(self, eta: float) -> None:
        self.eta = eta

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: CRPOParams,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], CRPOParams]:
        active = jnp.greater(constraint + self.eta, 0.0) | jnp.greater(params.burnin, 0)
        if rest is not None:
            loss_constraint = rest
        else:
            loss_constraint = constraint
        actor_loss = jnp.where(
            active,
            actor_loss,
            -loss_constraint,
        )
        new_params = CRPOParams(jnp.clip(params.burnin - 1, a_min=-1))
        aux = {
            "crpo/burnin_counter": new_params.burnin,
            "crpo/active": active,
        }
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: Params
    ) -> tuple[dict[str, Any], Params]:
        return {}, params


class MultiAugmentedLagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    penalty_multiplier: jax.Array


class MultiAugmentedLagrangian:
    def __init__(self, penalty_multiplier_factor: float):
        self.penalty_multiplier_factor = penalty_multiplier_factor

    def __call__(
        self,
        actor_loss: jax.Array,
        constraints: jax.Array,
        params: MultiAugmentedLagrangianParams,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], MultiAugmentedLagrangianParams]:
        # --- Record original shapes of the params so we can restore them ---
        lam_in = jnp.asarray(params.lagrange_multiplier)
        c_in = jnp.asarray(params.penalty_multiplier)

        lam_shape = lam_in.shape  # either () or (k,)
        c_shape = c_in.shape  # either () or (k,)

        # --- Work with 1D vectors internally ---
        lam_vec = jnp.atleast_1d(lam_in)  # shape (L,) where L = 1 or k
        c_vec = jnp.atleast_1d(c_in)  # same

        # constraints should be vector of length k
        constraints = jnp.asarray(constraints)
        k = constraints.shape[0]
        # If lam_vec/c_vec are length 1 but we have k>1, broadcast/repeat them for computation
        if lam_vec.shape[0] == 1 and k > 1:
            lam_comp = jnp.repeat(lam_vec, k)
        else:
            lam_comp = lam_vec
        if c_vec.shape[0] == 1 and k > 1:
            c_comp = jnp.repeat(c_vec, k)
        else:
            c_comp = c_vec

        # Now do the standard augmented lagrangian elementwise on vectors of length k
        g = -constraints  # g>0 means violation
        cond = lam_comp + c_comp * g  # shape (k,)
        psi = jnp.where(
            cond > 0.0,
            lam_comp * g + 0.5 * c_comp * g**2,
            -0.5 * (lam_comp**2) / c_comp,
        )  # shape (k,)

        psi_total = jnp.sum(psi)  # scalar to add to actor_loss

        # updates (vector)
        new_lam_comp = jnp.maximum(0.0, cond)
        new_c_comp = jnp.where(
            cond > 0.0, c_comp * self.penalty_multiplier_factor, c_comp
        )
        # clip (optional) — mimic your previous update behavior if needed
        new_c_comp = jnp.clip(new_c_comp, a_min=c_comp, a_max=1.0)

        # --- Restore original shapes for outputs so input/output carry types match ---
        def restore_shape(vec, original_shape):
            """If original_shape is scalar (), return scalar vec[0]; else return vec reshaped to original."""
            if original_shape == ():
                # Return scalar (shape ()) — use vec[0] but keep JAX-friendly ops
                return jnp.reshape(vec[0], ())
            else:
                # If original_shape == (k,) we must ensure vec has length k
                # If vec length is 1 and original_shape is (k,), we repeat (shouldn't be the case here)
                if vec.shape[0] == 1 and original_shape[0] > 1:
                    return jnp.repeat(vec[0], original_shape[0])
                return jnp.reshape(vec, original_shape)

        new_lam_out = restore_shape(new_lam_comp, lam_shape)
        new_c_out = restore_shape(new_c_comp, c_shape)

        new_params = MultiAugmentedLagrangianParams(
            lagrange_multiplier=new_lam_out,
            penalty_multiplier=new_c_out,
        )

        # Build aux: choose representations consistent with original shapes.
        # For debugging it's often helpful to include both per-constraint and aggregated entries.
        aux = {
            "psi_per_constraint": psi,  # always vector (k,)
            "psi_total": psi_total,  # scalar
            "lagrangian_cond": cond,  # vector (k,)
            # also return lagrange_multiplier/penalty_multiplier in same shape as input
            "lagrange_multiplier": new_lam_out,
            "penalty_multiplier": new_c_out,
        }

        return actor_loss + psi_total, aux, new_params


class AugmentedLagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    penalty_multiplier: jax.Array


class AugmentedLagrangian:
    def __init__(self, penalty_multiplier_factor: float) -> None:
        self.penalty_multiplier_factor = penalty_multiplier_factor

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: AugmentedLagrangianParams,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        psi, cond = augmented_lagrangian(constraint, *params)
        new_params = update_augmented_lagrangian(
            cond, params.penalty_multiplier, self.penalty_multiplier_factor
        )
        aux = {
            "lagrangian_cond": cond,
            "lagrange_multiplier": new_params.lagrange_multiplier,
        }
        return actor_loss + psi, aux, new_params


def augmented_lagrangian(
    constraint: jax.Array,
    lagrange_multiplier: jax.Array,
    penalty_multiplier: jax.Array,
) -> jax.Array:
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    g = -constraint
    c = penalty_multiplier
    cond = lagrange_multiplier + c * g
    psi = jnp.where(
        jnp.greater(cond, 0.0),
        lagrange_multiplier * g + c / 2.0 * g**2,
        -1.0 / (2.0 * c) * lagrange_multiplier**2,
    )
    return psi, cond


def update_augmented_lagrangian(
    cond: jax.Array, penalty_multiplier: jax.Array, penalty_multiplier_factor: float
):
    new_penalty_multiplier = jnp.clip(
        penalty_multiplier * (1.0 + penalty_multiplier_factor), penalty_multiplier, 1.0
    )
    new_lagrange_multiplier = jnp.clip(cond, a_min=0.0, a_max=100.0)
    return AugmentedLagrangianParams(new_lagrange_multiplier, new_penalty_multiplier)


class LagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    optimizer_state: optax.OptState


class Lagrangian:
    def __init__(self, multiplier_lr: float) -> None:
        self.optimizer = optax.adam(learning_rate=multiplier_lr)
        self.learning_rate = multiplier_lr

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: LagrangianParams,
        *,
        rest: Any,
    ) -> tuple[jax.Array, dict[str, Any], LagrangianParams]:
        cost_advantage = -rest
        lagrange_multiplier = params.lagrange_multiplier
        actor_loss += lagrange_multiplier * cost_advantage
        aux: dict[str, Any] = {}
        new_params = params
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: LagrangianParams
    ) -> tuple[jax.Array, LagrangianParams]:
        new_lagrange_multiplier = update_lagrange_multiplier(
            constraint, params.lagrange_multiplier, self.learning_rate
        )
        aux = {"lagrange_multiplier": new_lagrange_multiplier}
        return aux, LagrangianParams(new_lagrange_multiplier, params.optimizer_state)


def update_lagrange_multiplier(
    constraint: jax.Array, lagrange_multiplier: jax.Array, learning_rate: float
) -> jax.Array:
    new_multiplier = jnp.maximum(lagrange_multiplier - learning_rate * constraint, 0.0)
    return new_multiplier


class LBSGDParams(NamedTuple):
    eta: float


class LBSGD:
    def __init__(self, eta_rate: float, epsilon: float = 1e-7):
        self.eta_rate = eta_rate
        self.epsilon = epsilon

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: LBSGDParams,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], LBSGDParams]:
        active = jnp.greater(constraint, 0.0)
        constraint_loss = jnp.where(active, constraint, 0.0)
        constraint_loss = -jnp.log(constraint_loss + self.epsilon)
        combined_loss = actor_loss + params.eta * constraint_loss
        loss = jnp.where(
            active,
            combined_loss,
            -constraint,
        )

        new_params = LBSGDParams(params.eta * self.eta_rate)
        aux = {
            "lbsgd/eta": new_params.eta,
            "lbsgd/active": active,
        }

        return loss, aux, new_params


def get_penalizer(cfg):
    if "penalizer" not in cfg.agent:
        return None, None
    if cfg.agent.penalizer.name == "lagrangian":
        penalizer = AugmentedLagrangian(cfg.agent.penalizer.penalty_multiplier_factor)
        penalizer_state = AugmentedLagrangianParams(
            cfg.agent.penalizer.lagrange_multiplier,
            cfg.agent.penalizer.penalty_multiplier,
        )
    elif cfg.agent.penalizer.name == "crpo":
        penalizer = CRPO(cfg.agent.penalizer.eta)
        penalizer_state = CRPOParams(cfg.agent.penalizer.burnin)
    elif cfg.agent.penalizer.name == "ppo_lagrangian":
        penalizer = Lagrangian(cfg.agent.penalizer.multiplier_lr)
        init_lagrange_multiplier = cfg.agent.penalizer.initial_lagrange_multiplier
        penalizer_state = LagrangianParams(
            init_lagrange_multiplier,
            penalizer.optimizer.init(init_lagrange_multiplier),
        )
    elif cfg.agent.penalizer.name == "saute":
        return None, None
    elif cfg.agent.penalizer.name == "lbsgd":
        eta_rate = cfg.agent.penalizer.eta_rate + 1.0
        penalizer = LBSGD(
            eta_rate,
            cfg.agent.penalizer.epsilon,
        )
        penalizer_state = LBSGDParams(cfg.agent.penalizer.initial_eta)
    elif cfg.agent.penalizer.name == "multi_lagrangian":
        penalizer = MultiAugmentedLagrangian(
            cfg.agent.penalizer.penalty_multiplier_factor
        )
        penalizer_state = MultiAugmentedLagrangianParams(
            cfg.agent.penalizer.lagrange_multiplier,
            cfg.agent.penalizer.penalty_multiplier,
        )
    else:
        raise ValueError(f"Unknown penalizer {cfg.agent.penalizer.name}")
    return penalizer, penalizer_state
