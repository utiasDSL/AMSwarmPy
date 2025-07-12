from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jp
from jax import Array
from numpy.typing import NDArray

from .constraint import EqualityConstraint, InequalityConstraint, PolarInequalityConstraint
from .data import SolverData, solver_data_vmap_axes
from .settings import SolverSettings


def solve(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[Array[bool], Array[int], SolverData]:
    # The horizon is dynamically shaped based on which waypoints are in the current horizon. This is
    # therefore the only function we cannot compile with jax.jit.
    data = set_horizon(float(t), data, settings)
    # After setting the horizon, everything else is static and hence gets compiled
    return solve_swarm(states, data, settings)


@jax.jit
def solve_swarm(
    states: NDArray, data: SolverData, settings: SolverSettings
) -> tuple[Array[bool], Array[int], SolverData]:
    distances = compute_swarm_distances(data, settings)
    # Set the initial state and distances
    data = data.replace(x_0=states, distances=distances)
    # We want to use the data in shared without vmapping its members. We cannot make it static
    # because it contains Arrays. Hence we broadcast and prevent a growth in dimensions by resetting
    # shared after the vmap.
    shared = data.shared
    success, iters, data = jax.vmap(solve_drone, in_axes=(solver_data_vmap_axes, None))(
        data, settings
    )
    # Reset shared to default dimensions using shared (not data.shared) and set pos to current pos
    data = data.replace(shared=shared.replace(pos=data.pos))  # Reset to default dimensions
    return success, iters, data


def set_horizon(t: float, data: SolverData, settings: SolverSettings) -> SolverData:
    start, end, t_discrete = filter_horizon(data.waypoints["time"][0], t, settings.K, settings.freq)
    in_horizon = jp.arange(start, end)
    if len(in_horizon) < 1:
        raise RuntimeError(
            "Error: no waypoints within current horizon. Increase horizon or add waypoints."
        )
    return data.replace(
        shared=data.shared.replace(in_horizon=in_horizon, t_discrete=t_discrete, current_time=t)
    )


@jax.jit
def compute_swarm_distances(data: SolverData, settings: SolverSettings) -> Array:
    col = 1.0 / settings.collision_envelope
    distances = jp.linalg.norm((data.pos[None, ...] - data.pos[:, None, ...]) * col, axis=-1)
    distances = jp.where(jp.eye(settings.n_drones, dtype=bool)[..., None], jp.inf, distances)
    return distances


@jax.jit
def solve_drone(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Main solve function to be called by user."""
    data = reset_cost_matrices(data)
    data = reset_constraints(data)
    data = add_constraints(data, settings)
    success, iters, data = am_solve(data, settings)
    data = spline2states(data, settings)
    # Ensure constraints are None so pytree stays consistent for jax.lax.scan
    data = reset_constraints(data)
    return success, iters, data


@jax.jit
def reset_cost_matrices(data: SolverData) -> SolverData:
    Q_init = data.quad_cost.at[...].set(data.shared.quad_cost_init)
    q_init = jp.zeros_like(data.linear_cost)
    return data.replace(quad_cost=Q_init, linear_cost=q_init)


@jax.jit
def reset_constraints(data: SolverData) -> SolverData:
    """Reset constraints to initial values"""
    data = data.replace(
        pos_constraint=None,
        vel_constraint=None,
        acc_constraint=None,
        input_continuity_constraint=None,
        max_pos_constraint=None,
        max_vel_constraint=None,
        max_acc_constraint=None,
        collision_constraints=None,
    )
    return data


@jax.jit
def add_constraints(data: SolverData, settings: SolverSettings) -> SolverData:
    """Setup optimization problem before solving.

    Override of AMSolver method that configures constraints and cost functions.
    """
    assert data.zeta is not None, "Zeta must be initialized"
    assert data.shared.in_horizon is not None, "In horizon must be initialized"
    # Separate and reshape waypoints into position, velocity, and acceleration vectors
    des_pos = data.waypoints["pos"][data.shared.in_horizon].flatten()
    des_vel = data.waypoints["vel"][data.shared.in_horizon].flatten()
    des_acc = data.waypoints["acc"][data.shared.in_horizon].flatten()

    # Extract penalized steps from first column of waypoints
    # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
    step_idx = data.shared.t_discrete[data.shared.in_horizon]
    t_idx = jp.repeat(step_idx, 3) * 3 + jp.tile(jp.arange(3, dtype=int), len(step_idx))

    x_0 = data.x_0
    linear_cost = data.linear_cost
    quad_cost = data.quad_cost
    # Output smoothness cost
    linear_cost += data.shared.linear_cost_smoothness_const @ x_0

    # --- Add constraints - see thesis document for derivations ---
    # Waypoint position cost and/or equality constraint
    matrices = data.shared.matrices
    G_wp = matrices.M_p_S_u_W_input[t_idx]
    h_wp = des_pos - matrices.M_p_S_x[t_idx] @ x_0

    quad_cost += 2 * settings.pos_weight * G_wp.T @ G_wp
    linear_cost += -2 * settings.pos_weight * G_wp.T @ h_wp
    if settings.pos_constraints:
        data = data.replace(
            pos_constraint=EqualityConstraint.init(G_wp, h_wp, settings.waypoints_pos_tol)
        )

    # Waypoint velocity cost and/or equality constraint
    G_wv = matrices.M_v_S_u_W_input[t_idx]
    h_wv = des_vel - matrices.M_v_S_x[t_idx] @ x_0
    quad_cost += 2 * settings.vel_weight * G_wv.T @ G_wv
    linear_cost += -2 * settings.vel_weight * G_wv.T @ h_wv
    if settings.vel_constraints:
        data.vel_constraint = EqualityConstraint.init(G_wv, h_wv, settings.waypoints_vel_tol)

    # Waypoint acceleration cost and/or equality constraint
    G_wa = matrices.M_a_S_u_prime_W_input[t_idx]
    h_wa = des_acc - matrices.M_a_S_x_prime[t_idx] @ x_0
    quad_cost += 2 * settings.acc_weight * G_wa.T @ G_wa
    linear_cost += -2 * settings.acc_weight * G_wa.T @ h_wa
    if settings.acc_constraints:
        data.acc_constraint = EqualityConstraint.init(G_wa, h_wa, settings.waypoints_acc_tol)

    # Input continuity cost and/or equality constraint
    u_0 = data.u_pos[0]
    u_dot_0 = data.u_vel[0]
    u_ddot_0 = data.u_acc[0]
    h_u = jp.concatenate([u_0, u_dot_0, u_ddot_0])
    linear_cost += -2 * settings.input_continuity_weight * matrices.G_u.T @ h_u
    if settings.input_continuity_constraints:
        data = data.replace(
            input_continuity_constraint=EqualityConstraint.init(
                matrices.G_u, h_u, settings.input_continuity_tol
            )
        )

    # Position constraint
    upper = jp.tile(settings.pos_max, settings.K + 1) - matrices.M_p_S_x @ x_0
    lower = -jp.tile(settings.pos_min, settings.K + 1) + matrices.M_p_S_x @ x_0
    h_p = jp.concatenate([upper, lower])
    data = data.replace(
        max_pos_constraint=InequalityConstraint.init(matrices.G_p, h_p, settings.pos_limit_tol)
    )

    # Velocity constraint
    c_v = matrices.M_v_S_x @ x_0
    data = data.replace(
        max_vel_constraint=PolarInequalityConstraint.init(
            matrices.M_v_S_u_W_input, c_v, upr_bound=settings.vel_max, tol=settings.vel_limit_tol
        )
    )

    # Acceleration constraint
    c_a = matrices.M_a_S_x_prime @ x_0
    data = data.replace(
        max_acc_constraint=PolarInequalityConstraint.init(
            matrices.M_a_S_u_prime_W_input,
            c_a,
            upr_bound=settings.acc_max,
            tol=settings.acc_limit_tol,
        )
    )

    # Collision constraints
    n_collisions = min(settings.max_collisions, settings.n_drones - 1)
    min_dist = jp.min(data.distances, axis=-1)
    closest_drones = jp.argsort(min_dist)[:n_collisions]
    G_c_batched = jp.zeros((n_collisions, 3 * (settings.K + 1), 3 * (settings.N + 1)))
    c_c_batched = jp.zeros((n_collisions, 3 * (settings.K + 1)))

    envelope = jp.tile(1 / settings.collision_envelope, settings.K + 1)
    for i, d in enumerate(closest_drones):
        c_c = envelope * (matrices.M_p_S_x @ x_0 - data.shared.pos[d].flatten())
        G_c_batched = G_c_batched.at[i].set(envelope[:, None] * matrices.M_p_S_u_W_input)
        c_c_batched = c_c_batched.at[i].set(c_c)
    active = jp.zeros(n_collisions, dtype=bool)
    active = active.at[:n_collisions].set(min_dist[closest_drones] <= 1.0)
    data = data.replace(
        collision_constraints=PolarInequalityConstraint.init(
            G_c_batched,
            c_c_batched,
            lwr_bound=1.0,
            tol=settings.collision_tol,
            active=active,
        )
    )
    data = data.replace(linear_cost=data.linear_cost.at[...].set(linear_cost))
    data = data.replace(quad_cost=data.quad_cost.at[...].set(quad_cost))
    return data


@jax.jit
def spline2states(data: SolverData, settings: SolverSettings) -> SolverData:
    """Extract position, velocity, and acceleration trajectories from solution coefficients."""
    K = settings.K
    matrices = data.shared.matrices
    zeta, x_0 = data.zeta, data.x_0
    # Extract trajectories from zeta and x_0
    pos = (matrices.S_x @ x_0 + matrices.S_u_W_input @ zeta).T.reshape((K + 1, 6))[:, :3]
    u_pos = (matrices.W @ zeta).T.reshape((K, 3))
    u_vel = (matrices.W_dot @ zeta).T.reshape((K, 3))
    u_acc = (matrices.W_ddot @ zeta).T.reshape(K, 3)
    data = data.replace(pos=pos, u_pos=u_pos, u_vel=u_vel, u_acc=u_acc)
    return data


@jax.jit
def am_solve(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Conducts actual solving process implementing optimization algorithm.

    Not meant to be overridden by child classes.
    """
    rho = settings.rho_init
    zeta = data.zeta  # Previously was zero initialized, now uses previous solution
    bregman_mult = jp.zeros(data.quad_cost.shape[0])  # Bregman multiplier

    # Aggregate quadratic and linear terms from all constraints
    Q_cnstr = quadratic_constraint_costs(data)
    q_cnstr = linear_constraint_costs(data)

    def cond_fn(
        val: tuple[int, Array, float, Array, Array, SolverData, SolverSettings, Array],
    ) -> bool:
        i, zeta, _, _, _, _, data, settings = val
        return (i < settings.max_iters) & ~constraints_satisfied(zeta, data)

    def loop_fn(
        val: tuple[int, Array, float, Array, Array, SolverData, SolverSettings, Array],
    ) -> tuple[int, Array, SolverData]:
        # Unpack values
        i, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings = val
        # Solve QP
        Q = data.quad_cost + rho * Q_cnstr
        q = data.linear_cost + rho * (q_cnstr - bregman_mult)
        zeta = jp.linalg.solve(Q, -q)
        # Update constraints
        data = data.replace(
            max_pos_constraint=InequalityConstraint.update(data.max_pos_constraint, zeta),
            max_vel_constraint=PolarInequalityConstraint.update(data.max_vel_constraint, zeta),
            max_acc_constraint=PolarInequalityConstraint.update(data.max_acc_constraint, zeta),
            collision_constraints=PolarInequalityConstraint.update(
                data.collision_constraints, zeta
            ),
        )
        # Calculate Bregman multiplier
        q_cnstr = linear_constraint_costs(data)
        bregman_mult = bregman_mult - 0.5 * (Q_cnstr @ zeta + q_cnstr)
        # Increase penalty parameter
        rho = jp.clip(rho * settings.rho_init, max=settings.rho_max)
        return i + 1, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings

    # Compiled equivalent to
    # while cond_fn(...):
    #     loop_fn(...)
    i, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings = jax.lax.while_loop(
        cond_fn, loop_fn, (0, zeta, rho, bregman_mult, q_cnstr, Q_cnstr, data, settings)
    )
    data = data.replace(zeta=zeta)
    return i != settings.max_iters, i, data


@jax.jit
def quadratic_constraint_costs(data: SolverData) -> Array:
    Q_cnstr = jp.zeros_like(data.quad_cost)
    if data.pos_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.pos_constraint)
    if data.vel_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.vel_constraint)
    if data.acc_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.acc_constraint)
    if data.input_continuity_constraint is not None:
        Q_cnstr += EqualityConstraint.quadratic_term(data.input_continuity_constraint)
    Q_cnstr += InequalityConstraint.quadratic_term(data.max_pos_constraint)
    Q_cnstr += PolarInequalityConstraint.quadratic_term(data.max_vel_constraint)
    Q_cnstr += PolarInequalityConstraint.quadratic_term(data.max_acc_constraint)
    Q_cnstr += jp.sum(PolarInequalityConstraint.quadratic_term(data.collision_constraints), axis=0)
    return Q_cnstr


@jax.jit
def linear_constraint_costs(data: SolverData) -> Array:
    q_cnstr = jp.zeros_like(data.linear_cost)
    if data.pos_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.pos_constraint)
    if data.vel_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.vel_constraint)
    if data.acc_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.acc_constraint)
    if data.input_continuity_constraint is not None:
        q_cnstr += EqualityConstraint.linear_term(data.input_continuity_constraint)
    q_cnstr += InequalityConstraint.linear_term(data.max_pos_constraint)
    q_cnstr += PolarInequalityConstraint.linear_term(data.max_vel_constraint)
    q_cnstr += PolarInequalityConstraint.linear_term(data.max_acc_constraint)
    q_cnstr += jp.sum(PolarInequalityConstraint.linear_term(data.collision_constraints), axis=0)
    return q_cnstr


@jax.jit
def constraints_satisfied(zeta: Array, data: SolverData) -> Array:
    """Check if all constraints are satisfied"""
    satisfied = jp.all(PolarInequalityConstraint.satisfied(data.collision_constraints, zeta))
    satisfied &= InequalityConstraint.satisfied(data.max_pos_constraint, zeta)
    satisfied &= PolarInequalityConstraint.satisfied(data.max_vel_constraint, zeta)
    satisfied &= PolarInequalityConstraint.satisfied(data.max_acc_constraint, zeta)
    if data.pos_constraint:
        satisfied &= EqualityConstraint.satisfied(data.pos_constraint, zeta)
    if data.vel_constraint:
        satisfied &= EqualityConstraint.satisfied(data.vel_constraint, zeta)
    if data.acc_constraint:
        satisfied &= EqualityConstraint.satisfied(data.acc_constraint, zeta)
    if data.input_continuity_constraint:
        satisfied &= EqualityConstraint.satisfied(data.input_continuity_constraint, zeta)
    return satisfied


@partial(jax.jit, static_argnums=(2, 3))
def filter_horizon(times: Array, t: float, K: int, mpc_freq: float) -> tuple[int, int, Array]:
    """Extract waypoints in current horizon.

    Args:
        times: Waypoints times
        t: Current time
        K: Horizon length
        mpc_freq: MPC frequency

    Returns:
        A mask of waypoints within current horizon and the rounded times for all waypoints
    """
    assert isinstance(times, Array), f"Waypoints must be an Array, is {type(times)}"
    assert times.ndim == 1, "Waypoints must be a 2D array"
    # Round times to nearest discrete time step relative to current time
    rounded_times = jp.asarray(jp.round((times - t) * mpc_freq), dtype=int)
    # Find time steps with waypoints within current horizon
    in_horizon = (rounded_times > 0) & (rounded_times <= K)
    # Get the first and last index where in_horizon is True
    start = jp.argmax(in_horizon)
    end = len(in_horizon) - jp.argmax(jp.flip(in_horizon))
    return start, end, rounded_times
