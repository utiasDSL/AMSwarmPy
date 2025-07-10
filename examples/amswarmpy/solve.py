from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .constraint import EqualityConstraint, InequalityConstraint, PolarInequalityConstraint
from .data import SolverData
from .settings import SolverSettings


def solve_swarm(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[list[bool], list[int], SolverData]:
    # The horizon is dynamically shaped based on which waypoints are in the current horizon. This is
    # therefore the only function we cannot compile with jax.jit.
    data = set_horizon(data, settings)
    # After setting the horizon, everything else is static and hence gets compiled
    return _solve_swarm(states, t, data, settings)


def _solve_swarm(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[list[bool], list[int], SolverData]:
    distances = compute_swarm_distances(data, settings)
    # Set the initial state and current time
    data = data.replace(x_0=states, current_time=t, distance_matrix=distances)
    # Solve for each drone
    is_success = np.zeros(data.n_drones, dtype=bool)
    iters = np.zeros(data.n_drones)
    for i in range(data.n_drones):
        data = data.replace(rank=i)
        success, num_iters, data = solve_drone(data, settings)
        is_success[i] = success
        iters[i] = num_iters
    return is_success, iters, data


def set_horizon(data: SolverData, settings: SolverSettings) -> SolverData:
    in_horizon, t_discrete = filter_horizon(
        data.waypoints["time"][:, data.rank], data.current_time, settings.K, settings.freq
    )
    in_horizon = jp.where(in_horizon)[0]
    if len(in_horizon) < 1:
        raise RuntimeError(
            "Error: no waypoints within current horizon. Increase horizon or add waypoints."
        )
    return data.replace(in_horizon=in_horizon, t_discrete=t_discrete)


@jax.jit
def compute_swarm_distances(data: SolverData, settings: SolverSettings) -> Array:
    pos = data.previous_trajectory.pos
    col = 1.0 / settings.collision_envelope
    distances = jp.linalg.norm((pos[None, ...] - pos[:, None, ...]) * col, axis=-1)
    distances = jp.where(jp.eye(data.n_drones, dtype=bool)[..., None], jp.inf, distances)
    return distances


def solve_drone(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Main solve function to be called by user."""
    # Reset cost and clear carryover constraints from previous solves
    data = reset_cost_matrices(data)
    data = reset_constraints(data)
    data = add_constraints(data, settings)  # Build new constraints and add to cost matrices
    success, iters, data = am_solve(data, settings)  # Solve with AM algorithm
    data = spline2states(data, settings)
    return success, iters, data


@jax.jit
def reset_cost_matrices(data: SolverData) -> SolverData:
    Q_init = data.quad_cost.at[...].set(data.quad_cost_init)
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


def add_constraints(data: SolverData, settings: SolverSettings) -> SolverData:
    """Setup optimization problem before solving.

    Override of AMSolver method that configures constraints and cost functions.
    """
    assert data.zeta is not None, "Zeta must be initialized before adding constraints"
    assert data.in_horizon is not None, "In horizon must be initialized before adding constraints"
    # Separate and reshape waypoints into position, velocity, and acceleration vectors
    des_pos = data.waypoints["pos"][:, data.rank][data.in_horizon].flatten()
    des_vel = data.waypoints["vel"][:, data.rank][data.in_horizon].flatten()
    des_acc = data.waypoints["acc"][:, data.rank][data.in_horizon].flatten()

    # Extract penalized steps from first column of waypoints
    # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
    step_idx = data.t_discrete[data.in_horizon]
    t_idx = jp.repeat(step_idx, 3) * 3 + jp.tile(jp.arange(3, dtype=int), len(step_idx))

    x_0 = data.x_0[data.rank]
    linear_cost = data.linear_cost[data.rank]
    quad_cost = data.quad_cost[data.rank]
    # Output smoothness cost
    linear_cost += data.linear_cost_smoothness_const @ x_0

    # --- Add constraints - see thesis document for derivations ---
    # Waypoint position cost and/or equality constraint
    G_wp = data.matrices.M_p_S_u_W_input[t_idx]
    h_wp = des_pos - data.matrices.M_p_S_x[t_idx] @ x_0

    quad_cost += 2 * settings.pos_weight * G_wp.T @ G_wp
    linear_cost += -2 * settings.pos_weight * G_wp.T @ h_wp
    if settings.pos_constraints:
        data = data.replace(
            pos_constraint=EqualityConstraint.init(G_wp, h_wp, settings.waypoints_pos_tol)
        )

    # Waypoint velocity cost and/or equality constraint
    G_wv = data.matrices.M_v_S_u_W_input[t_idx]
    h_wv = des_vel - data.matrices.M_v_S_x[t_idx] @ x_0
    quad_cost += 2 * settings.vel_weight * G_wv.T @ G_wv
    linear_cost += -2 * settings.vel_weight * G_wv.T @ h_wv
    if settings.vel_constraints:
        data.vel_constraint = EqualityConstraint.init(G_wv, h_wv, settings.waypoints_vel_tol)

    # Waypoint acceleration cost and/or equality constraint
    G_wa = data.matrices.M_a_S_u_prime_W_input[t_idx]
    h_wa = des_acc - data.matrices.M_a_S_x_prime[t_idx] @ x_0
    quad_cost += 2 * settings.acc_weight * G_wa.T @ G_wa
    linear_cost += -2 * settings.acc_weight * G_wa.T @ h_wa
    if settings.acc_constraints:
        data.acc_constraint = EqualityConstraint.init(G_wa, h_wa, settings.waypoints_acc_tol)

    # Input continuity cost and/or equality constraint
    u_0 = data.previous_trajectory.u_pos[data.rank, 0]
    u_dot_0 = data.previous_trajectory.u_vel[data.rank, 0]
    u_ddot_0 = data.previous_trajectory.u_acc[data.rank, 0]
    h_u = np.concatenate([u_0, u_dot_0, u_ddot_0])
    linear_cost += -2 * settings.input_continuity_weight * data.matrices.G_u.T @ h_u
    if settings.input_continuity_constraints:
        data = data.replace(
            input_continuity_constraint=EqualityConstraint.init(
                data.matrices.G_u, h_u, settings.input_continuity_tol
            )
        )

    # Position constraint
    upper = np.tile(settings.pos_max, settings.K + 1) - data.matrices.M_p_S_x @ x_0
    lower = -np.tile(settings.pos_min, settings.K + 1) + data.matrices.M_p_S_x @ x_0
    h_p = np.concatenate([upper, lower])
    data = data.replace(
        max_pos_constraint=InequalityConstraint.init(data.matrices.G_p, h_p, settings.pos_limit_tol)
    )

    # Velocity constraint
    c_v = data.matrices.M_v_S_x @ x_0
    data = data.replace(
        max_vel_constraint=PolarInequalityConstraint.init(
            data.matrices.M_v_S_u_W_input,
            c_v,
            upr_bound=settings.vel_max,
            tol=settings.vel_limit_tol,
        )
    )

    # Acceleration constraint
    c_a = data.matrices.M_a_S_x_prime @ x_0
    data = data.replace(
        max_acc_constraint=PolarInequalityConstraint.init(
            data.matrices.M_a_S_u_prime_W_input,
            c_a,
            upr_bound=settings.acc_max,
            tol=settings.acc_limit_tol,
        )
    )

    # Collision constraints
    constraints = []
    for i in range(data.n_drones):
        if i == data.rank:
            continue
        if np.any(data.distance_matrix <= 1.0, axis=-1)[i, data.rank]:
            envelope = np.tile(1 / settings.collision_envelope, settings.K + 1)
            G_c = envelope[:, None] * data.matrices.M_p_S_u_W_input
            c_c = envelope * (
                data.matrices.M_p_S_x @ x_0 - data.previous_trajectory.pos[i].flatten()
            )
            constraints.append(
                PolarInequalityConstraint.init(G_c, c_c, lwr_bound=1.0, tol=settings.collision_tol)
            )
    data = data.replace(collision_constraints=constraints)
    data = data.replace(linear_cost=data.linear_cost.at[data.rank].set(linear_cost))
    data = data.replace(quad_cost=data.quad_cost.at[data.rank].set(quad_cost))
    return data


@jax.jit
def spline2states(data: SolverData, settings: SolverSettings) -> SolverData:
    """Extract position, velocity, and acceleration trajectories from solution coefficients."""
    K = settings.K
    # Extract position trajectory from state trajectory
    zeta = data.zeta[data.rank]
    x_0 = data.x_0[data.rank]
    pos = (data.matrices.S_x @ x_0 + data.matrices.S_u_W_input @ zeta).T.reshape((K + 1, 6))[:, :3]
    u_pos = (data.matrices.W @ zeta).T.reshape((K, 3))
    u_vel = (data.matrices.W_dot @ zeta).T.reshape((K, 3))
    u_acc = (data.matrices.W_ddot @ zeta).T.reshape(K, 3)
    # Get input position, velocity and acceleration from spline coefficients
    trajectory = data.trajectory.replace(
        pos=data.trajectory.pos.at[data.rank].set(pos),
        u_pos=data.trajectory.u_pos.at[data.rank].set(u_pos),
        u_vel=data.trajectory.u_vel.at[data.rank].set(u_vel),
        u_acc=data.trajectory.u_acc.at[data.rank].set(u_acc),
    )
    return data.replace(trajectory=trajectory)


def am_solve(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Conducts actual solving process implementing optimization algorithm.

    Not meant to be overridden by child classes.
    """
    rho = settings.rho_init
    zeta = data.zeta[data.rank]  # Previously was zero initialized, now uses previous solution
    bregman_mult = np.zeros(data.quad_cost[data.rank].shape[0])  # Bregman multiplier

    # Aggregate quadratic and linear terms from all constraints
    Q_cnstr = quadratic_constraint_costs(data)
    q_cnstr = linear_constraint_costs(data)

    # Iteratively solve until solution found or max iterations reached
    for i in range(settings.max_iters):
        Q = data.quad_cost[data.rank] + rho * Q_cnstr

        # Construct linear cost matrices
        q_cnstr -= bregman_mult
        q = data.linear_cost[data.rank] + rho * q_cnstr

        # Solve the QP
        zeta = jp.linalg.solve(Q, -q)

        # Update constraints
        data = data.replace(
            max_pos_constraint=InequalityConstraint.update(data.max_pos_constraint, zeta),
            max_vel_constraint=PolarInequalityConstraint.update(data.max_vel_constraint, zeta),
            max_acc_constraint=PolarInequalityConstraint.update(data.max_acc_constraint, zeta),
        )
        for i, c in enumerate(data.collision_constraints):
            data.collision_constraints[i] = c.update(c, zeta)

        if constraints_satisfied(zeta, data):
            data = data.replace(zeta=data.zeta.at[data.rank].set(zeta))
            return True, i, data  # Exit loop, indicate success

        # Recalculate linear term for Bregman multiplier
        q_cnstr = linear_constraint_costs(data)

        # Calculate Bregman multiplier
        bregman_mult -= 0.5 * (Q_cnstr @ zeta + q_cnstr)

        # Gradually increase penalty parameter
        rho *= settings.rho_init
        rho = min(rho, settings.rho_max)

    data = data.replace(zeta=data.zeta.at[data.rank].set(zeta))
    return False, i, data  # Indicate failure but still return vector


@jax.jit
def quadratic_constraint_costs(data: SolverData) -> Array:
    Q_cnstr = jp.zeros_like(data.quad_cost[data.rank])
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
    for c in data.collision_constraints:
        Q_cnstr += c.quadratic_term(c)
    return Q_cnstr


@jax.jit
def linear_constraint_costs(data: SolverData) -> Array:
    q_cnstr = jp.zeros_like(data.linear_cost[data.rank])
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
    for c in data.collision_constraints:
        q_cnstr += c.linear_term(c)
    return q_cnstr


@jax.jit
def constraints_satisfied(zeta: Array, data: SolverData) -> Array:
    """Check if all constraints are satisfied"""
    satisfied = jp.all(jp.array([c.satisfied(c, zeta) for c in data.collision_constraints]))
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
def filter_horizon(times: Array, t: float, K: int, mpc_freq: float) -> tuple[Array, Array]:
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
    return in_horizon, rounded_times
