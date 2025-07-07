from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constraint import EqualityConstraint, InequalityConstraint, PolarInequalityConstraint
from .data import SolverData
from .settings import SolverSettings


def solve_swarm(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[list[bool], list[int], SolverData]:
    n_drones = len(states)
    pos = data.previous_trajectory.pos
    col = 1.0 / settings.collision_envelope
    data.distance_matrix = np.linalg.norm((pos[None, ...] - pos[:, None, ...]) * col, axis=-1)
    data.x_0 = states
    data.current_time = t

    # Solve for each drone
    is_success = np.zeros(n_drones, dtype=bool)
    iters = np.zeros(n_drones)
    for i in range(n_drones):
        data.rank = i
        success, num_iters, data = solve_drone(data, settings)
        is_success[i] = success
        iters[i] = num_iters
    return is_success, iters, data


def add_constraints(data: SolverData, settings: SolverSettings) -> SolverData:
    """Setup optimization problem before solving.

    Override of AMSolver method that configures constraints and cost functions.
    """
    # Extract waypoints in current horizon. Each row is a waypoint of form:
    # [k, x, y, z, vx, vy, vz, ax, ay, az]. k is discrete STEP in current horizon
    K, freq = settings.K, settings.freq
    assert data.zeta is not None, "Zeta must be initialized before adding constraints"

    mask, steps = filter_horizon(data.waypoints["time"][:, data.rank], data.current_time, K, freq)
    # Separate and reshape waypoints into position, velocity, and acceleration vectors
    des_pos = data.waypoints["pos"][:, data.rank][mask].flatten()
    des_vel = data.waypoints["vel"][:, data.rank][mask].flatten()
    des_acc = data.waypoints["acc"][:, data.rank][mask].flatten()

    # Extract penalized steps from first column of waypoints
    # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
    wp_idx = np.repeat(steps, 3) * 3 + np.tile(np.arange(3, dtype=int), len(steps))

    # Output smoothness cost
    x_0 = data.x_0[data.rank]
    data.linear_cost[data.rank] += data.linear_cost_smoothness_const @ x_0

    # --- Add constraints - see thesis document for derivations ---
    # Waypoint position cost and/or equality constraint
    G_wp = data.matrices.M_p_S_u_W_input[wp_idx]
    h_wp = des_pos - data.matrices.M_p_S_x[wp_idx] @ x_0

    data.quad_cost[data.rank] += 2 * settings.pos_weight * G_wp.T @ G_wp
    data.linear_cost[data.rank] += -2 * settings.pos_weight * G_wp.T @ h_wp
    if settings.pos_constraints:
        data.constraints.append(EqualityConstraint(G_wp, h_wp, settings.waypoints_pos_tol))

    # Waypoint velocity cost and/or equality constraint
    G_wv = data.matrices.M_v_S_u_W_input[wp_idx]
    h_wv = des_vel - data.matrices.M_v_S_x[wp_idx] @ x_0
    data.quad_cost[data.rank] += 2 * settings.vel_weight * G_wv.T @ G_wv
    data.linear_cost[data.rank] += -2 * settings.vel_weight * G_wv.T @ h_wv
    if settings.vel_constraints:
        data.constraints.append(EqualityConstraint(G_wv, h_wv, settings.waypoints_vel_tol))

    # Waypoint acceleration cost and/or equality constraint
    G_wa = data.matrices.M_a_S_u_prime_W_input[wp_idx]
    h_wa = des_acc - data.matrices.M_a_S_x_prime[wp_idx] @ x_0
    data.quad_cost[data.rank] += 2 * settings.acc_weight * G_wa.T @ G_wa
    data.linear_cost[data.rank] += -2 * settings.acc_weight * G_wa.T @ h_wa
    if settings.acc_constraints:
        data.constraints.append(EqualityConstraint(G_wa, h_wa, settings.waypoints_acc_tol))

    # Input continuity cost and/or equality constraint
    u_0 = data.previous_trajectory.u_pos[data.rank, 0]
    u_dot_0 = data.previous_trajectory.u_vel[data.rank, 0]
    u_ddot_0 = data.previous_trajectory.u_acc[data.rank, 0]
    h_u = np.concatenate([u_0, u_dot_0, u_ddot_0])
    data.linear_cost[data.rank] += -2 * settings.input_continuity_weight * data.matrices.G_u.T @ h_u
    if settings.input_continuity_constraints:
        data.constraints.append(
            EqualityConstraint(data.matrices.G_u, h_u, settings.input_continuity_tol)
        )

    # Position constraint
    upper = np.tile(settings.pos_max, K + 1) - data.matrices.M_p_S_x @ x_0
    lower = -np.tile(settings.pos_min, K + 1) + data.matrices.M_p_S_x @ x_0
    h_p = np.concatenate([upper, lower])
    data.constraints.append(InequalityConstraint(data.matrices.G_p, h_p, settings.pos_limit_tol))

    # Velocity constraint
    c_v = data.matrices.M_v_S_x @ x_0
    data.constraints.append(
        PolarInequalityConstraint(
            data.matrices.M_v_S_u_W_input,
            c_v,
            -float("inf"),
            settings.vel_max,
            1.0,
            settings.vel_limit_tol,
        )
    )

    # Acceleration constraint
    c_a = data.matrices.M_a_S_x_prime @ x_0
    data.constraints.append(
        PolarInequalityConstraint(
            data.matrices.M_a_S_u_prime_W_input,
            c_a,
            -float("inf"),
            settings.acc_max,
            1.0,
            settings.acc_limit_tol,
        )
    )

    # Collision constraints
    for i in range(data.n_drones):
        if i == data.rank:
            continue
        if np.any(data.distance_matrix <= 1.0, axis=-1)[i, data.rank]:
            envelope = np.tile(1 / settings.collision_envelope, K + 1)
            G_c = envelope[:, None] * data.matrices.M_p_S_u_W_input
            c_c = envelope * (
                data.matrices.M_p_S_x @ x_0 - data.previous_trajectory.pos[i].flatten()
            )
            data.constraints.append(
                PolarInequalityConstraint(
                    G_c, c_c, 1.0, float("inf"), settings.bf_gamma, settings.collision_tol
                )
            )
    return data


def spline2states(data: SolverData, settings: SolverSettings) -> SolverData:
    """Extract position, velocity, and acceleration trajectories from solution coefficients."""
    K = settings.K
    # Extract position trajectory from state trajectory
    zeta = data.zeta[data.rank]
    x_0 = data.x_0[data.rank]
    pos = (data.matrices.S_x @ x_0 + data.matrices.S_u_W_input @ zeta).T.reshape((K + 1, 6))[:, :3]
    # Get input position, velocity and acceleration from spline coefficients
    data.trajectory.pos[data.rank] = pos
    data.trajectory.u_pos[data.rank] = (data.matrices.W @ zeta).T.reshape((K, 3))
    data.trajectory.u_vel[data.rank] = (data.matrices.W_dot @ zeta).T.reshape((K, 3))
    data.trajectory.u_acc[data.rank] = (data.matrices.W_ddot @ zeta).T.reshape(K, 3)
    return data


def am_solve(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Conducts actual solving process implementing optimization algorithm.

    Not meant to be overridden by child classes.
    """
    rho = settings.rho_init
    zeta = data.zeta[data.rank]  # Previously was zero initialized, now uses previous solution
    bregman_mult = np.zeros(data.quad_cost[data.rank].shape[0])  # Bregman multiplier

    # Aggregate quadratic and linear terms from all constraints
    quad_constraint_terms = np.zeros_like(data.quad_cost[data.rank])
    linear_constraint_terms = np.zeros(data.linear_cost[data.rank].shape[0])

    for constraint in data.constraints:
        quad_constraint_terms += constraint.get_quadratic_term()
        linear_constraint_terms += constraint.get_linear_term()

    # Iteratively solve until solution found or max iterations reached
    for i in range(settings.max_iters):
        Q = data.quad_cost[data.rank] + rho * quad_constraint_terms

        # Construct linear cost matrices
        linear_constraint_terms -= bregman_mult
        q = data.linear_cost[data.rank] + rho * linear_constraint_terms

        # Solve the QP
        zeta = np.linalg.solve(Q, -q)

        # Update constraints
        for c in data.constraints:
            c.update(zeta)

        if all(c.is_satisfied(zeta) for c in data.constraints):
            data.zeta[data.rank] = zeta
            return True, i, data  # Exit loop, indicate success

        # Recalculate linear term for Bregman multiplier
        linear_constraint_terms[...] = 0
        for constraint in data.constraints:
            linear_constraint_terms += constraint.get_linear_term()

        # Calculate Bregman multiplier
        bregman_mult -= 0.5 * (quad_constraint_terms @ zeta + linear_constraint_terms)

        # Gradually increase penalty parameter
        rho *= settings.rho_init
        rho = min(rho, settings.rho_max)

    data.zeta[data.rank] = zeta
    return False, i, data  # Indicate failure but still return vector


def solve_drone(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Main solve function to be called by user."""
    # Reset cost and clear carryover constraints from previous solves
    data = reset_cost_matrices(data)
    data.constraints.clear()  # Ensure no carryover updates from previous solve
    data = add_constraints(data, settings)  # Build new constraints and add to cost matrices
    success, iters, data = am_solve(data, settings)  # Solve with AM algorithm
    data = spline2states(data, settings)
    return success, iters, data


def reset_cost_matrices(data: SolverData) -> SolverData:
    """Reset cost matrices to initial values"""
    data.quad_cost[...] = data.quad_cost_init
    data.linear_cost[...] = 0
    return data


def filter_horizon(times: NDArray, t: float, K: int, mpc_freq: float) -> tuple[NDArray, NDArray]:
    """Extract waypoints in current horizon.

    Args:
        waypoints: Waypoints matrix
        t: Current time
        K: Horizon length
        mpc_freq: MPC frequency

    Returns:
        Filtered waypoints matrix containing only waypoints within current horizon
    """
    assert isinstance(times, np.ndarray), "Waypoints must be a numpy array"
    assert times.ndim == 1, "Waypoints must be a 2D array"
    # Round times to nearest discrete time step relative to current time
    rounded_times = np.asarray(np.round((times - t) * mpc_freq), dtype=int)
    # Find time steps with waypoints within current horizon
    in_horizon = np.where((rounded_times > 0) & (rounded_times <= K))[0]
    if len(in_horizon) == 0:
        raise RuntimeError(
            "Error: no waypoints within current horizon. Increase horizon or add waypoints."
        )
    return in_horizon, rounded_times[in_horizon]
