from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constraint import (
    EqualityConstraint,
    InequalityConstraint,
    PolarInequalityConstraint,
)
from .data import Result, SolverData, SolverSettings


def add_constraints(data: SolverData, settings: SolverSettings) -> SolverData:
    """Setup optimization problem before solving.

    Override of AMSolver method that configures constraints and cost functions.
    """
    # Extract waypoints in current horizon. Each row is a waypoint of form:
    # [k, x, y, z, vx, vy, vz, ax, ay, az]. k is discrete STEP in current horizon
    K, freq = settings.mpc.K, settings.mpc.freq
    if data.zeta is None:
        data.zeta = np.zeros(data.cost.quad.shape[0])  # Init optimization variable

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
    data.cost.linear += data.cost.linear_smoothness_const @ x_0

    # --- Add constraints - see thesis document for derivations ---
    # Waypoint position cost and/or equality constraint
    G_wp = data.matrices.M_p_S_u_W_input[wp_idx]
    h_wp = des_pos - data.matrices.M_p_S_x[wp_idx] @ x_0
    data.cost.quad += 2 * settings.weights.pos * G_wp.T @ G_wp
    data.cost.linear += -2 * settings.weights.pos * G_wp.T @ h_wp
    if settings.constraints.pos:
        data.constraints.append(EqualityConstraint(G_wp, h_wp, settings.mpc.waypoints_pos_tol))

    # Waypoint velocity cost and/or equality constraint
    G_wv = data.matrices.M_v_S_u_W_input[wp_idx]
    h_wv = des_vel - data.matrices.M_v_S_x[wp_idx] @ x_0
    data.cost.quad += 2 * settings.weights.vel * G_wv.T @ G_wv
    data.cost.linear += -2 * settings.weights.vel * G_wv.T @ h_wv
    if settings.constraints.vel:
        data.constraints.append(EqualityConstraint(G_wv, h_wv, settings.mpc.waypoints_vel_tol))

    # Waypoint acceleration cost and/or equality constraint
    G_wa = data.matrices.M_a_S_u_prime_W_input[wp_idx]
    h_wa = des_acc - data.matrices.M_a_S_x_prime[wp_idx] @ x_0
    data.cost.quad += 2 * settings.weights.acc * G_wa.T @ G_wa
    data.cost.linear += -2 * settings.weights.acc * G_wa.T @ h_wa
    if settings.constraints.acc:
        data.constraints.append(EqualityConstraint(G_wa, h_wa, settings.mpc.waypoints_acc_tol))

    # Input continuity cost and/or equality constraint
    h_u = np.concatenate([data.u_0, data.u_dot_0, data.u_ddot_0])
    data.cost.linear += -2 * settings.weights.input_continuity * data.matrices.G_u.T @ h_u
    if settings.constraints.input_continuity:
        data.constraints.append(
            EqualityConstraint(data.matrices.G_u, h_u, settings.mpc.input_continuity_tol)
        )

    # Position constraint
    upper = np.tile(settings.limits.pos_max, K + 1) - data.matrices.M_p_S_x @ x_0
    lower = -np.tile(settings.limits.pos_min, K + 1) + data.matrices.M_p_S_x @ x_0
    h_p = np.concatenate([upper, lower])
    data.constraints.append(InequalityConstraint(data.matrices.G_p, h_p, settings.mpc.pos_tol))

    # Velocity constraint
    c_v = data.matrices.M_v_S_x @ x_0
    data.constraints.append(
        PolarInequalityConstraint(
            data.matrices.M_v_S_u_W_input,
            c_v,
            -float("inf"),
            settings.limits.vel_max,
            1.0,
            settings.mpc.vel_tol,
        )
    )

    # Acceleration constraint
    c_a = data.matrices.M_a_S_x_prime @ x_0
    data.constraints.append(
        PolarInequalityConstraint(
            data.matrices.M_a_S_u_prime_W_input,
            c_a,
            -float("inf"),
            settings.limits.acc_max,
            1.0,
            settings.mpc.acc_tol,
        )
    )

    # Collision constraints
    for pos, envelope in zip(data.obstacle_positions, data.obstacle_envelopes, strict=True):
        envelope = np.tile(envelope, K + 1)
        G_c = envelope[:, None] * data.matrices.M_p_S_u_W_input
        c_c = envelope * (data.matrices.M_p_S_x @ x_0 - pos)
        data.constraints.append(
            PolarInequalityConstraint(
                G_c,
                c_c,
                1.0,
                float("inf"),
                settings.mpc.bf_gamma,
                settings.mpc.collision_tol,
            )
        )
    return data


def spline2states(data: SolverData, settings: SolverSettings) -> SolverData:
    """Extract position, velocity, and acceleration trajectories from solution coefficients."""
    K = settings.mpc.K
    # Extract position trajectory from state trajectory
    x_0 = data.x_0[data.rank]
    pos = (data.matrices.S_x @ x_0 + data.matrices.S_u_W_input @ data.zeta).T.reshape((K + 1, 6))[
        :, :3
    ]
    # Get input position, velocity and acceleration from spline coefficients
    u_pos = (data.matrices.W @ data.zeta).T.reshape((K, 3))
    u_vel = (data.matrices.W_dot @ data.zeta).T.reshape((K, 3))
    u_acc = (data.matrices.W_ddot @ data.zeta).T.reshape(K, 3)
    data.results[data.rank] = Result(pos=pos, u_pos=u_pos, u_vel=u_vel, u_acc=u_acc, zeta=data.zeta)
    return data


def am_solve(data: SolverData, settings: SolverSettings) -> tuple[bool, int, SolverData]:
    """Conducts actual solving process implementing optimization algorithm.

    Not meant to be overridden by child classes.
    """
    # Initialize solver components
    rho = settings.rho_init

    # Initialize optimization variables and matrices
    Q = np.zeros_like(data.cost.quad)  # Combined quadratic terms
    q = np.zeros(data.cost.quad.shape[0])  # Combined linear terms
    zeta = data.zeta
    zeta = np.zeros(data.cost.quad.shape[0])  # Optimization variable

    bregman_mult = np.zeros(data.cost.quad.shape[0])  # Bregman multiplier

    # Aggregate quadratic and linear terms from all constraints
    quad_constraint_terms = np.zeros_like(data.cost.quad)
    linear_constraint_terms = np.zeros(data.cost.linear.shape[0])

    for constraint in data.constraints:
        quad_constraint_terms += constraint.get_quadratic_term()
        linear_constraint_terms += constraint.get_linear_term()

    # Iteratively solve until solution found or max iterations reached
    for i in range(settings.max_iters):
        Q = data.cost.quad + rho * quad_constraint_terms

        # Construct linear cost matrices
        linear_constraint_terms -= bregman_mult
        q = data.cost.linear + rho * linear_constraint_terms

        # Solve the QP
        zeta = np.linalg.solve(Q, -q)

        # Update constraints
        for c in data.constraints:
            c.update(zeta)

        if all(c.is_satisfied(zeta) for c in data.constraints):
            data.zeta = zeta
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

    data.zeta = zeta
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
    data.cost.quad = data.cost.quad_init.copy()
    data.cost.linear[...] = 0
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
