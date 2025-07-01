from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from .constraint import (
    Constraint,
    EqualityConstraint,
    InequalityConstraint,
    PolarInequalityConstraint,
)


class Drone:
    """The Drone class solves the drone trajectory optimization problem.

    Args:
        solver_settings: Configuration for the AMSolver
        waypoints: Matrix of waypoints the drone should follow
        mpc_settings: Configuration for the MPC optimization problem
        weights: Weights for the MPC optimization problem
        limits: Physical limits for the drone
        dynamics: Dynamics matrices for the drone
    """

    def __init__(self, settings: SolverSettings, dynamics: Dynamics):
        # Initialize Bernstein and dynamics matrices
        K, N = settings.mpc.K, settings.mpc.N
        self.W, self.W_dot, self.W_ddot, W_input = bernstein_matrices(K, N, settings.mpc.freq)
        self.S_x, S_u, S_x_prime, S_u_prime = full_horizon_dynamics_matrices(dynamics, K)
        # Precompute matrices that don't change at solve time
        self.S_u_W_input = S_u @ W_input

        # Create an index of 0:3, 6:9, 12:15, ...
        p_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., :3].flatten()
        # Create an index of 3:6, 9:12, 15:18, ...
        v_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        a_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        self.M_p_S_u_W_input = self.S_u_W_input[p_idx]
        self.M_v_S_u_W_input = self.S_u_W_input[v_idx]
        self.M_a_S_u_prime_W_input = S_u_prime[a_idx] @ W_input

        self.M_p_S_x = self.S_x[p_idx]
        self.M_v_S_x = self.S_x[v_idx]
        self.M_a_S_x_prime = S_x_prime[a_idx]

        # Precompute constraint matrices
        s = (slice(3), slice(3 * (settings.mpc.N + 1)))  # [:3, :3 * (mpc_settings.n + 1)]
        self.G_u = np.concat((self.W[s], self.W_dot[s], self.W_ddot[s]))
        self.G_p = np.concat((self.M_p_S_u_W_input, -self.M_p_S_u_W_input))

        # Initialize cost matrices
        weights = settings.weights
        self.initial_quad_cost = 2 * weights.input_smoothness * (self.W_ddot.T @ self.W_ddot)
        self.initial_quad_cost += (
            2 * weights.smoothness * W_input.T @ S_u_prime[a_idx].T @ S_u_prime[a_idx] @ W_input
        )

        self.initial_quad_cost += 2 * weights.input_continuity * self.G_u.T @ self.G_u
        self.linear_cost = np.zeros(3 * (settings.mpc.N + 1))
        self.linear_cost_smoothness_const_term = (
            2 * weights.smoothness * self.M_a_S_u_prime_W_input.T @ self.M_a_S_x_prime
        )

    def pre_solve(self, data: SolverData, settings: SolverSettings):
        """Setup optimization problem before solving.

        Override of AMSolver method that configures constraints and cost functions.
        """
        # Extract waypoints in current horizon. Each row is a waypoint of form:
        # [k, x, y, z, vx, vy, vz, ax, ay, az]. k is discrete STEP in current horizon
        K, freq = settings.mpc.K, settings.mpc.freq
        if data.zeta is None:
            data.zeta = np.zeros(self.quad_cost.shape[0])  # Init optimization variable

        mask, steps = filter_horizon(data.waypoints["time"], data.current_time, K, freq)
        # Separate and reshape waypoints into position, velocity, and acceleration vectors
        des_pos = data.waypoints["pos"][mask].flatten()
        des_vel = data.waypoints["vel"][mask].flatten()
        des_acc = data.waypoints["acc"][mask].flatten()

        # Extract penalized steps from first column of waypoints
        # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
        wp_idx = np.repeat(steps, 3) * 3 + np.tile(np.arange(3, dtype=int), len(steps))

        # Plot the waypoint selection matrix
        # Output smoothness cost
        self.linear_cost += self.linear_cost_smoothness_const_term @ data.x_0

        # --- Add constraints - see thesis document for derivations ---
        # Waypoint position cost and/or equality constraint
        G_wp = self.M_p_S_u_W_input[wp_idx]
        h_wp = des_pos - self.M_p_S_x[wp_idx] @ data.x_0
        self.quad_cost += 2 * settings.weights.pos * G_wp.T @ G_wp
        self.linear_cost += -2 * settings.weights.pos * G_wp.T @ h_wp
        if settings.constraints.pos:
            data.constraints.append(EqualityConstraint(G_wp, h_wp, settings.mpc.waypoints_pos_tol))

        # Waypoint velocity cost and/or equality constraint
        G_wv = self.M_v_S_u_W_input[wp_idx]
        h_wv = des_vel - self.M_v_S_x[wp_idx] @ data.x_0
        self.quad_cost += 2 * settings.weights.vel * G_wv.T @ G_wv
        self.linear_cost += -2 * settings.weights.vel * G_wv.T @ h_wv
        if settings.constraints.vel:
            data.constraints.append(EqualityConstraint(G_wv, h_wv, settings.mpc.waypoints_vel_tol))

        # Waypoint acceleration cost and/or equality constraint
        G_wa = self.M_a_S_u_prime_W_input[wp_idx]
        h_wa = des_acc - self.M_a_S_x_prime[wp_idx] @ data.x_0
        self.quad_cost += 2 * settings.weights.acc * G_wa.T @ G_wa
        self.linear_cost += -2 * settings.weights.acc * G_wa.T @ h_wa
        if settings.constraints.acc:
            data.constraints.append(EqualityConstraint(G_wa, h_wa, settings.mpc.waypoints_acc_tol))

        # Input continuity cost and/or equality constraint
        h_u = np.concatenate([data.u_0, data.u_dot_0, data.u_ddot_0])
        self.linear_cost += -2 * settings.weights.input_continuity * self.G_u.T @ h_u
        if settings.constraints.input_continuity:
            data.constraints.append(
                EqualityConstraint(self.G_u, h_u, settings.mpc.input_continuity_tol)
            )

        # Position constraint
        upper = np.tile(settings.limits.pos_max, K + 1) - self.M_p_S_x @ data.x_0
        lower = -np.tile(settings.limits.pos_min, K + 1) + self.M_p_S_x @ data.x_0
        h_p = np.concatenate([upper, lower])
        data.constraints.append(InequalityConstraint(self.G_p, h_p, settings.mpc.pos_tol))

        # Velocity constraint
        c_v = self.M_v_S_x @ data.x_0
        data.constraints.append(
            PolarInequalityConstraint(
                self.M_v_S_u_W_input,
                c_v,
                -float("inf"),
                settings.limits.vel_max,
                1.0,
                settings.mpc.vel_tol,
            )
        )

        # Acceleration constraint
        c_a = self.M_a_S_x_prime @ data.x_0
        data.constraints.append(
            PolarInequalityConstraint(
                self.M_a_S_u_prime_W_input,
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
            G_c = envelope[:, None] * self.M_p_S_u_W_input
            c_c = envelope * (self.M_p_S_x @ data.x_0 - pos)
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

    def post_solve(self, data: SolverData, settings: SolverSettings) -> Result:
        """Process optimization results into Result object.

        Override of AMSolver method that extracts trajectories from solution.
        """
        """Process optimization results into Result object.

        Override of AMSolver method that extracts trajectories from solution.

        Args:
            zeta: Solution vector from optimization

        Returns:
            Result containing optimized trajectories
        """
        K = settings.mpc.K
        # Extract position trajectory from state trajectory
        pos = (self.S_x @ data.x_0 + self.S_u_W_input @ data.zeta).T.reshape((K + 1, 6))[:, :3]
        # Get input position, velocity and acceleration from spline coefficients
        u_pos = (self.W @ data.zeta).T.reshape((K, 3))
        u_vel = (self.W_dot @ data.zeta).T.reshape((K, 3))
        u_acc = (self.W_ddot @ data.zeta).T.reshape(K, 3)

        return Result(pos=pos, u_pos=u_pos, u_vel=u_vel, u_acc=u_acc, zeta=data.zeta)

    def actual_solve(
        self, data: SolverData, settings: SolverSettings
    ) -> tuple[bool, int, SolverData]:
        """Conducts actual solving process implementing optimization algorithm.

        Not meant to be overridden by child classes.
        """
        # Initialize solver components
        rho = settings.rho_init

        # Initialize optimization variables and matrices
        Q = np.zeros_like(self.quad_cost)  # Combined quadratic terms
        q = np.zeros(self.quad_cost.shape[0])  # Combined linear terms
        zeta = data.zeta
        zeta = np.zeros(self.quad_cost.shape[0])  # Optimization variable

        bregman_mult = np.zeros(self.quad_cost.shape[0])  # Bregman multiplier

        # Aggregate quadratic and linear terms from all constraints
        quad_constraint_terms = np.zeros_like(self.quad_cost)
        linear_constraint_terms = np.zeros(self.linear_cost.shape[0])

        for constraint in data.constraints:
            quad_constraint_terms += constraint.get_quadratic_term()
            linear_constraint_terms += constraint.get_linear_term()

        # Iteratively solve until solution found or max iterations reached
        for i in range(settings.max_iters):
            Q = self.quad_cost + rho * quad_constraint_terms

            # Construct linear cost matrices
            linear_constraint_terms -= bregman_mult
            q = self.linear_cost + rho * linear_constraint_terms

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

    def solve(self, data: SolverData, settings: SolverSettings) -> tuple[bool, int, any]:
        """Main solve function to be called by user.

        Contains main solving workflow (pre_solve, actual_solve, post_solve).
        Not meant to be overridden.
        """
        # Reset cost and clear carryover constraints from previous solves
        self.reset_cost_matrices()
        data.constraints.clear()
        # Build new constraints and add to cost matrices
        data = self.pre_solve(data, settings)
        # Ensure no carryover updates from previous solve
        for c in data.constraints:
            c.reset()
        # Execute solve process to get raw solution vector
        success, iters, data = self.actual_solve(data, settings)
        # Post-process solution according to derived class implementation
        return success, iters, self.post_solve(data, settings)

    def reset_cost_matrices(self) -> None:
        """Resets cost matrices to initial values"""
        self.quad_cost = self.initial_quad_cost
        self.linear_cost[...] = 0


@dataclass
class Result:
    """Results from drone trajectory optimization"""

    pos: np.ndarray  # K+1 x 3 matrix, each row is position at a timestep
    u_pos: np.ndarray  # K x 3 matrix
    u_vel: np.ndarray  # K x 3 matrix
    u_acc: np.ndarray  # K x 3 matrix
    zeta: np.ndarray | None = None  # Spline coefficients for input parameterization

    @staticmethod
    def initial_result(initial_pos: NDArray, K: int) -> Result:
        """Generate initial Result with stationary trajectories at initial position"""
        # Generate position trajectory by replicating initial position
        pos = np.tile(initial_pos, (K + 1, 1))
        u_pos = np.tile(initial_pos, (K, 1))
        return Result(pos=pos, u_pos=u_pos, u_vel=np.zeros((K, 3)), u_acc=np.zeros((K, 3)))

    def advance_for_next_solve_step(self):
        """Advance trajectories by one step for next solve iteration"""
        # Extrapolate last position by adding the difference between last two positions
        extrapolated_pos = 2 * self.pos[-1] - self.pos[-2]
        self.pos[:-1] = self.pos[1:]
        self.pos[-1] = extrapolated_pos
        # Advance input trajectories - only shift values since we only check first row
        self.u_pos[:-1] = self.u_pos[1:]
        self.u_vel[:-1] = self.u_vel[1:]
        self.u_acc[:-1] = self.u_acc[1:]


@dataclass
class SolverData:
    """Solver data"""

    waypoints: dict[str, NDArray]

    current_time: float = 0.0
    obstacle_envelopes: list[np.ndarray] = None
    obstacle_positions: list[np.ndarray] = None
    x_0: np.ndarray = field(default_factory=lambda: np.zeros(6))  # Initial state [pos, vel]
    u_0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Initial input pos reference
    u_dot_0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Initial input vel reference
    u_ddot_0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Initial input acc reference
    zeta: NDArray | None = None

    constraints: list[Constraint] = field(default_factory=lambda: [])


@dataclass
class SolverSettings:
    rho_init: float  # Initial value of rho
    rho_max: float  # Maximum allowable value of rho
    max_iters: int  # Maximum number of iterations

    constraints: ConstraintSettings
    weights: Weights
    limits: Limits
    mpc: MPCSettings


@dataclass
class ConstraintSettings:
    """Configuration for optimization constraints"""

    pos: bool = True
    vel: bool = True
    acc: bool = True
    input_continuity: bool = True


@dataclass
class Weights:
    """Weights for MPC cost function terms"""

    pos: float = 7000.0  # Waypoint position tracking
    vel: float = 1000.0  # Waypoint velocity tracking
    acc: float = 100.0  # Waypoint acceleration tracking
    smoothness: float = 100.0  # Trajectory smoothness
    input_smoothness: float = 1000.0  # Input smoothness
    input_continuity: float = 100.0  # Input continuity


@dataclass
class MPCSettings:
    """Configuration parameters for MPC problem"""

    K: int = 25  # Number of timesteps in horizon
    N: int = 10  # Number of spline coefficients
    freq: float = 8.0  # MPC solver frequency (Hz)
    bf_gamma: float = 1.0  # Barrier function gamma parameter
    waypoints_pos_tol: float = 1e-2  # Waypoint position constraint tolerance
    waypoints_vel_tol: float = 1e-2  # Waypoint velocity constraint tolerance
    waypoints_acc_tol: float = 1e-2  # Waypoint acceleration constraint tolerance
    input_continuity_tol: float = 1e-2  # Input continuity constraint tolerance
    pos_tol: float = 1e-2  # Position constraint tolerance
    vel_tol: float = 1e-2  # Velocity constraint tolerance
    acc_tol: float = 1e-2  # Acceleration constraint tolerance
    collision_tol: float = 1e-2  # Collision constraint tolerance


@dataclass
class Limits:
    """Physical limits and constraints for the drone"""

    pos_min: np.ndarray = field(default_factory=lambda: np.full(3, -10))  # Minimum position bounds
    pos_max: np.ndarray = field(default_factory=lambda: np.full(3, 10))  # Maximum position bounds
    vel_max: float = 1.73  # Maximum velocity
    acc_max: float = 0.75 * 9.81  # Maximum acceleration (0.75g)
    collision_x: float = 0.25  # Collision envelope width in x
    collision_y: float = 0.25  # Collision envelope width in y
    collision_z: float = 2.0 / 3.0  # Collision envelope width in z


@dataclass
class Dynamics:
    """Matrices for drone dynamics"""

    A: np.ndarray  # State transition matrix
    B: np.ndarray  # Input matrix
    A_prime: np.ndarray  # State derivative transition matrix
    B_prime: np.ndarray  # Input derivative matrix

    def __init__(self, A: np.ndarray, B: np.ndarray, A_prime: np.ndarray, B_prime: np.ndarray):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.A_prime = np.asarray(A_prime)
        self.B_prime = np.asarray(B_prime)


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


def bernstein_matrices(K: int, N: int, freq: int) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Initialize Bernstein polynomial matrices used for input parameterization.

    Args:
        mpc_settings: MPC configuration parameters

    Returns:
        Tuple of (W, W_dot, W_ddot, W_input) matrices where:
        - W: Position basis matrix
        - W_dot: Velocity basis matrix
        - W_ddot: Acceleration basis matrix
        - W_input: Combined position/velocity input matrix
    """
    W = np.zeros((3 * K, 3 * (N + 1)))
    W_dot = np.zeros((3 * K, 3 * (N + 1)))
    W_ddot = np.zeros((3 * K, 3 * (N + 1)))

    t_f = (K - 1) / freq

    for k in range(K):
        t = k / freq
        t_f_minus_t = t_f - t
        t_pow_n = t_f**N

        for m in range(N + 1):
            n_choose_m = comb(N, m, exact=True)
            # Position basis
            val = (t**m) * n_choose_m * (t_f_minus_t ** (N - m)) / t_pow_n

            # Velocity basis
            if k == 0 and m == 0:
                dot_val = -N / t_f
            elif k == (K - 1) and m == N:
                dot_val = N / t_f
            else:
                dot_val = (
                    t_f**-N
                    * n_choose_m
                    * (
                        m * (t ** (m - 1)) * (t_f_minus_t ** (N - m))
                        - (t**m) * (N - m) * (t_f_minus_t ** (N - m - 1))
                    )
                )

            # Acceleration basis
            if k == 0 and m == 0:
                dotdot_val = N * (N - 1) / (t_f**2)
            elif k == K - 1 and m == N:
                dotdot_val = N * (N - 1) / (t_f**2)
            elif k == 0 and m == 1:
                dotdot_val = -2 * N * (N - 1) / (t_f**2)
            elif k == K - 1 and m == N - 1:
                dotdot_val = -2 * N * (N - 1) / (t_f**2)
            else:
                dotdot_val = (
                    (t_f**-N)
                    * n_choose_m
                    * (
                        m * (m - 1) * (t ** (m - 2)) * (t_f_minus_t ** (N - m))
                        - 2 * m * (N - m) * (t ** (m - 1)) * (t_f_minus_t ** (N - m - 1))
                        + (N - m) * (N - m - 1) * (t**m) * (t_f_minus_t ** (N - m - 2))
                    )
                )

            # Fill matrices if values are non-zero
            row_idx = [3 * k, 3 * k + 1, 3 * k + 2]
            col_idx = [m, m + N + 1, m + 2 * (N + 1)]

            W[row_idx, col_idx] = val
            W_dot[row_idx, col_idx] = dot_val
            W_ddot[row_idx, col_idx] = dotdot_val

    # Matrix of rows [x0, y0, z0, vx0, vy0, vz0, x1, y1, z1, vx1, vy1, vz1, ...]
    W_view = W.reshape(-1, W.shape[1] * 3)
    W_dot_view = W_dot.reshape(-1, W_dot.shape[1] * 3)
    W_input = np.concat((W_view, W_dot_view), axis=-1).reshape(-1, W.shape[1])

    return W, W_dot, W_ddot, W_input


def full_horizon_dynamics_matrices(
    dynamics: Dynamics, K: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Initialize full horizon dynamics matrices for MPC.

    See thesis document for derivation of these matrices.

    Args:
        dynamics: Sparse dynamics matrices A, B, A_prime, B_prime
        K: Number of timesteps in horizon

    Returns:
        Tuple of (S_x, S_u, S_x_prime, S_u_prime) matrices
    """
    n_states = dynamics.A.shape[0]
    n_inputs = dynamics.B.shape[1]

    # Initialize matrices
    S_x = np.zeros((n_states * (K + 1), n_states))
    S_x_prime = np.zeros((n_states * (K + 1), n_states))
    S_u = np.zeros((n_states * (K + 1), n_inputs * K))
    S_u_prime = np.zeros((n_states * (K + 1), n_inputs * K))

    # Build S_x and S_x_prime
    for k in range(K + 1):
        S_x_block = np.linalg.matrix_power(dynamics.A, k)

        if k == 0:
            S_x_prime_block = np.zeros((n_states, n_states))
        else:
            S_x_prime_block = dynamics.A_prime @ np.linalg.matrix_power(dynamics.A, k - 1)

        S_x[k * n_states : (k + 1) * n_states, :] = S_x_block
        S_x_prime[k * n_states : (k + 1) * n_states, :] = S_x_prime_block

    # Build S_u and S_u_prime
    S_u_col = np.zeros((n_states * K, n_inputs))
    S_u_prime_col = np.zeros((n_states * K, n_inputs))

    for k in range(K):
        S_u_col_block = np.linalg.matrix_power(dynamics.A, k) @ dynamics.B
        S_u_col[k * n_states : (k + 1) * n_states, :] = S_u_col_block

    S_u_prime_col[0:n_states, :] = dynamics.B_prime
    for k in range(1, K):
        A_pow_k_minus_1 = np.linalg.matrix_power(dynamics.A, k - 1)
        S_u_prime_col_block = dynamics.A_prime @ A_pow_k_minus_1 @ dynamics.B
        S_u_prime_col[k * n_states : (k + 1) * n_states, :] = S_u_prime_col_block

    for k in range(K):
        row_start = (k + 1) * n_states
        col_start, col_end = k * n_inputs, (k + 1) * n_inputs

        S_u[row_start:, col_start:col_end] = S_u_col[: (K - k) * n_states, :]
        S_u_prime[row_start:, col_start:col_end] = S_u_prime_col[: (K - k) * n_states, :]

    return S_x, S_u, S_x_prime, S_u_prime
