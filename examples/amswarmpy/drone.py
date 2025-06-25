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
        solver_config: Configuration for the AMSolver
        waypoints: Matrix of waypoints the drone should follow
        mpc_config: Configuration for the MPC optimization problem
        weights: Weights for the MPC optimization problem
        limits: Physical limits for the drone
        dynamics: Dynamics matrices for the drone
    """

    def __init__(
        self,
        solver_config: AMSolverConfig,
        waypoints: np.ndarray,
        mpc_config: MPCConfig,
        weights: MPCWeights,
        limits: PhysicalLimits,
        dynamics: SparseDynamics,
    ):
        self.constraints: list[Constraint] = []
        self.solver_config = solver_config

        self.waypoints = waypoints
        self.mpc_config = mpc_config
        self.weights = weights
        self.limits = limits

        # Print initialization parameters
        # TODO: REMOVE. THIS IS REPLICATING A BUG IN THE CPP CODE.
        self.limits.y_collision_envelope = self.limits.x_collision_envelope
        self.limits.z_collision_envelope = self.limits.x_collision_envelope

        # Initialize collision envelope matrix
        self.collision_envelope = 1.0 / np.array(
            [limits.x_collision_envelope, limits.y_collision_envelope, limits.z_collision_envelope]
        )

        # Initialize Bernstein and dynamics matrices
        self.W, self.W_dot, self.W_ddot, W_input = self.init_bernstein_matrices(mpc_config)
        self.S_x, S_u, S_x_prime, S_u_prime = self.init_full_horizon_dynamics_matrices(dynamics)
        # Precompute matrices that don't change at solve time
        self.S_u_W_input = S_u @ W_input

        # Create an index of 0:3, 6:9, 12:15, ...
        p_idx = np.arange((mpc_config.K + 1) * 6).reshape(-1, 6)[..., :3].flatten()
        # Create an index of 3:6, 9:12, 15:18, ...
        v_idx = np.arange((mpc_config.K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        a_idx = np.arange((mpc_config.K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        self.M_p_S_u_W_input = self.S_u_W_input[p_idx]
        self.M_v_S_u_W_input = self.S_u_W_input[v_idx]
        self.M_a_S_u_prime_W_input = S_u_prime[a_idx] @ W_input

        self.M_p_S_x = self.S_x[p_idx]
        self.M_v_S_x = self.S_x[v_idx]
        self.M_a_S_x_prime = S_x_prime[a_idx]

        # Precompute constraint matrices
        s = (slice(3), slice(3 * (mpc_config.n + 1)))  # [:3, :3 * (mpc_config.n + 1)]
        self.G_u = np.concat((self.W[s], self.W_dot[s], self.W_ddot[s]))
        self.G_p = np.concat((self.M_p_S_u_W_input, -self.M_p_S_u_W_input))
        # Initialize cost matrices
        self.initial_quad_cost = 2 * weights.input_smoothness * (self.W_ddot.T @ self.W_ddot)
        self.initial_quad_cost += (
            2 * weights.smoothness * W_input.T @ S_u_prime[a_idx].T @ S_u_prime[a_idx] @ W_input
        )

        self.initial_quad_cost += 2 * weights.input_continuity * self.G_u.T @ self.G_u
        self.initial_linear_cost = np.zeros(3 * (mpc_config.n + 1))
        self.linear_cost_smoothness_const_term = (
            2 * weights.smoothness * self.M_a_S_u_prime_W_input.T @ self.M_a_S_x_prime
        )

    def pre_solve(self, args: dict) -> None:
        """Setup optimization problem before solving.

        Override of AMSolver method that configures constraints and cost functions.
        """
        args = DroneSolveArgs(**args)
        # Extract waypoints in current horizon. Each row is a waypoint of form:
        # [k, x, y, z, vx, vy, vz, ax, ay, az]. k is discrete STEP in current horizon
        mask, steps = filter_horizon(
            self.waypoints["time"], args.current_time, self.mpc_config.K, self.mpc_config.mpc_freq
        )

        # Separate and reshape waypoints into position, velocity, and acceleration vectors
        des_pos = self.waypoints["pos"][mask].flatten()
        des_vel = self.waypoints["vel"][mask].flatten()
        des_acc = self.waypoints["acc"][mask].flatten()

        # Extract penalized steps from first column of waypoints
        # First possible penalized step is 1, NOT 0 (input cannot affect initial state)
        # Create matrix that selects timesteps corresponding to waypoints
        M_waypoints = np.zeros((3 * len(steps), 3 * (self.mpc_config.K + 1)))
        eye3 = np.eye(3)
        for i in range(len(steps)):
            M_waypoints[3 * i : 3 * (i + 1), 3 * int(steps[i]) : 3 * int(steps[i]) + 3] = eye3
        # Plot the waypoint selection matrix
        # Output smoothness cost
        self.linear_cost += self.linear_cost_smoothness_const_term @ args.x_0

        # --- Add constraints - see thesis document for derivations ---
        # Waypoint position cost and/or equality constraint
        G_wp = M_waypoints @ self.M_p_S_u_W_input
        h_wp = des_pos - M_waypoints @ self.M_p_S_x @ args.x_0
        self.quad_cost += 2 * self.weights.waypoints_pos * G_wp.T @ G_wp
        self.linear_cost += -2 * self.weights.waypoints_pos * G_wp.T @ h_wp
        if args.constraint_config.waypoints_pos:
            self.add_constraint(EqualityConstraint(G_wp, h_wp, self.mpc_config.waypoints_pos_tol))

        # Waypoint velocity cost and/or equality constraint
        G_wv = M_waypoints @ self.M_v_S_u_W_input
        h_wv = des_vel - M_waypoints @ self.M_v_S_x @ args.x_0
        self.quad_cost += 2 * self.weights.waypoints_vel * G_wv.T @ G_wv
        self.linear_cost += -2 * self.weights.waypoints_vel * G_wv.T @ h_wv
        if args.constraint_config.waypoints_vel:
            self.add_constraint(EqualityConstraint(G_wv, h_wv, self.mpc_config.waypoints_vel_tol))

        # Waypoint acceleration cost and/or equality constraint
        G_wa = M_waypoints @ self.M_a_S_u_prime_W_input
        h_wa = des_acc - M_waypoints @ self.M_a_S_x_prime @ args.x_0
        self.quad_cost += 2 * self.weights.waypoints_acc * G_wa.T @ G_wa
        self.linear_cost += -2 * self.weights.waypoints_acc * G_wa.T @ h_wa
        if args.constraint_config.waypoints_acc:
            self.add_constraint(EqualityConstraint(G_wa, h_wa, self.mpc_config.waypoints_acc_tol))

        # Input continuity cost and/or equality constraint
        h_u = np.concatenate([args.u_0, args.u_dot_0, args.u_ddot_0])
        self.linear_cost += -2 * self.weights.input_continuity * self.G_u.T @ h_u
        if args.constraint_config.input_continuity:
            self.add_constraint(
                EqualityConstraint(self.G_u, h_u, self.mpc_config.input_continuity_tol)
            )

        # Position constraint
        h_p = np.concatenate(
            [
                np.tile(self.limits.p_max, self.mpc_config.K + 1) - self.M_p_S_x @ args.x_0,
                -np.tile(self.limits.p_min, self.mpc_config.K + 1) + self.M_p_S_x @ args.x_0,
            ]
        )
        self.add_constraint(InequalityConstraint(self.G_p, h_p, self.mpc_config.pos_tol))

        # Velocity constraint
        c_v = self.M_v_S_x @ args.x_0
        self.add_constraint(
            PolarInequalityConstraint(
                self.M_v_S_u_W_input,
                c_v,
                -float("inf"),
                self.limits.v_bar,
                1.0,
                self.mpc_config.vel_tol,
            )
        )

        # Acceleration constraint
        c_a = self.M_a_S_x_prime @ args.x_0
        self.add_constraint(
            PolarInequalityConstraint(
                self.M_a_S_u_prime_W_input,
                c_a,
                -float("inf"),
                self.limits.a_bar,
                1.0,
                self.mpc_config.acc_tol,
            )
        )

        # Collision constraints
        for i in range(args.num_obstacles):
            G_c = args.obstacle_envelopes[i] @ self.M_p_S_u_W_input
            c_c = args.obstacle_envelopes[i] @ (
                self.M_p_S_x @ args.x_0 - args.obstacle_positions[i]
            )
            self.add_constraint(
                PolarInequalityConstraint(
                    G_c,
                    c_c,
                    1.0,
                    float("inf"),
                    self.mpc_config.bf_gamma,
                    self.mpc_config.collision_tol,
                )
            )

    def post_solve(self, zeta: np.ndarray, args: DroneSolveArgs) -> DroneResult:
        """Process optimization results into DroneResult object.

        Override of AMSolver method that extracts trajectories from solution.
        """
        """Process optimization results into DroneResult object.

        Override of AMSolver method that extracts trajectories from solution.

        Args:
            zeta: Solution vector from optimization
            args: Arguments used in optimization

        Returns:
            DroneResult containing optimized trajectories
        """
        args = DroneSolveArgs(**args)
        # Get state trajectory vector from spline coefficients, reshape into matrix where each row
        # is state at a time step
        state_trajectory_vector = self.S_x @ args.x_0 + self.S_u_W_input @ zeta
        state_trajectory = state_trajectory_vector.reshape((6, self.mpc_config.K + 1)).T

        # Extract position trajectory from state trajectory
        p_idx = np.arange((self.mpc_config.K + 1) * 6).reshape(-1, 6)[..., :3].flatten()
        position_trajectory_vector = state_trajectory_vector[p_idx]
        position_trajectory = position_trajectory_vector.reshape((3, self.mpc_config.K + 1)).T

        # Get input position reference from spline coefficients
        input_position_trajectory_vector = self.W @ zeta
        input_position_trajectory = (
            input_position_trajectory_vector.reshape((3, self.mpc_config.K))
        ).T

        # Get input velocity reference from spline coefficients
        input_velocity_trajectory_vector = self.W_dot @ zeta
        input_velocity_trajectory = (
            input_velocity_trajectory_vector.reshape((3, self.mpc_config.K))
        ).T

        # Get input acceleration reference from spline coefficients
        input_acceleration_trajectory_vector = self.W_ddot @ zeta
        input_acceleration_trajectory = input_acceleration_trajectory_vector.reshape(
            (3, self.mpc_config.K)
        ).T

        # Store spline coefficients directly from optimization results
        spline_coeffs = zeta
        drone_result = DroneResult(
            position_trajectory=position_trajectory,
            position_trajectory_vector=position_trajectory_vector,
            state_trajectory=state_trajectory,
            state_trajectory_vector=state_trajectory_vector,
            input_position_trajectory=input_position_trajectory,
            input_position_trajectory_vector=input_position_trajectory_vector,
            input_velocity_trajectory=input_velocity_trajectory,
            input_velocity_trajectory_vector=input_velocity_trajectory_vector,
            input_acceleration_trajectory=input_acceleration_trajectory,
            input_acceleration_trajectory_vector=input_acceleration_trajectory_vector,
            spline_coeffs=spline_coeffs,
        )

        return drone_result

    def actual_solve(self, args) -> tuple[bool, int, np.ndarray]:
        """Conducts actual solving process implementing optimization algorithm.

        Not meant to be overridden by child classes.
        """
        # Initialize solver components
        iters = 0
        rho = self.solver_config.rho_init

        # Initialize optimization variables and matrices
        Q = np.zeros_like(self.quad_cost)  # Combined quadratic terms
        q = np.zeros(self.quad_cost.shape[0])  # Combined linear terms
        x = np.zeros(self.quad_cost.shape[0])  # Optimization variable
        bregman_mult = np.zeros(self.quad_cost.shape[0])  # Bregman multiplier

        # Aggregate quadratic and linear terms from all constraints
        quad_constraint_terms = np.zeros_like(self.quad_cost)
        linear_constraint_terms = np.zeros(self.linear_cost.shape[0])

        for constraint in self.constraints:
            quad_constraint_terms += constraint.get_quadratic_term()
            linear_constraint_terms += constraint.get_linear_term()

        # Plot heatmaps of Q matrices side by side

        # Iteratively solve until solution found or max iterations reached
        while iters < self.solver_config.max_iters:
            Q = self.quad_cost + rho * quad_constraint_terms

            # Construct linear cost matrices
            linear_constraint_terms -= bregman_mult
            q = self.linear_cost + rho * linear_constraint_terms

            # Solve the QP
            x = np.linalg.solve(Q, -q)

            # Update constraints
            self.update_constraints(x)

            # Check if all constraints are satisfied
            all_constraints_satisfied = all(
                constraint.is_satisfied(x) for constraint in self.constraints
            )

            if all_constraints_satisfied:
                return True, iters, x  # Exit loop, indicate success

            # Recalculate linear term for Bregman multiplier
            linear_constraint_terms[...] = 0
            for constraint in self.constraints:
                linear_constraint_terms += constraint.get_linear_term()

            # Calculate Bregman multiplier
            bregman_update = 0.5 * (quad_constraint_terms @ x + linear_constraint_terms)
            bregman_mult -= bregman_update

            # Gradually increase penalty parameter
            rho *= self.solver_config.rho_init
            rho = min(rho, self.solver_config.max_rho)
            iters += 1

        return False, iters, x  # Indicate failure but still return vector

    def solve(self, args) -> tuple[bool, int, any]:
        """Main solve function to be called by user.

        Contains main solving workflow (pre_solve, actual_solve, post_solve).
        Not meant to be overridden.
        """
        # Reset cost and clear carryover constraints from previous solves
        self.reset_cost_matrices()
        self.constraints.clear()
        # Build new constraints and add to cost matrices
        self.pre_solve(args)
        # Ensure no carryover updates from previous solve
        self.reset_constraints()
        # Execute solve process to get raw solution vector
        success, iters, result = self.actual_solve(args)
        # Post-process solution according to derived class implementation
        return success, iters, self.post_solve(result, args)

    def reset_cost_matrices(self) -> None:
        """Resets cost matrices to initial values"""
        self.quad_cost = self.initial_quad_cost
        self.linear_cost = self.initial_linear_cost

    def add_constraint(self, constraint: Constraint) -> None:
        """Adds a constraint to the solver"""
        self.constraints.append(constraint)

    def update_constraints(self, x: np.ndarray) -> None:
        """Updates all non-constant constraints based on current optimization variables"""
        for constraint in self.constraints:
            constraint.update(x)

    def reset_constraints(self) -> None:
        """Resets constraints to initial state"""
        for constraint in self.constraints:
            constraint.reset()

    def init_bernstein_matrices(
        self, mpc_config: MPCConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize Bernstein polynomial matrices used for input parameterization.

        Args:
            mpc_config: MPC configuration parameters

        Returns:
            Tuple of (W, W_dot, W_ddot, W_input) matrices where:
            - W: Position basis matrix
            - W_dot: Velocity basis matrix
            - W_ddot: Acceleration basis matrix
            - W_input: Combined position/velocity input matrix
        """
        # Initialize sparse matrices
        W = np.zeros((3 * mpc_config.K, 3 * (mpc_config.n + 1)))
        W_dot = np.zeros((3 * mpc_config.K, 3 * (mpc_config.n + 1)))
        W_ddot = np.zeros((3 * mpc_config.K, 3 * (mpc_config.n + 1)))
        W_input = np.zeros((6 * mpc_config.K, 3 * (mpc_config.n + 1)))

        t_f = (1 / mpc_config.mpc_freq) * (mpc_config.K - 1)

        for k in range(mpc_config.K):
            t = k * (1 / mpc_config.mpc_freq)
            t_f_minus_t = t_f - t
            t_pow_n = t_f**mpc_config.n

            for m in range(mpc_config.n + 1):
                # Position basis
                val = (
                    (t**m)
                    * comb(mpc_config.n, m, exact=True)
                    * (t_f_minus_t ** (mpc_config.n - m))
                    / t_pow_n
                )

                # Velocity basis
                if k == 0 and m == 0:
                    dot_val = -mpc_config.n / t_f
                elif k == (mpc_config.K - 1) and m == mpc_config.n:
                    dot_val = mpc_config.n / t_f
                else:
                    dot_val = (
                        (t_f ** (-mpc_config.n))
                        * comb(mpc_config.n, m, exact=True)
                        * (
                            m * (t ** (m - 1)) * (t_f_minus_t ** (mpc_config.n - m))
                            - (t**m) * (mpc_config.n - m) * (t_f_minus_t ** (mpc_config.n - m - 1))
                        )
                    )

                # Acceleration basis
                if k == 0 and m == 0:
                    dotdot_val = mpc_config.n * (mpc_config.n - 1) / (t_f**2)
                elif k == mpc_config.K - 1 and m == mpc_config.n:
                    dotdot_val = mpc_config.n * (mpc_config.n - 1) / (t_f**2)
                elif k == 0 and m == 1:
                    dotdot_val = -2 * mpc_config.n * (mpc_config.n - 1) / (t_f**2)
                elif k == mpc_config.K - 1 and m == mpc_config.n - 1:
                    dotdot_val = -2 * mpc_config.n * (mpc_config.n - 1) / (t_f**2)
                else:
                    dotdot_val = (
                        (t_f**-mpc_config.n)
                        * comb(mpc_config.n, m, exact=True)
                        * (
                            m * (m - 1) * (t ** (m - 2)) * (t_f_minus_t ** (mpc_config.n - m))
                            - 2
                            * m
                            * (mpc_config.n - m)
                            * (t ** (m - 1))
                            * (t_f_minus_t ** (mpc_config.n - m - 1))
                            + (mpc_config.n - m)
                            * (mpc_config.n - m - 1)
                            * (t**m)
                            * (t_f_minus_t ** (mpc_config.n - m - 2))
                        )
                    )

                # Fill matrices if values are non-zero
                if val != 0:
                    W[
                        [3 * k, 3 * k + 1, 3 * k + 2],
                        [m, m + mpc_config.n + 1, m + 2 * (mpc_config.n + 1)],
                    ] = val
                if dot_val != 0:
                    W_dot[
                        [3 * k, 3 * k + 1, 3 * k + 2],
                        [m, m + mpc_config.n + 1, m + 2 * (mpc_config.n + 1)],
                    ] = dot_val
                if dotdot_val != 0:
                    W_ddot[
                        [3 * k, 3 * k + 1, 3 * k + 2],
                        [m, m + mpc_config.n + 1, m + 2 * (mpc_config.n + 1)],
                    ] = dotdot_val

        # Construct input matrix
        for block in range(mpc_config.K):
            W_input[6 * block : 6 * block + 3, :] = W[3 * block : 3 * block + 3, :]
            W_input[6 * block + 3 : 6 * block + 6, :] = W_dot[3 * block : 3 * block + 3, :]

        return W, W_dot, W_ddot, W_input

    def init_full_horizon_dynamics_matrices(
        self, dynamics: SparseDynamics
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize full horizon dynamics matrices for MPC.

        See thesis document for derivation of these matrices.

        Args:
            dynamics: Sparse dynamics matrices A, B, A_prime, B_prime

        Returns:
            Tuple of (S_x, S_u, S_x_prime, S_u_prime) matrices
        """
        num_states = dynamics.A.shape[0]
        num_inputs = dynamics.B.shape[1]

        # Initialize matrices
        S_x = np.zeros((num_states * (self.mpc_config.K + 1), num_states))
        S_x_prime = np.zeros((num_states * (self.mpc_config.K + 1), num_states))
        S_u = np.zeros((num_states * (self.mpc_config.K + 1), num_inputs * self.mpc_config.K))
        S_u_prime = np.zeros((num_states * (self.mpc_config.K + 1), num_inputs * self.mpc_config.K))

        # Build S_x and S_x_prime
        for k in range(self.mpc_config.K + 1):
            temp_S_x_block = np.linalg.matrix_power(dynamics.A, k)

            if k == 0:
                temp_S_x_prime_block = np.zeros((num_states, num_states))
            else:
                temp_S_x_prime_block = dynamics.A_prime @ np.linalg.matrix_power(dynamics.A, k - 1)

            S_x[k * num_states : (k + 1) * num_states, :] = temp_S_x_block
            S_x_prime[k * num_states : (k + 1) * num_states, :] = temp_S_x_prime_block

        # Build S_u and S_u_prime
        S_u_col = np.zeros((num_states * self.mpc_config.K, num_inputs))
        S_u_prime_col = np.zeros((num_states * self.mpc_config.K, num_inputs))

        for k in range(self.mpc_config.K):
            temp_S_u_col_block = np.linalg.matrix_power(dynamics.A, k) @ dynamics.B
            S_u_col[k * num_states : (k + 1) * num_states, :] = temp_S_u_col_block

        S_u_prime_col[0:num_states, :] = dynamics.B_prime
        for k in range(1, self.mpc_config.K):
            temp_S_u_prime_col_block = (
                dynamics.A_prime @ np.linalg.matrix_power(dynamics.A, k - 1) @ dynamics.B
            )
            S_u_prime_col[k * num_states : (k + 1) * num_states, :] = temp_S_u_prime_col_block

        for k in range(self.mpc_config.K):
            S_u[(k + 1) * num_states :, k * num_inputs : (k + 1) * num_inputs] = S_u_col[
                0 : (self.mpc_config.K - k) * num_states, :
            ]
            S_u_prime[(k + 1) * num_states :, k * num_inputs : (k + 1) * num_inputs] = (
                S_u_prime_col[0 : (self.mpc_config.K - k) * num_states, :]
            )

        return S_x, S_u, S_x_prime, S_u_prime


@dataclass
class DroneResult:
    """Results from drone trajectory optimization"""

    position_trajectory: np.ndarray  # K+1 x 3 matrix, each row is position at a timestep
    position_trajectory_vector: np.ndarray  # 3(K+1) x 1 vector
    state_trajectory: np.ndarray  # K+1 x 6 matrix, each row is [position, velocity]
    state_trajectory_vector: np.ndarray  # 6(K+1) x 1 vector
    input_position_trajectory: np.ndarray  # K x 3 matrix
    input_position_trajectory_vector: np.ndarray  # 3K x 1 vector
    input_velocity_trajectory: np.ndarray  # K x 3 matrix
    input_velocity_trajectory_vector: np.ndarray  # 3K x 1 vector
    input_acceleration_trajectory: np.ndarray  # K x 3 matrix
    input_acceleration_trajectory_vector: np.ndarray  # 3K x 1 vector
    spline_coeffs: np.ndarray  # Spline coefficients for input parameterization

    @staticmethod
    def generate_initial_drone_result(initial_position: np.ndarray, K: int) -> DroneResult:
        """Generate initial DroneResult with stationary trajectories at initial position"""
        # Generate state trajectory by appending zero velocity to initial position and replicating
        initial_state = np.zeros(6)
        initial_state[:3] = initial_position
        state_trajectory_vector = np.tile(initial_state, K + 1)
        state_trajectory = state_trajectory_vector.reshape(K + 1, 6)

        # Generate position trajectory by replicating initial position
        position_trajectory_vector = np.tile(initial_position, K + 1)
        position_trajectory = position_trajectory_vector.reshape(K + 1, 3)

        # Generate input position trajectory by replicating initial position K times
        input_position_trajectory_vector = np.tile(initial_position, K)
        input_position_trajectory = input_position_trajectory_vector.reshape(K, 3)

        # Generate input velocity trajectory by replicating zero K times
        input_velocity_trajectory_vector = np.zeros(3 * K)
        input_velocity_trajectory = input_velocity_trajectory_vector.reshape(K, 3)

        # Generate input acceleration trajectory by replicating zero K times
        input_acceleration_trajectory_vector = np.zeros(3 * K)
        input_acceleration_trajectory = input_acceleration_trajectory_vector.reshape(K, 3)

        return DroneResult(
            position_trajectory=position_trajectory,
            position_trajectory_vector=position_trajectory_vector,
            state_trajectory=state_trajectory,
            state_trajectory_vector=state_trajectory_vector,
            input_position_trajectory=input_position_trajectory,
            input_position_trajectory_vector=input_position_trajectory_vector,
            input_velocity_trajectory=input_velocity_trajectory,
            input_velocity_trajectory_vector=input_velocity_trajectory_vector,
            input_acceleration_trajectory=input_acceleration_trajectory,
            input_acceleration_trajectory_vector=input_acceleration_trajectory_vector,
            spline_coeffs=None,
        )

    def advance_for_next_solve_step(self):
        """Advance trajectories by one step for next solve iteration"""
        # Advance state and position trajectories
        self.position_trajectory_vector[:-3] = self.position_trajectory_vector[3:]
        # Extrapolate last position by adding the difference between last two positions
        extrapolated_position = self.position_trajectory_vector[-3:] + (
            self.position_trajectory_vector[-3:] - self.position_trajectory_vector[-6:-3]
        )
        self.position_trajectory_vector[-3:] = extrapolated_position
        self.position_trajectory = self.position_trajectory_vector.reshape(-1, 3)

        # Advance input trajectories - only shift values since we only check first row
        self.input_position_trajectory_vector[:-3] = self.input_position_trajectory_vector[3:]
        self.input_velocity_trajectory_vector[:-3] = self.input_velocity_trajectory_vector[3:]
        self.input_acceleration_trajectory_vector[:-3] = self.input_acceleration_trajectory_vector[
            3:
        ]

        # Reshape input trajectories
        self.input_position_trajectory = self.input_position_trajectory_vector.reshape(-1, 3)
        self.input_velocity_trajectory = self.input_velocity_trajectory_vector.reshape(-1, 3)
        self.input_acceleration_trajectory = self.input_acceleration_trajectory_vector.reshape(
            -1, 3
        )


@dataclass
class ConstraintConfig:
    """Configuration for optimization constraints"""

    waypoints_pos: bool = True
    waypoints_vel: bool = True
    waypoints_acc: bool = True
    input_continuity: bool = True

    def set_waypoints_constraints(self, pos: bool, vel: bool, acc: bool):
        self.waypoints_pos = pos
        self.waypoints_vel = vel
        self.waypoints_acc = acc

    def set_input_continuity_constraints(self, flag: bool):
        self.input_continuity = flag


@dataclass
class MPCWeights:
    """Weights for MPC cost function terms"""

    waypoints_pos: float = 7000.0  # Waypoint position tracking
    waypoints_vel: float = 1000.0  # Waypoint velocity tracking
    waypoints_acc: float = 100.0  # Waypoint acceleration tracking
    smoothness: float = 100.0  # Trajectory smoothness
    input_smoothness: float = 1000.0  # Input smoothness
    input_continuity: float = 100.0  # Input continuity


@dataclass
class MPCConfig:
    """Configuration parameters for MPC problem"""

    K: int = 25  # Number of timesteps in horizon
    n: int = 10  # Number of spline coefficients
    mpc_freq: float = 8.0  # MPC solver frequency (Hz)
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
class PhysicalLimits:
    """Physical limits and constraints for the drone"""

    p_min: np.ndarray = field(default_factory=lambda: np.full(3, -10))  # Minimum position bounds
    p_max: np.ndarray = field(default_factory=lambda: np.full(3, 10))  # Maximum position bounds
    v_bar: float = 1.73  # Maximum velocity
    a_bar: float = 0.75 * 9.81  # Maximum acceleration (0.75g)
    x_collision_envelope: float = 0.25  # Collision envelope width in x
    y_collision_envelope: float = 0.25  # Collision envelope width in y
    z_collision_envelope: float = 2.0 / 3.0  # Collision envelope width in z


@dataclass
class SparseDynamics:
    """Sparse matrices for drone dynamics"""

    A: np.ndarray  # State transition matrix
    B: np.ndarray  # Input matrix
    A_prime: np.ndarray  # State derivative transition matrix
    B_prime: np.ndarray  # Input derivative matrix

    def __init__(self, A: np.ndarray, B: np.ndarray, A_prime: np.ndarray, B_prime: np.ndarray):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.A_prime = np.asarray(A_prime)
        self.B_prime = np.asarray(B_prime)


@dataclass
class DroneSolveArgs:
    """Arguments for drone trajectory optimization"""

    current_time: float = 0.0
    num_obstacles: int = 0
    obstacle_envelopes: list[np.ndarray] = None
    obstacle_positions: list[np.ndarray] = None
    x_0: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # Initial state [position, velocity]
    u_0: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Initial input position reference
    u_dot_0: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Initial input velocity reference
    u_ddot_0: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # Initial input acceleration reference
    constraint_config: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class AMSolverConfig:
    """Configuration settings for the AMSolver"""

    rho_init: float  # Initial value of rho
    max_rho: float  # Maximum allowable value of rho
    max_iters: int  # Maximum number of iterations


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
    rounded_times = np.round((times - t) * mpc_freq)
    # Find time steps with waypoints within current horizon
    in_horizon = np.where((rounded_times > 0) & (rounded_times <= K))[0]
    if len(in_horizon) == 0:
        raise RuntimeError(
            "Error: no waypoints within current horizon. Increase horizon or add waypoints."
        )
    return in_horizon, rounded_times[in_horizon]
