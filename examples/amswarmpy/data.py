from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jp
import numpy as np
from einops import rearrange
from flax.struct import dataclass as flax_dataclass
from jax import Array
from numpy.typing import NDArray

from .constraint import Constraint
from .spline import bernstein_input, bernstein_matrices


@dataclass
class SolverData:
    # Constants
    n_drones: int
    waypoints: dict[str, NDArray]
    matrices: Matrices

    # Shared data across drones
    current_time: float
    rank: int  # TODO: Remove
    quad_cost_init: Array  # 3 * (N + 1) x 3 * (N + 1)
    linear_cost_smoothness_const: Array  # 3 * (N + 1)

    # Swarm data. Tensors of shape (n_drones, ...)
    quad_cost: Array  # n_drones x 3 * (N + 1) x 3 * (N + 1)
    linear_cost: Array  # n_drones x 3 * (N + 1)
    zeta: NDArray  # n_drones x 3 * (N + 1)
    x_0: NDArray  # n_drones x 6 (pos, vel)
    trajectory: Trajectory
    previous_trajectory: Trajectory
    obstacle_positions: list[NDArray]  # TODO: Move to fixed-sized tensor
    distance_matrix: NDArray  # n_drones x n_drones
    # Constraints
    constraints: list[Constraint]
    pos_constraint: EqualityConstraint | None = None
    vel_constraint: EqualityConstraint | None = None
    acc_constraint: EqualityConstraint | None = None
    input_continuity_constraint: EqualityConstraint | None = None
    pos_bound_constraint: InequalityConstraint | None = None

    @staticmethod
    def init(
        waypoints: dict[str, NDArray],
        K: int,
        N: int,
        A: NDArray,
        B: NDArray,
        A_prime: NDArray,
        B_prime: NDArray,
        freq: int,
        smoothness_weight: float,
        input_smoothness_weight: float,
        input_continuity_weight: float,
    ) -> SolverData:
        n_drones = waypoints["pos"].shape[1]
        trajectory = Trajectory.init(waypoints["pos"][0, :], K, n_drones)
        # Init optimization variable
        zeta = np.zeros((n_drones, 3 * (N + 1)))
        x_0 = np.concat((waypoints["pos"][0, :], waypoints["vel"][0, :]), axis=-1)
        matrices = Matrices.from_dynamics(A, B, A_prime, B_prime, K, N, freq)

        quad_cost, linear_cost_smoothness_const = init_cost(
            smoothness_weight, input_smoothness_weight, input_continuity_weight, matrices, n_drones
        )

        return SolverData(
            n_drones=n_drones,
            waypoints=waypoints,
            matrices=matrices,
            current_time=waypoints["time"][0, 0],
            rank=0,
            quad_cost=quad_cost,
            quad_cost_init=quad_cost[0].copy(),
            linear_cost=np.zeros((n_drones, 3 * (N + 1))),
            linear_cost_smoothness_const=linear_cost_smoothness_const,
            zeta=zeta,
            x_0=x_0,
            trajectory=deepcopy(trajectory),
            previous_trajectory=trajectory,
            obstacle_positions=[],
            distance_matrix=np.zeros((n_drones, n_drones)),
            constraints=[],
        )


@dataclass
class EqualityConstraint:
    G: Array
    h: Array
    _G_T_G: Array
    _G_T_h: Array
    tol: float = 1e-2

    @staticmethod
    def init(G: Array, h: Array, tol: float = 1e-2) -> EqualityConstraint:
        G_T_G = G.T @ G
        G_T_h = G.T @ h
        return EqualityConstraint(G, h, G_T_G, G_T_h, tol)

    @staticmethod
    def quadratic_term(cnstr: EqualityConstraint) -> Array:
        return cnstr._G_T_G

    @staticmethod
    def linear_term(cnstr: EqualityConstraint) -> Array:
        return -cnstr._G_T_h

    @staticmethod
    def satisfied(cnstr: EqualityConstraint, zeta: Array) -> bool:
        return np.max(np.abs(cnstr.G @ zeta - cnstr.h)) <= cnstr.tol


@dataclass
class InequalityConstraint:
    G: Array
    h: Array
    _G_T_G: Array
    _G_T_h: Array
    slack: Array
    tol: float = 1e-2

    @staticmethod
    def init(G: Array, h: Array, tol: float = 1e-2) -> InequalityConstraint:
        G_T_G = G.T @ G
        G_T_h = G.T @ h
        slack = np.zeros_like(h)
        return InequalityConstraint(G, h, G_T_G, G_T_h, slack, tol)

    @staticmethod
    def quadratic_term(cnstr: InequalityConstraint) -> Array:
        return cnstr._G_T_G

    @staticmethod
    def linear_term(cnstr: InequalityConstraint) -> Array:
        return -cnstr._G_T_h + cnstr.G.T @ cnstr.slack

    @staticmethod
    def update(cnstr: InequalityConstraint, x: Array) -> None:
        cnstr.slack = np.maximum(0, -cnstr.G @ x + cnstr.h)

    @staticmethod
    def satisfied(cnstr: InequalityConstraint, zeta: Array) -> bool:
        return np.max(cnstr.G @ zeta - cnstr.h) < cnstr.tol


@dataclass
class PolarInequalityConstraint:
    """Manages polar inequality constraints of a specialized form.

    This class is designed to handle constraints defined by polar coordinates that conform to the
    formula
    Gx + c = h(alpha, beta, d)

    with the boundary condition lwr_bound <= d <= upr_bound.

    Here, 'alpha', 'beta', and 'd' are vectors with a length of K+1, where 'd' represents the
    distance from the origin, 'alpha' the azimuthal angle, and 'beta' the polar angle. The vector
    'h' has a length of 3(K+1), where each set of three elements in 'h()' represents a point in 3D
    space expressed as:

    d[k] * [cos(alpha[k]) * sin(beta[k]), sin(alpha[k]) * sin(beta[k]), cos(beta[k])]^T

    This represents a unit vector defined by angles 'alpha[k]' and 'beta[k]', scaled by 'd[k]',
    where 'k' is an index running from 0 to K. The index range from 0 to K can be interpreted as
    discrete time steps, allowing this constraint to serve as a Barrier Function (BF) constraint to
    manage the rate at which a constraint boundary is approached over successive time steps.
    """

    G: Array
    c: Array
    h: Array
    _G_T_G: Array
    lwr_bound: float | None
    upr_bound: float | None
    bf_gamma: float
    tol: float

    @staticmethod
    def init(
        G: np.ndarray,
        c: np.ndarray,
        lwr_bound: float | None = None,
        upr_bound: float | None = None,
        bf_gamma: float = 1.0,
        tol: float = 1e-2,
    ) -> PolarInequalityConstraint:
        return PolarInequalityConstraint(G, c, -c, G.T @ G, lwr_bound, upr_bound, bf_gamma, tol)

    @staticmethod
    def quadratic_term(cnstr: PolarInequalityConstraint) -> Array:
        return cnstr._G_T_G

    @staticmethod
    def linear_term(cnstr: PolarInequalityConstraint) -> Array:
        return -cnstr.G.T @ cnstr.h

    @staticmethod
    def update(cnstr: PolarInequalityConstraint, x: Array) -> PolarInequalityConstraint:
        if cnstr.G.shape[1] != x.shape[0]:
            raise ValueError("G and x are not compatible sizes")
        assert not (cnstr.upr_bound is not None and cnstr.lwr_bound is not None)

        if cnstr.bf_gamma == 1.0:
            cnstr.h = cnstr._fast_h(cnstr, x) - cnstr.c
            return cnstr

        h = cnstr.G @ x + cnstr.c
        prev_norm = 0
        h = h.reshape(-1, 3)
        h_norm = np.linalg.norm(h, axis=-1)

        prev_norm = cnstr.upr_bound if cnstr.upr_bound is not None else cnstr.lwr_bound

        for i in range(0, h.shape[0]):
            # Calculate norm of current time segment
            segment_norm = h_norm[i]

            # Apply upper bound if not infinite
            if cnstr.upr_bound is not None:
                bound = cnstr.bf_gamma * cnstr.upr_bound + (1.0 - cnstr.bf_gamma) * prev_norm

                if segment_norm > bound:
                    h[i] *= bound / segment_norm
                    segment_norm = bound

            # Apply lower bound if not infinite
            if cnstr.lwr_bound is not None:
                bound = cnstr.bf_gamma * cnstr.lwr_bound + (1.0 - cnstr.bf_gamma) * prev_norm
                assert cnstr.bf_gamma == 1.0

                if segment_norm < bound:
                    h[i] *= bound / segment_norm
                    segment_norm = bound

            # Track norm for next iteration
            prev_norm = segment_norm

        h = h.flatten()
        cnstr.h = h - cnstr.c
        return cnstr

    @staticmethod
    def _fast_h(cnstr: PolarInequalityConstraint, x: Array) -> Array:
        assert cnstr.bf_gamma == 1.0, "Fast-path barrier function gamma must be zero"

        h = cnstr.G @ x + cnstr.c
        h = h.reshape(-1, 3)
        h_norm = np.linalg.norm(h, axis=-1)

        if cnstr.upr_bound is not None:
            mask = h_norm > cnstr.upr_bound
            bound = cnstr.upr_bound
        elif cnstr.lwr_bound is not None:
            mask = h_norm < cnstr.lwr_bound
            bound = cnstr.lwr_bound
        else:
            raise ValueError("Must be either upper or lower")
        h[mask] = h[mask] / h_norm[mask][:, None] * bound
        return h.flatten()

    @staticmethod
    def satisfied(cnstr: PolarInequalityConstraint, zeta: Array) -> bool:
        return np.max(np.abs(cnstr.G @ zeta - cnstr.h)) < cnstr.tol

    @staticmethod
    def reset(cnstr: PolarInequalityConstraint) -> None:
        cnstr.h = -cnstr.c


@dataclass
class Trajectory:
    """Swarm trajectories"""

    pos: np.ndarray  # n_drones x K+1 x 3 matrix, each row is position at a timestep
    u_pos: np.ndarray  # n_drones x K x 3 matrix
    u_vel: np.ndarray  # n_drones x K x 3 matrix
    u_acc: np.ndarray  # n_drones x K x 3 matrix

    @staticmethod
    def init(pos: NDArray, K: int, n_drones: int) -> Trajectory:
        """Generate initial trajectory"""
        assert pos.shape == (n_drones, 3), f"{pos.shape} != {(n_drones, 3)}"
        u_pos = np.tile(pos[:, None, :], (1, K, 1))
        assert u_pos.shape == (n_drones, K, 3), f"{u_pos.shape} != {(n_drones, K, 3)}"
        pos = np.tile(pos[:, None, :], (1, K + 1, 1))
        assert pos.shape == (n_drones, K + 1, 3), f"{pos.shape} != {(n_drones, K + 1, 3)}"
        return Trajectory(
            pos=pos, u_pos=u_pos, u_vel=np.zeros((n_drones, K, 3)), u_acc=np.zeros((n_drones, K, 3))
        )

    def step(self):
        """Advance trajectories by one step for next solve iteration"""
        # Extrapolate last position by adding the difference between last two positions
        extrapolated_pos = 2 * self.pos[:, -1] - self.pos[:, -2]
        self.pos[:, :-1] = self.pos[:, 1:]
        self.pos[:, -1] = extrapolated_pos
        # Advance input trajectories - only shift values since we only check first row
        self.u_pos[:, :-1] = self.u_pos[:, 1:]
        self.u_vel[:, :-1] = self.u_vel[:, 1:]
        self.u_acc[:, :-1] = self.u_acc[:, 1:]


@flax_dataclass
class Matrices:
    """Matrices for drone trajectory optimization"""

    W: Array
    W_dot: Array
    W_ddot: Array
    W_input: Array
    S_x: Array
    S_u: Array
    S_u_W_input: Array
    S_x_prime: Array
    S_u_prime: Array
    M_p_S_u_W_input: Array
    M_v_S_u_W_input: Array
    M_a_S_u_prime_W_input: Array
    M_p_S_x: Array
    M_v_S_x: Array
    M_a_S_x_prime: Array
    G_u: Array
    G_p: Array

    @staticmethod
    def from_dynamics(A, B, A_prime, B_prime, K: int, N: int, freq: int):
        W, W_dot, W_ddot = bernstein_matrices(K, N, freq)
        W, W_dot, W_ddot = np.asarray(W), np.asarray(W_dot), np.asarray(W_ddot)
        W_input = np.asarray(bernstein_input(W, W_dot))

        S_x, S_u, S_x_prime, S_u_prime = full_horizon_dynamics(A, B, A_prime, B_prime, K)
        S_x = np.asarray(S_x)
        S_u, S_x_prime, S_u_prime = np.asarray(S_u), np.asarray(S_x_prime), np.asarray(S_u_prime)
        # Precompute matrices that don't change at solve time
        S_u_W_input = S_u @ W_input

        # Create an index of 0:3, 6:9, 12:15, ...
        p_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., :3].flatten()
        # Create an index of 3:6, 9:12, 15:18, ...
        v_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        a_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        M_p_S_u_W_input = S_u_W_input[p_idx]
        M_v_S_u_W_input = S_u_W_input[v_idx]
        M_a_S_u_prime_W_input = S_u_prime[a_idx] @ W_input

        M_p_S_x = S_x[p_idx]
        M_v_S_x = S_x[v_idx]
        M_a_S_x_prime = S_x_prime[a_idx]

        # Precompute constraint matrices
        G_u = np.concat((W[:3], W_dot[:3], W_ddot[:3]))
        G_p = np.concat((M_p_S_u_W_input, -M_p_S_u_W_input))

        return Matrices(
            W=W,
            W_dot=W_dot,
            W_ddot=W_ddot,
            W_input=W_input,
            S_x=S_x,
            S_u=S_u,
            S_u_W_input=S_u_W_input,
            S_x_prime=S_x_prime,
            S_u_prime=S_u_prime,
            M_p_S_u_W_input=M_p_S_u_W_input,
            M_v_S_u_W_input=M_v_S_u_W_input,
            M_a_S_u_prime_W_input=M_a_S_u_prime_W_input,
            M_p_S_x=M_p_S_x,
            M_v_S_x=M_v_S_x,
            M_a_S_x_prime=M_a_S_x_prime,
            G_u=G_u,
            G_p=G_p,
        )


def init_cost(
    smoothness_weight: float,
    input_smoothness_weight: float,
    input_continuity_weight: float,
    matrices: Matrices,
    n_drones: int,
) -> tuple[Array, Array]:
    K = matrices.W.shape[0] // 3
    quad = 2 * input_smoothness_weight * (matrices.W_ddot.T @ matrices.W_ddot)
    a_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
    quad += (
        2
        * smoothness_weight
        * matrices.W_input.T
        @ matrices.S_u_prime[a_idx].T
        @ matrices.S_u_prime[a_idx]
        @ matrices.W_input
    )
    quad += 2 * input_continuity_weight * matrices.G_u.T @ matrices.G_u
    quad = np.tile(quad, (n_drones, 1, 1))
    linear_smoothness_const = (
        2 * smoothness_weight * matrices.M_a_S_u_prime_W_input.T @ matrices.M_a_S_x_prime
    )
    return quad, linear_smoothness_const


@partial(jax.jit, static_argnames=("K"))
def full_horizon_dynamics(
    A: Array, B: Array, A_prime: Array, B_prime: Array, K: int
) -> tuple[Array, Array, Array, Array]:
    """Initialize full horizon dynamics matrices for MPC.

    See thesis document for derivation of these matrices.

    Args:
        dynamics: Sparse dynamics matrices A, B, A_prime, B_prime
        K: Number of timesteps in horizon

    Returns:
        Tuple of (S_x, S_u, S_x_prime, S_u_prime) matrices
    """
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    A_pow = jax.lax.scan(lambda X, _: (A @ X, X), jp.eye(n_states), length=K + 1)[1]
    S_x = rearrange(A_pow, "k n d -> (k n) d")
    S_x_prime = rearrange(A_pow[:-1] @ A_prime, "k n d -> (k n) d")  # Time step 1 to K
    S_x_prime = jp.concat((jp.zeros((n_states, n_states)), S_x_prime), axis=0)
    # U matrices
    S_u_single = rearrange(A_pow[:-1] @ B, "k n d -> (k n) d")
    A_prime_A_pow_B = rearrange(A_prime @ A_pow[:-1] @ B, "k n d -> (k n) d")
    S_u_prime_single = jp.concat((B_prime, A_prime_A_pow_B), axis=0)

    # TODO: Improve by vectorizing or scanning?
    S_u = jp.zeros((n_states * (K + 1), n_inputs * K))
    S_u_prime = jp.zeros((n_states * (K + 1), n_inputs * K))
    for k in range(K):
        row_start = (k + 1) * n_states
        col_start, col_end = k * n_inputs, (k + 1) * n_inputs
        S_u = S_u.at[row_start:, col_start:col_end].set(S_u_single[: (K - k) * n_states, :])
        S_u_prime = S_u_prime.at[row_start:, col_start:col_end].set(
            S_u_prime_single[: (K - k) * n_states, :]
        )
    return S_x, S_u, S_x_prime, S_u_prime
