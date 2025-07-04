from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
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
    """Solver data"""

    waypoints: dict[str, NDArray]

    zeta: NDArray  # n_drones x 3 * (N + 1)
    trajectory: Trajectory
    previous_trajectory: Trajectory

    matrices: Matrices | None = None
    current_time: float = 0.0
    x_0: NDArray | None = None  # n_drones x 6 (pos, vel)
    cost: CostData | None = None

    obstacle_positions: list[NDArray] | None = None

    constraints: list[Constraint] = field(default_factory=lambda: [])
    rank: int = 0  # TODO: Remove

    @staticmethod
    def init(waypoints: dict[str, NDArray], K: int, N: int) -> SolverData:
        n_drones = waypoints["pos"].shape[1]
        trajectory = Trajectory.init(waypoints["pos"][0, :], K, n_drones)
        # Init optimization variable
        zeta = np.zeros((n_drones, 3 * (N + 1)))
        return SolverData(
            waypoints=waypoints,
            zeta=zeta,
            trajectory=deepcopy(trajectory),
            previous_trajectory=trajectory,
        )

    def init_matrices(self, A, B, A_prime, B_prime, K: int, N: int, freq: int):
        self.matrices = Matrices.from_dynamics(A, B, A_prime, B_prime, K, N, freq)

    def init_cost(self, weights: Weights):
        assert self.matrices is not None, "Matrices must be initialized before cost"
        self.cost = CostData.init(weights, self.matrices)


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


@dataclass
class CostData:
    """Cost data"""

    quad: Array
    quad_init: Array
    linear: Array
    linear_smoothness_const: Array

    @staticmethod
    def init(weights: Weights, matrices: Matrices):
        K, N = matrices.W.shape[0] // 3, matrices.W.shape[1] // 3 - 1
        quad = 2 * weights.input_smoothness * (matrices.W_ddot.T @ matrices.W_ddot)
        a_idx = np.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        quad += (
            2
            * weights.smoothness
            * matrices.W_input.T
            @ matrices.S_u_prime[a_idx].T
            @ matrices.S_u_prime[a_idx]
            @ matrices.W_input
        )
        quad += 2 * weights.input_continuity * matrices.G_u.T @ matrices.G_u
        linear = np.zeros(3 * (N + 1))
        linear_smoothness_const = (
            2 * weights.smoothness * matrices.M_a_S_u_prime_W_input.T @ matrices.M_a_S_x_prime
        )
        return CostData(
            quad=quad,
            quad_init=quad.copy(),
            linear=linear,
            linear_smoothness_const=linear_smoothness_const,
        )


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
    collision: NDArray | None = None


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
