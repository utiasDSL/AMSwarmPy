from __future__ import annotations

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jp
from einops import rearrange
from flax.struct import dataclass, field
from jax import Array
from numpy.typing import NDArray

from .constraint import EqualityConstraint, InequalityConstraint, PolarInequalityConstraint
from .spline import bernstein_input, bernstein_matrices


@dataclass
class SolverData:
    # Constants
    n_drones: int = field(pytree_node=False)
    waypoints: dict[str, NDArray]
    matrices: Matrices

    # Shared data across drones
    current_time: float
    rank: int  # TODO: Remove
    quad_cost_init: Array  # 3 * (N + 1) x 3 * (N + 1)
    linear_cost_smoothness_const: Array  # 3 * (N + 1)
    in_horizon: Array | None  # dynamic shape! If the shape changes, jax recompiles
    t_discrete: Array | None  # T

    # Swarm data. Tensors of shape (n_drones, ...)
    quad_cost: Array  # n_drones x 3 * (N + 1) x 3 * (N + 1)
    linear_cost: Array  # n_drones x 3 * (N + 1)
    zeta: Array  # n_drones x 3 * (N + 1)
    x_0: Array  # n_drones x 6 (pos, vel)
    trajectory: Trajectory
    previous_trajectory: Trajectory
    distance_matrix: Array  # n_drones x n_drones
    # Constraints
    collision_constraints: PolarInequalityConstraint | None = None
    pos_constraint: EqualityConstraint | None = None
    vel_constraint: EqualityConstraint | None = None
    acc_constraint: EqualityConstraint | None = None
    input_continuity_constraint: EqualityConstraint | None = None
    max_pos_constraint: InequalityConstraint | None = None
    max_vel_constraint: InequalityConstraint | None = None
    max_acc_constraint: InequalityConstraint | None = None

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
        n_drones = waypoints["pos"].shape[0]
        trajectory = Trajectory.init(waypoints["pos"][:, 0], K, n_drones)
        # Init optimization variable
        zeta = jp.zeros((n_drones, 3 * (N + 1)))
        x_0 = jp.concat((waypoints["pos"][:, 0], waypoints["vel"][:, 0]), axis=-1)
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
            in_horizon=None,
            t_discrete=None,
            quad_cost=quad_cost,
            quad_cost_init=quad_cost[0].copy(),
            linear_cost=jp.zeros((n_drones, 3 * (N + 1))),
            linear_cost_smoothness_const=linear_cost_smoothness_const,
            zeta=zeta,
            x_0=x_0,
            trajectory=deepcopy(trajectory),
            previous_trajectory=trajectory,
            distance_matrix=jp.zeros((n_drones, n_drones)),
        )


@dataclass(frozen=False)
class Trajectory:
    """Swarm trajectories"""

    pos: Array  # n_drones x K+1 x 3 matrix, each row is position at a timestep
    u_pos: Array  # n_drones x K x 3 matrix
    u_vel: Array  # n_drones x K x 3 matrix
    u_acc: Array  # n_drones x K x 3 matrix

    @staticmethod
    @partial(jax.jit, static_argnames=("K", "n_drones"))
    def init(pos: Array, K: int, n_drones: int) -> Trajectory:
        """Generate initial trajectory"""
        assert pos.shape == (n_drones, 3), f"{pos.shape} != {(n_drones, 3)}"
        u_pos = jp.tile(pos[:, None, :], (1, K, 1))
        assert u_pos.shape == (n_drones, K, 3), f"{u_pos.shape} != {(n_drones, K, 3)}"
        pos = jp.tile(pos[:, None, :], (1, K + 1, 1))
        assert pos.shape == (n_drones, K + 1, 3), f"{pos.shape} != {(n_drones, K + 1, 3)}"
        return Trajectory(
            pos=pos, u_pos=u_pos, u_vel=jp.zeros((n_drones, K, 3)), u_acc=jp.zeros((n_drones, K, 3))
        )

    @staticmethod
    @jax.jit
    def step(traj: Trajectory) -> Trajectory:
        """Advance trajectories by one step for next solve iteration"""
        # Extrapolate last position by adding the difference between last two positions
        extrapolated_pos = 2 * traj.pos[:, -1] - traj.pos[:, -2]
        pos = traj.pos.at[:, :-1].set(traj.pos[:, 1:])
        pos = pos.at[:, -1].set(extrapolated_pos)
        # Advance input trajectories - only shift values since we only check first row
        u_pos = traj.u_pos.at[:, :-1].set(traj.u_pos[:, 1:])
        u_vel = traj.u_vel.at[:, :-1].set(traj.u_vel[:, 1:])
        u_acc = traj.u_acc.at[:, :-1].set(traj.u_acc[:, 1:])
        return traj.replace(pos=pos, u_pos=u_pos, u_vel=u_vel, u_acc=u_acc)


@dataclass
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
    @partial(jax.jit, static_argnames=("K", "N", "freq"))
    def from_dynamics(A, B, A_prime, B_prime, K: int, N: int, freq: int):
        W, W_dot, W_ddot = bernstein_matrices(K, N, freq)
        W_input = bernstein_input(W, W_dot)

        S_x, S_u, S_x_prime, S_u_prime = full_horizon_dynamics(A, B, A_prime, B_prime, K)
        # Precompute matrices that don't change at solve time
        S_u_W_input = S_u @ W_input

        # Create an index of 0:3, 6:9, 12:15, ...
        p_idx = jp.arange((K + 1) * 6).reshape(-1, 6)[..., :3].flatten()
        # Create an index of 3:6, 9:12, 15:18, ...
        v_idx = jp.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        a_idx = jp.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
        M_p_S_u_W_input = S_u_W_input[p_idx]
        M_v_S_u_W_input = S_u_W_input[v_idx]
        M_a_S_u_prime_W_input = S_u_prime[a_idx] @ W_input

        M_p_S_x = S_x[p_idx]
        M_v_S_x = S_x[v_idx]
        M_a_S_x_prime = S_x_prime[a_idx]

        # Precompute constraint matrices
        G_u = jp.concat((W[:3], W_dot[:3], W_ddot[:3]))
        G_p = jp.concat((M_p_S_u_W_input, -M_p_S_u_W_input))

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


@partial(jax.jit, static_argnames=("n_drones"))
def init_cost(
    smoothness_weight: float,
    input_smoothness_weight: float,
    input_continuity_weight: float,
    matrices: Matrices,
    n_drones: int,
) -> tuple[Array, Array]:
    K = matrices.W.shape[0] // 3
    quad = 2 * input_smoothness_weight * (matrices.W_ddot.T @ matrices.W_ddot)
    a_idx = jp.arange((K + 1) * 6).reshape(-1, 6)[..., 3:].flatten()
    quad += (
        2
        * smoothness_weight
        * matrices.W_input.T
        @ matrices.S_u_prime[a_idx].T
        @ matrices.S_u_prime[a_idx]
        @ matrices.W_input
    )
    quad += 2 * input_continuity_weight * matrices.G_u.T @ matrices.G_u
    quad = jp.tile(quad, (n_drones, 1, 1))
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
