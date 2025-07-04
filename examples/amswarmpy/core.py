from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from .drone import SolverData, SolverSettings, solve_drone


def solve_swarm(
    states: NDArray, t: float, data: SolverData, settings: SolverSettings
) -> tuple[list[bool], list[int], SolverData]:
    n_drones = len(states)
    envelope = 1.0 / settings.limits.collision
    # Alternatively, use assign_tuples(list(combinations(range(n_drones), 2)))
    avoidance_map = {i: [j for j in range(n_drones) if j != i] for i in range(n_drones)}

    # Initialize results
    is_success = np.zeros(n_drones, dtype=bool)
    iters = np.zeros(n_drones)

    # Solve for each drone
    for i in range(n_drones):
        obstacle_positions = []
        obstacle_envelopes = []

        # Check for potential collisions with drones this drone needs to avoid
        for avoid_drone in avoidance_map[i]:
            # Time taken: 5.00e-05 seconds
            intersect = check_intersection(
                data.previous_results[i].pos, data.previous_results[avoid_drone].pos, envelope
            )
            if intersect:
                obstacle_positions.append(data.previous_results[avoid_drone].pos.flatten())
                obstacle_envelopes.append(envelope)

        data.rank = i
        data.current_time = t
        data.obstacle_positions = obstacle_positions
        data.obstacle_envelopes = obstacle_envelopes
        data.x_0 = states[i]
        data.u_0 = data.previous_results[i].u_pos[0]
        data.u_dot_0 = data.previous_results[i].u_vel[0]
        data.u_ddot_0 = data.previous_results[i].u_acc[0]
        data.waypoints = {k: v[:, i] for k, v in waypoints.items()}

        # Solve for this drone
        success, num_iters, data = solve_drone(data, settings)
        is_success[i] = success
        iters[i] = num_iters

    return is_success, iters, data


@jax.jit
def check_intersection(traj_a: Array, traj_b: Array, envelope: Array) -> Array:
    """Check if two trajectories intersect given their positions and a collision envelope matrix.

    Returns:
        True if trajectories intersect, False otherwise
    """
    assert traj_a.shape == traj_b.shape
    assert traj_a.shape[1] == 3, f"Shape {traj_a.shape} != (N, 3)"
    assert envelope.shape == (3,)
    return jp.any(jp.linalg.norm((traj_a - traj_b) * envelope, axis=-1) <= 1.0)


def assign_tuples(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Assign each pair to the endpoint with fewer assignments.

    This function is greedy. Can be improved by Integer Linear Programming (ILP) with e.g. pulp.
    """
    assigned = {i: [] for i in range(len(pairs))}
    for i, j in pairs:
        idx_source, idx_target = (i, j) if len(assigned[i]) <= len(assigned[j]) else (j, i)
        assigned[idx_source].append(idx_target)
    return assigned
