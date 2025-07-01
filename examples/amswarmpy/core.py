from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .drone import Drone, Result, SolverData, SolverSettings


def solve(
    drones: list[Drone],
    current_time: float,
    initial_states: list[np.ndarray],
    waypoints: dict[str, NDArray],
    previous_results: list[Result],
    settings: SolverSettings,
) -> tuple[list[bool], list[int], list[Result]]:
    """Solves the navigation and collision avoidance problem for the entire swarm.

    Takes the current time, initial states of each drone, results from previous
    computations, and constraint configurations to compute the next trajectories for each
    drone. The past results are used for drones to predict where other drones will be, so they
    can avoid them. Also, the past results are used for the input continuity constraint.

    Args:
        current_time: The current time
        initial_states: List of initial states for each drone. Each initial state consists of
            [x, y, z, vx, vy, vz]
        previous_results: List of results from previous computation for each drone. If no previous
            results, can initialize with Result.initial_result(...)
        settings: Solver settings

    Returns:
        Tuple containing:
        - List of success flags for each drone
        - List of iteration counts
        - List of results for the current computation
    """
    # Validate input sizes
    n_drones = len(drones)
    if len(initial_states) != n_drones or len(previous_results) != n_drones:
        raise ValueError("Input lists must all have same length as number of drones in swarm")
    assert isinstance(settings, SolverSettings), f"Unexpected type: {type(settings)}"

    limits = settings.limits
    envelope = 1.0 / np.array([limits.collision_x, limits.collision_y, limits.collision_z])
    # Alternatively, use assign_tuples(list(combinations(range(n_drones), 2)))
    avoidance_map = {i: [j for j in range(n_drones) if j != i] for i in range(n_drones)}

    # Initialize results
    is_success = np.zeros(n_drones, dtype=bool)
    iters = np.zeros(n_drones)
    results = [None] * n_drones

    # Solve for each drone
    for i in range(n_drones):
        obstacle_positions = []
        obstacle_envelopes = []

        # Check for potential collisions with drones this drone needs to avoid
        for avoid_drone in avoidance_map[i]:
            # Time taken: 5.00e-05 seconds
            intersect = check_intersection(
                previous_results[i].pos, previous_results[avoid_drone].pos, envelope
            )
            if intersect:
                obstacle_positions.append(previous_results[avoid_drone].pos.flatten())
                obstacle_envelopes.append(envelope)

        data = SolverData(
            current_time=current_time,
            obstacle_positions=obstacle_positions,
            obstacle_envelopes=obstacle_envelopes,
            x_0=initial_states[i],
            u_0=previous_results[i].u_pos[0],
            u_dot_0=previous_results[i].u_vel[0],
            u_ddot_0=previous_results[i].u_acc[0],
            waypoints={k: v[:, i] for k, v in waypoints.items()},
        )

        # Solve for this drone
        success, num_iters, result = drones[i].solve(data, settings)
        is_success[i] = success
        iters[i] = num_iters
        results[i] = result

    return is_success, iters, results


def check_intersection(traj1: np.ndarray, traj2: np.ndarray, envelope: np.ndarray) -> bool:
    """Check if two trajectories intersect given their positions and a collision envelope matrix.

    Args:
        traj1: Position trajectory of first drone
        traj2: Position trajectory of second drone
        envelope: Collision envelope matrix

    Returns:
        True if trajectories intersect, False otherwise
    """
    assert traj1.shape == traj2.shape
    assert traj1.shape[1] == 3, f"Shape {traj1.shape} != (N, 3)"
    assert envelope.shape == (3,)
    return np.any(np.linalg.norm((traj1 - traj2) * envelope, axis=-1) <= 1.0)


def assign_tuples(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Assign each pair to the endpoint with fewer assignments.

    This function is greedy. Can be improved by Integer Linear Programming (ILP) with e.g. pulp.
    """
    assigned = {i: [] for i in range(len(pairs))}
    for i, j in pairs:
        idx_source, idx_target = (i, j) if len(assigned[i]) <= len(assigned[j]) else (j, i)
        assigned[idx_source].append(idx_target)
    return assigned
