from __future__ import annotations

import numpy as np

from .drone import ConstraintConfig, Drone, DroneResult


class Swarm:
    """Provides a convenient interface to manage trajectory planning for many drones.

    Rather than calling Drone.solve() for each drone individually and keeping track of
    all the predicted trajectories, passing the necessary info for collision avoidance
    to each drone, etc., this class handles that through a single Swarm.solve() call.

    Args:
        drones: List of Drone objects that are part of the swarm
    """

    def __init__(self, drones: list[Drone]):
        self.num_drones = len(drones)
        self.drones = drones
        # Collision envelope for each drone at each time step
        self.all_obstacle_envelopes: list[np.ndarray] = []
        self._reduced_collision_envelopes: list[np.ndarray] = []
        if self.drones:
            # Create identity matrix of size K+1 x K+1
            eye_kp1 = np.eye(self.drones[0].mpc_config.K + 1)

            # Create collision matrices over all time steps for each drone
            for i in range(self.num_drones):
                # At each time step, each drone will take the relevant collision envelopes
                # from this vector according to which drones they need to avoid
                envelope = self.drones[i].collision_envelope
                self.all_obstacle_envelopes.append(np.kron(eye_kp1, envelope * np.eye(3)))
                self._reduced_collision_envelopes.append(envelope)
        self.avoidance_map = {
            i: [j for j in range(self.num_drones) if j != i] for i in range(self.num_drones)
        }
        # self.avoidance_map = assign_tuples(list(combinations(range(self.num_drones), 2)))

    def solve(
        self,
        current_time: float,
        initial_states: list[np.ndarray],
        previous_results: list[DroneResult],
        constraint_configs: list[ConstraintConfig],
    ) -> tuple[list[bool], list[int], list[DroneResult]]:
        """Solves the navigation and collision avoidance problem for the entire swarm.

        Takes the current time, initial states of each drone, results from previous
        computations, and constraint configurations to compute the next trajectories for each
        drone. The past results are used for drones to predict where other drones will be, so they
        can avoid them. Also, the past results are used for the input continuity constraint.

        Args:
            current_time: The current time
            initial_states: List of initial states for each drone. Each initial state consists of [x, y, z, vx, vy, vz]
            previous_results: List of results from previous computation for each drone. If no previous results,
                            can initialize with DroneResult.generate_initial_drone_result(...)
            constraint_configs: List of constraint configurations for each drone

        Returns:
            Tuple containing:
            - List of success flags for each drone
            - List of iteration counts
            - List of results for the current computation
        """
        # Validate input sizes
        if (
            len(initial_states) != self.num_drones
            or len(previous_results) != self.num_drones
            or len(constraint_configs) != self.num_drones
        ):
            raise ValueError("Input lists must all have same length as number of drones in swarm")

        # Initialize results
        is_success = np.zeros(self.num_drones, dtype=bool)
        iters = np.zeros(self.num_drones)
        results = [None] * self.num_drones

        # Solve for each drone
        for i in range(len(self.drones)):
            obstacle_positions = []
            obstacle_envelopes = []
            num_obstacles = 0

            # Check for potential collisions with drones this drone needs to avoid
            for avoid_drone in self.avoidance_map[i]:
                # Time taken: 5.00e-05 seconds
                intersect = check_intersection(
                    previous_results[i].positions,
                    previous_results[avoid_drone].positions,
                    self._reduced_collision_envelopes[avoid_drone],
                )
                if intersect:
                    obstacle_positions.append(previous_results[avoid_drone].positions.flatten())
                    obstacle_envelopes.append(self.all_obstacle_envelopes[avoid_drone])
                    num_obstacles += 1

            # Set up arguments for drone solve
            args = {
                "current_time": current_time,
                "num_obstacles": num_obstacles,
                "obstacle_positions": obstacle_positions,
                "obstacle_envelopes": obstacle_envelopes,
                "x_0": initial_states[i],
                "u_0": previous_results[i].input_positions[0],
                "u_dot_0": previous_results[i].input_velocities[0],
                "u_ddot_0": previous_results[i].input_accelerations[0],
                "constraint_config": constraint_configs[i],
            }

            # Solve for this drone
            success, num_iters, result = self.drones[i].solve(args)
            is_success[i] = success
            iters[i] = num_iters
            results[i] = result

        return is_success, iters, results


def check_intersection(traj1: np.ndarray, traj2: np.ndarray, theta: np.ndarray) -> bool:
    """Check if two trajectories intersect given their positions and a collision envelope matrix.

    Args:
        traj1: Position trajectory of first drone
        traj2: Position trajectory of second drone
        theta: Collision envelope matrix

    Returns:
        True if trajectories intersect, False otherwise
    """
    assert traj1.shape == traj2.shape
    assert traj1.shape[1] == 3, f"Shape {traj1.shape} != (N, 3)"
    assert theta.shape == (3,)
    return np.any(np.linalg.norm((traj1 - traj2) * theta, axis=-1) <= 1.0)


def assign_tuples(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Assign each pair to the endpoint with fewer assignments.

    This function is greedy. Can be improved by Integer Linear Programming (ILP) with e.g. pulp.
    """
    assigned = {i: [] for i in range(len(pairs))}
    for i, j in pairs:
        idx_source, idx_target = (i, j) if len(assigned[i]) <= len(assigned[j]) else (j, i)
        assigned[idx_source].append(idx_target)
    return assigned
