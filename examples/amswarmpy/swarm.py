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
        if self.drones:
            K = self.drones[0].get_K()  # TODO: check if all drones have same K

            # Create identity matrix of size K+1 x K+1
            eye_kp1 = np.eye(K + 1)

            # Create collision matrices over all time steps for each drone
            for i in range(self.num_drones):
                # At each time step, each drone will take the relevant collision envelopes
                # from this vector according to which drones they need to avoid
                self.all_obstacle_envelopes.append(
                    np.kron(eye_kp1, self.drones[i].get_collision_envelope())
                )

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

        # Initialize avoidance responsibility counters and map
        avoidance_counts = [0] * self.num_drones
        avoidance_map = {i: [] for i in range(self.num_drones)}

        # Determine avoidance responsibilities
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                # Assign avoidance responsibility to drone with fewer existing responsibilities
                if avoidance_counts[i] <= avoidance_counts[j]:
                    avoidance_map[i].append(j)
                    avoidance_counts[i] += 1
                else:
                    avoidance_map[j].append(i)
                    avoidance_counts[j] += 1

        # Initialize results
        is_success = [False] * self.num_drones
        iters = [0] * self.num_drones
        results = [None] * self.num_drones

        # Solve for each drone
        for i in range(len(self.drones)):
            obstacle_positions = []
            obstacle_envelopes = []
            num_obstacles = 0

            # Check for potential collisions with drones this drone needs to avoid
            for avoid_drone in avoidance_map[i]:
                if self._check_intersection(
                    previous_results[i].position_trajectory_vector,
                    previous_results[avoid_drone].position_trajectory_vector,
                    0.9 * self.all_obstacle_envelopes[avoid_drone],  # TODO: remove magic number
                ):
                    obstacle_positions.append(
                        previous_results[avoid_drone].position_trajectory_vector
                    )
                    obstacle_envelopes.append(self.all_obstacle_envelopes[avoid_drone])
                    num_obstacles += 1

            # Set up arguments for drone solve
            args = {
                "current_time": current_time,
                "num_obstacles": num_obstacles,
                "obstacle_positions": obstacle_positions,
                "obstacle_envelopes": obstacle_envelopes,
                "x_0": initial_states[i],
                "u_0": previous_results[i].input_position_trajectory[0],
                "u_dot_0": previous_results[i].input_velocity_trajectory[0],
                "u_ddot_0": previous_results[i].input_acceleration_trajectory[0],
                "constraint_config": constraint_configs[i],
            }

            # Solve for this drone
            success, num_iters, result = self.drones[i].solve(args)
            is_success[i] = success
            iters[i] = num_iters
            results[i] = result

        return is_success, iters, results

    def _check_intersection(self, traj1: np.ndarray, traj2: np.ndarray, theta: np.ndarray) -> bool:
        """Checks if two trajectories intersect given their positions and a collision envelope matrix.

        Args:
            traj1: Position trajectory of first drone
            traj2: Position trajectory of second drone
            theta: Collision envelope matrix

        Returns:
            True if trajectories intersect, False otherwise
        """
        # theta accounts for collision envelopes by scaling difference between trajectories
        diff = theta @ (traj1 - traj2)
        # Iterate over chunks of 3 rows (x,y,z positions for one time)
        for i in range(0, diff.shape[0], 3):
            norm = np.linalg.norm(diff[i : i + 3])
            # If norm of any chunk is <= 1, return True for intersection
            if norm <= 1.0:
                return True
        # If no chunk's norm was <= 1, then no intersection detected
        return False
