import numpy as np


def generate_time_stamps(num_waypoints, min_interval=1.0, max_interval=4):
    """Generate a sorted array of time stamps for waypoint creation.

    Note: If num_waypoints is 1 or less, the function returns an array containing only the initial
        time stamp 0.

    Returns:
        An array of time stamps, starting from 0, sorted in ascending order.
    """
    if num_waypoints <= 1:
        return np.array([0])

    intervals = np.random.uniform(min_interval, max_interval, size=num_waypoints - 1)
    time_stamps = np.insert(np.cumsum(intervals), 0, 0)
    return time_stamps


def generate_random_positions(
    n_drones, min_distance, max_change, scaling=1.0, previous_positions=None
):
    """Generate random positions for drones that are at least min_distance from each other."""
    if previous_positions is None:
        pos = (np.random.rand(n_drones, 3) - 0.5) * 2 * scaling
        pos[..., 2] += scaling
    else:
        pos = previous_positions + (np.random.rand(n_drones, 3) - 0.5) * 2 * max_change
    pos[..., 2] = np.clip(pos[..., 2], 0.1, 10)

    # Resample positions that are too close until all are at least min_distance apart
    max_attempts = 1000
    for _ in range(max_attempts):
        # Compute pairwise distances
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
        # Set diagonal to a large value to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        # Find pairs that are too close
        too_close = np.where(dists < min_distance)
        if len(too_close[0]) == 0:
            break  # All positions are valid
        # Resample positions for drones that are too close to any other
        to_resample = set(too_close[0]) | set(too_close[1])
        for idx in to_resample:
            if previous_positions is None:
                pos[idx] = (np.random.rand(3) - 0.5) * 2 * scaling
            else:
                pos[idx] = previous_positions[idx] + (np.random.rand(3) - 0.5) * 2 * max_change
            pos[..., 2] = np.clip(pos[..., 2], 0.1, 10)
    else:
        raise RuntimeError(
            "Could not generate positions with required min_distance after many attempts."
        )
    return pos


def generate_random_waypoints(n_drones, num_waypoints, min_distance=0.5, duration=10.0):
    max_distance = 1.0
    time_stamps = np.linspace(0, duration, num_waypoints)
    scaling = 2.0 * n_drones ** (1 / 3)
    starting_positions = generate_random_positions(n_drones, min_distance, max_distance, scaling)
    drone_waypoints = np.zeros((n_drones, num_waypoints, 10))  # Initialize waypoint array
    drone_waypoints[..., 0] = time_stamps  # Set time stamps
    drone_waypoints[:, 0, 1:4] = starting_positions

    for i in range(1, num_waypoints):
        drone_waypoints[:, i, 1:4] = generate_random_positions(
            n_drones, min_distance, max_distance, scaling, drone_waypoints[:, i - 1, 1:4]
        )

    waypoints = {i: drone_waypoints[i] for i in range(n_drones)}
    return waypoints
