import mujoco
import numpy as np
from crazyflow import Sim
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


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


def draw_line(
    sim: Sim,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        sim: The crazyflow simulation.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    if np.any(np.isnan(points)):
        return
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_points(sim: Sim, points: NDArray, rgba: NDArray | None = None, size: float = 3.0):
    """Draw points into the simulation.

    Args:
        sim: The crazyflow simulation.
        points: An array of [N, 3] points.
        rgba: The color of the line.
        size: The size of points.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many points. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    size = np.ones(3) * size
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = np.eye(3).flatten()
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size, pos=points[i], mat=mats, rgba=rgba
        )


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    v = p2 - p1
    vnorm = np.linalg.norm(p2 - p1, axis=-1, keepdims=True)
    # print(p1.shape, vnorm.shape)
    # <add eps to points that are identical to avoid singularity issues
    p1 = np.where(vnorm < 1e-6, p1 + 1e-4, p1)
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))
