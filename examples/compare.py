from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import amswarm
import amswarmpy
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from crazyflow import Sim
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

np.random.seed(0)
rgbas = np.random.rand(5, 4)
rgbas[..., 3] = 1.0


def render_solutions(sim, drone_results):
    for i, result in enumerate(drone_results):
        points = result.position_trajectory
        draw_points(sim, points, rgba=rgbas[i], size=0.01)
        draw_line(sim, points, rgba=rgbas[i])


def generate_waypoints(n_drones: int, n_points: int = 4, duration_sec: float = 10.0):
    radius = 0.75
    phase = np.linspace(0, 2 * (1 - 1 / n_drones) * np.pi, n_drones)[:, None]
    t = np.linspace(0, duration_sec, n_points)
    x = np.cos(0.1 * t * np.pi + phase) * radius
    y = np.sin(0.1 * t * np.pi + phase) * radius
    z = np.ones_like(x) * np.linspace(0.5, 1.5, n_points)
    t = np.repeat(t[None, :], n_drones, axis=0)
    wpt = np.stack([t, x, y, z], axis=-1)
    wpt = np.concat([wpt, np.zeros((n_drones, n_points, 6))], axis=-1)
    return {i: wpt[i] for i in range(n_drones)}


def solve_swarm(swarm, current_time, initial_states, input_drone_results, constraint_configs):
    """Solve the optimization problem for the swarm."""
    for cfg in constraint_configs:
        if hasattr(cfg, "setWaypointsConstraints"):
            cfg.setWaypointsConstraints(True, False, False)
        elif hasattr(cfg, "set_waypoints_constraints"):
            cfg.set_waypoints_constraints(True, False, False)
        else:
            raise ValueError(
                "Constraint config does not have setWaypointsConstraints or set_waypoints_constraints"
            )

    solve_success, iters, drone_results = swarm.solve(
        current_time, initial_states, input_drone_results, constraint_configs
    )
    if not solve_success:
        print("Warning: Solve failed")
    return drone_results


def simulate_amswarm(sim, waypoints, render=False):
    """Run the AMSwarm simulation and record timing for solve steps.

    Returns a dict with 'positions', 'timestamps', and timing info.
    """
    with open(Path(__file__).resolve().parents[1] / "params/model_params.yaml") as f:
        settings = yaml.safe_load(f)

    num_drones = len(waypoints)
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    # Setup simulation parameters
    mpc_freq = settings["MPCConfig"]["mpc_freq"]
    duration_sec = waypoints[0][-1, 0]
    num_steps = int(duration_sec * mpc_freq)

    # Initialize results storage
    results = {
        "positions": {i: [] for i in range(num_drones)},
        "timestamps": np.linspace(0, duration_sec, num_steps),
    }

    drone_results = [
        amswarm.DroneResult.generateInitialDroneResult(
            initial_positions[k], settings["MPCConfig"]["K"]
        )
        for k in waypoints
    ]

    # Initialize drones and swarm
    amswarm_kwargs = {
        "solverConfig": amswarm.AMSolverConfig(**settings["AMSolverConfig"]),
        "mpcConfig": amswarm.MPCConfig(**settings["MPCConfig"]),
        "weights": amswarm.MPCWeights(**settings["MPCWeights"]),
        "limits": amswarm.PhysicalLimits(**settings["PhysicalLimits"]),
        "dynamics": amswarm.SparseDynamics(**settings["Dynamics"]),
    }

    drones = [amswarm.Drone(waypoints=waypoints[key], **amswarm_kwargs) for key in waypoints]
    swarm = amswarm.Swarm(drones)

    # Set initial states
    initial_states = [np.concatenate((initial_positions[k], [0, 0, 0])) for k in waypoints]
    constraint_configs = [amswarm.ConstraintConfig() for k in waypoints]

    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    current_positions = {k: initial_positions[k] for k in waypoints}

    sim.reset()
    # Set initial position states to first waypoint for each drone
    pos = np.stack([waypoints[i][0, 1:4] for i in range(num_drones)], axis=0).reshape(1, -1, 3)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))

    for step in range(num_steps):
        current_time = step / mpc_freq

        initial_states = [np.concatenate((current_positions[k], [0, 0, 0])) for k in waypoints]
        drone_results = solve_swarm(
            swarm, current_time, initial_states, drone_results, constraint_configs
        )
        for result in drone_results:
            result.advanceForNextSolveStep()
        control = np.stack([r.position_trajectory[1] for r in drone_results], axis=0)
        control = np.concat([control, np.zeros((control.shape[0], 10))], axis=-1)
        control = control[None, ...]

        sim.state_control(control)
        sim.step(sim.freq // mpc_freq)
        if render:
            render_solutions(sim, drone_results)
            for i in range(num_drones):
                draw_points(sim, waypoints[i][:, 1:4], rgba=rgbas[i], size=0.02)
            sim.render()

        for i in range(num_drones):
            current_positions[i] = np.asarray(sim.data.states.pos[0, i])
            results["positions"][i].append(current_positions[i])

    return {
        "positions": np.array([results["positions"][i] for i in range(num_drones)]),
    }


def simulate_amswarmpy(sim, waypoints, render=False):
    """Run the AMSwarmPy simulation.

    Args:
        sim: Simulation object containing parameters
        waypoints: Dictionary of waypoints for each drone

    Returns:
        Dictionary containing trajectory positions
    """
    with open(Path(__file__).resolve().parents[1] / "params/model_params.yaml") as f:
        settings = yaml.safe_load(f)

    num_drones = sim.n_drones
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    # Setup simulation parameters
    mpc_freq = settings["MPCConfig"]["mpc_freq"]
    duration_sec = waypoints[0][-1, 0]
    num_steps = int(duration_sec * mpc_freq)

    # Initialize results storage
    results = {
        "positions": {i: [] for i in range(num_drones)},
        "timestamps": np.linspace(0, duration_sec, num_steps),
    }

    drone_results = [
        amswarmpy.DroneResult.generate_initial_drone_result(
            initial_positions[k], settings["MPCConfig"]["K"]
        )
        for k in waypoints
    ]

    # Initialize drones and swarm
    amswarm_kwargs = {
        "solver_config": amswarmpy.AMSolverConfig(**settings["AMSolverConfig"]),
        "mpc_config": amswarmpy.MPCConfig(**settings["MPCConfig"]),
        "weights": amswarmpy.MPCWeights(**settings["MPCWeights"]),
        "limits": amswarmpy.PhysicalLimits(**settings["PhysicalLimits"]),
        "dynamics": amswarmpy.SparseDynamics(**settings["Dynamics"]),
    }

    drones = [amswarmpy.Drone(waypoints=waypoints[key], **amswarm_kwargs) for key in waypoints]
    swarm = amswarmpy.Swarm(drones)

    # Set initial states
    initial_states = [np.concatenate((initial_positions[k], [0, 0, 0])) for k in waypoints]
    constraint_configs = [amswarmpy.ConstraintConfig() for k in waypoints]

    drone_results = solve_swarm(swarm, 0, initial_states, drone_results, constraint_configs)
    current_positions = {k: initial_positions[k] for k in waypoints}

    sim.reset()
    # Set initial position states to first waypoint for each drone
    pos = np.stack([waypoints[i][0, 1:4] for i in range(num_drones)], axis=0).reshape(1, -1, 3)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))

    for step in range(num_steps):
        current_time = step / mpc_freq

        initial_states = [np.concatenate((current_positions[k], [0, 0, 0])) for k in waypoints]
        drone_results = solve_swarm(
            swarm, current_time, initial_states, drone_results, constraint_configs
        )
        for result in drone_results:
            result.advance_for_next_solve_step()
        control = np.stack([r.position_trajectory[1] for r in drone_results], axis=0)
        control = np.concat([control, np.zeros((control.shape[0], 10))], axis=-1)
        control = control[None, ...]

        sim.state_control(control)
        sim.step(sim.freq // mpc_freq)
        if render:
            render_solutions(sim, drone_results)
            for i in range(num_drones):
                draw_points(sim, waypoints[i][:, 1:4], rgba=rgbas[i], size=0.02)
            sim.render()

        for i in range(num_drones):
            current_positions[i] = np.asarray(sim.data.states.pos[0, i])
            results["positions"][i].append(current_positions[i])

    return {
        "positions": np.array([results["positions"][i] for i in range(num_drones)]),
    }


def plot_trajectories(sim, waypoints, results_amswarm, results_amswarmpy):
    """Plot comparison of trajectories between AMSwarm and AMSwarmPy implementations.

    Args:
        sim: Simulation object containing number of drones
        waypoints: Dictionary of waypoints for each drone
        results_amswarm: Dictionary containing AMSwarm trajectory results
        results_amswarmpy: Dictionary containing AMSwarmPy trajectory results
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each drone's trajectory
    for i in range(sim.n_drones):
        # Plot AMSwarm trajectory
        if results_amswarm:
            ax.plot(
                results_amswarm["positions"][i][:, 0],
                results_amswarm["positions"][i][:, 1],
                results_amswarm["positions"][i][:, 2],
                label=f"AMSwarm Drone {i}",
                color=rgbas[i],
            )

        # Plot AMSwarmPy trajectory
        if results_amswarmpy:
            ax.plot(
                results_amswarmpy["positions"][i][:, 0],
                results_amswarmpy["positions"][i][:, 1],
                results_amswarmpy["positions"][i][:, 2],
                "--",
                label=f"AMSwarmPy Drone {i}",
                color=rgbas[i],
            )

        # Plot waypoints
        ax.scatter(
            waypoints[i][:, 1], waypoints[i][:, 2], waypoints[i][:, 3], marker="x", color=rgbas[i]
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Trajectories Comparison")
    ax.legend()
    plt.show()


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


def main():
    sim = Sim(n_drones=5, freq=400, state_freq=80, attitude_freq=400, control="state")
    n_points = 7
    waypoints = generate_waypoints(sim.n_drones, n_points=n_points)

    results_amswarm = simulate_amswarm(sim, waypoints, render=True)
    results_amswarmpy = simulate_amswarmpy(sim, waypoints, render=True)
    sim.close()

    plot_trajectories(sim, waypoints, results_amswarm, results_amswarmpy)


if __name__ == "__main__":
    main()
