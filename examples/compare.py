from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import amswarm
import amswarmpy
import fire
import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml
from crazyflow import Sim
from utils import draw_line, draw_points

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


np.random.seed(0)
rgbas = np.random.rand(5, 4)
rgbas[..., 3] = 1.0


def render_solutions(sim, trajectories: list[np.ndarray]):
    for i, trajectory in enumerate(trajectories):
        draw_points(sim, trajectory, rgba=rgbas[i], size=0.01)
        draw_line(sim, trajectory, rgba=rgbas[i])


def generate_waypoints(n_drones: int, n_points: int = 4, duration_sec: float = 10.0):
    """Waypoints have the following shape: [T, n_drones, 3]."""
    radius = 0.75
    phase = np.linspace(0, 2 * (1 - 1 / n_drones) * np.pi, n_drones)
    t = np.tile(np.linspace(0, duration_sec, n_points), (n_drones, 1)).T
    x = np.cos(0.1 * t * np.pi + phase) * radius
    y = np.sin(0.1 * t * np.pi + phase) * radius
    z = np.ones_like(x) * np.linspace(0.5, 1.5, n_points)[:, None]
    pos = np.stack([x, y, z], axis=-1)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    assert pos.shape == (n_points, n_drones, 3), f"Shape {pos.shape} != ({n_points}, {n_drones}, 3)"
    return {"time": t, "pos": pos, "vel": vel, "acc": acc}


def solve_swarm(swarm, current_time, initial_states, input_drone_results, constraint_configs):
    """Solve the optimization problem for the swarm."""
    for cfg in constraint_configs:
        if isinstance(cfg, amswarm.ConstraintConfig):
            cfg.setWaypointsConstraints(True, False, False)
        else:
            raise ValueError(
                "Constraint config does not have setWaypointsConstraints or set_waypoints_constraints"
            )
    success, iters, drone_results = swarm.solve(
        current_time, initial_states, input_drone_results, constraint_configs
    )
    if not all(success):
        print("Warning: Solve failed")
    return drone_results


def legacy_waypoints(waypoints):
    """Convert waypoints to the legacy format."""
    res = {}
    n_drones = waypoints["pos"].shape[1]
    t = waypoints["time"]
    for i in range(n_drones):
        res[i] = np.concat(
            (t[:, i, None], waypoints["pos"][:, i], waypoints["vel"][:, i], waypoints["acc"][:, i]),
            axis=-1,
        )
    return res


def simulate_amswarm(sim, waypoints, render=False) -> NDArray:
    """Run the AMSwarm simulation and record timing for solve steps.

    Returns the simulated positions for each time step and drone.
    """
    with open(Path(__file__).resolve().parents[1] / "params/model_params.yaml") as f:
        settings = yaml.safe_load(f)

    waypoints = legacy_waypoints(waypoints)

    num_drones = len(waypoints)
    initial_positions = {k: waypoints[k][0, 1:4] for k in waypoints}

    # Setup simulation parameters
    mpc_freq = settings["MPCConfig"]["mpc_freq"]
    duration_sec = waypoints[0][-1, 0]
    n_steps = int(duration_sec * mpc_freq)

    # Initialize results storage
    simulated_pos = np.zeros((n_steps, num_drones, 3))

    drone_results = [
        amswarm.DroneResult.generateInitialDroneResult(
            initial_positions[k], settings["MPCConfig"]["K"]
        )
        for k in waypoints
    ]

    # Initialize drones and swarm

    drones = [
        amswarm.Drone(
            waypoints=waypoints[key],
            solverConfig=amswarm.AMSolverConfig(**settings["AMSolverConfig"]),
            mpcConfig=amswarm.MPCConfig(**settings["MPCConfig"]),
            weights=amswarm.MPCWeights(**settings["MPCWeights"]),
            limits=amswarm.PhysicalLimits(**settings["PhysicalLimits"]),
            dynamics=amswarm.SparseDynamics(**settings["Dynamics"]),
        )
        for key in waypoints
    ]
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

    for step in range(n_steps):
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
            render_solutions(sim, [r.position_trajectory for r in drone_results])
            for i in range(num_drones):
                draw_points(sim, waypoints[i][:, 1:4], rgba=rgbas[i], size=0.02)
            sim.render()

        current_positions = np.asarray(sim.data.states.pos[0])
        simulated_pos[step] = current_positions

    return simulated_pos


def simulate_amswarmpy(sim, waypoints, render=False) -> NDArray:
    """Run the AMSwarmPy simulation.

    Args:
        sim: Simulation object containing parameters
        waypoints: Dictionary of waypoints for each drone

    Returns:
        Dictionary containing trajectory positions
    """
    with open(Path(__file__).resolve().parents[1] / "params/settings.yaml") as f:
        settings = yaml.safe_load(f)

    # Setup simulation parameters
    n_drones = sim.n_drones
    mpc_freq = settings["MPCSettings"]["freq"]
    n_steps = int(waypoints["time"][-1, 0] * mpc_freq)

    trajectories = np.zeros((n_steps, n_drones, 3))  # Initialize trajectories storage

    solver_data = amswarmpy.SolverData.init(waypoints=waypoints, K=settings["MPCSettings"]["K"])

    # Initialize data and settings
    weights = amswarmpy.Weights(**settings["Weights"])
    limits = amswarmpy.Limits(**settings["Limits"])
    mpc_settings = amswarmpy.MPCSettings(**settings["MPCSettings"])
    solver_settings = amswarmpy.SolverSettings(
        **settings["SolverSettings"],
        constraints=amswarmpy.ConstraintSettings(),
        weights=weights,
        limits=limits,
        mpc=mpc_settings,
    )
    solver_settings.constraints.pos = True
    solver_settings.constraints.vel = False
    solver_settings.constraints.acc = False
    solver_settings.constraints.input_continuity = True
    dynamics = settings["Dynamics"]
    A, B = np.asarray(dynamics["A"]), np.asarray(dynamics["B"])
    A_prime, B_prime = np.asarray(dynamics["A_prime"]), np.asarray(dynamics["B_prime"])
    solver_data.init_matrices(
        A, B, A_prime, B_prime, mpc_settings.K, mpc_settings.N, mpc_settings.freq
    )
    solver_data.init_cost(weights)

    states = np.concat((waypoints["pos"][0], np.zeros((n_drones, 3))), axis=-1)
    success, _, solver_data = amswarmpy.solve(0, states, waypoints, solver_data, solver_settings)
    if not all(success):
        print("Warning: Solve failed")

    pos, vel = waypoints["pos"][0], np.zeros((n_drones, 3))

    sim.reset()
    # Set initial position states to first waypoint for each drone
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos[None, ...]))

    for step in range(n_steps):
        t = step / mpc_freq

        states = np.concat((pos, vel), axis=-1)
        success, _, solver_data = amswarmpy.solve(
            t, states, waypoints, solver_data, solver_settings
        )
        if not all(success):
            print("Warning: Solve failed")

        for i in range(n_drones):
            solver_data.results[i].advance_for_next_solve_step()
            solver_data.previous_results[i] = deepcopy(solver_data.results[i])
        control = np.stack([r.pos[1] for r in solver_data.results], axis=0)
        control = np.concat([control, np.zeros((control.shape[0], 10))], axis=-1)
        control = control[None, ...]

        sim.state_control(control)
        sim.step(sim.freq // mpc_freq)
        if render:
            render_solutions(sim, [r.pos for r in solver_data.results])
            for i in range(n_drones):
                draw_points(sim, waypoints["pos"][:, i, :], rgba=rgbas[i], size=0.02)
            sim.render()

        pos, vel = np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0])
        trajectories[step] = pos

    return trajectories


def plot_trajectories(sim, waypoints, pos_amswarm, pos_amswarmpy):
    """Plot comparison of trajectories between AMSwarm and AMSwarmPy implementations.

    Args:
        sim: Simulation object containing number of drones
        waypoints: Dictionary of waypoints for each drone
        pos_amswarm: Dictionary containing AMSwarm trajectory results
        pos_amswarmpy: Dictionary containing AMSwarmPy trajectory results
    """

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each drone's trajectory
    for i in range(sim.n_drones):
        # Plot AMSwarm trajectory
        if pos_amswarm is not None:
            pos = pos_amswarm[:, i, :]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"AMSwarm Drone {i}", color=rgbas[i])

        # Plot AMSwarmPy trajectory
        if pos_amswarmpy is not None:
            pos = pos_amswarmpy[:, i, :]
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2], "--", label=f"AMSwarmPy Drone {i}", color=rgbas[i]
            )

        # Plot waypoints
        pos = waypoints["pos"][:, i, :]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], marker="x", color=rgbas[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Trajectories Comparison")
    ax.legend()
    plt.show()


def main(render: bool = False):
    sim = Sim(n_drones=5, freq=400, state_freq=80, attitude_freq=400, control="state")
    n_points = 7
    waypoints = generate_waypoints(sim.n_drones, n_points=n_points)
    # results_amswarm = simulate_amswarm(sim, waypoints, render=render)

    results_amswarm = None
    t1 = time.perf_counter()
    # results_amswarm = simulate_amswarm(sim, waypoints, render=render)
    t2 = time.perf_counter()
    print(f"AMSwarm (cpp) time: {t2 - t1:.2f} seconds")
    tstart = time.perf_counter()
    results_amswarmpy = None
    results_amswarmpy = simulate_amswarmpy(sim, waypoints, render=render)
    tstop = time.perf_counter()
    print(f"AMSwarmPy time: {tstop - tstart:.2f} seconds")
    # Solve time v00: 20s
    sim.close()

    plot_trajectories(sim, waypoints, results_amswarm, results_amswarmpy)


if __name__ == "__main__":
    fire.Fire(main)
