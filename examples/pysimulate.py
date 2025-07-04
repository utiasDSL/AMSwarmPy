"""
AMSwarmPy Simulation with Line Formation and Benchmarking

This module simulates drone swarm behavior using AMSwarmPy for trajectory planning,
with drones arranged in a line on the x-axis flying from y=-1 to y=1.
It also includes benchmarking functionality to measure AMSwarmPy's performance.
"""

import time
from pathlib import Path

import amswarmpy
import matplotlib.pyplot as plt
import numpy as np
import yaml
from utils import generate_random_waypoints


def solve_swarm(swarm, current_time, initial_states, input_drone_results, constraint_configs):
    """Solve the optimization problem for the swarm."""
    for cfg in constraint_configs:
        cfg.set_waypoints_constraints(True, False, False)
    solve_success, iters, drone_results = swarm.solve(
        current_time, initial_states, input_drone_results, constraint_configs
    )
    if not solve_success:
        raise RuntimeError(f"Solver failed to find a valid solution at time {current_time}")
    return drone_results


def simulate(waypoints):
    """Run the AMSwarmPy simulation and record timing for solve steps.

    Returns a dict with 'positions', 'timestamps', and timing info.
    """
    with open(Path(__file__).resolve().parents[1] / "params/settings.yaml") as f:
        settings = yaml.safe_load(f)

    num_drones = waypoints["pos"].shape[1]
    initial_positions = {i: waypoints["pos"][0, i] for i in range(num_drones)}

    # Setup simulation parameters
    mpc_freq = settings["MPCSettings"]["freq"]
    duration_sec = (
        waypoints["time"][-1, 0] if waypoints["time"].ndim == 2 else waypoints["time"][-1]
    )
    num_steps = int(duration_sec * mpc_freq)

    # Initialize results storage
    results = {
        "positions": {i: [] for i in range(num_drones)},
        "timestamps": np.linspace(0, duration_sec, num_steps),
    }
    timings = {"solve_times": [], "advance_times": [], "solve_steps": []}

    t0 = time.perf_counter()
    drone_results = [
        amswarmpy.Trajectory.init(initial_positions[i], settings["MPCSettings"]["K"])
        for i in range(num_drones)
    ]
    t1 = time.perf_counter()
    timings["init_time"] = t1 - t0

    constraint_settings = amswarmpy.ConstraintSettings(
        pos=True, vel=False, acc=False, input_continuity=True
    )
    weights = amswarmpy.Weights(**settings["Weights"])
    limits = amswarmpy.Limits(**settings["Limits"])
    mpc_settings = amswarmpy.MPCSettings(**settings["MPCSettings"])
    solver_settings = amswarmpy.SolverSettings(
        **settings["SolverSettings"],
        constraints=constraint_settings,
        weights=weights,
        limits=limits,
        mpc=mpc_settings,
    )
    dynamics = amswarmpy.Dynamics(**settings["Dynamics"])
    drones = [
        amswarmpy.Drone(settings=solver_settings, dynamics=dynamics) for _ in range(num_drones)
    ]

    # Set initial states
    initial_states = np.concatenate((waypoints["pos"][0], np.zeros((num_drones, 3))), axis=-1)
    solve_success, iters, drone_results = amswarmpy.solve(
        drones, 0, initial_states, waypoints, drone_results, solver_settings
    )
    if not all(solve_success):
        print("Warning: Solver failed to find a valid solution")
    current_positions = waypoints["pos"][0]

    # Main simulation loop
    for step in range(num_steps):
        current_time = step / mpc_freq

        if step % int(mpc_freq) == 0:
            initial_states = np.concatenate((current_positions, np.zeros((num_drones, 3))), axis=-1)
            t0 = time.perf_counter()
            solve_success, iters, drone_results = amswarmpy.solve(
                drones, current_time, initial_states, waypoints, drone_results, solver_settings
            )
            t1 = time.perf_counter()
            timings["solve_times"].append(t1 - t0)
            t2 = time.perf_counter()
            for result in drone_results:
                result.advance_for_next_solve_step()
            t3 = time.perf_counter()
            timings["advance_times"].append(t3 - t2)

        for i, result in enumerate(drone_results):
            planned_pos = result.pos[step % int(mpc_freq), :]
            current_positions[i] = planned_pos
            results["positions"][i].append(planned_pos)

    return {
        "positions": results["positions"],
        "timestamps": results["timestamps"],
        "solve_times": timings["solve_times"],
        "advance_times": timings["advance_times"],
        "init_time": timings["init_time"],
    }


def plot_trajectories(results, waypoints, filename="trajectories.png"):
    """Plot the simulated trajectories and waypoints and save to file."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for drone_id in results["positions"]:
        pos = np.array(results["positions"][drone_id])
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"Drone {drone_id} Trajectory")
        wpts = waypoints["pos"][:, drone_id, :]
        ax.scatter(wpts[:, 0], wpts[:, 1], wpts[:, 2], marker="*", s=100)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()  # Remove legend to avoid overcrowding
    plt.title("Drone Swarm Trajectories")
    plt.savefig(filename)
    plt.show()
    plt.close(fig)


def check_swarm_success(results, waypoints, tol=0.1):
    """
    Check if at least 50% of drones reach their final waypoint within tol (meters).
    Returns True if successful, False otherwise.
    """
    num_drones = waypoints["pos"].shape[1]
    num_success = 0
    for i in range(num_drones):
        final_pos = np.array(results["positions"][i][-1])
        goal = waypoints["pos"][-1, i, :]
        if np.linalg.norm(final_pos - goal) <= tol:
            num_success += 1
    return num_success >= num_drones / 2


def benchmark(swarm_sizes: list[int]) -> dict[str, list[float]]:
    # Initialize timing results
    times = {"solve_time": [], "swarm_size": swarm_sizes, "init_time": [], "advance_time": []}

    for num_drones in swarm_sizes:
        print(f"\nBenchmarking swarm size: {num_drones}")

        waypoints = generate_random_waypoints(num_drones, num_waypoints=4, min_distance=0.5)

        results = simulate(waypoints)
        times["solve_time"].append(np.mean(results["solve_times"]))
        times["advance_time"].append(np.mean(results["advance_times"]))
        times["init_time"].append(results["init_time"])
        print(f"Average solve time over simulation: {results['solve_times'][-1]:.3f}s")

    return times


def plot_benchmark_results(times: dict[str, list[float]], path: Path):
    """Plot the benchmark results and save to file."""
    plt.figure(figsize=(12, 6))

    plt.plot(times["swarm_size"], times["init_time"], "o-", label="Initialization")
    plt.plot(times["swarm_size"], times["advance_time"], "s-", label="Advance")
    plt.plot(times["swarm_size"], times["solve_time"], "^-", label="Solve")

    plt.xlabel("Swarm Size")
    plt.ylabel("Time (seconds)")
    plt.title("AMSwarmPy Performance Benchmark")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Use log scale for better visualization
    plt.savefig(path)
    plt.close()


def main():
    np.random.seed(42)
    np.seterr(invalid="raise")
    np.set_printoptions(precision=5, suppress=True)

    # Run benchmark
    swarm_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    times = benchmark(swarm_sizes)
    plot_benchmark_results(times, path=Path(__file__).parent / "benchmark_py.png")


if __name__ == "__main__":
    main()
