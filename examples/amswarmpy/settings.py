from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SolverSettings:
    rho_init: float  # Initial value of rho
    rho_max: float  # Maximum allowable value of rho
    max_iters: int  # Maximum number of iterations

    constraints: ConstraintSettings
    weights: Weights
    limits: Limits
    mpc: MPCSettings


@dataclass
class ConstraintSettings:
    """Configuration for optimization constraints"""

    pos: bool = True
    vel: bool = True
    acc: bool = True
    input_continuity: bool = True


@dataclass
class Weights:
    """Weights for MPC cost function terms"""

    pos: float = 7000.0  # Waypoint position tracking
    vel: float = 1000.0  # Waypoint velocity tracking
    acc: float = 100.0  # Waypoint acceleration tracking
    smoothness: float = 100.0  # Trajectory smoothness
    input_smoothness: float = 1000.0  # Input smoothness
    input_continuity: float = 100.0  # Input continuity


@dataclass
class MPCSettings:
    """Configuration parameters for MPC problem"""

    K: int = 25  # Number of timesteps in horizon
    N: int = 10  # Number of spline coefficients
    freq: float = 8.0  # MPC solver frequency (Hz)
    bf_gamma: float = 1.0  # Barrier function gamma parameter
    waypoints_pos_tol: float = 1e-2  # Waypoint position constraint tolerance
    waypoints_vel_tol: float = 1e-2  # Waypoint velocity constraint tolerance
    waypoints_acc_tol: float = 1e-2  # Waypoint acceleration constraint tolerance
    input_continuity_tol: float = 1e-2  # Input continuity constraint tolerance
    pos_tol: float = 1e-2  # Position constraint tolerance
    vel_tol: float = 1e-2  # Velocity constraint tolerance
    acc_tol: float = 1e-2  # Acceleration constraint tolerance
    collision_tol: float = 1e-2  # Collision constraint tolerance


@dataclass
class Limits:
    """Physical limits and constraints for the drone"""

    pos_min: np.ndarray = field(default_factory=lambda: np.full(3, -10))  # Minimum position bounds
    pos_max: np.ndarray = field(default_factory=lambda: np.full(3, 10))  # Maximum position bounds
    vel_max: float = 1.73  # Maximum velocity
    acc_max: float = 0.75 * 9.81  # Maximum acceleration (0.75g)
    collision: NDArray | None = None
