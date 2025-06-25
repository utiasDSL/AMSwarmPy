from typing import Protocol

import numpy as np


class Constraint(Protocol):
    """Protocol defining the interface for constraints.

    Defines the interface for constraints and provides common functionality
    for derived constraint classes.
    """

    def get_quadratic_term(self) -> np.ndarray:
        """Retrieves the quadratic term of the constraint-as-penalty.

        All constraints can be reformulated as quadratic penalties ||Ax - b||^2
        in the cost function instead of hard constraints. This penalty can be
        expanded into a quadratic term of the form x^T*Q*x, a linear term of the
        form c^T * x, and a constant term. This function returns the sparse
        matrix Q representing the quadratic term.
        """
        ...

    def get_linear_term(self) -> np.ndarray:
        """Retrieves the linear term of the constraint-as-penalty.

        All constraints can be formulated as quadratic penalties ||Gx - h||^2
        in the cost function instead of hard constraints. This penalty can be
        expanded into a quadratic term of the form x^T*Q*x, a linear term of the
        form c^T * x, and a constant term. This function returns the vector c
        representing the linear term.
        """
        ...

    def get_bregman_update(self, x: np.ndarray) -> np.ndarray:
        """Calculates and returns the Bregman "multiplier" update based on the current point.

        See thesis document for more information on Bregman iteration.
        """
        ...

    def update(self, x: np.ndarray) -> None:
        """Updates the internal state of the constraint if necessary (e.g. slack variables)."""
        ...

    def is_satisfied(self, x: np.ndarray) -> bool:
        """Checks if the constraint is satisfied at the given point."""
        ...

    def reset(self) -> None:
        """Resets the internal state of the constraint to its initial state."""
        ...


class EqualityConstraint:
    """Handles equality constraints of the form Gx = h."""

    def __init__(self, G: np.ndarray, h: np.ndarray, tolerance: float = 1e-2):
        self.G = G
        self.h = h
        self.G_T = G.T
        self.G_T_G = G.T @ G
        self.G_T_h = G.T @ h
        self.tolerance = tolerance

    def get_quadratic_term(self) -> np.ndarray:
        return self.G_T_G

    def get_linear_term(self) -> np.ndarray:
        return -self.G_T_h

    def update(self, x: np.ndarray) -> None: ...

    def is_satisfied(self, x: np.ndarray) -> bool:
        return np.max(np.abs(self.G @ x - self.h)) <= self.tolerance

    def reset(self): ...


class InequalityConstraint:
    """Handles inequality constraints of the form Gx <= h."""

    def __init__(self, G: np.ndarray, h: np.ndarray, tolerance: float = 1e-2):
        self.G = G
        self.h = h
        self.G_T = G.T
        self.G_T_G = G.T @ G
        self.G_T_h = G.T @ h
        self.slack = np.zeros_like(h)
        self.tolerance = tolerance

    def get_quadratic_term(self) -> np.ndarray:
        return self.G_T_G

    def get_linear_term(self) -> np.ndarray:
        return -self.G_T_h + self.G_T @ self.slack

    def update(self, x: np.ndarray) -> None:
        self.slack = np.maximum(0, -self.G @ x + self.h)

    def is_satisfied(self, x: np.ndarray) -> bool:
        return np.max(self.G @ x - self.h) < self.tolerance

    def reset(self):
        self.slack = np.zeros_like(self.h)


class PolarInequalityConstraint:
    """Manages polar inequality constraints of a specialized form.

    This class is designed to handle constraints defined by polar coordinates that conform to the formula:
    Gx + c = h(alpha, beta, d), with the boundary condition lwr_bound <= d <= upr_bound.

    Here, 'alpha', 'beta', and 'd' are vectors with a length of K+1, where 'd' represents the distance from the origin,
    'alpha' the azimuthal angle, and 'beta' the polar angle. The vector 'h' has a length of 3(K+1), where each set of three elements
    in 'h()' represents a point in 3D space expressed as:

    d[k] * [cos(alpha[k]) * sin(beta[k]), sin(alpha[k]) * sin(beta[k]), cos(beta[k])]^T

    This represents a unit vector defined by angles 'alpha[k]' and 'beta[k]', scaled by 'd[k]', where 'k' is an index running from 0 to K.
    The index range from 0 to K can be interpreted as discrete time steps, allowing this constraint to serve as a Barrier Function (BF)
    constraint to manage the rate at which a constraint boundary is approached over successive time steps.
    """

    def __init__(
        self,
        G: np.ndarray,
        c: np.ndarray,
        lwr_bound: float,
        upr_bound: float,
        bf_gamma: float = 1.0,
        tolerance: float = 1e-2,
    ):
        self.G = G
        self.G_T = G.T
        self.G_T_G = G.T @ G
        self.c = c
        self.h = -c
        self.lwr_bound = lwr_bound
        self.upr_bound = upr_bound
        self.apply_upr_bound = not np.isinf(upr_bound)
        self.apply_lwr_bound = not np.isinf(lwr_bound)
        self.bf_gamma = bf_gamma
        self.tolerance = tolerance

    def get_quadratic_term(self) -> np.ndarray:
        return self.G_T_G

    def get_linear_term(self) -> np.ndarray:
        return -self.G_T @ self.h

    def update(self, x: np.ndarray) -> None:
        if self.G.shape[1] != x.shape[0]:
            raise ValueError("G and x are not compatible sizes")

        h_tmp = self.G @ x + self.c
        prev_norm = 0

        for i in range(0, h_tmp.shape[0], 3):
            # Calculate norm of current time segment
            segment = h_tmp[i : i + 3]
            segment_norm = np.linalg.norm(segment)

            # Apply upper bound if not infinite
            if self.apply_upr_bound:
                if i > 0:
                    bound = self.upr_bound - (1.0 - self.bf_gamma) * (self.upr_bound - prev_norm)
                else:
                    bound = self.upr_bound

                if segment_norm > bound:
                    h_tmp[i : i + 3] *= bound / segment_norm
                    segment_norm = bound

            # Apply lower bound if not infinite
            if self.apply_lwr_bound:
                if i > 0:
                    bound = self.lwr_bound + (1.0 - self.bf_gamma) * (prev_norm - self.lwr_bound)
                else:
                    bound = self.lwr_bound

                if segment_norm < bound:
                    h_tmp[i : i + 3] *= bound / segment_norm
                    segment_norm = bound

            # Track norm for next iteration
            prev_norm = segment_norm

        self.h = h_tmp - self.c

    def is_satisfied(self, x: np.ndarray) -> bool:
        return np.max(np.abs(self.G @ x - self.h)) < self.tolerance

    def reset(self):
        self.h = -self.c
