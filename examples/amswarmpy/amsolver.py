from dataclasses import dataclass

import numpy as np

from .constraint import Constraint


@dataclass
class AMSolverConfig:
    """Configuration settings for the AMSolver"""

    rho_init: float  # Initial value of rho
    max_rho: float  # Maximum allowable value of rho
    max_iters: int  # Maximum number of iterations


class AMSolver:
    """Abstract base class for Alternating Minimization (AM) Solver.

    The AMSolver is designed to solve optimization problems by alternating between solving for the optimization variables and updating
    dual variables, using a specified set of constraints. The solver is customizable, allowing for various types
    of constraints and optimization problems to be solved.

    Args:
        config: Configuration settings for the solver
    """

    def __init__(self, config: AMSolverConfig):
        # Constant constraints that do not change between solves
        self.const_constraints: list[Constraint] = []
        self.non_const_constraints: list[Constraint] = []
        self.solver_config = config

        # Cost of the form 0.5 * x^T * quad_cost * x + x^T * linear_cost
        self.initial_quad_cost: np.ndarray = None
        self.initial_linear_cost: np.ndarray = None
        self.quad_cost: np.ndarray = None  # Current quadratic cost matrix (modified during solving)
        self.linear_cost: np.ndarray = None  # Current linear cost vector (modified during solving)

    def pre_solve(self, args) -> None:
        """Called before solve process begins to setup optimization problem.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def post_solve(self, x: np.ndarray, args) -> any:
        """Called after solve process to process results.

        Must be implemented by child classes.
        """
        raise NotImplementedError

    def actual_solve(self, args) -> tuple[bool, int, np.ndarray]:
        """Conducts actual solving process implementing optimization algorithm.

        Not meant to be overridden by child classes.
        """
        # Initialize solver components
        iters = 0
        rho = self.solver_config.rho_init

        # Initialize optimization variables and matrices
        Q = np.zeros_like(self.quad_cost)  # Combined quadratic terms
        q = np.zeros(self.quad_cost.shape[0])  # Combined linear terms
        x = np.zeros(self.quad_cost.shape[0])  # Optimization variable
        bregman_mult = np.zeros(self.quad_cost.shape[0])  # Bregman multiplier

        # Aggregate quadratic and linear terms from all constraints
        quad_constraint_terms = np.zeros_like(self.quad_cost)
        linear_constraint_terms = np.zeros(self.linear_cost.shape[0])

        for constraint in self.const_constraints:
            quad_constraint_terms += constraint.get_quadratic_term()
            linear_constraint_terms += constraint.get_linear_term()
        for constraint in self.non_const_constraints:
            quad_constraint_terms += constraint.get_quadratic_term()
            linear_constraint_terms += constraint.get_linear_term()

        # Plot heatmaps of Q matrices side by side

        # Iteratively solve until solution found or max iterations reached
        while iters < self.solver_config.max_iters:
            Q = self.quad_cost + rho * quad_constraint_terms

            # Construct linear cost matrices
            linear_constraint_terms -= bregman_mult
            q = self.linear_cost + rho * linear_constraint_terms

            # Solve the QP
            x = np.linalg.solve(Q, -q)

            # Update constraints
            self.update_constraints(x)

            # Check if all constraints are satisfied
            all_constraints_satisfied = all(
                constraint.is_satisfied(x) for constraint in self.const_constraints
            ) and all(constraint.is_satisfied(x) for constraint in self.non_const_constraints)

            if all_constraints_satisfied:
                return True, iters, x  # Exit loop, indicate success

            # Recalculate linear term for Bregman multiplier
            linear_constraint_terms[...] = 0
            for constraint in self.const_constraints:
                linear_constraint_terms += constraint.get_linear_term()
            for constraint in self.non_const_constraints:
                linear_constraint_terms += constraint.get_linear_term()

            # Calculate Bregman multiplier
            bregman_update = 0.5 * (quad_constraint_terms @ x + linear_constraint_terms)
            bregman_mult -= bregman_update

            # Gradually increase penalty parameter
            rho *= self.solver_config.rho_init
            rho = min(rho, self.solver_config.max_rho)
            iters += 1

        return False, iters, x  # Indicate failure but still return vector

    def reset_cost_matrices(self) -> None:
        """Resets cost matrices to initial values"""
        self.quad_cost = self.initial_quad_cost
        self.linear_cost = self.initial_linear_cost

    def add_constraint(self, constraint: Constraint, is_constant: bool) -> None:
        """Adds a constraint to the solver"""
        if is_constant:
            self.const_constraints.append(constraint)
        else:
            self.non_const_constraints.append(constraint)

    def update_constraints(self, x: np.ndarray) -> None:
        """Updates all non-constant constraints based on current optimization variables"""
        for constraint in self.const_constraints:
            constraint.update(x)
        for constraint in self.non_const_constraints:
            constraint.update(x)

    def reset_constraints(self) -> None:
        """Resets constraints to initial state"""
        for constraint in self.const_constraints:
            constraint.reset()
        for constraint in self.non_const_constraints:
            constraint.reset()

    def solve(self, args) -> tuple[bool, int, any]:
        """Main solve function to be called by user.

        Contains main solving workflow (pre_solve, actual_solve, post_solve).
        Not meant to be overridden.
        """
        # Reset cost and clear carryover constraints from previous solves
        self.reset_cost_matrices()
        self.non_const_constraints.clear()

        # Build new constraints and add to cost matrices
        self.pre_solve(args)

        # Ensure no carryover updates from previous solve
        self.reset_constraints()

        # Execute solve process to get raw solution vector
        success, iters, result = self.actual_solve(args)

        # Post-process solution according to derived class implementation
        return success, iters, self.post_solve(result, args)
