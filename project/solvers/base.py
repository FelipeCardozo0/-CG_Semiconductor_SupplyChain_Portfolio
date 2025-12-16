"""
Base class for iterative solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import time


@dataclass
class SolverResult:
    """Container for solver results and diagnostics."""
    solution: np.ndarray
    converged: bool
    iterations: int
    final_residual_norm: float
    residual_history: List[float]
    elapsed_time: float
    solver_name: str
    
    # Additional diagnostics
    initial_residual_norm: float = 0.0
    relative_residual: float = 0.0
    
    def __post_init__(self):
        if self.initial_residual_norm > 0:
            self.relative_residual = self.final_residual_norm / self.initial_residual_norm


class IterativeSolver(ABC):
    """
    Abstract base class for iterative linear system solvers.
    
    Solves: A @ x = b
    
    All solvers share identical interface for fair comparison.
    """
    
    def __init__(self, 
                 tolerance: float = 1e-8,
                 max_iterations: int = 1000,
                 verbose: bool = False):
        """
        Initialize solver with convergence parameters.
        
        Parameters
        ----------
        tolerance : float
            Stopping criterion for residual norm ||b - Ax||
        max_iterations : int
            Maximum number of iterations before termination
        verbose : bool
            Print iteration progress
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.name = "IterativeSolver"
    
    @abstractmethod
    def _solve_impl(self, 
                    A: np.ndarray, 
                    b: np.ndarray, 
                    x0: np.ndarray) -> Tuple[np.ndarray, List[float], int, bool]:
        """
        Internal solve implementation.
        
        Parameters
        ----------
        A : np.ndarray
            System matrix (n x n), must be symmetric positive definite
        b : np.ndarray
            Right-hand side vector (n,)
        x0 : np.ndarray
            Initial guess (n,)
            
        Returns
        -------
        tuple
            (solution, residual_history, iterations, converged)
        """
        pass
    
    def solve(self, 
              A: np.ndarray, 
              b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve the linear system A @ x = b.
        
        Parameters
        ----------
        A : np.ndarray
            System matrix (n x n), symmetric positive definite
        b : np.ndarray
            Right-hand side vector (n,)
        x0 : np.ndarray, optional
            Initial guess. If None, uses zero vector.
            
        Returns
        -------
        SolverResult
            Solution and convergence diagnostics
        """
        n = len(b)
        
        # Ensure proper shapes
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).flatten()
        
        if x0 is None:
            x0 = np.zeros(n, dtype=np.float64)
        else:
            x0 = np.asarray(x0, dtype=np.float64).flatten()
        
        # Compute initial residual
        r0 = b - A @ x0
        initial_residual_norm = np.linalg.norm(r0)
        
        # Time the solve
        start_time = time.perf_counter()
        x, residual_history, iterations, converged = self._solve_impl(A, b, x0)
        elapsed_time = time.perf_counter() - start_time
        
        # Final residual
        final_residual = b - A @ x
        final_residual_norm = np.linalg.norm(final_residual)
        
        return SolverResult(
            solution=x,
            converged=converged,
            iterations=iterations,
            final_residual_norm=final_residual_norm,
            residual_history=residual_history,
            elapsed_time=elapsed_time,
            solver_name=self.name,
            initial_residual_norm=initial_residual_norm,
        )
    
    def _check_convergence(self, residual_norm: float) -> bool:
        """Check if solver has converged."""
        return residual_norm <= self.tolerance
    
    def _log(self, iteration: int, residual_norm: float):
        """Log iteration progress."""
        if self.verbose:
            print(f"  {self.name} iter {iteration:4d}: ||r|| = {residual_norm:.6e}")


def verify_spd(A: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Verify that matrix A is symmetric positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix to verify
    tol : float
        Tolerance for symmetry check
        
    Returns
    -------
    tuple
        (is_spd, min_eigenvalue)
    """
    # Check symmetry
    if not np.allclose(A, A.T, atol=tol):
        return False, 0.0
    
    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(A)
    min_eig = eigenvalues.min()
    
    return min_eig > 0, min_eig


