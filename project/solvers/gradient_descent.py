"""
Gradient Descent Solver for Linear Systems
==========================================

Solves A @ x = b using steepest descent with optimal step size.

For symmetric positive definite A, the optimal step size is:
    α = (r^T r) / (r^T A r)

where r = b - A @ x is the residual.

Reference: Shewchuk, "An Introduction to the Conjugate Gradient Method 
Without the Agonizing Pain", 1994.
"""

from typing import List, Tuple
import numpy as np
from .base import IterativeSolver


class GradientDescentSolver(IterativeSolver):
    """
    Gradient Descent (Steepest Descent) solver for SPD linear systems.
    
    Uses optimal step size for quadratic minimization.
    Convergence rate depends on condition number κ(A).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Gradient Descent"
    
    def _solve_impl(self, 
                    A: np.ndarray, 
                    b: np.ndarray, 
                    x0: np.ndarray) -> Tuple[np.ndarray, List[float], int, bool]:
        """
        Solve using steepest descent with optimal step size.
        
        Algorithm:
        1. r = b - A @ x
        2. α = (r^T r) / (r^T A r)
        3. x = x + α * r
        4. Repeat until convergence
        """
        x = x0.copy()
        r = b - A @ x
        residual_history = []
        
        residual_norm = np.linalg.norm(r)
        residual_history.append(residual_norm)
        
        converged = self._check_convergence(residual_norm)
        iteration = 0
        
        while not converged and iteration < self.max_iterations:
            # Compute A @ r
            Ar = A @ r
            
            # Optimal step size: α = (r^T r) / (r^T A r)
            rTr = np.dot(r, r)
            rTAr = np.dot(r, Ar)
            
            # Guard against division by zero
            if abs(rTAr) < 1e-15:
                break
                
            alpha = rTr / rTAr
            
            # Update solution
            x = x + alpha * r
            
            # Update residual: r_new = r - α * A @ r
            r = r - alpha * Ar
            
            # Track convergence
            residual_norm = np.linalg.norm(r)
            residual_history.append(residual_norm)
            
            iteration += 1
            converged = self._check_convergence(residual_norm)
            
            self._log(iteration, residual_norm)
        
        return x, residual_history, iteration, converged


