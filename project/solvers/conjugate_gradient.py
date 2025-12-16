"""
Conjugate Gradient Solver for Linear Systems
============================================

Solves A @ x = b using the Conjugate Gradient method.

For symmetric positive definite A, CG converges in at most n iterations
(in exact arithmetic) and typically much faster for well-conditioned systems.

The method generates A-conjugate search directions that span the Krylov subspace:
    K_k(A, r_0) = span{r_0, A r_0, A^2 r_0, ..., A^{k-1} r_0}

Reference: Shewchuk, "An Introduction to the Conjugate Gradient Method 
Without the Agonizing Pain", 1994.
"""

from typing import List, Tuple
import numpy as np
from .base import IterativeSolver


class ConjugateGradientSolver(IterativeSolver):
    """
    Conjugate Gradient solver for SPD linear systems.
    
    Key properties:
    - Generates A-conjugate search directions
    - Optimal in Krylov subspace at each iteration
    - Convergence in at most n iterations (exact arithmetic)
    - Convergence rate: O(sqrt(κ(A))) vs O(κ(A)) for gradient descent
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Conjugate Gradient"
    
    def _solve_impl(self, 
                    A: np.ndarray, 
                    b: np.ndarray, 
                    x0: np.ndarray) -> Tuple[np.ndarray, List[float], int, bool]:
        """
        Solve using Conjugate Gradient method.
        
        Algorithm:
        1. r = b - A @ x
        2. d = r (initial search direction)
        3. For each iteration:
           a. α = (r^T r) / (d^T A d)
           b. x = x + α * d
           c. r_new = r - α * A @ d
           d. β = (r_new^T r_new) / (r^T r)
           e. d = r_new + β * d
        """
        x = x0.copy()
        r = b - A @ x
        d = r.copy()  # Search direction
        
        residual_history = []
        
        # Initial residual quantities
        rTr = np.dot(r, r)
        residual_norm = np.sqrt(rTr)
        residual_history.append(residual_norm)
        
        converged = self._check_convergence(residual_norm)
        iteration = 0
        
        while not converged and iteration < self.max_iterations:
            # Compute A @ d
            Ad = A @ d
            
            # Step size: α = (r^T r) / (d^T A d)
            dTAd = np.dot(d, Ad)
            
            # Guard against division by zero
            if abs(dTAd) < 1e-15:
                break
            
            alpha = rTr / dTAd
            
            # Update solution
            x = x + alpha * d
            
            # Update residual
            r = r - alpha * Ad
            
            # Compute new residual norm squared
            rTr_new = np.dot(r, r)
            residual_norm = np.sqrt(rTr_new)
            residual_history.append(residual_norm)
            
            iteration += 1
            converged = self._check_convergence(residual_norm)
            
            self._log(iteration, residual_norm)
            
            if converged:
                break
            
            # Compute conjugate direction coefficient: β = (r_new^T r_new) / (r^T r)
            beta = rTr_new / rTr
            
            # Update search direction
            d = r + beta * d
            
            # Update for next iteration
            rTr = rTr_new
        
        return x, residual_history, iteration, converged


