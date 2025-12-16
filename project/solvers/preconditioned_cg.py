"""
Preconditioned Conjugate Gradient Solver for Linear Systems
==========================================================

Solves A @ x = b using Preconditioned Conjugate Gradient (PCG).

Preconditioning transforms the system to:
    M^{-1} A x = M^{-1} b

where M ≈ A is easy to invert. The effective condition number becomes
κ(M^{-1} A) << κ(A), accelerating convergence.

Available preconditioners:
- Jacobi (diagonal): M = diag(A)
- SSOR (Symmetric Successive Over-Relaxation)
- Incomplete Cholesky

Reference: Shewchuk, "An Introduction to the Conjugate Gradient Method 
Without the Agonizing Pain", 1994.
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from .base import IterativeSolver


# =============================================================================
# PRECONDITIONERS
# =============================================================================

def jacobi_preconditioner(A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Jacobi (diagonal) preconditioner.
    
    M = diag(A)
    M^{-1} r = r / diag(A)
    
    Simple and cheap, but only effective when A is diagonally dominant.
    """
    diag_inv = 1.0 / np.diag(A)
    
    def apply(r: np.ndarray) -> np.ndarray:
        return diag_inv * r
    
    return apply


def ssor_preconditioner(A: np.ndarray, omega: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Symmetric Successive Over-Relaxation (SSOR) preconditioner.
    
    M = (D/ω + L) @ (D/ω)^{-1} @ (D/ω + L^T)
    
    where D = diag(A), L = strict lower triangle of A.
    
    Parameters
    ----------
    A : np.ndarray
        System matrix
    omega : float
        Relaxation parameter (0 < ω < 2). ω = 1 gives symmetric Gauss-Seidel.
    """
    n = A.shape[0]
    D = np.diag(A)
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    # M = (D/ω + L) @ inv(D) @ (D/ω + U)
    # For SSOR, we solve by forward then backward substitution
    
    D_omega = D / omega
    DL = np.diag(D_omega) + L
    DU = np.diag(D_omega) + U
    D_inv = 1.0 / D
    
    def apply(r: np.ndarray) -> np.ndarray:
        # Forward solve: (D/ω + L) y = r
        y = np.zeros(n)
        for i in range(n):
            y[i] = (r[i] - np.dot(L[i, :i], y[:i])) / D_omega[i]
        
        # Scale: z = D @ y
        z = D * y
        
        # Backward solve: (D/ω + U) x = z
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (z[i] - np.dot(U[i, i+1:], x[i+1:])) / D_omega[i]
        
        return omega * (2 - omega) * x
    
    return apply


def incomplete_cholesky_preconditioner(A: np.ndarray, 
                                        fill_factor: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Incomplete Cholesky (IC) preconditioner.
    
    Computes L such that A ≈ L @ L^T, where L has the same sparsity pattern as
    the lower triangle of A (IC(0)) or allows limited fill-in.
    
    For dense matrices, this reduces to standard Cholesky.
    
    Parameters
    ----------
    A : np.ndarray
        System matrix (SPD)
    fill_factor : float
        Drop tolerance for fill-in (0 = no fill, IC(0))
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    
    # Compute incomplete Cholesky factorization
    for j in range(n):
        # Diagonal element
        sum_sq = np.dot(L[j, :j], L[j, :j])
        L[j, j] = np.sqrt(max(A[j, j] - sum_sq, 1e-10))
        
        # Off-diagonal elements in column j
        for i in range(j+1, n):
            if abs(A[i, j]) > fill_factor or fill_factor == 0:
                sum_prod = np.dot(L[i, :j], L[j, :j])
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    def apply(r: np.ndarray) -> np.ndarray:
        # Solve L @ y = r (forward substitution)
        y = np.zeros(n)
        for i in range(n):
            y[i] = (r[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        
        # Solve L^T @ x = y (backward substitution)
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
        
        return x
    
    return apply


# Preconditioner registry
PRECONDITIONERS = {
    'jacobi': jacobi_preconditioner,
    'ssor': ssor_preconditioner,
    'ichol': incomplete_cholesky_preconditioner,
}


# =============================================================================
# PCG SOLVER
# =============================================================================

class PreconditionedCGSolver(IterativeSolver):
    """
    Preconditioned Conjugate Gradient solver for SPD linear systems.
    
    Key properties:
    - Reduces effective condition number via preconditioning
    - Maintains CG optimality in transformed space
    - Choice of preconditioner trades setup cost vs. iteration reduction
    """
    
    def __init__(self, 
                 preconditioner: str = 'jacobi',
                 precond_params: Optional[dict] = None,
                 *args, **kwargs):
        """
        Initialize PCG solver.
        
        Parameters
        ----------
        preconditioner : str
            Preconditioner type: 'jacobi', 'ssor', 'ichol'
        precond_params : dict, optional
            Additional parameters for preconditioner (e.g., omega for SSOR)
        """
        super().__init__(*args, **kwargs)
        self.preconditioner_name = preconditioner
        self.precond_params = precond_params or {}
        self.name = f"PCG ({preconditioner})"
        
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"Unknown preconditioner: {preconditioner}. "
                           f"Available: {list(PRECONDITIONERS.keys())}")
    
    def _build_preconditioner(self, A: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Build preconditioner for given matrix."""
        precond_factory = PRECONDITIONERS[self.preconditioner_name]
        
        if self.preconditioner_name == 'ssor':
            omega = self.precond_params.get('omega', 1.0)
            return precond_factory(A, omega=omega)
        elif self.preconditioner_name == 'ichol':
            fill_factor = self.precond_params.get('fill_factor', 0.0)
            return precond_factory(A, fill_factor=fill_factor)
        else:
            return precond_factory(A)
    
    def _solve_impl(self, 
                    A: np.ndarray, 
                    b: np.ndarray, 
                    x0: np.ndarray) -> Tuple[np.ndarray, List[float], int, bool]:
        """
        Solve using Preconditioned Conjugate Gradient.
        
        Algorithm (with preconditioner M):
        1. r = b - A @ x
        2. z = M^{-1} @ r
        3. d = z
        4. For each iteration:
           a. α = (r^T z) / (d^T A d)
           b. x = x + α * d
           c. r_new = r - α * A @ d
           d. z_new = M^{-1} @ r_new
           e. β = (r_new^T z_new) / (r^T z)
           f. d = z_new + β * d
        """
        # Build preconditioner
        M_inv = self._build_preconditioner(A)
        
        x = x0.copy()
        r = b - A @ x
        z = M_inv(r)
        d = z.copy()
        
        residual_history = []
        
        rTz = np.dot(r, z)
        residual_norm = np.linalg.norm(r)
        residual_history.append(residual_norm)
        
        converged = self._check_convergence(residual_norm)
        iteration = 0
        
        while not converged and iteration < self.max_iterations:
            # Compute A @ d
            Ad = A @ d
            
            # Step size
            dTAd = np.dot(d, Ad)
            
            if abs(dTAd) < 1e-15:
                break
            
            alpha = rTz / dTAd
            
            # Update solution
            x = x + alpha * d
            
            # Update residual
            r = r - alpha * Ad
            residual_norm = np.linalg.norm(r)
            residual_history.append(residual_norm)
            
            iteration += 1
            converged = self._check_convergence(residual_norm)
            
            self._log(iteration, residual_norm)
            
            if converged:
                break
            
            # Apply preconditioner to new residual
            z = M_inv(r)
            
            # Compute conjugate direction coefficient
            rTz_new = np.dot(r, z)
            beta = rTz_new / rTz
            
            # Update search direction
            d = z + beta * d
            
            # Update for next iteration
            rTz = rTz_new
        
        return x, residual_history, iteration, converged


