"""
Iterative Solvers for Portfolio Optimization
=============================================

This package contains implementations of iterative methods for solving
the mean-variance portfolio optimization problem:

    Σ w = λ μ

Solvers:
- Gradient Descent (GD)
- Conjugate Gradient (CG)
- Preconditioned Conjugate Gradient (PCG)
"""

from .gradient_descent import GradientDescentSolver
from .conjugate_gradient import ConjugateGradientSolver
from .preconditioned_cg import PreconditionedCGSolver
from .base import IterativeSolver

__all__ = [
    'IterativeSolver',
    'GradientDescentSolver',
    'ConjugateGradientSolver',
    'PreconditionedCGSolver',
]


