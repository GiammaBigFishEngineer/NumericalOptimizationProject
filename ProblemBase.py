import numpy as np
import scipy.sparse as sp
from typing import Callable
from abc import ABC, abstractmethod # Used to define an abstract class

class ProblemBase(ABC):
    """
    Abstract Base Class for a large-scale optimization problem.

    This class provides a universal interface for optimization problems
    and includes built-in methods for finite difference (FD) Hessian
    approximations, as required by the assignment (Point 3).
    
    Any child class (like BroydenTridiagonal) that inherits from this
    MUST implement the 'F' and 'F_gradient' methods.
    """
    
    def __init__(self, n: int):
        """
        Initializes the problem with its dimension.
        """
        if n < 1:
            raise ValueError("Dimension 'n' must be at least 1.")
        self.n = n
        self.name = "Abstract Problem"

    # --- Abstract Methods (Must be implemented by child classes) ---

    @abstractmethod
    def F(self, x: np.ndarray) -> float:
        """
        Calculates the scalar objective function value F(x).
        """
        pass

    @abstractmethod
    def F_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the exact gradient vector, grad(F(x)).
        """
        pass

    @abstractmethod
    def get_starting_point(self) -> np.ndarray:
        """
        Returns the problem's suggested starting point vector.
        This must be implemented by the child class.
        """
        pass

    # --- Finite Difference Hessian Methods (Inherited by all child classes) ---

    def F_hessian_product(self, x, v, epsilon=1e-8):
        """
        Approssima il prodotto Hessiana-Vettore H(x) * v usando differenze finite
        sul gradiente. Non calcola mai la matrice H esplicitamente.
        Formula: Hv ≈ (grad(x + epsilon * v) - grad(x)) / epsilon
        """
        grad_x = self.F_gradient(x)
        grad_x_eps = self.F_gradient(x + epsilon * v)
        return (grad_x_eps - grad_x) / epsilon
    
    def F_hessian_product_from_full_matrix(self, x, v):
        # 1. Calculate the FULL dense matrix (Expensive!)
        H = self.F_hessian_approx_full(x, h=1e-5)
        
        # 2. Perform Matrix-Vector multiplication
        return H @ v
    
    @abstractmethod
    def F_hessian_approx_full(self, x: np.ndarray, h: float) -> np.ndarray:
        """
        Calculates the FULL approximate Hessian matrix using finite differences
        optimized with GRAPH COLORING (grouping).
        """
        pass