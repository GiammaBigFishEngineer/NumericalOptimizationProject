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
    
    @abstractmethod
    def F_hessian_approx(self, x: np.ndarray, h: float) -> np.ndarray:
        """
        Calculates the FULL approximate Hessian matrix using finite differences
        optimized with GRAPH COLORING (grouping).
        """
        pass