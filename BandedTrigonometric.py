import numpy as np
import scipy.sparse as sp
from ProblemBase import ProblemBase

class BandedTrigonometric(ProblemBase):
    """
    Implementation of the Banded Trigonometric Problem (Problem 16).
    
    Objective Function:
    F(x) = sum_{i=1}^n i * [ (1 - cos(x_i)) + sin(x_{i-1}) - sin(x_{i+1}) ]
    with boundary conditions x_0 = 0, x_{n+1} = 0.
    
    Note: Despite the name "Banded", the Hessian of this specific formulation
    is strictly DIAGONAL, making it extremely efficient for large-scale tests.
    """

    def __init__(self, n):
        """
        Initializes the problem.
        """
        super().__init__(n) 
        self.name = f"Banded Trigonometric (n={n})"
    
    def get_starting_point(self) -> np.ndarray:
        """
        Returns the suggested starting point.
        According to[cite: 165]: x_i = 1 for i >= 1.
        """
        return np.full(self.n, 1.0, dtype=np.float64)
    
    def F(self, x: np.ndarray) -> float:
        """
        Calculates the scalar objective function F(x).
        Sum i * [ (1 - cos(x_i)) + sin(x_{i-1}) - sin(x_{i+1}) ]
        """
        x = x.astype(np.float64)
        n = self.n
        
        # Indices i from 1 to n
        i_vec = np.arange(1, n + 1, dtype=np.float64)
        
        # Term 1: i * (1 - cos(x_i))
        term1 = i_vec * (1.0 - np.cos(x))
        
        # Term 2: i * sin(x_{i-1})
        # x_{i-1} corresponds to shifting x to the right (insert 0 at start)
        x_prev = np.concatenate(([0.0], x[:-1]))
        term2 = i_vec * np.sin(x_prev)
        
        # Term 3: -i * sin(x_{i+1})
        # x_{i+1} corresponds to shifting x to the left (insert 0 at end)
        x_next = np.concatenate((x[1:], [0.0]))
        term3 = -1.0 * i_vec * np.sin(x_next)
        
        # Sum all components
        return np.sum(term1 + term2 + term3)
    
    def F_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the Exact Gradient.
        
        Analytic derivation shows that the gradient component g_k depends
        only on x_k (due to the cancellation of cross-terms in the summation).
        
        Formula derived from[cite: 163]:
        g_k = k * sin(x_k) + 2 * cos(x_k)      (for k < n)
        g_n = n * sin(x_n) - (n-1) * cos(x_n)  (boundary case)
        """
        x = x.astype(np.float64)
        n = self.n
        k_vec = np.arange(1, n + 1, dtype=np.float64)
        
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        
        # General formula: g_k = k*sin(x_k) + 2*cos(x_k)
        grad = k_vec * sin_x + 2.0 * cos_x
        
        # Fix boundary condition for the last element (k=n)
        # The term from sin(x_{i+1}) is missing for the last element
        # Correct coeff for cos(x_n) is: (n) - (n-1) - n (wait, derivation check)
        #
        # Re-derivation for last element:
        # From Term 1 (i=n): n * sin(x_n)
        # From Term 2 (i=n+1): Does not exist in sum.
        # From Term 3 (i=n-1): -(n-1) * (deriv of sin(x_n)) = -(n-1)*cos(x_n)
        # Result: n*sin(x_n) - (n-1)*cos(x_n)
        
        grad[-1] = n * sin_x[-1] - (n - 1.0) * cos_x[-1]
        
        return grad

    def F_hessian_exact(self, x: np.ndarray) -> sp.csc_matrix:
        """
        Calculates the Exact Hessian.
        
        Since g_k depends only on x_k, the Hessian is DIAGONAL.
        H_kk = d(g_k) / d(x_k)
        """
        x = x.astype(np.float64)
        n = self.n
        k_vec = np.arange(1, n + 1, dtype=np.float64)
        
        sin_x = np.sin(x)
        cos_x = np.cos(x)
        
        # General formula derivative:
        # d/dx (k*sin + 2*cos) = k*cos - 2*sin
        diag_val = k_vec * cos_x - 2.0 * sin_x
        
        # Boundary correction for last element:
        # d/dx (n*sin - (n-1)*cos) = n*cos + (n-1)*sin
        diag_val[-1] = n * cos_x[-1] + (n - 1.0) * sin_x[-1]
        
        # Build diagonal sparse matrix
        H = sp.diags(
            [diag_val], 
            [0], 
            shape=(n, n), 
            format='csc'
        )
        
        return H
    
    def F_hessian_approx(self, x: np.ndarray, h: float) -> sp.spmatrix:
        """
        Calculates the FULL approximate Hessian matrix using finite differences
        optimized with GRAPH COLORING (grouping).
        
        For the Banded Trigonometric Problem, the Hessian is DIAGONAL.
        This means the bandwidth is 0.
        
        Applying the coloring logic:
        Number of groups = 2 * bandwidth + 1 = 2*0 + 1 = 1 group.
        
        Therefore, we can approximate the entire Diagonal Hessian with
        just a SINGLE perturbation vector d = [1, 1, ..., 1].
        
        This reduces the cost from N iterations to just 1 iteration.
        
        Returns:
            scipy.sparse.csc_matrix: The approximate Hessian in sparse format.
        """
        n = self.n
        
        # 1. Create the perturbation vector 'd' for the single group.
        # Since columns never overlap (diagonal), we perturb all x_i simultaneously.
        d = np.ones(n, dtype=np.float64)
        
        # 2. Calculate gradient difference (Central Differences)
        # This requires only 2 evaluations of the Exact Gradient.
        g_plus = self.F_gradient(x + h * d)
        g_minus = self.F_gradient(x - h * d)
        
        # The vector 'diff' contains the approximations of the diagonal elements.
        # diff[i] ≈ d(g_i) / d(x_i) = H_ii
        diff = (g_plus - g_minus) / (2.0 * h)
        
        # 3. Construct the Sparse Diagonal Matrix directly
        # sp.diags is more efficient than lil_matrix for pure diagonals
        H_approx = sp.diags(
            diff, 
            0, # Offset 0 = main diagonal
            shape=(n, n), 
            format='csc'
        )

        return H_approx