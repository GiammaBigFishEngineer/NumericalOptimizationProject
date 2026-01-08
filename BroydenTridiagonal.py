import numpy as np
import scipy.sparse as sp

# Assicurati che il nome di questo file sia 'problem_base.py'
from ProblemBase import ProblemBase 

class BroydenTridiagonal(ProblemBase):
    """
    Implementation of the Broyden Tridiagonal function (Problem 31).
    
    This class inherits the finite difference Hessian methods
    (get_fd_hessian_product_func and F_hessian_approx_full)
    from the 'ProblemBase' class.
    """

    def __init__(self, n):
        """
        Initializes the problem.
        Calls the parent constructor 'super().__init__(n)'.
        """
        super().__init__(n) 
        self.name = f"Broyden Tridiagonal (n={n})"
    
    def get_starting_point(self) -> np.ndarray:
        """Returns the suggested starting point vector (all -1s)."""
        return np.full(self.n, -1.0)
    
    def F(self, x):
        """Calculates the objective function F(x) = 0.5 * sum(f_k(x)^2)"""
        f_vec = self.f(x)
        return 0.5 * np.dot(f_vec, f_vec)
    
    def f(self, x):
        """Calculates the vector of components f(x) = [f_1, ..., f_n]"""
        x_prev = np.concatenate(([0.0], x[:-1]))
        x_next = np.concatenate((x[1:], [0.0]))
        return (3 - 2 * x) * x - x_prev - 2 * x_next + 1
    
    def F_gradient(self, x):
        """
        Computes the EXACT Gradient in an optimized manner.
        
        Instead of constructing the Jacobian matrix J (which is computationally expensive),
        we directly calculate the gradient vector using the analytical formula:
        g_i = f_i * (3 - 4x_i) - 2*f_{i-1} - f_{i+1}
        """
        # First, calculate the residual vector f(x)
        # Ensure float64 precision to avoid numerical errors
        f_vec = self.f(x).astype(np.float64)
        x = x.astype(np.float64)
        
        # Term 1: f_i * (3 - 4x_i)
        term1 = f_vec * (3.0 - 4.0 * x)
        
        # Term 2: -2 * f_{i-1}
        # "f_{i-1}" is f_vec shifted right (padded with 0 at the start)
        f_prev = np.concatenate(([0.0], f_vec[:-1]))
        term2 = -2.0 * f_prev
        
        # Term 3: -1 * f_{i+1}
        # "f_{i+1}" is f_vec shifted left (padded with 0 at the end)
        f_next = np.concatenate((f_vec[1:], [0.0]))
        term3 = -1.0 * f_next
        
        # Sum all components
        return term1 + term2 + term3

    def F_hessian_exact(self, x: np.ndarray) -> sp.csc_matrix:
        """
        Computes the EXACT Hessian in an optimized (hard-coded) manner.
        
        Instead of explicitly building J and performing the matrix product J.T * J,
        we directly construct the 5 diagonals of the resulting matrix 
        using analytical formulas.
        
        Result: A pentadiagonal (5-banded) sparse matrix.
        """
        n = self.n
        x = x.astype(np.float64)
        f_vec = self.f(x)  # Required for the second-order term
        
        # Recurrent term: D_i = (3 - 4x_i)
        D = 3.0 - 4.0 * x
        
        # --- 1. Main Diagonal (Offset 0) ---
        # Base formula: D_i^2 + 5
        main_diag = D**2 + 5.0
        
        # Boundary corrections (due to the finite tridiagonal structure of J)
        main_diag[0] -= 4.0   # First element: D_0^2 + 1
        main_diag[-1] -= 1.0  # Last element: D_{n-1}^2 + 4
        
        # Add the Second-Order Term (sum of f_k * H_k)
        # Since H_k has only -4 on its diagonal
        main_diag -= 4.0 * f_vec
        
        # --- 2. First Off-Diagonal (Offsets +1 and -1) ---
        # Formula: -2*D_i - D_{i+1}
        # D[:-1] covers indices 0 to n-2 (i)
        # D[1:]  covers indices 1 to n-1 (i+1)
        off_diag_1 = -2.0 * D[:-1] - D[1:]
        
        # --- 3. Second Off-Diagonal (Offsets +2 and -2) ---
        # Formula: Constant 2.0
        off_diag_2 = np.full(n - 2, 2.0, dtype=np.float64)
        
        # --- 4. Matrix Assembly ---
        # Leverage the symmetry of the Hessian
        H = sp.diags(
            [off_diag_2, off_diag_1, main_diag, off_diag_1, off_diag_2],
            [-2, -1, 0, 1, 2],
            shape=(n, n),
            format='csc'
        )
        
        return H
    
    def F_hessian_approx(self, x: np.ndarray, h: float) -> sp.spmatrix:
        """
        Calculates the FULL approximate Hessian matrix using finite differences
        optimized with GRAPH COLORING (grouping).
        
        Based on the observation that the problem 
        has LOCAL DEPENDENCIES, the Hessian is a BANDED matrix.
        For Broyden Tridiagonal, the bandwidth is 2 (5 diagonals total).
        
        Instead of N iterations (which fails for N=10^5), we only need 
        (2 * bandwidth + 1) = 5 iterations to build the whole matrix.
        
        Returns:
            scipy.sparse.csc_matrix: The approximate Hessian in sparse format.
        """
        n = self.n
        
        # 1. Use Sparse Matrix (LIL format is fast for constructing)
        # This solves the MemoryError for N=10^5
        H_approx = sp.lil_matrix((n, n), dtype=np.float64)
        
        # Bandwidth of the Hessian (for Broyden it's 2)
        bw = 2 
        
        # Number of "colors" or groups needed
        # Columns separated by this distance don't overlap
        num_groups = 2 * bw + 1 
        
        # 2. Iterate only over the groups (5 iterations total!)
        for group in range(num_groups):
            
            # Create a perturbation vector 'd' that perturbs 
            # every (num_groups)-th variable simultaneously
            d = np.zeros(n)
            d[group::num_groups] = 1.0 
            
            # Calculate gradient difference for the whole group at once
            # (Uses exact gradient as per Point 3.1)
            g_plus = self.F_gradient(x + h * d)
            g_minus = self.F_gradient(x - h * d)
            
            diff = (g_plus - g_minus) / (2.0 * h)
            
            # 3. Map the result back to the specific columns
            # We iterate over the columns belonging to this group
            for col_idx in range(group, n, num_groups):
                
                # We only extract the rows that are within the bandwidth
                # (The non-zero elements for this column)
                row_start = max(0, col_idx - bw)
                row_end = min(n, col_idx + bw + 1)
                
                # Assign the relevant segment of the difference vector 
                # to the sparse matrix column
                H_approx[row_start:row_end, col_idx] = diff[row_start:row_end]

        # Convert to Compressed Sparse Column format for efficient solving
        return H_approx.tocsc()