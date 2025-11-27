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
        Calcola il Gradiente ESATTO in modo ottimizzato.
        
        Invece di costruire la matrice Jacobiana J (che è costoso),
        calcoliamo direttamente il vettore gradiente usando la formula analitica:
        g_i = f_i * (3 - 4x_i) - 2*f_{i-1} - f_{i+1}
        """
        # Calcola prima il vettore dei residui f(x)
        # (Assicurati che self.f restituisca float64 per evitare errori numerici)
        f_vec = self.f(x).astype(np.float64)
        x = x.astype(np.float64)
        
        # Termine 1: f_i * (3 - 4x_i)
        term1 = f_vec * (3.0 - 4.0 * x)
        
        # Termine 2: -2 * f_{i-1}
        # "f_{i-1}" è f_vec shiftato a destra (con 0 all'inizio)
        f_prev = np.concatenate(([0.0], f_vec[:-1]))
        term2 = -2.0 * f_prev
        
        # Termine 3: -1 * f_{i+1}
        # "f_{i+1}" è f_vec shiftato a sinistra (con 0 alla fine)
        f_next = np.concatenate((f_vec[1:], [0.0]))
        term3 = -1.0 * f_next
        
        # Somma tutto
        return term1 + term2 + term3
    
    def F_hessian_exact(self, x: np.ndarray) -> sp.csc_matrix:
        """
        Calcola l'Hessiana ESATTA in modo ottimizzato (Hard-coded).
        
        Invece di costruire J e fare il prodotto matriciale J.T * J (costoso),
        costruiamo direttamente le 5 diagonali della matrice risultante
        usando le formule analitiche.
        
        Risultato: Una matrice pentadiagonale (5-banded).
        """
        n = self.n
        x = x.astype(np.float64)
        f_vec = self.f(x) # Serve per il secondo termine
        
        # Termine ricorrente: D_i = (3 - 4x_i)
        D = 3.0 - 4.0 * x
        
        # --- 1. Diagonale Principale (Offset 0) ---
        # Formula base: D_i^2 + 5
        main_diag = D**2 + 5.0
        
        # Correzioni ai bordi (perché J è tridiagonale finita)
        main_diag[0] -= 4.0 # Primo elemento: D_0^2 + 1
        main_diag[-1] -= 1.0 # Ultimo elemento: D_{n-1}^2 + 4
        
        # Aggiunta del Secondo Termine (sum f_k * H_k)
        # H_k ha solo -4 sulla diagonale
        main_diag -= 4.0 * f_vec
        
        # --- 2. Diagonale 1 (Offset +1 e -1) ---
        # Formula: -2*D_i - D_{i+1}
        # D[:-1] prende da 0 a n-2 (i)
        # D[1:]  prende da 1 a n-1 (i+1)
        off_diag_1 = -2.0 * D[:-1] - D[1:]
        
        # --- 3. Diagonale 2 (Offset +2 e -2) ---
        # Formula: Costante 2
        off_diag_2 = np.full(n - 2, 2.0, dtype=np.float64)
        
        # --- 4. Assemblaggio Matrice ---
        # Sfruttiamo la simmetria dell'Hessiana
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