import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Callable, Tuple, Union

class OptimizationClass():
    """
    Contains a collection of line-search-based optimization methods
    for large-scale problems.
    """

    def __init__(self):
        """
        Initializes the optimization class.
        """
        pass
    
    # --- 1. The Common Backtracking Method ---
    
    def _backtracking_line_search(
        self,
        f: Callable,
        xk: np.ndarray,
        fk: float,
        gradfk: np.ndarray,
        pk: np.ndarray,
        c1: float,
        rho: float,
        btmax: int, 
        k: int
    ) -> float:
        """
        Performs a backtracking line search (Armijo condition).
        """
        alpha = 1.0
        armijo_term = c1 * np.dot(gradfk, pk)
        fnew = f(xk + alpha * pk)
        bt = 0

        while fnew > fk + alpha * armijo_term and bt < btmax:
            alpha = rho * alpha
            fnew = f(xk + alpha * pk)
            bt += 1

        if bt == btmax and fnew > fk + alpha * armijo_term:
            print(f"Warning (Iter {k}): Backtracking failed.")
            return 1e-12 

        return alpha

    # --- 2. Modified Newton Method ---

    def modified_newton_method(
        self,
        f: Callable,
        gradf: Callable,
        Hessf: Callable,
        x0: np.ndarray,
        kmax: int,
        tolgrad: float,
        c1: float = 1e-4,
        rho: float = 0.5,
        btmax: int = 20,
        max_n_dense: int = 10**4  # STOP if N > 10 for avoid crash RAM
    ) -> Tuple[np.ndarray, int, float, list]:
        """
        Solves min f(x) using the Modified Newton method.
        (Uses sparse/dense solvers correctly)
        """
        print("--- Running Modified Newton ---")
        n = len(x0)

        n = len(x0)
        
        # --- SAFEGUARD: Dimension Check ---
        if n >= max_n_dense:
            print(f"ABORT: Dimension n={n} is too large for Modified Newton (Limit={max_n_dense}).")
            print("Dense Cholesky O(N^3) would freeze the system.")
            return x0, 0, np.linalg.norm(gradf(x0)), [np.linalg.norm(gradf(x0))]
        
        xk = x0.copy().astype(np.float64)
        k = 0
        
        fk = f(xk)
        gradfk = gradf(xk).astype(np.float64)
        gradfk_norm = np.linalg.norm(gradfk)
        grad_history = [gradfk_norm]

        while k < kmax and gradfk_norm >= tolgrad:
            
            H = Hessf(xk)

            try:
                # Try to solve directly ---
                if sp.issparse(H):
                    H_dense = H.toarray()
                else:
                    H_dense = H
                
                # Test for PD
                np.linalg.cholesky(H_dense)
                
                # If test succeeds, solve
                if sp.issparse(H):
                    H_solve = H.tocsc()
                    pk = spla.spsolve(H_solve, -gradfk) # Use SPARSE solver
                else:
                    H_solve = H
                    pk = np.linalg.solve(H_solve, -gradfk) # Use DENSE solver
                
                # Check for descent direction
                if np.dot(gradfk, pk) > -1e-12:
                    raise np.linalg.LinAlgError("Not a descent direction")

            except np.linalg.LinAlgError as e:
                # Solve failed OR pk was not a descent direction ---
                print(f"Warning (Iter {k}): H not PD or bad direction. Modifying. ({e})")
                
                min_diag = np.min(H.diagonal()) if sp.issparse(H) else np.min(np.diag(H)) 
                tau = max(1e-4, -min_diag + 1e-4)
                
                if sp.issparse(H):
                    H_mod = H + tau * sp.eye(n, format=H.format)
                    H_mod_solve = H_mod.tocsc()
                    pk = spla.spsolve(H_mod_solve, -gradfk)
                else:
                    H_mod = H + tau * np.eye(n)
                    H_mod_solve = H_mod
                    pk = np.linalg.solve(H_mod_solve, -gradfk)
            
            # --- Backtracking & Update ---
            alpha = self._backtracking_line_search(
                f, xk, fk, gradfk, pk, c1, rho, btmax, k
            )
            
            if alpha < 1e-11:
                print(f"Error (Iter {k}): Backtracking failed. Stopping.")
                break 
            
            xk = xk + alpha * pk
            fk = f(xk)
            gradfk = gradf(xk).astype(np.float64) # Ensure float64
            gradfk_norm = np.linalg.norm(gradfk)
            k += 1
            grad_history.append(gradfk_norm)

        print(f"Modified Newton finished in {k} iterations.")
        return xk, k, gradfk_norm, grad_history

    # --- 3. Our Own Conjugate Gradient (CG) Method ---
    
    def truncated_newton_method(
        self,
        f: Callable,
        gradf: Callable,
        hess_vec_prod: Callable,
        x0: np.ndarray,
        kmax: int,
        tolgrad: float,
        c1: float = 1e-4,
        rho: float = 0.5,
        btmax: int = 20,
        h: float = 1e-8
    ) -> Tuple[np.ndarray, int, float, list]:
        """
        Solves min f(x) using the Truncated Newton (Newton-CG) method.
        
        Args:
            hess_vec_prod: Function that computes H(x) * v efficiently 
                           without forming H. Signature: func(x, v) -> np.ndarray
        """
        print("--- Running Truncated Newton (Newton-CG) ---")
        n = len(x0)
        xk = x0.copy().astype(np.float64)
        k = 0
        
        fk = f(xk)
        gradfk = gradf(xk).astype(np.float64)
        gradfk_norm = np.linalg.norm(gradfk)
        grad_history = [gradfk_norm]

        while k < kmax and gradfk_norm >= tolgrad:
            
            # --- 1. Define CG tolerance (Forcing Sequence) ---
            # Typically: min(0.5, sqrt(norm(g))) * norm(g) gives superlinear convergence
            eta_k = min(0.5, np.sqrt(gradfk_norm)) * gradfk_norm
            
            # --- 2. Inner Loop: Conjugate Gradient (Truncated) ---
            # Solves H * pk = -gradfk approximately
            pk = self._truncated_cg(
                hess_vec_prod = hess_vec_prod, 
                x = xk, 
                b = -gradfk, 
                tol = eta_k, 
                max_iter = n*2,
                h = h
            )
            
            # Safety check: if CG returns zero vector (rare), use gradient
            if np.linalg.norm(pk) < 1e-14:
                pk = -gradfk

            # --- 3. Backtracking Line Search ---
            alpha = self._backtracking_line_search(
                f, xk, fk, gradfk, pk, c1, rho, btmax, k
            )
            
            if alpha < 1e-11:
                print(f"Warning (Iter {k}): Backtracking failed (alpha too small). Stopping.")
                break
            
            # --- 4. Update ---
            xk = xk + alpha * pk
            fk = f(xk)
            gradfk = gradf(xk).astype(np.float64)
            gradfk_norm = np.linalg.norm(gradfk)
            
            k += 1
            grad_history.append(gradfk_norm)

        print(f"Truncated Newton finished in {k} iterations.")
        return xk, k, gradfk_norm, grad_history

    def _truncated_cg(
        self, 
        hess_vec_prod: Callable, 
        x: np.ndarray, 
        b: np.ndarray, 
        tol: float, 
        max_iter: int,
        h: float
    ) -> np.ndarray:
        """
        Inner solver: Computes d approx H^-1 b using Conjugate Gradient.
        Handles negative curvature (if H is indefinite) by truncating.
        """
        d = np.zeros_like(b)
        r = b.copy()  # Residual: r = b - H*d (since d=0, r=b)
        p = r.copy()  # Search direction
        
        rs_old = np.dot(r, r)
        
        for i in range(max_iter):
            if np.sqrt(rs_old) < tol:
                break
            
            # 1. H_sparse now holds the full sparse matrix H(x)
            #    This is what your hess_vec_prod(x) currently returns.
            H_sparse = hess_vec_prod(x)
            
            # 2. Correctly calculate the vector product Hp = H * p
            Hp = H_sparse.dot(p) 
            
            # 3. Calculate the scalar p^T H p
            pHp_raw = np.dot(p, Hp) 
            
            # Ensure the scalar value is extracted for safe comparison (Fix for NotImplementedError)
            try:
                pHp = pHp_raw.item()
            except AttributeError:
                pHp = pHp_raw
            
            # --- Truncation Rule: Negative Curvature ---
            if pHp <= 1e-16: # Use 1e-16 for numerical stability
                if i == 0:
                    return b 
                else:
                    return d
            
            alpha = rs_old / pHp
            d = d + alpha * p
            r = r - alpha * Hp
            
            rs_new = np.dot(r, r)
            
            if np.sqrt(rs_new) < tol:
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            
        return d
