import numpy as np
import pandas as pd
import time  # Importiamo il modulo time
from BroydenTridiagonal import BroydenTridiagonal
from OptimizationClass import OptimizationClass

def generate_starting_points(problem, num_random=5):
    """
    Generates the suggested starting point and 5 random points 
    within the hypercube [x_bar-1, x_bar+1]. 
    """
    x_bar = problem.get_starting_point()
    points = [x_bar] # The first point is the standard suggestion
    
    # Generate random uniform points
    for _ in range(num_random):
        random_perturbation = np.random.uniform(-1, 1, size=problem.n)
        x_rnd = x_bar + random_perturbation
        points.append(x_rnd)
        
    return points
    
def build_row(method, xk_mn, iterations, n, x0, h, grad_norm_mn, tol, elapsed_time):
    """
    Helper function to create a dictionary row for the DataFrame.
    Now includes 'elapsed_time'.
    """
    res = {
            "method": method,
            "xk_final": xk_mn,       # Renamed for clarity
            "iterations": iterations,
            "n": n,
            "x0_index": x0,          # Renamed for clarity (index of start point)
            "h": h,
            "grad_norm": grad_norm_mn,
            "converged": grad_norm_mn <= tol,
            "time_sec": elapsed_time # Added time column
        }
    return res

# --- Main Execution ---

optimizer = OptimizationClass()
n_arr = [10**5]
h_arr = [10**(-4)]
TOL = 1e-5
MAX_ITER = 500

results_list = [] 

for n in n_arr:
    print(f"\n ----------- Starting n={n} computation --------------- \n")
    
    # Initialize problem
    try:
        problem = BroydenTridiagonal(n)
    except Exception as e:
        print(f"Error initializing problem for n={n}: {e}")
        continue

    points = generate_starting_points(problem)
    
    for i, x0 in enumerate(points):
        print(f"Processing Starting Point {i}...")
        
        # --- 1. EXACT DERIVATIVES ---
        h = None 
        
        # A. Modified Newton (Exact)
        start_time = time.perf_counter() # Start timer
        xk_mn, k_mn, grad_norm_mn, history_mn = optimizer.modified_newton_method(
                problem.F,
                problem.F_gradient,
                problem.F_hessian_exact, 
                x0,
                MAX_ITER,
                TOL
        )
        end_time = time.perf_counter() # Stop timer
        results_list.append(build_row("MN (Exact)", xk_mn, k_mn, n, i, h, grad_norm_mn, TOL, end_time - start_time))

        # B. Truncated Newton (Exact) 
        
        start_time = time.perf_counter() # Start timer
        xk_tn, k_tn, grad_norm_tn, history_tn = optimizer.truncated_newton_method(
                problem.F,
                problem.F_gradient,
                problem.F_hessian_exact, 
                x0,
                MAX_ITER,
                TOL
        )
        end_time = time.perf_counter() # Stop timer
        results_list.append(build_row("TN (Exact)", xk_tn, k_tn, n, i, h, grad_norm_tn, TOL, end_time - start_time))
        
        # --- 2. FINITE DIFFERENCES (APPROXIMATE) ---
        for h in h_arr:
            print(f"  -> Trying Finite Differences with h={h}")
            
            # C. Modified Newton (Approximate - Full Matrix)
            # WARNING: This is the "slow" method. For n=10^5 it might be extremely slow or crash memory.
            # Ensure your 'F_hessian_approx_full' handles sparse matrices or check memory!
            hessian_matrix_func = lambda x: problem.F_hessian_approx_full(x, h=h)
            xk_mn, k_mn, grad_norm_mn, history_mn = optimizer.modified_newton_method(
                problem.F,
                problem.F_gradient,
                hessian_matrix_func,  # <--- Passing the lambda wrapper here
                x0,
                MAX_ITER,
                TOL
            )
            end_time = time.perf_counter() # Stop timer
            results_list.append(build_row("MN (FD)", xk_mn, k_mn, n, i, h, grad_norm_mn, TOL, end_time - start_time))
            
            # D. Truncated Newton (Approximate - Operator)
            
            start_time = time.perf_counter() # Start timer
            xk_tn, k_tn, grad_norm_tn, history_tn = optimizer.truncated_newton_method(
                problem.F,
                problem.F_gradient,
                hessian_matrix_func,  # <--- Passing the lambda wrapper here
                x0,
                MAX_ITER,
                TOL
            )
            end_time = time.perf_counter() # Stop timer
            results_list.append(build_row("TN (FD)", xk_tn, k_tn, n, i, h, grad_norm_tn, TOL, end_time - start_time))

# --- Create and Print DataFrame ---
df_final = pd.DataFrame(results_list)

# Reorder columns for better readability
cols = ["n", "x0_index", "method", "h", "iterations", "grad_norm", "converged", "time_sec"]
df_final = df_final[cols]

print("\n--- Optimization Results ---")
print(df_final)

# Optional: Save to CSV
df_final.to_csv("final_results.csv", index=False)