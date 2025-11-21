import numpy as np
import pandas as pd
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
    # bounds: [x_bar_i - 1, x_bar_i + 1]
    for _ in range(num_random):
        # Uniform returns [0, 1), scale to [low, high)
        random_perturbation = np.random.uniform(-1, 1, size=problem.n)
        x_rnd = x_bar + random_perturbation
        points.append(x_rnd)
        
    return points
    
def build_row(method, xk_mn, iterations, n, x0, h, grad_norm_mn, tol):
    res = {
            "method": method,
            "xk_mn": xk_mn,
            "iterations": iterations,
            "n": n,
            "x0 number": x0,
            "h": h,
            "converged": grad_norm_mn <= tol
        }
    return res

# Generate points and select one
optimizer = OptimizationClass()
n_arr = [10**5]
h_arr = [10**(-4)]
TOL = 1e-5
MAX_ITER = 500

results_list = [] # Lista vuota

for n in n_arr:
    print("\n ----------- Starting n=",n," computation --------------- \n")
    problem = BroydenTridiagonal(n)
    points = generate_starting_points(problem)
    for i, x0 in enumerate(points):
        h = None
        #EXACT HESSIAN
        xk_mn, k_mn, grad_norm_mn, history_mn = optimizer.modified_newton_method(
                problem.F,
                problem.F_gradient,
                problem.F_hessian_exact, # Passing the exact Hessian matrix
                x0,
                MAX_ITER,
                TOL
        )
        results_list.append(build_row("MN", xk_mn, k_mn, n, i, h, grad_norm_mn, TOL))

        xk_tn, k_tn, grad_norm_tn, history_tn = optimizer.truncated_newton_method(
                problem.F,
                problem.F_gradient,
                problem.F_hessian_exact, # Passing the matrix-vector product function
                x0,
                MAX_ITER,
                TOL
        )
        results_list.append(build_row("TN", xk_tn, k_tn, n, i, h, grad_norm_tn, TOL))
        
        for h in h_arr:
            #APPROXIMATE HESSIAN
            print("Trying h=",h)
            hessian_matrix_func = lambda x: problem.F_hessian_approx_full(x, h=h)
            xk_mn, k_mn, grad_norm_mn, history_mn = optimizer.modified_newton_method(
                problem.F,
                problem.F_gradient,
                hessian_matrix_func,  # <--- Passing the lambda wrapper here
                x0,
                MAX_ITER,
                TOL
            )
            results_list.append(build_row("MN", xk_mn, k_mn, n, i, h, grad_norm_mn, TOL))
            
            xk_tn, k_tn, grad_norm_tn, history_tn = optimizer.truncated_newton_method(
                problem.F,
                problem.F_gradient,
                hessian_matrix_func,  # <--- Passing the lambda wrapper here
                x0,
                MAX_ITER,
                TOL
            )
            results_list.append(build_row("TN", xk_tn, k_tn, n, i, h, grad_norm_tn, TOL))

            

df_final = pd.DataFrame(results_list)
print(df_final)
        