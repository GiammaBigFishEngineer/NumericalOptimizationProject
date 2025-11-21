import matplotlib.pyplot as plt
import numpy as np
from BroydenTridiagonal import BroydenTridiagonal
from OptimizationClass import OptimizationClass

# --- Setup Problem ---
n = 10**3
problem = BroydenTridiagonal(n)

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

# Generate points and select one
points = generate_starting_points(problem)
x0 = points[5] # Using the 5th random point

# Initialize Optimizer
optimizer = OptimizationClass()

# --- Parameters ---
tol = 1e-7
max_iter = 500
h = 10**(-12)

print(f"--- Start Comparison (n={n}, h={h}) ---")

print("--- Testing with Full Finite Difference Hessian ---")

# A. Modified Newton

hessian_matrix_func = lambda x: problem.F_hessian_approx_full(x, h=h)
print("\nRunning Modified Newton (Full FD Hessian)...")
xk_mn, k_mn, grad_norm_mn, history_mn = optimizer.modified_newton_method(
    problem.F,
    problem.F_gradient,
    hessian_matrix_func,  # <--- Passing the lambda wrapper here
    x0,
    max_iter,
    tol
)
print(f"MN Iterations: {k_mn}")

# B. Truncated Newton
print("\nRunning Truncated Newton (Full FD Hessian)...")
xk_tn, k_tn, grad_norm_tn, history_tn = optimizer.truncated_newton_method(
    problem.F,
    problem.F_gradient,
    problem.F_hessian_product_from_full_matrix, # <--- Passing the wrapper function here
    x0,
    max_iter,
    tol
)
print(f"TN Iterations: {k_tn}")

# --- 3. Plotting Results ---

plt.figure(figsize=(10, 6))

# Plot Modified Newton (MN)
plt.semilogy(range(len(history_mn)), history_mn, 
             label=f'Modified Newton (Iter: {k_mn})', 
             color='blue', marker='o', markersize=4, linestyle='-')

# Plot Truncated Newton (TN)
plt.semilogy(range(len(history_tn)), history_tn, 
             label=f'Truncated Newton (Iter: {k_tn})', 
             color='orange', marker='x', markersize=6, linestyle='--')

# Tolerance Line
plt.axhline(y=tol, color='red', linestyle=':', linewidth=1.5, label=f'Tolerance ({tol})')

# Plot Formatting
plt.xlabel('Iterations ($k$)', fontsize=12)
plt.ylabel(r'Gradient Norm $\|\nabla f(x_k)\|$', fontsize=12)
plt.title(f'Convergence Comparison: Broyden Tridiagonal (n={n})', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend(fontsize=11)

# Optional: Highlight the start point
plt.plot(0, history_mn[0], 'ko', label='Start') 

plt.tight_layout()
plt.show()