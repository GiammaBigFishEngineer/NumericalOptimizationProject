import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from typing import List, Tuple

def plot_average_and_std(
    ax: plt.Axes, 
    histories: List[list], 
    label: str, 
    style: str, 
    marker: str
):
    """
    Helper function to plot the average of multiple convergence histories
    with a shaded standard deviation area.
    """
    if not histories:
        return

    # --- 1. Pad histories to the same length ---
    # Find the length of the longest run
    max_len = max(len(h) for h in histories)
    
    padded_histories = np.zeros((len(histories), max_len))
    
    for i, h in enumerate(histories):
        # Get the last gradient norm
        pad_val = h[-1]
        # Pad the list to max_len with the last value
        pad_len = max_len - len(h)
        padded_histories[i, :] = np.pad(
            h, (0, pad_len), 'constant', constant_values=pad_val
        )

    # --- 2. Calculate statistics ---
    mean_history = np.mean(padded_histories, axis=0)
    std_history = np.std(padded_histories, axis=0)
    iterations = np.arange(max_len)

    # --- 3. Plot ---
    ax.semilogy(
        iterations, 
        mean_history, 
        label=label, 
        linestyle=style, 
        marker=marker, 
        linewidth=1.5
    )
    
    # Add the shaded standard deviation area
    ax.fill_between(
        iterations, 
        mean_history - std_history, 
        mean_history + std_history, 
        alpha=0.15
    )

def plot_by_dimension(df: pd.DataFrame, unique_dims: list, tol: float):
    """
    Generates one plot for each dimension 'n'.
    Each plot compares the performance of all (Method, Derivative, h)
    combinations, averaged over the random starting points.
    """
    print("\n--- Generating plots grouped by dimension (n) ---")
    
    for n in unique_dims:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        df_n = df[df['n'] == n]
        
        # Group by everything *except* the start point
        grouped = df_n.groupby(['Method', 'Derivative', 'h_FD'])
        
        for name, group in grouped:
            method, derivative, h = name
            
            # Get all histories for this group (from the different starts)
            histories = group['Gradient History'].tolist()
            
            # Build a clear label
            label = (
                f"{method} ({derivative}, "
                f"h={h if pd.notna(h) else 'N/A'}) "
                f"(avg. {len(histories)} starts)"
            )
            style = '--' if "Modified" in method else '-'
            
            plot_average_and_std(ax, histories, label, style, marker='.')

        # --- Configure the plot ---
        ax.set_title(f"Method Convergence (Dimension n = {n})", fontsize=16)
        ax.set_xlabel("Iteration (k)", fontsize=12)
        ax.set_ylabel("Gradient Norm ||∇F(x_k)|| (Log Scale)", fontsize=12)
        ax.axhline(tol, color='red', linestyle=':', lw=2, label=f'Tolerance ({tol})')
        ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        output_filename = f"convergence_plot_BY_N_{n}.png"
        plt.savefig(output_filename)
        print(f"Plot saved to: {output_filename}")
        plt.show()

def plot_by_h_step(df: pd.DataFrame, unique_h_steps: list, tol: float):
    """
    Generates one plot for each h_FD step (including 'Exact').
    Each plot compares the scalability (performance vs. 'n') 
    of the two methods, averaged over the random starting points.
    """
    print("\n--- Generating plots grouped by h_FD step ---")
    
    for h in unique_h_steps:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        h_label = "Exact" if pd.isna(h) else f"h = {h}"
        
        # Filter for this h value
        if pd.isna(h):
            df_h = df[df['h_FD'].isna()]
        else:
            df_h = df[df['h_FD'] == h]
            
        # Group by Method and Dimension 'n'
        grouped = df_h.groupby(['Method', 'n'])
        
        for name, group in grouped:
            method, n = name
            
            # Get all histories for this group (from the different starts)
            histories = group['Gradient History'].tolist()
            
            # Build a clear label
            label = (
                f"{method} (n={n}) "
                f"(avg. {len(histories)} starts)"
            )
            style = '--' if "Modified" in method else '-'
            
            plot_average_and_std(ax, histories, label, style, marker='.')

        # --- Configure the plot ---
        ax.set_title(f"Method Scalability (Derivatives: {h_label})", fontsize=16)
        ax.set_xlabel("Iteration (k)", fontsize=12)
        ax.set_ylabel("Gradient Norm ||∇F(x_k)|| (Log Scale)", fontsize=12)
        ax.axhline(tol, color='red', linestyle=':', lw=2, label=f'Tolerance ({tol})')
        ax.legend(fontsize=9, bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        output_filename = f"convergence_plot_BY_H_{h_label}.png"
        plt.savefig(output_filename)
        print(f"Plot saved to: {output_filename}")
        plt.show()


def main_plotter(csv_filename="optimization_results.csv"):
    """
    Loads results from CSV and generates all required plots.
    """
    
    print(f"Loading results from {csv_filename}...")
    try:
        results_df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        print("Please run 'run_experiments.py' first to generate the results.")
        return

    # --- Data Conversion ---
    valid_results = results_df[
        (results_df['Status'] == 'Success') & 
        (results_df['Gradient History'].notna())
    ].copy()
    
    if valid_results.empty:
        print("No 'Success' results with valid history to plot.")
        return
        
    try:
        valid_results['Gradient History'] = valid_results['Gradient History'].apply(ast.literal_eval)
    except Exception as e:
        print(f"Error converting 'Gradient History' column: {e}")
        return

    # --- Get plot groupings ---
    unique_dims = valid_results['n'].unique()
    unique_dims.sort()
    
    # Get h values (and add np.nan for 'Exact')
    unique_h_steps = valid_results['h_FD'].unique()
    # Sort, putting np.nan first
    unique_h_steps = sorted(
        unique_h_steps, 
        key=lambda x: (pd.isna(x), x)
    )

    # Import tolerance from main script
    try:
        from Project.script_exact_grad import SOLVER_PARAMS
    except ImportError:
        SOLVER_PARAMS = {"tolgrad": 1e-6}
    tol = SOLVER_PARAMS.get('tolgrad', 1e-6)

    # --- Generate all plots ---
    plot_by_dimension(valid_results, unique_dims, tol)
    plot_by_h_step(valid_results, unique_h_steps, tol)
    
    print("\n--- All plots generated. ---")

# --- Run the script ---
if __name__ == "__main__":
    main_plotter()