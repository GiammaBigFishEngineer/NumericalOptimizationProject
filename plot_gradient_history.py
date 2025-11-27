import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import os

def plot_gradient_descent_history(csv_filename="final_results.csv"):
    """
    Reads the optimization results CSV and generates detailed convergence plots.
    
    Output Structure:
    - Creates a folder named 'plots_history'.
    - Generates one PNG file per Dimension (n) and Step (h).
    - Each PNG contains a grid of subplots comparing MN vs TN for every Starting Point.
    - **Note:** Only plots the history if the method Converged.
    """
    
    print(f"--- Loading data from {csv_filename} ---")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    # --- 1. Data Preprocessing ---
    
    # Convert the string representation of lists "[...]" back to actual Python lists
    print("Parsing gradient history arrays...")
    try:
        # Use ast.literal_eval safely on string entries
        df['history'] = df['history'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except Exception as e:
        print(f"Error parsing 'history' column: {e}")
        return

    # Handle 'h' column: Replace NaNs (Exact derivatives) with "Exact" for easier grouping
    df['h_label'] = df['h'].fillna('Exact')

    # Create output directory if it doesn't exist
    output_dir = "plots_history"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get unique values for loops
    unique_ns = sorted(df['n'].unique())
    # Sort steps: 'Exact' first, then numerical values
    unique_hs = sorted(df['h_label'].unique(), key=lambda x: (x != 'Exact', x)) 
    unique_x0s = sorted(df['x0_index'].unique())

    print(f"Found Dimensions n: {unique_ns}")
    print(f"Found Steps h: {unique_hs}")
    print(f"Found Start Points: {unique_x0s}")

    # --- 2. Plotting Loop ---

    for n in unique_ns:
        for h_label in unique_hs:
            
            # Filter data for the current Dimension (n) and Step (h)
            subset = df[(df['n'] == n) & (df['h_label'] == h_label)]
            
            if subset.empty:
                continue

            print(f"Generating plot for n={n}, h={h_label}...")

            # Create a figure with subplots (Assuming 6 start points: 2 rows x 3 cols)
            # You can adjust figsize if you have more points
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Gradient Descent History | Dimension n={n} | h={h_label}", fontsize=16)
            
            # Flatten axes array for easy iteration (from 2D grid to 1D list)
            axes_flat = axes.flatten()

            # Loop over each starting point
            for i, x0_idx in enumerate(unique_x0s):
                if i >= len(axes_flat): break # Safety check
                
                ax = axes_flat[i]
                
                # Get data rows for this specific starting point
                row_data = subset[subset['x0_index'] == x0_idx]
                
                lines_plotted = False
                
                # --- Plot Modified Newton (MN) ---
                # We look for methods containing "MN"
                mn_data = row_data[row_data['method'].str.contains("MN")]
                if not mn_data.empty:
                    # Check if converged
                    is_converged = mn_data.iloc[0]['converged']
                    
                    if is_converged:
                        hist = mn_data.iloc[0]['history']
                        lbl = mn_data.iloc[0]['method'] # e.g. "MN (Exact)"
                        # Plot on Logarithmic Scale
                        ax.semilogy(hist, label=lbl, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                        lines_plotted = True

                # --- Plot Truncated Newton (TN) ---
                # We look for methods containing "TN"
                tn_data = row_data[row_data['method'].str.contains("TN")]
                if not tn_data.empty:
                    # Check if converged
                    is_converged = tn_data.iloc[0]['converged']
                    
                    if is_converged:
                        hist = tn_data.iloc[0]['history']
                        lbl = tn_data.iloc[0]['method'] # e.g. "TN (Exact)"
                        # Plot on Logarithmic Scale
                        ax.semilogy(hist, label=lbl, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
                        lines_plotted = True

                # Titles and Labels for the subplot
                point_name = "Suggested" if x0_idx == 0 else f"Random {x0_idx}"
                ax.set_title(f"Start Point: {point_name} (ID: {x0_idx})", fontsize=11)
                ax.set_xlabel("Iterations")
                ax.set_ylabel("||∇f(x)|| (Log Scale)")
                ax.grid(True, which="both", linestyle='--', alpha=0.4)
                
                # Add a red dotted line for the Tolerance (1e-5)
                # (Adjust 'y=1e-5' if you used a different tolerance)
                ax.axhline(y=1e-5, color='red', linestyle=':', alpha=0.5, label='Tol (1e-5)')

                if lines_plotted:
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No Converged Data", ha='center', va='center', transform=ax.transAxes)

            # Adjust layout to prevent overlap of titles and axes
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure to the output directory
            # We replace dots in filename to avoid extension issues (e.g. h=1.0e-4)
            filename = f"history_n{n}_h{str(h_label).replace('.', 'p')}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            
            # Close the figure to free up memory
            plt.close()

    print(f"\n--- Done! All plots have been saved in the '{output_dir}' folder. ---")

if __name__ == "__main__":
    plot_gradient_descent_history()