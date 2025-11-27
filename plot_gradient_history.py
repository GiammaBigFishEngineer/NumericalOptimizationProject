import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import os

def plot_gradient_descent_history(csv_filename="final_results.csv"):
    """
    Reads the optimization results CSV and generates detailed convergence plots.
    
    Structure:
    - Creates a folder 'plots_history'
    - Generates one PNG file per Dimension (n) and Step (h).
    - Each PNG contains subplots comparing MN vs TN for every Starting Point.
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
        df['history'] = df['history'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    except Exception as e:
        print(f"Error parsing 'history' column: {e}")
        return

    # Handle 'h' column: Replace NaNs (Exact derivatives) with "Exact" for easier grouping
    df['h_label'] = df['h'].fillna('Exact')

    # Create output directory
    output_dir = "plots_history"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get unique values for loops
    unique_ns = sorted(df['n'].unique())
    unique_hs = sorted(df['h_label'].unique(), key=lambda x: (x != 'Exact', x)) # Sorts Exact first, then numbers
    unique_x0s = sorted(df['x0_index'].unique())

    print(f"Found Dimensions n: {unique_ns}")
    print(f"Found Steps h: {unique_hs}")
    print(f"Found Start Points: {unique_x0s}")

    # --- 2. Plotting Loop ---

    for n in unique_ns:
        for h_label in unique_hs:
            
            # Filter data for current N and H
            subset = df[(df['n'] == n) & (df['h_label'] == h_label)]
            
            if subset.empty:
                continue

            print(f"Generating plot for n={n}, h={h_label}...")

            # Create a figure with subplots (Assuming 6 start points: 2 rows x 3 cols)
            # Adjust figsize as needed
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"Gradient Descent History | Dimension n={n} | h={h_label}", fontsize=16)
            
            # Flatten axes array for easy iteration (from 2D to 1D)
            axes_flat = axes.flatten()

            for i, x0_idx in enumerate(unique_x0s):
                if i >= len(axes_flat): break # Safety check
                
                ax = axes_flat[i]
                
                # Get data for this specific starting point
                row_data = subset[subset['x0_index'] == x0_idx]
                
                # Plot Lines
                lines_plotted = False
                
                # --- Plot Modified Newton ---
                mn_data = row_data[row_data['method'].str.contains("MN")]
                if not mn_data.empty:
                    hist = mn_data.iloc[0]['history']
                    lbl = mn_data.iloc[0]['method'] # e.g. "MN (Exact)"
                    ax.semilogy(hist, label=lbl, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
                    lines_plotted = True

                # --- Plot Truncated Newton ---
                tn_data = row_data[row_data['method'].str.contains("TN")]
                if not tn_data.empty:
                    hist = tn_data.iloc[0]['history']
                    lbl = tn_data.iloc[0]['method'] # e.g. "TN (Exact)"
                    ax.semilogy(hist, label=lbl, color='orange', linestyle='-', linewidth=1.5, alpha=0.8)
                    lines_plotted = True

                # Titles and Labels
                point_name = "Suggested" if x0_idx == 0 else f"Random {x0_idx}"
                ax.set_title(f"Start Point: {point_name} (ID: {x0_idx})", fontsize=11)
                ax.set_xlabel("Iterations")
                ax.set_ylabel("||∇f(x)|| (Log)")
                ax.grid(True, which="both", linestyle='--', alpha=0.4)
                
                # Add Tolerance Line (1e-5 based on your previous script)
                ax.axhline(y=1e-5, color='red', linestyle=':', alpha=0.5, label='Tol (1e-5)')

                if lines_plotted:
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save file
            filename = f"history_n{n}_h{h_label}.png".replace(".", "p") # Replace dots in filename
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close() # Close memory to free RAM

    print(f"\n--- Done! All plots saved in folder '{output_dir}' ---")

if __name__ == "__main__":
    plot_gradient_descent_history()