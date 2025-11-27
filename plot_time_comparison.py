import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_execution_time(csv_filename="final_results.csv"):
    """
    Reads the optimization results CSV and generates bar charts to compare
    the execution times of the two methods.
    
    Structure:
    - Creates a folder 'plots_time'
    - Generates one PNG file per Dimension (n) and Step (h).
    - Each chart shows time bars for MN and TN for every starting point.
    """
    
    print(f"--- Loading data from {csv_filename} ---")
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: File {csv_filename} not found.")
        return

    # --- 1. Preprocessing ---
    
    # Handle 'h' column: Replace NaNs (Exact derivatives) with "Exact"
    df['h_label'] = df['h'].fillna('Exact')

    # Create output directory
    output_dir = "plots_time"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Unique values for loops
    unique_ns = sorted(df['n'].unique())
    unique_hs = sorted(df['h_label'].unique(), key=lambda x: (x != 'Exact', x))
    unique_x0s = sorted(df['x0_index'].unique())

    print(f"Found Dimensions n: {unique_ns}")
    print(f"Found Steps h: {unique_hs}")

    # --- 2. Plotting Loop ---

    for n in unique_ns:
        for h_label in unique_hs:
            
            # Filter data for current configuration n, h
            subset = df[(df['n'] == n) & (df['h_label'] == h_label)]
            
            if subset.empty:
                continue

            print(f"Generating time plot for n={n}, h={h_label}...")

            # Prepare data for bar chart
            start_points = []
            times_mn = []
            times_tn = []
            
            for x0_idx in unique_x0s:
                row_data = subset[subset['x0_index'] == x0_idx]
                
                # Point name
                p_name = "Suggested" if x0_idx == 0 else f"Rand {x0_idx}"
                start_points.append(p_name)
                
                # Modified Newton Time
                mn_row = row_data[row_data['method'].str.contains("MN")]
                if not mn_row.empty:
                    times_mn.append(mn_row.iloc[0]['time_sec'])
                else:
                    times_mn.append(0) # No data

                # Truncated Newton Time
                tn_row = row_data[row_data['method'].str.contains("TN")]
                if not tn_row.empty:
                    times_tn.append(tn_row.iloc[0]['time_sec'])
                else:
                    times_tn.append(0) # No data

            # --- Create Bar Chart ---
            plt.figure(figsize=(10, 6))
            
            # Bar positions
            x = np.arange(len(start_points))
            width = 0.35  # Width of bars

            # Draw bars
            bars1 = plt.bar(x - width/2, times_mn, width, label='Modified Newton', color='blue', alpha=0.7)
            bars2 = plt.bar(x + width/2, times_tn, width, label='Truncated Newton', color='orange', alpha=0.7)

            # Labels and Titles
            plt.xlabel('Starting Point')
            plt.ylabel('Execution Time (seconds)')
            plt.title(f'Time Comparison | Dimension n={n} | h={h_label}')
            plt.xticks(x, start_points)
            plt.legend()
            
            # Add a light horizontal grid
            plt.grid(axis='y', linestyle='--', alpha=0.4)

            # Optional: Add values on top of bars
            # If the difference is huge (e.g. 40s vs 0.1s), consider log scale:
            # plt.yscale('log') # Uncomment if you want logarithmic scale

            plt.tight_layout()
            
            # Save
            filename = f"time_n{n}_h{str(h_label).replace('.', 'p')}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close()

    print(f"\n--- Done! Plots saved in folder '{output_dir}' ---")

if __name__ == "__main__":
    plot_execution_time()