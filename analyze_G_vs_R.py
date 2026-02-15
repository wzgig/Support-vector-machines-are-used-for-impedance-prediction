import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def analyze_r_vs_g():
    file_path = 'extracted_RL_Series_Y11_wide.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Filter R_e based on the user's range
    r_col = 'R_e'
    if r_col not in df.columns:
        print(f"Column {r_col} not found in {df.columns}")
        return

    # Filter
    mask = (df[r_col] >= -90000) & (df[r_col] <= 90000)
    df_filtered = df[mask].copy()
    
    # Calculate G_e
    df_filtered['G_e'] = 1.0 / df_filtered[r_col].replace(0, np.nan)
    df_filtered = df_filtered.dropna(subset=['G_e'])

    print("\n--- Statistics for G_e (1/R_e) ---")
    g_e = df_filtered['G_e']
    print(f"Min: {g_e.min():.6e}")
    print(f"Max: {g_e.max():.6e}")
    print(f"Mean: {g_e.mean():.6e}")
    print(f"Std Dev: {g_e.std():.6e}")
    print(f"Skewness: {skew(g_e):.4f}")
    print(f"Kurtosis: {kurtosis(g_e):.4f}")
    
    print("\n--- Statistics for R_e (Filtered) ---")
    r_e = df_filtered[r_col]
    print(f"Min: {r_e.min():.4f}")
    print(f"Max: {r_e.max():.4f}")
    print(f"Mean: {r_e.mean():.4f}")
    print(f"Std Dev: {r_e.std():.4f}")
    print(f"Skewness: {skew(r_e):.4f}")
    print(f"Kurtosis: {kurtosis(r_e):.4f}")

    print("\n--- Histogram of G_e ---")
    counts, bin_edges = np.histogram(g_e, bins=10)
    for i in range(len(counts)):
        print(f"Range [{bin_edges[i]:.2e}, {bin_edges[i+1]:.2e}]: {counts[i]}")

    print("\n--- Zero Analysis ---")
    # Check for values close to zero (Large R)
    small_g_threshold = 1e-4 
    near_zero = g_e[g_e.abs() < small_g_threshold]
    print(f"Number of G_e values with abs(G) < {small_g_threshold} (implies |R| > {1/small_g_threshold}): {len(near_zero)}")

    print("\n--- Sample Data Slice (P, Q constant) ---")
    if 'P' in df_filtered.columns and 'Q' in df_filtered.columns:
        mode_p = df_filtered['P'].mode()[0]
        mode_q = df_filtered['Q'].mode()[0]
        df_slice = df_filtered[(df_filtered['P'] == mode_p) & (df_filtered['Q'] == mode_q)].sort_values('xi')
        
        # Select 5 evenly spaced points
        if len(df_slice) >= 5:
            indices = np.linspace(0, len(df_slice)-1, 5, dtype=int)
            sample = df_slice.iloc[indices]
            print(f"Slice for P={mode_p}, Q={mode_q}")
            print(f"{'xi':>10} | {'R_e':>15} | {'G_e':>15}")
            print("-" * 45)
            for _, row in sample.iterrows():
                print(f"{row['xi']:10.4f} | {row[r_col]:15.4f} | {row['G_e']:15.6e}")

    print("\n--- Error Amplification Analysis ---")
    # Simulation: Predict G with error epsilon = 1e-6
    epsilon = 1e-6
    
    # True G
    G_true = g_e
    # Predicted G (perturbed)
    G_pred = G_true + epsilon
    
    # Invert back to R
    R_pred = 1.0 / G_pred
    R_true = df_filtered[r_col]
    
    abs_error_R = np.abs(R_pred - R_true)
    
    print(f"Assuming constant prediction error in G of epsilon = {epsilon}")
    print(f"Max Absolute Error in R: {abs_error_R.max():.4f}")
    print(f"Mean Absolute Error in R: {abs_error_R.mean():.4f}")
    print(f"R at Max Error: {R_true.loc[abs_error_R.idxmax()]:.4f}")

if __name__ == "__main__":
    analyze_r_vs_g()
