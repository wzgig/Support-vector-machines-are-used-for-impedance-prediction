
import pandas as pd
import numpy as np

def analyze_re_distribution():
    file_path = "extracted_RL_Series_Y11_wide.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}, shape: {df.shape}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return

    # Check columns
    if "R_e" not in df.columns:
        print("Column R_e not found.")
        return
    
    # Filter R_e based on current existing logic [-90000, 90000]
    # We want to see what happens INSIDE this range and OUTSIDE too.
    
    print("\n--- Original Data Stats (R_e) ---")
    print(df["R_e"].describe())
    
    current_min, current_max = -90000, 90000
    df_filtered = df[(df["R_e"] >= current_min) & (df["R_e"] <= current_max)].copy()
    
    print(f"\n--- Data within [{current_min}, {current_max}] ---")
    print(f"Count: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.2f}%)")
    print(df_filtered["R_e"].describe())
    
    # Analyze relationship with xi
    # R_e ~ 1/xi. Let's look at xi values where |R_e| is large
    
    # Define "Large" R_e within the filtered set (e.g., > 50000 or < -50000)
    high_threshold = 50000
    large_R = df_filtered[abs(df_filtered["R_e"]) > high_threshold]
    
    print(f"\n--- Large |R_e| (> {high_threshold}) Analysis ---")
    print(f"Count: {len(large_R)}")
    if len(large_R) > 0:
        print("Xi stats for large R_e:")
        print(large_R["xi"].describe())
        
        # Check if they cluster near xi=0
        near_zero_xi_count = len(large_R[abs(large_R["xi"]) < 100]) # Assuming xi units, check threshold
        print(f"Number of large R_e samples with |xi| < 100: {near_zero_xi_count}")

    # Percentile analysis
    print("\n--- Percentile Analysis for Filtering ---")
    # Let's look at 1% and 99% quantiles
    q01 = df_filtered["R_e"].quantile(0.01)
    q05 = df_filtered["R_e"].quantile(0.05)
    q95 = df_filtered["R_e"].quantile(0.95)
    q99 = df_filtered["R_e"].quantile(0.99)
    
    print(f"1% Quantile: {q01:.2f}")
    print(f"5% Quantile: {q05:.2f}")
    print(f"95% Quantile: {q95:.2f}")
    print(f"99% Quantile: {q99:.2f}")
    
    # Check xi distribution for the extremes (top 1% and bottom 1%)
    extreme_bottom = df_filtered[df_filtered["R_e"] < q01]
    extreme_top = df_filtered[df_filtered["R_e"] > q99]
    
    print("\n--- Xi characteristics of extremes ---")
    if not extreme_bottom.empty:
        print(f"Bottom 1% R_e (avg {extreme_bottom['R_e'].mean():.2f}) has mean |xi|: {abs(extreme_bottom['xi']).mean():.2f}")
    if not extreme_top.empty:
        print(f"Top 1% R_e (avg {extreme_top['R_e'].mean():.2f}) has mean |xi|: {abs(extreme_top['xi']).mean():.2f}")

    # Check regular data (middle 90%)
    middle = df_filtered[(df_filtered["R_e"] > q05) & (df_filtered["R_e"] < q95)]
    print(f"\n--- Middle 90% Data (R_e between {q05:.2f} and {q95:.2f}) ---")
    print(f"Count: {len(middle)}")
    print(f"Std Dev: {middle['R_e'].std():.2f}")
    
    IQR = df_filtered["R_e"].quantile(0.75) - df_filtered["R_e"].quantile(0.25)
    print(f"IQR of filtered data: {IQR:.2f}")

if __name__ == "__main__":
    analyze_re_distribution()
