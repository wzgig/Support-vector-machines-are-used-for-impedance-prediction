
import pandas as pd
import numpy as np
import os
import sys

# 设置输入路径
INPUT_CSV = os.path.join("csv_data", "equivalent_circuit_parameters_optimized_Y11.csv")
if not os.path.exists(INPUT_CSV):
    INPUT_CSV = "equivalent_circuit_parameters_optimized_Y11.csv"
    if not os.path.exists(INPUT_CSV):
         INPUT_CSV = r"e:\ruanjian\GitHubDesktop\Support-vector-machines-are-used-for-impedance-prediction\equivalent_circuit_parameters_optimized_Y11.csv"

def inspect_data():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File not found at {INPUT_CSV}")
        return

    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Filter Parallel branch
    branch_name = "Parallel"
    df_part = df[df['Branch_ID'] == branch_name].copy()
    
    if len(df_part) == 0:
        print(f"No Branch_ID found for {branch_name}")
        return

    R = df_part['R']
    
    print("\n--- Basic Statistics of R (Parallel) ---")
    print(R.describe())
    
    # Check negative values
    neg_count = (R < 0).sum()
    print(f"\nNegative count: {neg_count} ({neg_count/len(R)*100:.2f}%)")
    
    # Check "High Impedance" candidates (> 1e9)
    high_imp_count = (R > 1e9).sum()
    print(f"High Impedance (>1e9) count: {high_imp_count} ({high_imp_count/len(R)*100:.2f}%)")
    
    # Check "High Negative" candidates (< -1e9)
    high_neg_count = (R < -1e9).sum()
    print(f"High Negative (< -1e9) count: {high_neg_count} ({high_neg_count/len(R)*100:.2f}%)")

    # Apply Arcsinh to see distribution
    R_trans = np.arcsinh(R)
    print("\n--- Statistics of Arcsinh(R) ---")
    print(R_trans.describe())

    # Correlations
    features = ['P', 'Q', 'V', 'xi']
    print("\n--- Correlation with Features (using Arcsinh(R)) ---")
    df_part['R_trans'] = R_trans
    corr = df_part[features + ['R_trans']].corr()['R_trans']
    print(corr)

if __name__ == "__main__":
    inspect_data()
