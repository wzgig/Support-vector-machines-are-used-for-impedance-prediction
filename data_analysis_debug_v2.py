import pandas as pd
import numpy as np
import os

# Load the data
try:
    file_path = r'e:\ruanjian\GitHubDesktop\Support-vector-machines-are-used-for-impedance-prediction\equivalent_circuit_parameters_optimized_accurate_Y11.csv'
    if not os.path.exists(file_path):
        print(f"File not found at {file_path}")
        exit(1)
        
    df = pd.read_csv(file_path)
    
    # Filter for Branch_Type 'RL_Series' and Branch_ID 'e'
    branch_e = df[(df['Branch_ID'] == 'e') & (df['Branch_Type'] == 'RL_Series')].copy()
    
    # Sort by xi to see the trend
    branch_e = branch_e.sort_values(by='xi')
    
    print(f"Data for Branch_ID = 'e': {len(branch_e)} samples")
    print("-" * 80)
    print(f"{'min_real_part':>15} | {'xi':>10} | {'R':>20} | {'Abs(R)':>20} | {'log10(Abs(R))':>15}")
    print("-" * 80)
    
    # Select a subset to print: extremes (large R) and around 0 xi
    subset = branch_e.copy()
    subset['abs_R'] = subset['R'].abs()
    
    # Get top 10 largest R values
    top_large = subset.sort_values('abs_R', ascending=False).head(10)
    
    # Get values around xi = 0
    around_zero = subset[subset['xi'].abs() < 1.0].sort_values('xi')
    
    print("--- Top 10 Largest R ---")
    print(top_large[['xi', 'R', 'abs_R']].to_string())
    
    print("\n--- Data around xi=0 ---")
    print(around_zero[['xi', 'R', 'abs_R']].to_string())
    
    print("-" * 80)
    print("Statistics for R:")
    print(branch_e['R'].describe())
    
    # Check for sign changes in R
    positive_R = branch_e[branch_e['R'] > 0]
    negative_R = branch_e[branch_e['R'] < 0]
    print(f"\nPositive R samples: {len(positive_R)}")
    print(f"Negative R samples: {len(negative_R)}")
    
    # Check if P, Q, V also affect the singularity
    print("\nCorrelation with R:")
    print(branch_e[['P', 'Q', 'V', 'xi', 'R']].corr()['R'])

except Exception as e:
    print(f"Error: {e}")
