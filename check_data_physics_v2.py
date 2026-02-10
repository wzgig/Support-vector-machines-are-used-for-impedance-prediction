
import pandas as pd
import numpy as np
import os

# Load data
csv_path = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
if not os.path.exists(csv_path):
    if os.path.exists("csv_data/" + csv_path):
        csv_path = "csv_data/" + csv_path
    elif os.path.exists("equivalent_circuit_parameters_optimized_Y11.csv"):
        csv_path = "equivalent_circuit_parameters_optimized_Y11.csv"

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Filter like the training script
TARGET_BRANCH_TYPE = "RL_Series"
TARGET_BRANCH_ID = "e"
mask = (df['Branch_Type'] == TARGET_BRANCH_TYPE) & (df['Branch_ID'] == TARGET_BRANCH_ID)
df = df[mask].copy()

if df.empty:
    print("Error: Filtered dataframe is empty.")
    exit()

# 1. Check Physical Theory: R_calc = V^2 * P / (P^2 + Q^2)
P = df['P']
Q = df['Q']
V = df['V']
S2 = P**2 + Q**2
R_true = df['R']

# Calculate theoretical R assuming it's the real part of Z seen from terminals
R_theo = (V**2 * P) / (S2 + 1e-9)

# Analyze correlation
corr = np.corrcoef(R_true, R_theo)[0,1]
print(f"\nCorrelation between True R and Theoretical R (V^2*P/S^2): {corr:.4f}")

# 2. Analyze R vs Xi (Singularities)
print("\n--- R vs Xi Analysis ---")
print(df[['xi', 'R']].describe())

# Check for zero crossings or poles in R
print(f"Count R > 0: {(df['R'] > 0).sum()}")
print(f"Count R < 0: {(df['R'] < 0).sum()}")

# Check dynamic range
log_R_abs = np.log10(df['R'].abs() + 1e-9)
print(f"Log10(|R|) Range: {log_R_abs.min():.2f} to {log_R_abs.max():.2f}")

# Look at top errors from SVM output (manual check based on user log)
print("\nTop 5 Largest |R|:")
print(df['R'].abs().nlargest(5))

# Check samples where R is huge
huge_R_indices = df[df['R'].abs() > 1e5].index
print(f"\nNumber of samples with |R| > 100,000: {len(huge_R_indices)}")
if len(huge_R_indices) > 0:
    print(df.loc[huge_R_indices, ['P', 'Q', 'V', 'xi', 'R']].head())
