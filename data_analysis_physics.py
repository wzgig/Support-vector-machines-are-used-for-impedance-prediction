import pandas as pd
import numpy as np

INPUT_CSV = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
TARGET_BRANCH_TYPE = "RL_Series"
TARGET_BRANCH_ID = "e"

print(f"Reading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
mask = (df['Branch_Type'] == TARGET_BRANCH_TYPE) & (df['Branch_ID'] == TARGET_BRANCH_ID)
df = df[mask].copy()

# Add physical features
df['S_mag'] = np.sqrt(df['P']**2 + df['Q']**2)
df['Z_mag'] = df['V']**2 / (df['S_mag'] + 1e-9)
df['Y_mag'] = df['S_mag'] / (df['V']**2 + 1e-9)
df['Phase'] = np.arctan2(df['Q'], df['P'])
df['Q_over_P'] = df['Q'] / (df['P'] + 1e-9)

# Define extreme
threshold = 10000
df['is_extreme'] = df['R'].abs() > threshold

print("\n--- Correlation with R (All Data) ---")
print(df[['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'R']].corr()['R'].sort_values())

print("\n--- Correlation with Abs(R) (All Data) ---")
# Check what correlates with the MAGNITUDE of R
print(df[['P', 'Q', 'V', 'xi', 'inv_abs_xi', 'Z_mag', 'Phase']].assign(abs_R=df['R'].abs()).corr()['abs_R'].sort_values())

# Check extremes specifically
print("\n--- Extreme R samples features ---")
print(df[df['is_extreme']][['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'R']].head(5))

print("\n--- Normal R samples features ---")
print(df[~df['is_extreme']][['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'R']].head(5))
