
import pandas as pd
import numpy as np
import os

file_path = r'e:\ruanjian\GitHubDesktop\Support-vector-machines-are-used-for-impedance-prediction\extracted_RL_Series_Y11_wide.csv'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

df = pd.read_csv(file_path)

print(f"Total rows: {len(df)}")
print("Columns:", df.columns.tolist())

# Analyze R_b
if 'R_b' not in df.columns:
    print("Column R_b not found!")
    exit()
    
rb = df['R_b']

print("\n--- R_b Statistics ---")
print(rb.describe())
print(f"Skewness: {rb.skew()}")
print(f"Kurtosis: {rb.kurtosis()}")

print("\n--- Quantiles ---")
print(rb.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n--- Sign Distribution ---")
print(f"Positive: {(rb > 0).sum()}")
print(f"Negative: {(rb < 0).sum()}")
print(f"Zero: {(rb == 0).sum()}")

# Correlations
print("\n--- Correlation with Features ---")
features = ['P', 'Q', 'V', 'xi']
found_features = [f for f in features if f in df.columns]
print(df[found_features + ['R_b']].corr()['R_b'])

# Check for extreme values
threshold = 1000
print(f"\n--- Extreme Values (|val| > {threshold}) ---")
extremes = df[np.abs(rb) > threshold]
print(f"Count: {len(extremes)}")
if len(extremes) > 0:
    print(extremes[['P', 'Q', 'V', 'xi', 'R_b']].head(10))
    print("\n--- 'xi' distribution in extremes ---")
    print(extremes['xi'].describe())
