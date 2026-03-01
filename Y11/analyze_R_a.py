
import pandas as pd
import numpy as np
import os

def analyze(filepath, name):
    print(f"\n--- {name} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows.")
        
        target_col = 'R_a'
        if target_col not in df.columns:
            print(f"Column {target_col} not found!")
            return

        series = df[target_col]
        
        # Basic counts
        n_total = len(series)
        n_nan = series.isna().sum()
        n_pos = (series > 0).sum()
        n_zero = (series == 0).sum()
        n_neg = (series < 0).sum()
        
        print(f"Total: {n_total}")
        print(f"NaN: {n_nan}")
        print(f"Positive (>0): {n_pos} ({n_pos/n_total:.1%})")
        print(f"Zero (=0): {n_zero}")
        print(f"Negative (<0): {n_neg} ({n_neg/n_total:.1%})")
        
        # Distribution of negative values
        if n_neg > 0:
            neg_vals = series[series < 0]
            print("Negative Value Statistics:")
            print(neg_vals.describe())
            
            # Boxplot logic
            Q1 = neg_vals.quantile(0.25)
            Q3 = neg_vals.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((neg_vals < lower) | (neg_vals > upper)).sum()
            print(f"Outliers (IQR 1.5x): {outliers}")
            print(f"IQR Range: [{lower:.4f}, {upper:.4f}]")

    except Exception as e:
        print(f"Error: {e}")

analyze("Y11/Ideal_Power_Grid_Train_Processed_Y11_RL_Wide.csv", "Training Set")
analyze("Y11/Ideal_Power_Grid_Test_Processed_Y11_RL_Wide.csv", "Testing Set")
