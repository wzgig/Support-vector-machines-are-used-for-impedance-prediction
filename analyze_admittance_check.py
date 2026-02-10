import pandas as pd
import numpy as np
import sys

# Load Data
try:
    df = pd.read_csv('equivalent_circuit_parameters_optimized_accurate_Y11.csv')
    mask = (df['Branch_Type'] == 'RL_Series') & (df['Branch_ID'] == 'e')
    df = df[mask].copy()
    
    # Calculate G (Admittance)
    # Handle division by zero if R is exactly 0 (unlikely for float but good practice)
    df['G'] = 1.0 / df['R']

    print("-" * 30)
    print("ANALYSIS OF G (Admittance = 1/R)")
    print("-" * 30)
    print(df['G'].describe())
    print(f"Skewness: {df['G'].skew():.4f}")
    print(f"Kurtosis: {df['G'].kurt():.4f}")
    
    # Correlation with xi
    # If R ~ 1/xi, then G ~ xi. Correlation should be very high (linear).
    corr_xi = df[['xi', 'G']].corr().iloc[0, 1]
    print(f"Correlation (xi vs G): {corr_xi:.4f}")
    
    # Check for singularities in G (values near infinity?)
    # G would be infinite if R is 0.
    # Check max abs G
    print(f"\nMax Abs G: {df['G'].abs().max():.4e}")
    
    # Compare with R
    print("\n" + "-" * 30)
    print("COMPARISON WITH R")
    print("-" * 30)
    print(f"R Skewness: {df['R'].skew():.4f}")
    print(f"R Kurtosis: {df['R'].kurt():.4f}")
    print(f"Correlation (xi vs R): {df[['xi', 'R']].corr().iloc[0, 1]:.4f}") # Expected to be low due to singularity
    
    # Check if a linear model fits G better
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = df[['xi']]
    y = df['G']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"\nLinear Regression (G ~ xi) R2 Score: {r2_score(y, y_pred):.4f}")
    
except Exception as e:
    print(f"Error: {e}")
