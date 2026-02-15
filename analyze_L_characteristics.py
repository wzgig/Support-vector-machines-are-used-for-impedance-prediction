import pandas as pd
import os

file_path = 'e:\\ruanjian\\GitHubDesktop\\Support-vector-machines-are-used-for-impedance-prediction\\equivalent_circuit_parameters_optimized_accurate_Y11.csv'

if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Filter for RL_Series
        rl_series_df = df[df['Branch_Type'] == 'RL_Series']
        
        # Group by Branch_ID and calculate stats for 'L'
        stats = rl_series_df.groupby('Branch_ID')['L'].agg(['count', 'mean', 'std', 'min', 'max'])
        
        print("L Statistics by Branch_ID for RL_Series:")
        print(stats)
        
        # Check for NaN or Inf
        print("\nMissing or Infinite values in L:")
        print(rl_series_df['L'].isna().sum())
        
        # Display some info about range magnitude
        print("\nMagnitude analysis:")
        print(rl_series_df.groupby('Branch_ID')['L'].apply(lambda x: x.abs().describe()))

    except Exception as e:
        print(f"Error analyzing file: {e}")
else:
    print(f"File not found: {file_path}")
