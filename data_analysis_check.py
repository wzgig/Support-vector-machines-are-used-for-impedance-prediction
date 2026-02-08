import pandas as pd
import numpy as np
import os

INPUT_CSV = "equivalent_circuit_parameters_optimized_Y11.csv"

def analyze_data():
    if not os.path.exists(INPUT_CSV):
        print("File not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Reconstruct the pivot logic to see the actual training matrix Y
    unique_branches = df['Branch_ID'].unique()
    
    # Same sorting as before
    def branch_sorter(b_id):
        b_id = str(b_id)
        if b_id == 'Parallel': return (0, b_id)
        elif b_id.isalpha():   return (1, b_id)
        else:                  return (2, b_id)
            
    sorted_template = sorted(unique_branches, key=branch_sorter)
    
    pivot_df = df.pivot_table(
        index=['Filename', 'P', 'Q', 'V', 'xi'], 
        columns='Branch_ID', 
        values='R'
    )
    
    pivot_df = pivot_df.reindex(columns=sorted_template)
    
    # Analyze sparsity before filling
    print(f"{'Branch':<15} | {'Non-NaN Count':<15} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 80)
    
    for col in pivot_df.columns:
        series = pivot_df[col]
        non_nan = series.count()
        if non_nan > 0:
            print(f"{str(col):<15} | {non_nan:<15} | {series.min():.2e}   | {series.max():.2e}   | {series.mean():.2e}   | {series.std():.2e}")
        else:
            print(f"{str(col):<15} | 0")

    # Analyze Zero mixing
    filled_df = pivot_df.fillna(0.0)
    total_rows = len(filled_df)
    
    print("\n--- Zero Padding Impact ---")
    print(f"Total Samples (Rows): {total_rows}")
    print(f"{'Branch':<15} | {'Zeros (Missing)':<15} | {'% Missing':<10}")
    print("-" * 50)
    for col in filled_df.columns:
        zeros = (filled_df[col] == 0).sum()
        print(f"{str(col):<15} | {zeros:<15} | {zeros/total_rows*100:.1f}%")

if __name__ == "__main__":
    analyze_data()