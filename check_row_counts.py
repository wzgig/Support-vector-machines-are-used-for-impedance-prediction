import pandas as pd
import os

files = [
    "equivalent_circuit_parameters_optimized_accurate_Y11.csv",
    "equivalent_circuit_parameters_optimized_accurate_Y12.csv",
    "equivalent_circuit_parameters_optimized_accurate_Y21.csv",
    "equivalent_circuit_parameters_optimized_accurate_Y22.csv"
]

for f in files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
        
    df = pd.read_csv(f)
    print(f"--- {f} ---")
    print(f"Total rows: {len(df)}")
    
    unique_files = df['Filename'].unique()
    print(f"Unique source files: {len(unique_files)}")
    
    # Count Branch Types
    branch_counts = df['Branch_Type'].value_counts()
    print("Branch Types:")
    print(branch_counts)
    
    # Check rows per file
    rows_per_file = df.groupby('Filename').size()
    print(f"Most common row count per file: {rows_per_file.value_counts()}")
    
    # Identify files with unusual row counts
    mode_count = rows_per_file.mode()[0]
    unusual_files = rows_per_file[rows_per_file != mode_count]
    if not unusual_files.empty:
        print(f"Files with unusual row counts (not {mode_count}):")
        print(unusual_files)
    else:
        print(f"All files have {mode_count} rows.")
    print("\n")
