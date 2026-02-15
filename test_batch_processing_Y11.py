import os
import glob
import random
import pandas as pd
import numpy as np
import VF # Assumes VF.py is in the same directory

# Configuration mimicking batch_processing.py
INPUT_DIR = os.path.join("your_root", "csv_data")
TARGET_ELEMENT = "Y11"

def test_Y11_processing():
    print(f"Looking for CSV files in: {INPUT_DIR}")
    
    # 1. Find files
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {os.path.abspath(INPUT_DIR)}")
        # Check if dir exists
        if os.path.exists(INPUT_DIR):
            print("Directory exists but is empty of .csv files.")
        else:
            print("Directory does not exist.")
        return

    # 2. Pick random files (e.g., 3 random cases)
    num_samples = 3
    sample_files = random.sample(csv_files, min(num_samples, len(csv_files)))

    print(f"Found {len(csv_files)} files. Selected {len(sample_files)} random files for verification testing.\n")

    for i, fpath in enumerate(sample_files):
        print(f"===========================================================")
        print(f"TEST CASE {i+1}: {os.path.basename(fpath)}")
        print(f"===========================================================")
        
        # Load Data
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                print("Skipping empty file.")
                continue
        except Exception as e:
            print(f"Failed to read file: {e}")
            continue

        real_col = f"{TARGET_ELEMENT}_Real"
        imag_col = f"{TARGET_ELEMENT}_Imag"

        if 'Frequency_Hz' not in df.columns or real_col not in df.columns or imag_col not in df.columns:
            print(f"Missing required columns for {TARGET_ELEMENT}. Available: {df.columns.tolist()}")
            continue

        # Prepare Data
        freq_hz = df['Frequency_Hz'].values
        # Note: In batch_processing.py s_vec = 1j * 2 * np.pi * freq_hz
        s_vec = 1j * 2 * np.pi * freq_hz
        real_vals = df[real_col].values
        imag_vals = df[imag_col].values
        f_vec = real_vals + 1j * imag_vals
        
        # Check validation
        if np.isnan(f_vec).any():
            print("Data contains NaNs. Skipping.")
            continue
            
        print(f"[Data Info]")
        print(f"  Element: {TARGET_ELEMENT}")
        print(f"  Samples: {len(freq_hz)}")
        print(f"  Freq Range: {np.min(freq_hz):.1f} Hz - {np.max(freq_hz):.1f} Hz")
        print(f"  Y11 Magnitude Range: {np.min(np.abs(f_vec)):.4e} - {np.max(np.abs(f_vec)):.4e}")

        # Step 1: Vector Fitting
        print(f"\n[Step 1: Vector Fitting]")
        print("  Calling VF.vectfit_find_best_order with:")
        print("    min_poles=3, max_poles=3, weighting_policy='none'")
        
        poles, residues, d, h, metrics = VF.vectfit_find_best_order(
            f_vec, s_vec, 
            min_poles=3, max_poles=3, step=1,
            target_error=1e-5, 
            weighting_policy='none',
            silent=True
        )
        
        print(f"  -> Fitting Completed.")
        print(f"  -> Selected Order: {len(poles)}")
        print(f"  -> RMS Relative Error: {metrics['rms_rel']:.6%}")
        print(f"  -> Max Relative Error: {metrics['max_rel']:.6%}")
        
        # Step 2: Passivity Check
        print(f"\n[Step 2: Passivity Check]")
        is_passive, min_real, viol_freq = VF.check_passivity(s_vec, poles, residues, d, h)
        passivity_str = "PASS" if is_passive else "FAIL"
        print(f"  -> Passivity Status: {passivity_str}")
        print(f"  -> Min Real Part: {min_real:.4e}")
        if not is_passive:
             print(f"  -> Violation Freq: {viol_freq}")

        # Step 3: Equivalent Circuit Synthesis
        print(f"\n[Step 3: Equivalent Circuit Synthesis]")
        analyzer = VF.SystemAnalyzer()
        analyzer.load_fitting_result(poles, residues, d, h)
        
        # Output Results mimics batch_processing logic
        print("  -> Circuit Topology Extracted:")
        
        has_params = False
        
        # Parallel RC
        if analyzer.output_data['rc_params']:
            p = analyzer.output_data['rc_params']
            print(f"    [Parallel RC] R = {p['R']:.4e} Ohm, C = {p['C']:.4e} F")
            has_params = True
        
        # Series RL
        if analyzer.output_data['rl_params']:
            for p in analyzer.output_data['rl_params']:
                 print(f"    [Series RL]   ID={p['id']}: R = {p['R']:.4e} Ohm, L = {p['L']:.4e} H")
            has_params = True

        # Series RLC
        if analyzer.output_data['rlc_params']:
            for p in analyzer.output_data['rlc_params']:
                 print(f"    [Series RLC]  ID={p['id']}: R = {p['R']:.4e} Ohm, L = {p['L']:.4e} H, C = {p['C']:.4e} F")
            has_params = True
            
        if not has_params:
            print("    (No circuit parameters extracted)")

        print("\n")

if __name__ == "__main__":
    test_Y11_processing()
