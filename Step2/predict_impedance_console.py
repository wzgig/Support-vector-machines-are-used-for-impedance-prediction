"""
Impedance Prediction Console Tool
=================================
Interactive tool to predict circuit parameters (R_a, R_b, R_c, R_d, R_e, R_f)
using pre-trained SVM models.

Features:
- Validates model existence.
- Converts input xi from radians to degrees (as required by models).
- Applies specific feature engineering per branch.
- Outputs predictions for all branches.

Usage:
    Run this script and follow the prompts.
    Exit with 'q' or 'exit'.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# ==========================================
# 配置区域 (Configuration)
# ==========================================

# 1. 模型文件夹路径 (Model Directory)
# 默认指向当前脚本所在目录。如需指定其他路径，请修改引号内的内容。
# 例如: r"E:\path\to\models"
MODEL_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# 2. 参考数据文件路径 (Reference CSV Path)
# 指定一个包含已有工况的CSV文件，用于通过“最近邻”算法寻找 RC_Parallel 参数。
# 默认假设文件在当前目录下
REFERENCE_CSV_PATH = os.path.join(MODEL_DIR_PATH, "equivalent_circuit_parameters_optimized_accurate_Y11.csv")

# 3. 输出 CSV 文件名 (Output CSV Filename)
# 文件将生成在当前脚本所在的目录下
OUTPUT_CSV_NAME = "predicted_impedance_results.csv"

# 4. 新工况输入 (New Operating Conditions)
# 格式: [{'Case_Name': '工况1', 'P': ..., 'Q': ..., 'V': ..., 'xi': ...}, ...]
# 注意: xi 单位为弧度 (Radians)
# 您可以在此处添加任意多组工况
NEW_CONDITIONS = [
    {'Case_Name': 'Case_1', 'P': -1, 'Q': 0, 'V': 1, 'xi': 0.1}, 
    {'Case_Name': 'Case_2', 'P': -0.8, 'Q': 1, 'V': 1, 'xi': 0.1}, 
    {'Case_Name': 'Case_3', 'P': -0.8, 'Q': -1, 'V': 1, 'xi': -0.1}, 
]
# ==========================================

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ImpedancePredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        # We will load models for branches that exist
        self.branches = ['a', 'b', 'c', 'd', 'e', 'f']
        
        # Reference Data for Nearest Neighbor Search
        self.ref_csv_path = REFERENCE_CSV_PATH
        self.ref_df = None
        self.ref_stats = {} # Mean and Std for normalization
        
        if self.ref_csv_path:
            self.load_reference_data()

    def load_reference_data(self):
        """Load reference CSV and prepare for nearest neighbor search."""
        if not os.path.exists(self.ref_csv_path):
            print(f"[WARN] Reference CSV not found at: {self.ref_csv_path}")
            return

        print(f"Loading reference data from: {self.ref_csv_path}")
        try:
            df = pd.read_csv(self.ref_csv_path)
            
            # Filter for RC_Parallel rows to ensure we have R and C targets
            # Checking both 'Branch_Type' and 'Branch_ID' for robustness
            mask = (df['Branch_Type'] == 'RC_Parallel') | (df['Branch_ID'] == 'Parallel')
            self.ref_df = df[mask].copy().reset_index(drop=True)
            
            if self.ref_df.empty:
                print("[WARN] No 'RC_Parallel' or 'Parallel' branches found in reference CSV.")
                return
                
            # Compute stats for normalization of features (P, Q, V, xi)
            # Ensure columns exist
            req_cols = ['P', 'Q', 'V', 'xi']
            if not all(col in self.ref_df.columns for col in req_cols):
                 print(f"[ERROR] Reference CSV missing required columns: {req_cols}")
                 self.ref_df = None
                 return

            self.ref_stats['mean'] = self.ref_df[req_cols].mean()
            self.ref_stats['std'] = self.ref_df[req_cols].std()
            
            # Handle zero std dev to avoid division by zero
            self.ref_stats['std'] = self.ref_stats['std'].replace(0, 1.0)
            
            print(f"  [OK] Loaded {len(self.ref_df)} reference cases for RC_Parallel lookup.")
            
        except Exception as e:
            print(f"[ERROR] Failed to load reference data: {e}")

    def find_nearest_rc_parallel(self, P, Q, V, xi_deg):
        """
        Find 2 closest cases in reference data and average their R and C values.
        Returns: {'R': val, 'C': val} or None
        """
        if self.ref_df is None or self.ref_df.empty:
            return {'R': None, 'C': None}
            
        # 1. Prepare query vector
        query = pd.Series({'P': P, 'Q': Q, 'V': V, 'xi': xi_deg})
        
        # 2. Normalize query and ref data
        # (x - mean) / std
        query_norm = (query - self.ref_stats['mean']) / self.ref_stats['std']
        ref_norm = (self.ref_df[['P', 'Q', 'V', 'xi']] - self.ref_stats['mean']) / self.ref_stats['std']
        
        # 3. Calculate Euclidean distance
        # dist = sqrt(sum((ref - query)^2))
        dists = np.sqrt(((ref_norm - query_norm) ** 2).sum(axis=1))
        
        # 4. Find indexes of 2 smallest distances
        # nsmallest returns Series with index as original index
        closest_indices = dists.nsmallest(2).index
        
        if len(closest_indices) < 1:
            return {'R': None, 'C': None}
            
        # 5. Extract R and C values
        r_values = []
        c_values = []
        
        # Debug info about neighbors
        # print("  [DEBUG] Nearest Neighbors:")
        for idx in closest_indices:
            row = self.ref_df.loc[idx]
            # print(f"    idx={idx}: P={row['P']}, Q={row['Q']}, V={row['V']}, xi={row['xi']} -> R={row['R']}, C={row['C']}")
            r_values.append(pd.to_numeric(row['R'], errors='coerce'))
            c_values.append(pd.to_numeric(row['C'], errors='coerce'))
            
        # 6. Average
        # Use np.nanmean to handle potential missing values
        avg_r = np.nanmean(r_values) if r_values else None
        avg_c = np.nanmean(c_values) if c_values else None
        
        # If mean is nan (e.g. all values were nan), return None
        if np.isnan(avg_r): avg_r = None
        if np.isnan(avg_c): avg_c = None
            
        return {'R': avg_r, 'C': avg_c}
    
    def load_models(self):
        """Load all available .pkl models for branches (Result R & L)."""
        print("正在加载模型...")
        loaded_count = 0
        for branch in self.branches:
            # Load R model
            filename_r = f"svm_model_ArcSinh_Impedance_R_{branch}.pkl"
            # Try loading R model (new naming)
            path_r = os.path.join(self.model_dir, filename_r)
            if not os.path.exists(path_r):
                 # Try loading R model (old naming fallback)
                 filename_r_old = f"svm_model_ArcSinh_Impedance_{branch}.pkl"
                 path_r = os.path.join(self.model_dir, filename_r_old)

            if os.path.exists(path_r):
                try:
                    self.models[f"R_{branch}"] = joblib.load(path_r)
                    print(f"  [OK] Loaded model for R_{branch}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to load model for R_{branch}: {e}")
            else:
                print(f"  [MISSING] Model file not found: {filename_r}")

            # Load L model
            filename_l = f"svm_model_ArcSinh_Impedance_L_{branch}.pkl"
            path_l = os.path.join(self.model_dir, filename_l)
            
            if os.path.exists(path_l):
                try:
                    self.models[f"L_{branch}"] = joblib.load(path_l)
                    print(f"  [OK] Loaded model for L_{branch}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  [ERROR] Failed to load model for L_{branch}: {e}")
            else:
                print(f"  [MISSING] Model file not found: {filename_l}")
        
        if loaded_count == 0:
            print("\n警告: 没有加载到任何模型！请检查路径。")
            
    def _create_base_df(self, P, Q, V, xi_deg):
        """Create the base DataFrame with input values."""
        data = {
            'P': [float(P)],
            'Q': [float(Q)],
            'V': [float(V)],
            'xi': [float(xi_deg)]
        }
        return pd.DataFrame(data)

    def _engineer_features_a(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering for Branch 'a':
        cols: P, Q, V, xi, Z_mag, Phase, Y_mag, xi_2
        """
        X = df_input.copy()
        
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)
        X['xi_2'] = X['xi'] ** 2
        
        return X[['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'Y_mag', 'xi_2']]

    def _engineer_features_common(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering for Branch 'b', 'c', 'd', 'f':
        cols: P, Q, V, xi, Z_mag, Phase, inv_xi, xi_2, inv_xi_2
        """
        X = df_input.copy()
        
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9)
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)
        
        return X[['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'inv_xi', 'xi_2', 'inv_xi_2']]

    def _engineer_features_e(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering for Branch 'e':
        cols: P, Q, V, xi, Z_mag, Phase, inv_xi, xi_2, inv_xi_2, P_div_xi, Q_div_xi, V_div_xi
        """
        X = df_input.copy()
        
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9)
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)
        
        X['P_div_xi'] = X['P'] * X['inv_xi']
        X['Q_div_xi'] = X['Q'] * X['inv_xi']
        X['V_div_xi'] = X['V'] * X['inv_xi']
        
        return X[['P', 'Q', 'V', 'xi', 'Z_mag', 'Phase', 'inv_xi', 'xi_2', 'inv_xi_2', 'P_div_xi', 'Q_div_xi', 'V_div_xi']]

    def predict(self, P, Q, V, xi_rad):
        """
        Predict parameters for all loaded branches.
        Returns:
            results (dict): { 'a': {'R': val, 'L': val}, 'b': ... }
                            Maybe includes 'Parallel': {'R': val, 'C': val}
            xi_deg (float): converted angle
        """
        # Convert radians to degrees (Models trained on degrees)
        xi_deg = np.degrees(xi_rad)
        
        # Initialize results structure
        results = {b: {'R': None, 'L': None} for b in self.branches}
        
        # --- 1. SVM Predictions for RL Series Branches ---
        # Base dataframe
        df_base = self._create_base_df(P, Q, V, xi_deg)
        
        for key, model in self.models.items():
            # key format: "R_a" or "L_a"
            parts = key.split('_')
            param_type = parts[0] # R or L
            branch = parts[1]     # a, b, ...
            
            # print(f"[DEBUG] Processing {key} (Branch: {branch}, Type: {param_type})")
            
            try:
                # Select feature engineering function
                if branch == 'a':
                    X_features = self._engineer_features_a(df_base)
                elif branch == 'e':
                    X_features = self._engineer_features_e(df_base)
                else: # b, c, d, f share common features
                    X_features = self._engineer_features_common(df_base)
                
                # Predict
                # print(f"[DEBUG] Features for {key}: {X_features.columns.tolist()}")
                pred = model.predict(X_features)[0]
                # print(f"[DEBUG] Prediction for {key}: {pred}")
                results[branch][param_type] = pred
                
            except Exception as e:
                # print(f"[ERROR] details predicting {key}: {e}")
                logger.error(f"Error predicting {key}: {e}")
        
        # --- 2. Nearest Neighbor Lookup for RC Parallel Branch ---
        if self.ref_df is not None:
             rc_result = self.find_nearest_rc_parallel(P, Q, V, xi_deg)
             if rc_result['R'] is not None or rc_result['C'] is not None:
                 results['Parallel'] = rc_result
        
        return results, xi_deg

    def save_to_csv(self, filename: str, case_name: str, P, Q, V, xi_rad, results):
        """
        Appends prediction results to a CSV file.
        Includes RC_Parallel results if available.
        Format similar to equivalent_circuit_parameters_optimized_accurate_Y11.csv
        Cols: Case_Name, P, Q, V, xi, Branch_Type, Branch_ID, R, L, C
        """
        file_exists = os.path.isfile(filename)
        
        rows = []
        xi_deg = np.degrees(xi_rad)
        
        # 1. Process standard RL branches (a-f)
        for branch in self.branches:
            if branch not in results: continue
            
            r_val = results[branch].get('R')
            l_val = results[branch].get('L')
            
            # Skip if both are None
            if r_val is None and l_val is None:
                continue
                
            row = {
                'Case_Name': case_name,
                'P': P, 'Q': Q, 'V': V, 'xi': xi_deg,
                'Branch_Type': 'RL_Series',
                'Branch_ID': branch,
                'R': r_val if r_val is not None else '',
                'L': l_val if l_val is not None else '',
                'C': '' # RL_Series doesn't have C
            }
            rows.append(row)
            
        # 2. Process RC Parallel branch (if exists)
        if 'Parallel' in results:
             r_val = results['Parallel'].get('R')
             c_val = results['Parallel'].get('C')
             
             row_rc = {
                'Case_Name': case_name,
                'P': P, 'Q': Q, 'V': V, 'xi': xi_deg,
                'Branch_Type': 'RC_Parallel',
                'Branch_ID': 'Parallel',
                'R': r_val if r_val is not None else '',
                'L': '', # RC_Parallel doesn't have L
                'C': c_val if c_val is not None else ''
             }
             rows.insert(0, row_rc) # Insert at top like reference file often does for Y11
            
        df = pd.DataFrame(rows)
        
        # column order
        cols = ['Case_Name', 'P', 'Q', 'V', 'xi', 'Branch_Type', 'Branch_ID', 'R', 'L', 'C']
        df = df[cols]
        
        mode = 'a' if file_exists else 'w'
        header = not file_exists
        
        try:
            df.to_csv(filename, mode=mode, header=header, index=False)
            print(f"  [保存成功] 已追加到 {filename}")
        except Exception as e:
            print(f"  [保存失败] {e}")

def main():
    # Use config from the top of the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = MODEL_DIR_PATH
    
    # Check if model directory exists
    if not os.path.isdir(model_dir):
        # Fallback to current script directory if the configured path is invalid
        print(f"[WARN] Configured model path not found: {model_dir}")
        print(f"       Trying script directory: {script_dir}")
        model_dir = script_dir

    print("==========================================")
    print(" SVR 阻抗预测工具 (Batch Processing)")
    print("==========================================")
    print(f"模型路径: {model_dir}")
    
    predictor = ImpedancePredictor(model_dir)
    predictor.load_models()
    
    if not predictor.models:
        print("错误: 未能加载任何模型。请确认 .pkl 文件在上述路径中。")
        return

    # Construct output file path
    output_file_path = os.path.join(script_dir, OUTPUT_CSV_NAME)
    print(f"输出文件路径: {output_file_path}")

    print("\n------------------------------------------")
    print(f"开始处理 {len(NEW_CONDITIONS)} 组工况...")

    success_count = 0
    
    for i, condition in enumerate(NEW_CONDITIONS):
        try:
            # Use provided Case_Name or default to sequential name
            case_name = condition.get('Case_Name', f"Condition_{i+1}")
            P = float(condition['P'])
            Q = float(condition['Q'])
            V = float(condition['V'])
            xi_rad = float(condition['xi'])
            xi_deg = np.degrees(xi_rad)
            
            print(f"\n[工况 {i+1}/{len(NEW_CONDITIONS)}] - {case_name}")
            print(f"  输入: P={P}, Q={Q}, V={V}, xi={xi_rad} rad ({xi_deg:.2f} deg)")
            
            # Predict
            predictions, _ = predictor.predict(P, Q, V, xi_rad)
            
            # Display results briefly
            print("  预测结果摘要 (Branch R/L):")
            for branch in sorted(predictions.keys()):
                 r_val = predictions[branch]['R']
                 if r_val is not None:
                     if branch == 'Parallel':
                         print(f"    Branch {branch}: R={r_val:.4e}, C={predictions[branch]['C']:.6e}")
                     else:
                         print(f"    Branch {branch}: R={r_val:.4f}, L={predictions[branch]['L']:.6f}")

            # Save to CSV
            predictor.save_to_csv(output_file_path, case_name, P, Q, V, xi_rad, predictions)
            success_count += 1
            
        except Exception as e:
            print(f"  [ERROR] 处理工况 {i+1} 失败: {e}")

    print("\n------------------------------------------")
    print(f"处理完成。成功: {success_count}, 失败: {len(NEW_CONDITIONS) - success_count}")
    print(f"结果已保存至: {output_file_path}")

if __name__ == "__main__":
    main()
