import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_CSV = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
MODEL_PATH = "svm_model_G_Admittance.pkl"
TARGET_BRANCH_TYPE = "RL_Series"
TARGET_BRANCH_ID = "e"
FEATURES = ['P', 'Q', 'V', 'xi']

def main():
    # Load Data
    try:
        df = pd.read_csv(INPUT_CSV)
    except:
        df = pd.read_csv("equivalent_circuit_parameters_optimized_Y11.csv")
        
    mask = (df['Branch_Type'] == TARGET_BRANCH_TYPE) & (df['Branch_ID'] == TARGET_BRANCH_ID)
    df_filtered = df[mask].copy()

    # Recreate Features
    X = df_filtered[FEATURES].copy()
    X['S_mag'] = np.sqrt(X['P']**2 + X['Q']**2)
    X['Z_mag'] = X['V']**2 / (X['S_mag'] + 1e-9)
    # Be careful to match exact features from training
    X['Phase'] = np.arctan2(X['Q'], X['P'])
    
    # Missing feature that caused error
    X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)

    X['inv_xi_0.1'] = 1.0 / (X['xi'].abs() + 0.1)
    X['inv_xi_0.5'] = 1.0 / (X['xi'].abs() + 0.5)
    X.drop(columns=['S_mag'], inplace=True)

    # Target G
    G = 1.0 / df_filtered['R']
    R_true = df_filtered['R']

    # Split (Same random_state as training)
    indices = np.arange(len(X))
    X_train, X_test, idx_train, idx_test = train_test_split(X, indices, test_size=0.2, random_state=42)
    
    G_train = G.iloc[idx_train]
    G_test = G.iloc[idx_test]
    R_test = R_true.iloc[idx_test]

    # Load Model
    reg = joblib.load(MODEL_PATH)
    
    # Predict
    G_train_pred = reg.predict(X_train)
    G_test_pred = reg.predict(X_test)

    # Metrics on G
    print(f"Train R2 (G): {r2_score(G_train, G_train_pred):.4f}")
    print(f"Test R2 (G): {r2_score(G_test, G_test_pred):.4f}")
    
    # Metrics on R
    # Invert G_pred to get R_pred
    epsilon = 1e-7
    G_test_pred_safe = np.where(np.abs(G_test_pred) < epsilon, np.sign(G_test_pred)*epsilon, G_test_pred)
    G_test_pred_safe = np.where(G_test_pred_safe == 0, epsilon, G_test_pred_safe)
    
    R_pred = 1.0 / G_test_pred_safe
    
    r2_R = r2_score(R_test, R_pred)
    print(f"Test R2 (converted to R): {r2_R:.4f}")
    
    # Relative Error Stats
    diff = np.abs(R_test - R_pred)
    rel_err = diff / (np.abs(R_test) + 1e-9)
    print(f"MAPE (R): {np.mean(rel_err)*100:.2f}%")
    print(f"Median APE (R): {np.median(rel_err)*100:.2f}%")
    
    # Check "Physical" consistency
    # Where R is large, G should be small.
    # Check singular points
    print("\n[Analysis of Poles]")
    # Find points where true R is huge (>100k)
    high_R_mask = np.abs(R_test) > 100000
    if high_R_mask.sum() > 0:
        mape_high = np.mean(rel_err[high_R_mask]) * 100
        print(f"High Impedance Samples (R>100k) Count: {high_R_mask.sum()}")
        print(f"High Impedance MAPE: {mape_high:.2f}%")
        print(f"High Impedance R2: {r2_score(R_test[high_R_mask], R_pred[high_R_mask]):.4f}")

if __name__ == "__main__":
    main()
