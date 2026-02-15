import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load data
file_path = "extracted_RL_Series_Y11_wide.csv"
if not os.path.exists(file_path):
    file_path = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"

df = pd.read_csv(file_path)
print(f"Columns: {df.columns[:10]}")

# Filter for R_e
if 'R_e' in df.columns:
    df['R'] = df['R_e']
elif 'R' in df.columns: #Fallback
    df['R'] = df['R']
else:
    # Assuming long format for simplicity if needed, but context says wide is available
    print("Could not find R_e column.")
    exit()

# Filter Range (matching user's request)
df = df[(df['R'] >= -90000) & (df['R'] <= 90000)].copy()

print(f"Dataset size: {len(df)}")
print(f"R statistics: Min={df['R'].min():.2f}, Max={df['R'].max():.2f}, Mean={df['R'].mean():.2f}")
print(f"Abs(R) min: {df['R'].abs().min():.4f}")

# Check G
df['G'] = 1.0 / df['R']
print(f"G statistics: Min={df['G'].min():.4e}, Max={df['G'].max():.4e}")

# Check Correlation
corr_xi_R = df['xi'].corr(df['R'])
corr_xi_G = df['xi'].corr(df['G'])
print(f"Correlation xi vs R: {corr_xi_R:.4f}")
print(f"Correlation xi vs G: {corr_xi_G:.4f}")

# Quick ML Test
features = ['P', 'Q', 'V', 'xi']
X = df[features]
y_R = df['R']
y_G = df['G']

X_train, X_test, y_R_train, y_R_test, y_G_train, y_G_test = train_test_split(X, y_R, y_G, test_size=0.2, random_state=42)

# Train on R (ArcSinh)
print("\n--- Training on R (ArcSinh) ---")
# Mimic existing script simple version
svr_r = Pipeline([('scaler', RobustScaler()), ('svr', SVR(C=1000, gamma=0.1))])
y_R_train_trans = np.arcsinh(y_R_train)
svr_r.fit(X_train, y_R_train_trans)
y_R_pred_trans = svr_r.predict(X_test)
y_R_pred = np.sinh(y_R_pred_trans)

r2_r = r2_score(y_R_test, y_R_pred)
rmse_r = np.sqrt(mean_squared_error(y_R_test, y_R_pred))
print(f"R Model -> R2: {r2_r:.4f}, RMSE: {rmse_r:.2f}")

# Train on G
print("\n--- Training on G ---")
svr_g = Pipeline([('scaler', RobustScaler()), ('svr', SVR(C=1000, gamma=0.1))]) 
# G is small, maybe separate scaling helps, but RobustScaler handles inputs. SVR handles target scale implicitly or we scale target.
# Let's scale target G for fair comparison (multiply by 1000 to be in mS range?)
scale_factor = 1000.0
y_G_train_scaled = y_G_train * scale_factor
svr_g.fit(X_train, y_G_train_scaled)
y_G_pred_scaled = svr_g.predict(X_test)
y_G_pred = y_G_pred_scaled / scale_factor

# Invert back to R
y_R_pred_from_G = 1.0 / y_G_pred

# Handle division by zero or huge values?
# Clip G to avoid explosion?
# print stats of G_pred
print(f"G_pred min abs: {np.min(np.abs(y_G_pred))}")


r2_g = r2_score(y_R_test, y_R_pred_from_G)
rmse_g = np.sqrt(mean_squared_error(y_R_test, y_R_pred_from_G))
print(f"G Model (inverted) -> R2: {r2_g:.4f}, RMSE: {rmse_g:.2f}")

# Compare on subset of Large R
mask_large = y_R_test.abs() > 10000
if mask_large.sum() > 0:
    print(f"\n--- Large R (>10k) Performance ({mask_large.sum()} samples) ---")
    print(f"R Model RMSE: {np.sqrt(mean_squared_error(y_R_test[mask_large], y_R_pred[mask_large])):.2f}")
    print(f"G Model RMSE: {np.sqrt(mean_squared_error(y_R_test[mask_large], y_R_pred_from_G[mask_large])):.2f}")
    
# Compare on subset of Small R
mask_small = y_R_test.abs() < 1000
if mask_small.sum() > 0:
    print(f"\n--- Small R (<1k) Performance ({mask_small.sum()} samples) ---")
    print(f"R Model RMSE: {np.sqrt(mean_squared_error(y_R_test[mask_small], y_R_pred[mask_small])):.2f}")
    print(f"G Model RMSE: {np.sqrt(mean_squared_error(y_R_test[mask_small], y_R_pred_from_G[mask_small])):.2f}")
