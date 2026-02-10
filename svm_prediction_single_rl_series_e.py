import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 全局配置 ---
INPUT_CSV = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
MODEL_SAVE_PATH = "svm_model_R_RL_Series_e_Specific.pkl"
TARGET_BRANCH_TYPE = "RL_Series"
TARGET_BRANCH_ID = "e"
TARGET_COL = "R"
FEATURES = ['P', 'Q', 'V', 'xi']

def load_data(csv_path):
    print(f"正在读取数据: {csv_path}")
    if not os.path.exists(csv_path):
        # 尝试使用绝对路径或相对路径的回退机制
        alt_path = os.path.join("csv_data", csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        elif os.path.exists("equivalent_circuit_parameters_optimized_Y11.csv"):
             # Fallback 
             csv_path = "equivalent_circuit_parameters_optimized_Y11.csv"

    df = pd.read_csv(csv_path)
    
    # 筛选 Branch_Type == RL_Series AND Branch_ID == e
    print(f"筛选 Branch_Type='{TARGET_BRANCH_TYPE}' & Branch_ID='{TARGET_BRANCH_ID}' 的数据...")
    mask = (df['Branch_Type'] == TARGET_BRANCH_TYPE) & (df['Branch_ID'] == TARGET_BRANCH_ID)
    df_filtered = df[mask].copy()
    
    # 检查并清洗无效值
    if df_filtered.empty:
        raise ValueError("筛选后数据为空，请检查过滤条件！")
        
    # 提取特征和目标
    X = df_filtered[FEATURES].copy()
    Y = df_filtered[TARGET_COL]

    # --- 特征工程 ---
    # 1. 物理特征: 阻抗模值、视在功率、相位
    # R 往往与总阻抗 Z_mag 相关
    print("应用特征工程: 添加物理特征 (Z_mag, Phase)...")
    X['S_mag'] = np.sqrt(X['P']**2 + X['Q']**2)
    X['Z_mag'] = X['V']**2 / (X['S_mag'] + 1e-9)
    X['Phase'] = np.arctan2(X['Q'], X['P'])
    
    # 2. 奇点特征: 不同衰减因子的倒数特征
    # 极值与 xi 的倒数强相关，提供多个尺度的倒数特征供模型选择
    print("应用特征工程: 添加多尺度倒数特征...")
    X['inv_xi_0.1'] = 1.0 / (X['xi'].abs() + 0.1)
    X['inv_xi_0.5'] = 1.0 / (X['xi'].abs() + 0.5)
    X['inv_xi_1.0'] = 1.0 / (X['xi'].abs() + 1.0)
    
    # 移除原始 S_mag (它包含在 Z_mag 中，且与 P,Q 高度共线，保留 Z_mag 物理意义更强)
    X.drop(columns=['S_mag'], inplace=True)

    
    print(f"数据加载完成。样本数: {len(X)}")
    print(f"目标值 (R) 统计:\n  Max: {Y.max():.2f}\n  Min: {Y.min():.2f}\n  Mean: {Y.mean():.2f}\n  Std: {Y.std():.2f}")
    
    # 简单分析分布偏度
    skew = Y.skew()
    print(f"数据偏度 (Skewness): {skew:.2f}")

    # 分析分位数，辅助判断异常值情况
    print(f"分位数概览:\n{Y.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])}")

    return X, Y

def main():
    # 1. 准备数据
    X, Y = load_data(INPUT_CSV)
    
    # 2. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 3. 策略选择 (基于数据观察)
    # 之前尝试 QuantileTransformer 虽然改善了 R2，但对极值预测仍有偏差，且丢失了物理量级的距离感。
    # 分析发现极值主要集中在 xi 接近 0 的区域，呈现类似 1/x 的特性。
    # 我们改用 np.arcsinh (反双曲正弦) 变换。
    # arcsinh 在 0 附近近似线性，在极大值处近似对数 (log)，且支持负数。
    # 这样既能压缩 10^5 数量级的极值，又能保留数值间的相对大小关系。
    
    print("\n策略更新: 检测到极值与 xi 强相关。")
    print("1. 特征增强: 增加 1/(|xi|+0.5)")
    print("2. 目标变换: 使用 np.arcsinh 对数级压缩，保留正负号和相对距离。")
    
    # 4. 构建模型管道
    # 输入特征使用 RobustScaler
    base_regressor = Pipeline([
        ('scaler_X', RobustScaler()),
        ('svr', SVR())
    ])
    
    # 目标变换器: Arcsinh
    model = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.arcsinh,
        inverse_func=np.sinh
    )
    
    # 5. 定义超参数搜索空间
    # 使用 arcsinh 后，Y 的范围大约在 -15 到 +15 之间 (arcsinh(800000) ~= 14.2)
    # 这个尺度比 Quantile 的 N(0,1) 大，所以 epsilon 需要相应调大
    # C 仍然需要较大，以拟合复杂的曲面
    
    param_distributions = {
        'regressor__svr__C': [100, 1000, 5000, 10000, 20000, 50000], 
        'regressor__svr__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0], 
        'regressor__svr__gamma': ['scale', 0.1, 0.5, 1, 2], # gamma 控制核宽，可能需要较窄(大数值)来拟合尖峰
        'regressor__svr__kernel': ['rbf']
    }
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=60, # 增加搜索次数
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )
    
    # 6. 训练
    print("\n开始超参数搜索与训练...")
    start_time = time.time()
    search.fit(X_train, Y_train)
    end_time = time.time()
    
    print(f"训练完成，耗时: {end_time - start_time:.2f}s")
    print(f"最佳参数: {search.best_params_}")
    
    best_model = search.best_estimator_
    
    # 7. 预测与评估
    print("\n正在评估 (训练集 vs 测试集)...")
    Y_pred = best_model.predict(X_test)
    Y_train_pred = best_model.predict(X_train)
    
    # 计算指标
    r2_test = r2_score(Y_test, Y_pred)
    r2_train = r2_score(Y_train, Y_train_pred)
    
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred)
    
    print(f"训练集 R2 Score: {r2_train:.4f}")
    print(f"测试集 R2 Score: {r2_test:.4f}")
    print(f"测试集 RMSE: {rmse:.2f}")
    print(f"测试集 MAE: {mae:.2f}")
    
    # 展示几个具体的预测样例
    print("\n--- 预测样例分析 ---")
    results = pd.DataFrame({
        'True Value': Y_test.values,
        'Predicted': Y_pred
    })
    results['Diff'] = results['True Value'] - results['Predicted']
    results['Rel Err %'] = (results['Diff'].abs() / results['True Value'].replace(0, 1e-10).abs()) * 100
    
    print("\n[极值样本 (Top 5 Absolute True Values)]")
    top_5 = results.iloc[results['True Value'].abs().argsort()[-5:]]
    print(top_5.to_string(float_format=lambda x: "{:.2f}".format(x)))

    print("\n[中间区域样本 (Middle 5)]")
    mid_idx = len(results) // 2
    mid_5 = results.iloc[mid_idx-2:mid_idx+3]
    print(mid_5.to_string(float_format=lambda x: "{:.2f}".format(x)))

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
