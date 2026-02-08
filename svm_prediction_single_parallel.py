
import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- 全局配置 ---
INPUT_CSV = os.path.join("csv_data", "equivalent_circuit_parameters_optimized_Y11.csv")
MODEL_SAVE_PATH = "svm_model_R_Parallel_Specific.pkl"
TARGET_BRANCH = "Parallel"
TARGET_COL = "R"
FEATURES = ['P', 'Q', 'V', 'xi']

# --- 自定义变换函数 ---
# 使用 arcsinh 处理横跨多个数量级且包含负数的数据
# 1e16 -> ~37.5, -1e17 -> -39.8
def target_transform(y):
    return np.arcsinh(y)

def target_inverse_transform(y):
    return np.sinh(y)

def load_parallel_data(csv_path):
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 筛选只包含 Parallel 分支的数据
    print(f"筛选 Branch_ID == '{TARGET_BRANCH}' 的数据...")
    df_parallel = df[df['Branch_ID'] == TARGET_BRANCH].copy()
    
    # 提取特征和目标
    X = df_parallel[FEATURES]
    Y = df_parallel[TARGET_COL]
    
    print(f"数据加载完成。样本数: {len(X)}")
    print(f"目标值 (R) 统计:\n  Max: {Y.max():.2e}\n  Min: {Y.min():.2e}\n  Mean: {Y.mean():.2e}")
    
    return X, Y

def main():
    # 路径检查逻辑
    if os.path.exists(INPUT_CSV):
        csv_path = INPUT_CSV
    elif os.path.exists("equivalent_circuit_parameters_optimized_Y11.csv"):
        csv_path = "equivalent_circuit_parameters_optimized_Y11.csv"
    else:
        print(f"错误：找不到文件 {INPUT_CSV} 或当前目录下的同名文件")
        return

    # 1. 准备数据
    X, Y = load_parallel_data(csv_path)
    
    # 2. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 3. 构建模型管道
    # 内部 SVR 只需要处理标准化后的特征和被 Log(Arcsinh) 压缩后的目标值
    # 这使得 C 和 epsilon 的搜索空间可以保持在常规范围内 (如 0.1~1000)
    
    # 基础回归器 (处理 X 的标准化 + SVR)
    base_regressor = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])
    
    # 目标变换回归器 (处理 Y 的压缩与还原)
    model = TransformedTargetRegressor(
        regressor=base_regressor,
        func=target_transform,
        inverse_func=target_inverse_transform,
        check_inverse=False # 浮点精度原因，不用严格检查
    )
    
    # 4. 定义超参数搜索空间
    # 注意：这里的参数名需要加上前缀 'regressor__' (指向 Pipeline) 
    # 再加上 'svr__' (指向 Pipeline 里的 SVR)
    # 所以这就像俄罗斯套娃: model(TransformedTargetRegressor) -> regressor(Pipeline) -> svr(SVR)
    param_distributions = {
        'regressor__svr__C': np.logspace(0, 4, 20),       # 1 ~ 10000 (给大一点范围)
        'regressor__svr__epsilon': np.logspace(-3, 1, 10), # 0.001 ~ 10 (在 arcsinh 空间里的容忍度)
        'regressor__svr__gamma': ['scale', 'auto', 0.1, 1],
        'regressor__svr__kernel': ['rbf']
    }
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=50, # 搜索50次
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )
    
    # 5. 训练
    print("\n开始超参数搜索与训练 (目标值已应用 arcsinh 变换)...")
    start_time = time.time()
    search.fit(X_train, Y_train)
    end_time = time.time()
    
    print(f"训练完成，耗时: {end_time - start_time:.2f}s")
    print(f"最佳参数: {search.best_params_}")
    
    best_model = search.best_estimator_
    
    # 6. 预测与评估
    print("\n正在评估测试集...")
    Y_pred = best_model.predict(X_test)
    
    # 计算 R2
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    
    print(f"测试集 R2 Score: {r2:.4f}")
    print(f"测试集 RMSE (原始尺度): {rmse:.2e}")
    
    # 展示几个具体的预测样例
    print("\n--- 预测样例 (前5个) ---")
    print(f"{'True Value':>20} | {'Predicted Value':>20} | {'Rel Error %':>12}")
    print("-" * 60)
    for y_true, y_pred in zip(Y_test[:5], Y_pred[:5]):
        rel_err = abs((y_true - y_pred) / y_true) * 100 if y_true != 0 else 0
        print(f"{y_true:20.2e} | {y_pred:20.2e} | {rel_err:10.2f}%")
        
    # 7. 保存模型
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
