import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score

# --- 全局配置 ---
INPUT_CSV = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
MODEL_SAVE_PATH = "svm_model_R_Y11_tuned.pkl"
TARGET_COL = "R"  # 目标物理量
FEATURES = ['P', 'Q', 'V', 'xi']

# --- 超参数搜索空间 (针对 scaler 后的数据) ---
# 不同的分支可能适应不同的 C (正则化强度) 和 epsilon (不敏感区)
PARAM_DISTRIBUTIONS = {
    'regressor__C': np.logspace(-1, 3, 10),      # 0.1 ~ 1000
    'regressor__epsilon': np.logspace(-3, -1, 5), # 0.001 ~ 0.1
    'regressor__gamma': ['scale', 'auto', 0.1, 1],
    'regressor__kernel': ['rbf']                 # 通常 RBF 是最通用的
}

class IndependentTunedRegressor(BaseEstimator, RegressorMixin):
    """
    自定义回归器：
    1. 对 Y 的每一列独立建立模型。
    2. 对全零列使用 DummyRegressor (恒输出0)。
    3. 对非零列使用 RandomizedSearchCV 进行超参数调优。
    """
    def __init__(self, base_estimator=None, param_distributions=None, n_iter=10, cv=3, n_jobs=-1, verbose=1):
        self.base_estimator = base_estimator or SVR()
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.estimators_ = [] # 存储每个列的最佳模型
        self.best_params_ = [] # 存储每个列的最佳参数
        self.col_types_ = []   # 记录是 'tuned' 还是 'zero'

    def fit(self, X, Y):
        n_samples, n_targets = Y.shape
        self.estimators_ = [None] * n_targets
        self.best_params_ = [None] * n_targets
        self.col_types_ = [None] * n_targets

        print(f"开始训练 IndependentTunedRegressor (Targets={n_targets})...")
        start_time = time.time()

        # 这里我们按顺序处理每一列 (GridSearch 内部本身支持 n_jobs并行，
        # 如果列数很多，也可以考虑在外层做 joblib.Parallel，但在 Windows 下嵌套并行有时不稳定)
        # 为求稳健，我们这里用简单的循环，但让 Search 内部全速并行。
        
        for i in range(n_targets):
            y_col = Y[:, i]
            col_start = time.time()
            
            # --- 1. 检查是否为纯占位列 (全零) ---
            # 考虑到浮点误差，使用微小阈值
            if np.all(np.abs(y_col) < 1e-12):
                if self.verbose > 0:
                    print(f"  [Col {i}] 检测到全零列 (Placeholder)，使用 DummyRegressor。")
                self.estimators_[i] = DummyRegressor(strategy='constant', constant=0.0)
                self.estimators_[i].fit(X, y_col)
                self.col_types_[i] = 'zero'
                self.best_params_[i] = {}
            
            # --- 2. 独立调优训练 ---
            else:
                if self.verbose > 0:
                    print(f"  [Col {i}] 开始超参数寻优...", end="")
                
                # --- 智能自适应预处理 (Adaptive Output Processing) ---
                # 回答您的痛点：针对不同尺度的输出，使用不同的处理策略
                col_abs_max = np.max(np.abs(y_col))
                
                # 策略选择逻辑：
                # 1. 极端值 (Extreme): > 1e9 (如 Parallel, 1)。通常是奇异点。使用 Quantile 强力压缩。
                # 2. 普通值 (Normal): < 1e9 (如 a, b, c, d...)。使用 StandardScaler 保持物理分布线性。
                #    注意：之前 c, b 效果不好可能是因为 Quantile 甚至破坏了原本较弱的规律，
                #    或者数据本身需要更强的正则化。
                
                if col_abs_max > 1e9:
                    scaler_name = "QuantileTransformer"
                    y_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_col), 1000), random_state=42)
                else:
                    scaler_name = "StandardScaler"
                    y_scaler = StandardScaler()
                
                if self.verbose > 0:
                    print(f" [{scaler_name}] ", end="")

                y_col_scaled = y_scaler.fit_transform(y_col.reshape(-1, 1)).ravel()
                
                # 针对该 Pipeline 的 Search
                # pipeline 里是 ['scaler_X', 'regressor']
                # 用来 search 的 pipe 应该尽量简单，我们只优化 regressor 部分
                search_pipe = Pipeline([
                    ('scaler_X', StandardScaler()),
                    ('regressor', self.base_estimator)
                ])

                search = RandomizedSearchCV(
                    search_pipe, 
                    param_distributions=self.param_distributions,
                    n_iter=self.n_iter,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    scoring='neg_mean_squared_error',
                    random_state=42
                )
                
                search.fit(X, y_col_scaled)
                
                # 保存包含 scaler_Y 信息的复合对象，以便预测时还原
                best_model_bundle = {
                    'model': search.best_estimator_, # 这是一个 Pipeline(scaler_X, SVR)
                    'y_scaler': y_scaler
                }
                
                self.estimators_[i] = best_model_bundle
                self.best_params_[i] = search.best_params_
                self.col_types_[i] = 'tuned'
                
                if self.verbose > 0:
                    print(f" 完成。最佳参数: {search.best_params_['regressor__C']:.2f}, eps={search.best_params_['regressor__epsilon']:.3f} (耗时 {time.time()-col_start:.1f}s)")

        total_time = time.time() - start_time
        print(f"所有列训练完成，总耗时: {total_time:.2f}s")
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_targets = len(self.estimators_)
        Y_pred = np.zeros((n_samples, n_targets))

        for i, estimator in enumerate(self.estimators_):
            if self.col_types_[i] == 'zero':
                Y_pred[:, i] = estimator.predict(X)
            else:
                # 这是一个 bundle {'model': Pipeline, 'y_scaler': StandardScaler}
                bundle = estimator
                model = bundle['model']
                y_scaler = bundle['y_scaler']
                
                # 预测标准化后的值
                y_pred_scaled = model.predict(X)
                # 还原回物理值
                Y_pred[:, i] = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return Y_pred

# --- 辅助函数：数据加载与预处理 (保持不变) ---
def load_data_and_create_template(csv_path):
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    unique_branches = df['Branch_ID'].unique()
    
    # 物理含义排序
    def branch_sorter(b_id):
        b_id = str(b_id)
        if b_id == 'Parallel': return (0, b_id)
        elif b_id.isalpha():   return (1, b_id)
        else:                  return (2, b_id)
            
    sorted_template = sorted(unique_branches, key=branch_sorter)
    print(f"构建最大拓扑模板 (N={len(sorted_template)}): {sorted_template}")
    
    pivot_df = df.pivot_table(
        index=['Filename', 'P', 'Q', 'V', 'xi'], 
        columns='Branch_ID', 
        values=TARGET_COL
    )
    
    pivot_df = pivot_df.reindex(columns=sorted_template)
    pivot_df = pivot_df.fillna(0.0) # 占位符补零
    
    X = pivot_df.index.to_frame(index=False)[FEATURES]
    Y = pivot_df.values
    return X, Y, sorted_template

def main():
    # 检查CSV路径
    if os.path.exists(INPUT_CSV):
        csv_path = INPUT_CSV
    # 兼容性检查：如果当前位置找不到，尝试去 csv_data 找
    elif os.path.exists(os.path.join("csv_data", INPUT_CSV)):
        csv_path = os.path.join("csv_data", INPUT_CSV)
    else:
        print(f"错误：找不到文件 {INPUT_CSV}")
        return

    # 1. 准备数据
    X, Y, template = load_data_and_create_template(csv_path)
    
    # 2. 划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 3. 初始化并训练自定义回归器
    model = IndependentTunedRegressor(
        base_estimator=SVR(), # 基础模型
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=10, # 每个列尝试 10 种参数组合 (可根据时间增加)
        n_jobs=-1  # 全速并行搜索
    )

    model.fit(X_train, Y_train)

    # 4. 预测与评估
    print("\n正在评估测试集...")
    Y_pred = model.predict(X_test)
    
    # 整体 R2
    overall_r2 = r2_score(Y_test, Y_pred) # 默认 uniform_average
    print(f"整体平均 R2 Score: {overall_r2:.4f}")

    # 逐列 R2 (排除全零列，因为全零列 R2 通常定义为 1.0 或 0.0 取决于实现，这里主要看有意义的列)
    raw_r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')
    print("\n--- 各分支模型表现 ---")
    print(f"{'Branch ID':<12} | {'Type':<6} | {'R2 Score':<10} | {'Best Params'}")
    print("-" * 60)
    for i, name in enumerate(template):
        print(f"{str(name):<12} | {model.col_types_[i]:<6} | {raw_r2[i]:.4f}     | {model.best_params_[i]}")

    # 5. 保存
    joblib.dump({
        'model': model,
        'template': template,
        'features': FEATURES
    }, MODEL_SAVE_PATH)
    print(f"\n完整模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
