import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score

# --- 全局配置 ---
INPUT_CSV = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
MODEL_SAVE_PATH = "svm_model_L_Y11_tuned.pkl"
TARGET_COL = "L"  # 目标物理量
FEATURES = ['P', 'Q', 'V', 'xi']

# --- 超参数搜索空间 (针对 scaler 后的数据) ---
# 针对 L 的数值特性，可能需要更广的参数范围
PARAM_DISTRIBUTIONS = {
    'regressor__C': np.logspace(-1, 4, 15),       # 0.1 ~ 10000 (增加上限，应对大数值或复杂关系)
    'regressor__epsilon': np.logspace(-4, -1, 10), # 0.0001 ~ 0.1 (精细化)
    'regressor__gamma': ['scale', 'auto', 0.1, 1, 10],
    'regressor__kernel': ['rbf']
}

class IndependentTunedRegressor(BaseEstimator, RegressorMixin):
    """
    自定义回归器：
    1. 对 Y 的每一列独立建立模型。
    2. 对全零列使用 DummyRegressor (恒输出0)。
    3. 对非零列使用 RandomizedSearchCV 进行超参数调优。
    """
    def __init__(self, base_estimator=None, param_distributions=None, n_iter=15, cv=3, n_jobs=-1, verbose=1, feature_names=None):
        self.base_estimator = base_estimator or SVR()
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.feature_names = feature_names # 用于日志显示具体列名
        self.estimators_ = [] # 存储每个列的最佳模型
        self.best_params_ = [] # 存储每个列的最佳参数
        self.col_types_ = []   # 记录是 'tuned' 还是 'zero'

    def fit(self, X, Y):
        n_samples, n_targets = Y.shape
        self.estimators_ = [None] * n_targets
        self.best_params_ = [None] * n_targets
        self.col_types_ = [None] * n_targets

        print(f"开始训练 IndependentTunedRegressor (Targets={n_targets}, TargetType={TARGET_COL})...")
        start_time = time.time()

        for i in range(n_targets):
            y_col = Y[:, i]
            col_name = str(self.feature_names[i]) if self.feature_names is not None else str(i)
            col_start = time.time()
            
            # --- 1. 检查是否为纯占位列 ---
            # 对于 L，通常不会全为 0，但为了健壮性保留
            if np.all(np.abs(y_col) < 1e-12):
                if self.verbose > 0:
                    print(f"  [Branch {col_name}] 检测到全零列，使用 DummyRegressor。")
                self.estimators_[i] = DummyRegressor(strategy='constant', constant=0.0)
                self.estimators_[i].fit(X, y_col)
                self.col_types_[i] = 'zero'
                self.best_params_[i] = {}
            
            # --- 2. 独立调优训练 ---
            else:
                if self.verbose > 0:
                    print(f"  [Branch {col_name}] 开始超参数寻优...", end="")
                
                # --- 智能自适应预处理 ---
                col_max = np.max(y_col)
                col_min = np.min(y_col)
                col_abs_max = np.max(np.abs(y_col))
                col_std = np.std(y_col)
                
                # 针对 L 的特殊策略:
                # 'e' 分支发现有 ~10^4 数量级且波动巨大。
                # 如果绝对值最大值很大 (> 1000) 或者 标准差很大 (> 100)，使用 QuantileTransformer 归一化到正态分布
                # 这样可以压缩离群值，让 SVR 更好拟合
                
                if col_abs_max > 1000 or col_std > 100:
                    scaler_name = "QuantileTransformer"
                    # output_distribution='normal' 将数据映射为高斯分布，这是 SVR (RBF核) 最喜欢的
                    y_scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_col), 1000), random_state=42)
                
                # 针对非常小的值 (如 a, c ~0.02)，StandardScaler 也可以，但如果有长尾，RobustScaler 更好
                # 这里暂时保持 StandardScaler，除非发现其他问题
                else:
                    scaler_name = "StandardScaler"
                    y_scaler = StandardScaler()
                
                if self.verbose > 0:
                    print(f" (Range:[{col_min:.2f}, {col_max:.2f}], Std:{col_std:.2f}) -> 使用 [{scaler_name}] ", end="")

                y_col_scaled = y_scaler.fit_transform(y_col.reshape(-1, 1)).ravel()
                
                # 简单的 Pipeline
                search_pipe = Pipeline([
                    ('scaler_X', StandardScaler()), # 输入特征必须标准化
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
                
                try:
                    search.fit(X, y_col_scaled)
                    
                    best_model_bundle = {
                        'model': search.best_estimator_, 
                        'y_scaler': y_scaler
                    }
                    
                    self.estimators_[i] = best_model_bundle
                    self.best_params_[i] = search.best_params_
                    self.col_types_[i] = 'tuned'
                    
                    if self.verbose > 0:
                        print(f" 完成。C={search.best_params_['regressor__C']:.2f}, eps={search.best_params_['regressor__epsilon']:.4f} (R2_cv_inner: {search.best_score_:.3f})")
                        
                except Exception as e:
                    print(f" 训练失败: {e}")
                    # 回退到 Dummy
                    self.estimators_[i] = DummyRegressor(strategy='mean')
                    self.estimators_[i].fit(X, y_col)
                    self.col_types_[i] = 'error_fallback'

        total_time = time.time() - start_time
        print(f"所有列训练完成，总耗时: {total_time:.2f}s")
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_targets = len(self.estimators_)
        Y_pred = np.zeros((n_samples, n_targets))

        for i, estimator in enumerate(self.estimators_):
            if self.col_types_[i] in ['zero', 'error_fallback']:
                Y_pred[:, i] = estimator.predict(X)
            else:
                bundle = estimator
                model = bundle['model']
                y_scaler = bundle['y_scaler']
                
                y_pred_scaled = model.predict(X)
                Y_pred[:, i] = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return Y_pred

def load_data_and_create_template(csv_path):
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- 关键修改：只保留 RL_Series 分支 ---
    original_count = len(df)
    df = df[df['Branch_Type'] == 'RL_Series']
    print(f"过滤 RL_Series: {original_count} -> {len(df)} 行")

    unique_branches = df['Branch_ID'].unique()
    
    # 物理含义排序
    def branch_sorter(b_id):
        b_id = str(b_id)
        if b_id == 'Parallel': return (0, b_id)
        elif b_id.isalpha():   return (1, b_id)
        else:                  return (2, b_id)
            
    sorted_template = sorted(unique_branches, key=branch_sorter)
    print(f"目标分支 (N={len(sorted_template)}): {sorted_template}")
    
    # 透视表
    pivot_df = df.pivot_table(
        index=['Filename', 'P', 'Q', 'V', 'xi'], 
        columns='Branch_ID', 
        values=TARGET_COL
    )
    
    # 确保列顺序
    pivot_df = pivot_df.reindex(columns=sorted_template)
    
    # 对于 L，缺失值用 0 填充 (假设没有该元件就没有电感)
    # 注意：如果某个样本完全没有该分支数据，fillna(0) 是合理的。
    pivot_df = pivot_df.fillna(0.0) 
    
    X = pivot_df.index.to_frame(index=False)[FEATURES]
    Y = pivot_df.values
    return X, Y, sorted_template

def main():
    if os.path.exists(INPUT_CSV):
        csv_path = INPUT_CSV
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
        base_estimator=SVR(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=20, # 增加搜索次数以获得更好结果
        cv=3,
        n_jobs=-1,
        feature_names=template # 传入列名以便日志清楚
    )

    model.fit(X_train, Y_train)

    # 4. 预测与评估
    print("\n正在评估测试集...")
    Y_pred = model.predict(X_test)
    
    # 逐列 R2
    raw_r2 = r2_score(Y_test, Y_pred, multioutput='raw_values')
    
    print("\n--- 各分支模型表现 (Target: L) ---")
    print(f"{'Branch ID':<12} | {'Scaler':<20} | {'R2 Score':<10} | {'Params'}")
    print("-" * 80)
    for i, name in enumerate(template):
        # 获取 Scaler 名称
        scaler_name = "None"
        if model.estimators_[i] and isinstance(model.estimators_[i], dict):
            scaler_name = model.estimators_[i]['y_scaler'].__class__.__name__
        elif model.estimators_[i]:
            scaler_name = model.estimators_[i].__class__.__name__

        print(f"{str(name):<12} | {scaler_name:<20} | {raw_r2[i]:.4f}     | {model.best_params_[i]}")

    # 5. 保存
    joblib.dump({
        'model': model,
        'template': template,
        'features': FEATURES
    }, MODEL_SAVE_PATH)
    print(f"\n完整模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
