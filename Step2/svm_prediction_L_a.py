"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'L_a' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 L_a 分支进行预测。

Author: GitHub Copilot (Refactored for L_a)
Date: 2026-02-14
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'a' 的电感 L 进行预测的模型。
    
    问题复盘 (Problem Analysis for Branch 'a' - L):
    1. 数据分布 (Data Distribution): L_a 的值主要分布在负数区间 (-0.02 ~ -0.04)，数量级较小 (10^-2)。
       存在极少数正值离群点 (L > 0)，这些点被视为异常值并剔除。
    2. 策略选择 (Strategy): 
       - 目标值较小: L_a 的绝对值远小于 1。直接使用 defaults Epsilon (0.1) 会导致欠拟合（因为变化幅度小于 epsilon）。
       - 参数调整: Epsilon 必须调小至 1e-4 ~ 5e-3 量级。
       - ArcSinh: 尽管数值小，ArcSinh 近似线性，但仍保留其作为一般化处理手段。
       - 无加权回归: 样本分布相对集中。

Usage:
    python svm_prediction_L_a.py
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 配置日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 1. 配置管理 ---

@dataclass
class ExperimentConfig:
    """实验参数配置类"""
    input_filename: str = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
    model_save_path: str = "svm_model_L_a.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "a"
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    backup_filenames: Tuple[str, ...] = ('equivalent_circuit_parameters_optimized_accurate_Y11.csv',) # Added backup

    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 策略更新 (2026-02-14 - 针对 L_a 优化): 
                # L_a 数值很小 (~ -0.03)，需要较小的 Epsilon 来捕捉变化。
                
                'regressor__svr__C': [10, 100, 500, 1000, 3000], 
                
                'regressor__svr__gamma': ['scale', 0.1, 1.0, 10.0],
                
                # Epsilon 必须非常小，因为 Target 的 Scale 本身就很小
                # std(L_a) ~ 0.06, mean ~ -0.03.
                # Epsilon 应该小于 std.
                'regressor__svr__epsilon': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                'regressor__svr__kernel': ['rbf'] 
            }
        
        self.n_iter_search = 15  
        self.cv_folds = 5

# --- 2. 数据处理模块 ---
class DataProcessor:
    """处理数据加载、清洗和特征工程的类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _find_data_file(self) -> Path:
        candidates = [
            Path(self.config.input_filename),
            Path("csv_data") / self.config.input_filename,
            Path("Step2") / self.config.input_filename, # Added Step2 folder
        ] + [Path(f) for f in self.config.backup_filenames]
        
        for path in candidates:
            if path.exists():
                return path
        # Fallback for workspace root
        if Path(self.config.input_filename).exists():
             return Path(self.config.input_filename)
             
        raise FileNotFoundError(f"未找到输入数据文件。尝试查找路径: {[str(p) for p in candidates]}")

    def load_and_filter(self) -> pd.DataFrame:
        file_path = self._find_data_file()
        logger.info(f"正在读取数据文件: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 严格筛选目标分支 (a)
        mask = (df['Branch_Type'] == self.config.target_branch_type) & \
               (df['Branch_ID'] == self.config.target_branch_id)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (L_a 专用) ---
        # 现象: L_a 绝大多数为负值，仅有极个别正值。
        if self.config.target_branch_id == 'a':
            target_col = 'L' # 原始长格式文件列名为 L
            if target_col in df_filtered.columns:
                 # 剔除正值 (L > 0)
                mask_valid = df_filtered[target_col] < 0
                n_removed = (~mask_valid).sum()
                if n_removed > 0:
                    df_filtered = df_filtered[mask_valid]
                    logger.warning(f"针对 Branch 'L_a' 移除了 {n_removed} 个正值异常样本 (L > 0)")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df_filtered)}")
        return df_filtered

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)

        # --- 2. 形状特征 ---
        X['xi_2'] = X['xi'] ** 2
        
        # 目标变量 L
        y_values = df['L']
        return X, y_values

# --- 3. 模型训练模块 ---

class AdmittanceSVM:
    """SVM 模型封装"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_model = None
        self.search_results = None

    def train(self, X_train, y_train):
        # 1. Scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 2. Configure SVR and Search
        clean_param_dist = {
            k.replace('regressor__svr__', ''): v 
            for k, v in self.config.param_distributions.items()
        }
        
        base_svr = SVR(cache_size=1000, max_iter=100000, tol=1e-4) # tol needs to be smaller for small values
        
        search = RandomizedSearchCV(
            base_svr,
            param_distributions=clean_param_dist,
            n_iter=self.config.n_iter_search,
            cv=self.config.cv_folds,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1,
            random_state=self.config.random_state
        )
        
        # 3. TransformedTargetRegressor
        # 使用 ArcSinh 对 L 进行变换 (L 为负值时，ArcSinh 也是负值且单调)
        # 对于 L ~ -0.03，ArcSinh(L) ~ -0.03 (近似线性)
        # 但保留它以防有较大的负值出现。
        ttr_model = TransformedTargetRegressor(
            regressor=search,
            func=np.arcsinh,
            inverse_func=np.sinh,
            check_inverse=True
        )
        
        weights = None
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(L))...")
        
        start_time = time.time()
        ttr_model.fit(X_train_scaled, y_train, sample_weight=weights)
        elapsed = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")
        best_search = ttr_model.regressor_
        logger.info(f"最佳参数: {best_search.best_params_}")
        
        self.best_model = Pipeline([
            ('scaler', scaler),
            ('regressor', ttr_model)
        ])
        self.search_results = best_search

    def save(self):
        if self.best_model:
            joblib.dump(self.best_model, self.config.model_save_path)
            logger.info(f"模型已保存至: {self.config.model_save_path}")

# --- 4. 评估工具 ---

def evaluate_performance(model, X_test, L_test_true):
    """
    评估 (都在物理域 L 进行)
    """
    logger.info("\n=== 模型评估报告 ===")
    
    L_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(L_test_true, L_pred)
    
    # 2. MAPE
    diff = np.abs(L_test_true - L_pred)
    mape = np.mean(diff / (np.abs(L_test_true) + 1e-9)) * 100
    
    # 3. RMSE
    rmse = np.sqrt(mean_squared_error(L_test_true, L_pred))

    log_msg = f"[Physical Domain L] R2: {r2:.4f}\n[Physical Domain L] MAPE: {mape:.2f}%\n[Physical Domain L] RMSE: {rmse:.6f}"
    logger.info(log_msg)
    
    with open("results_L_a.txt", "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True L': L_test_true.values,
            'Pred L': L_pred,
            'Diff L (Abs)': np.abs(L_test_true.values - L_pred),
            'Rel Err %': np.abs(L_test_true.values - L_pred) / (np.abs(L_test_true.values) + 1e-9) * 100
        })
        results['Sign Match'] = np.sign(results['True L']) == np.sign(results['Pred L'])
        
        f.write("\n--- 最差预测样本 Top 5 ---\n")
        f.write(results.sort_values('Diff L (Abs)', ascending=False).head(5).to_string() + "\n")
        
    analyze_samples(L_test_true, L_pred)

def analyze_samples(L_true, L_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True L': L_true.values,
        'Pred L': L_pred,
        'Diff L (Abs)': np.abs(L_true.values - L_pred),
        'Rel Err %': np.abs(L_true.values - L_pred) / (np.abs(L_true.values) + 1e-9) * 100
    })
    
    print("\n--- 最差预测样本 Top 5 (按 L 绝对误差) ---")
    print(results.sort_values('Diff L (Abs)', ascending=False).head(5).to_string(float_format="%.6f"))
    
    print("\n--- 最佳预测样本 Top 5 (按 L 绝对误差) ---")
    print(results.sort_values('Diff L (Abs)', ascending=True).head(5).to_string(float_format="%.6f"))

# --- 主程序入口 ---

def main():
    try:
        config = ExperimentConfig()
        processor = DataProcessor(config)
        df = processor.load_and_filter()
        X, L = processor.engineer_features(df)
        
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=config.random_state
        )
        
        L_train = L.iloc[idx_train]
        L_test = L.iloc[idx_test]
        
        # 训练
        trainer = AdmittanceSVM(config)
        trainer.train(X_train, L_train)
        
        # 评估
        evaluate_performance(trainer.best_model, X_test, L_test)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
