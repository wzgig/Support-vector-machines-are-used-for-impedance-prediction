"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'L_c' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 L_c 分支进行预测。

Author: GitHub Copilot (Refactored for L_c)
Date: 2026-02-14
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'c' 的电感 L_c 进行预测的模型。
    
    问题复盘 (Problem Analysis for Branch 'L_c'):
    1. 数据分布 (Data Distribution): 
       - L_c 是一个高度稳定且数值极小的正值参数，绝大多数数据集中在 [0.015, 0.06] 区间，中心约 0.027。
       - 数据中存在极个别的负 outlier (如 -3.03)，物理上 L_c 应为正，需剔除。
       - L_c 的数值远小于 R_c (R_c ~ 2.0)，因此对模型的精度 (epsilon) 要求极高。
       
    2. 策略选择 (Strategy): 
       - 清洗策略: 强制 L_c > 0 且剔除极端大值 (例如 > 0.5)。
       - 目标变换 (ArcSinh): 尽管 L_c 分布平稳，ArcSinh 仍能提供良好的非负约束和平滑效果。
         由于 L_c 很小，ArcSinh(x) ≈ x，近似线性。
       - 特征工程: 保留物理特征 (P, Q, V, xi)。

Usage:
    python svm_prediction_L_c.py
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
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
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
    backup_filenames:Tuple[str, ...] = ("extracted_RL_Series_Y11_wide.csv",)
    model_save_path: str = "svm_model_ArcSinh_Impedance_L_c.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "c" # 目标分支为 c
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 20   # Restore for better search coverage
    cv_folds: int = 5         # Restore for robust evaluation
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 针对 Branch 'L_c' 优化: 
                # L_c 数值很小 (0.027)，噪声容忍度必须极低。
                # C 值: 需要足够大以减少偏差
                'regressor__svr__C': [100, 500, 1000, 2000], 
                
                # Gamma: RBF 核宽度
                # 对于 L_c 这种平滑且集中的数据，gamma 不宜过大
                'regressor__svr__gamma': ['scale', 0.1, 1.0],
                
                # Epsilon: 关键参数！
                # 0.0001 (High precision) -> 0.005 (Loose)
                'regressor__svr__epsilon': [1e-4, 5e-4, 1e-3, 5e-3],
                'regressor__svr__kernel': ['rbf'] 
            }
        
        # 增加搜索次数
        self.n_iter_search = 20
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
        ] + [Path(f) for f in self.config.backup_filenames]
        
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"未找到输入数据文件。尝试查找路径: {[str(p) for p in candidates]}")

    def load_and_filter(self) -> pd.DataFrame:
        file_path = self._find_data_file()
        logger.info(f"正在读取数据文件: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 目标列名为 L_c
        target_col = f"L_{self.config.target_branch_id}"
        
        # 兼容处理: 如果是 wide 表，查找 L_{id} 列
        if target_col in df.columns:
            logger.info(f"检测到宽表格式 (Wide Format)，提取 {target_col} 列...")
            df = df.copy()
            df['target'] = df[target_col] # 将目标列重命名为 target
            
            # 过滤掉非数值或缺失值
            df = df.dropna(subset=['target'])
             
        else:
            # Long format logic (legacy or for backup files)
            mask = (df['Branch_Type'] == self.config.target_branch_type) & \
                   (df['Branch_ID'] == self.config.target_branch_id)
            df = df[mask].copy()
            # 假设 long 表中电感列名为 'L'
            if 'L' in df.columns:
                df['target'] = df['L']
            else:
                raise ValueError("未找到目标列 (L_c 或 L)")
            
        if df.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (L_c 专用) ---
        # 现象: L_c 大部分在 0.027 附近。存在少量负异常值。
        # 策略: 严格仅保留正值，并剔除离谱的大数值 (> 0.5)。
        min_limit = 0.0
        max_limit = 0.5 
        
        mask_range = (df['target'] >= min_limit) & (df['target'] <= max_limit)
        n_removed_range = (~mask_range).sum()
        
        if n_removed_range > 0:
            df = df[mask_range]
            logger.warning(f"根据阈值 (L >= {min_limit} and L <= {max_limit}) 移除了 {n_removed_range} 个异常样本")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df)}")
        return df

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
        
        # --- 2. 形状特征 (Singularity Handling) ---
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9) 
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)

        y_values = df['target']
        return X, y_values

# --- 3. 模型训练模块 ---

class AdmittanceSVM:
    """SVM 模型封装"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_model = None
        self.search_results = None

    def train(self, X_train, y_train):
        # 1. Manual Scaling (Bypass Pipeline Sample Weight Issues in complex CV)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 2. Configure SVR and Search
        # Strip prefixes because we are no longer using a Pipeline inside CV directly for the search base
        clean_param_dist = {
            k.replace('regressor__svr__', ''): v 
            for k, v in self.config.param_distributions.items()
        }
        
        # Base estimator is SVR
        # 优化: 大幅度增加 max_iter 以解决 ConvergenceWarning
        base_svr = SVR(cache_size=1000, max_iter=200000, tol=1e-5)
        
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
        
        # 3. Wrap Search in TransformedTargetRegressor
        # This ensures CV optimizes for the transformed target space
        ttr_model = TransformedTargetRegressor(
            regressor=search,
            func=np.arcsinh,
            inverse_func=np.sinh,
            check_inverse=True
        )
        
        # 4. Calculate Sample Weights
        # 针对 Branch 'L_c' 优化: 均等权重
        weights = None
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(L_c))...")
        
        start_time = time.time()
        
        # TTR.fit passes kwargs to search.fit
        ttr_model.fit(X_train_scaled, y_train, sample_weight=weights)
        
        elapsed = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")
        # 获取 fit 后的 RandomizedSearchCV 对象需要从 ttr_model.regressor_ 获取
        best_search = ttr_model.regressor_
        logger.info(f"最佳参数: {best_search.best_params_}")
        logger.info(f"最佳分数 (Neg MSE in Transformed Domain): {best_search.best_score_:.4f}")
        
        # 5. Assemble Final Inference Pipeline
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

def evaluate_performance(model, X_test, y_test_true):
    """
    评估 (都在物理域 L 进行，因为模型会自动反变换)
    """
    logger.info("\n=== 模型评估报告 ===")
    
    # model is the Pipeline([scaler, ttr])
    y_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(y_test_true, y_pred)
    
    # 2. MSE & RMSE
    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    
    # 3. MAE & MAPE
    diff = np.abs(y_test_true - y_pred)
    mae = np.mean(diff)
    # L_c 值很小，MAPE 可能对误差非常敏感
    mape = np.mean(diff / (np.abs(y_test_true) + 1e-9)) * 100
    
    log_msg = f"[Physical Domain L] R2: {r2:.4f}\n[Physical Domain L] MAE: {mae:.6f}\n[Physical Domain L] RMSE: {rmse:.6f}\n[Physical Domain L] MAPE: {mape:.2f}%"
    logger.info(log_msg)
    
    results_file = "results_L_c.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True L': y_test_true.values,
            'Pred L': y_pred,
            'Diff L (Abs)': np.abs(y_test_true.values - y_pred),
            'Rel Err %': np.abs(y_test_true.values - y_pred) / (np.abs(y_test_true.values) + 1e-9) * 100
        })
        results['Sign Match'] = np.sign(results['True L']) == np.sign(results['Pred L'])
        
        f.write("\n--- 最差预测样本 Top 10 ---\n")
        f.write(results.sort_values('Diff L (Abs)', ascending=False).head(10).to_string() + "\n")
        
    analyze_samples(y_test_true, y_pred)

def analyze_samples(y_true, y_pred):
    """
    分段分析预测误差
    """
    regions = [
        ('Small (<0.02)', lambda y: y < 0.02),
        ('Typical [0.02, 0.04]', lambda y: (y >= 0.02) & (y <= 0.04)),
        ('Large (>0.04)', lambda y: y > 0.04)
    ]
    
    logger.info("\n--- 分段性能分析 ---")
    for name, mask_func in regions:
        mask = mask_func(y_true)
        if mask.any():
            subset_true = y_true[mask]
            subset_pred = y_pred[mask]
            r2 = r2_score(subset_true, subset_pred)
            mae = np.mean(np.abs(subset_true - subset_pred))
            logger.info(f"Region {name}: N={mask.sum()}, R2={r2:.4f}, MAE={mae:.6f}")

# --- 5. 主程序 ---
def main():
    start_total = time.time()
    logger.info("启动 L_c 阻抗预测任务 ...")
    
    # 1. Initialize
    config = ExperimentConfig()
    processor = DataProcessor(config)
    
    # 2. Data Loading
    try:
        df = processor.load_and_filter()
        X, y = processor.engineer_features(df)
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        return

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state
    )
    logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 4. Train
    model = AdmittanceSVM(config)
    model.train(X_train, y_train)
    
    # 5. Evaluate
    evaluate_performance(model.best_model, X_test, y_test)
    
    # 6. Save
    model.save()
    
    logger.info(f"任务完成，总耗时: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()
