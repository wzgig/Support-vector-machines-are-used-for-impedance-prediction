"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'c' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 R_c 分支进行预测。

Author: GitHub Copilot (Refactored for R_c)
Date: 2026-02-13
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'c' 的电阻 R_c 进行预测的模型。
    
    问题复盘 (Problem Analysis for Branch 'c'):
    1. 数据分布 (Data Distribution): 
       - R_c 是一个高度稳定的正值参数，绝大多数数据集中在 [1.5, 3.0] 区间。
       - 数据中存在极个别的负值异常点 (例如 -484)，这在物理上是不合理的，必须剔除。
       - 与 R_b 不同，R_c 没有表现出极端的奇异性或巨大的动态范围。
       
    2. 策略选择 (Strategy): 
       - 清洗策略: 强制 R_c > 0。
       - 目标变换 (ArcSinh): R_c 分布平稳，ArcSinh 仅作为轻量级的平滑变换 (类似于 log1p)，
         能很好地处理正偏态分布并保证预测值非负趋势。
       - 特征工程: 保留物理特征 (P, Q, V, xi)，R_c 与这些参数呈现较平滑的非线性关系。

Usage:
    python svm_prediction_R_c.py
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
    input_filename: str = "extracted_RL_Series_Y11_wide.csv" 
    backup_filenames:Tuple[str, ...] = ("equivalent_circuit_parameters_optimized_accurate_Y11.csv",)
    model_save_path: str = "svm_model_ArcSinh_Impedance_R_f.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "c" # 目标分支为 c
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 针对 Branch 'c' 优化: 
                # R_c 数据虽然整体平稳，但存在高值区域 (R > 30)，
                # 这些"尖峰"需要较高的 C 值才能拟合，否则会被视为噪声平滑掉。
                'regressor__svr__C': [100, 500, 1000, 2000, 5000], 
                
                # Gamma: 需要较高的 gamma 来捕捉高值区域的局部特征
                'regressor__svr__gamma': ['scale', 0.5, 1.0, 2.0],
                
                # Epsilon: 保持高精度
                'regressor__svr__epsilon': [0.001, 0.005, 0.01],
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
        
        target_col = f"R_{self.config.target_branch_id}"
        
        # 兼容处理: 如果是 wide 表，查找 R_{id} 列
        if target_col in df.columns:
            logger.info(f"检测到宽表格式 (Wide Format)，提取 {target_col} 列...")
            df = df.copy()
            df['R'] = df[target_col] # 将目标列重命名为 R 以匹配后续逻辑
            
            # 过滤掉非数值或缺失值
            df = df.dropna(subset=['R'])
             
        else:
            # Long format logic (legacy or for backup files)
            mask = (df['Branch_Type'] == self.config.target_branch_type) & \
                   (df['Branch_ID'] == self.config.target_branch_id)
            df = df[mask].copy()
            
        if df.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (R_c 专用) ---
        # 现象: R_c 应为正值，但数据中存在极个别负离群值 (如 -484)。
        # 策略: 严格仅保留正电阻值，并剔除离谱的大数值。
        min_limit = 0
        max_limit = 60.0 # R_c 大部分在 30 以内
        
        mask_range = (df['R'] >= min_limit) & (df['R'] <= max_limit)
        n_removed_range = (~mask_range).sum()
        
        if n_removed_range > 0:
            df = df[mask_range]
            logger.warning(f"根据阈值 (R >= {min_limit} and R <= {max_limit}) 移除了 {n_removed_range} 个异常样本")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df)}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程 - 针对 R_b 优化
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # --- 2. 形状特征 (Singularity Handling) ---
        # R_b 在 xi=0 附近表现出奇异性 (1/xi 行为)。
        # 引入 1/xi 特征对线性化 R_b 的行为至关重要。
        # 加一个小 epsilon 防止 xi=0 精确除零 (虽然数据中貌似只有 0.5)
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9) 
        
        # 平方项保留，对应对称性
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)

        R_values = df['R']
        return X, R_values

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
        
        # Base estimator is SVR (supports sample_weight natively)
        # 优化: 大幅度增加 max_iter 以解决 ConvergenceWarning
        base_svr = SVR(cache_size=1000, max_iter=100000, tol=1e-4)
        
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
        # 针对 Branch 'c' 优化: 均等权重
        weights = None
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(R_c))...")
        
        start_time = time.time()
        
        # TTR.fit passes kwargs to search.fit
        # search.fit splits sample_weight and passes to SVR.fit
        # This chain works in all sklearn versions.
        ttr_model.fit(X_train_scaled, y_train, sample_weight=weights)
        
        elapsed = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")
        # 修正：search 对象本身没有被 fit，而是被 TTR 克隆并 fit 了
        # 获取 fit 后的 RandomizedSearchCV 对象需要从 ttr_model.regressor_ 获取
        best_search = ttr_model.regressor_
        logger.info(f"最佳参数: {best_search.best_params_}")
        
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

def evaluate_performance(model, X_test, R_test_true):
    """
    评估 (都在物理域 R 进行，因为模型会自动反变换)
    """
    logger.info("\n=== 模型评估报告 ===")
    
    # model is the Pipeline([scaler, ttr])
    # predict calls transform -> ttr.predict -> search.predict -> inverse_func
    R_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(R_test_true, R_pred)
    
    # 2. MAPE (处理 R_true 为 0 的情况)
    diff = np.abs(R_test_true - R_pred)
    # 分母加一个小量，防止除零。对于 R~100000，1e-3 忽略不计。
    mape = np.mean(diff / (np.abs(R_test_true) + 1e-3)) * 100
    
    log_msg = f"[Physical Domain R] R2: {r2:.4f}\n[Physical Domain R] MAPE: {mape:.2f}%"
    logger.info(log_msg)
    
    with open("results_arcsinh_c.txt", "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True R': R_test_true.values,
            'Pred R': R_pred,
            'Diff R (Abs)': np.abs(R_test_true.values - R_pred),
            'Rel Err %': np.abs(R_test_true.values - R_pred) / (np.abs(R_test_true.values) + 1e-3) * 100
        })
        results['Sign Match'] = np.sign(results['True R']) == np.sign(results['Pred R'])
        
        f.write("\n--- 最差预测样本 Top 5 ---\n")
        f.write(results.sort_values('Diff R (Abs)', ascending=False).head(5).to_string() + "\n")
        
        f.write("\n--- 符号错误 ---\n")
        f.write(f"Count: {(~results['Sign Match']).sum()} / {len(results)}\n")
        
    analyze_samples(R_test_true, R_pred)

def analyze_samples(R_true, R_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True R': R_true.values,
        'Pred R': R_pred,
        'Diff R (Abs)': np.abs(R_true.values - R_pred),
        'Rel Err %': np.abs(R_true.values - R_pred) / (np.abs(R_true.values) + 1e-3) * 100
    })
    
    # 添加符号检查
    results['Sign Match'] = np.sign(results['True R']) == np.sign(results['Pred R'])
    
    print("\n--- 最差预测样本 Top 5 (按 R 绝对误差) ---")
    print(results.sort_values('Diff R (Abs)', ascending=False).head(5).to_string(float_format="%.4f"))
    
    print("\n--- 最佳预测样本 Top 5 (按 R 绝对误差) ---")
    print(results.sort_values('Diff R (Abs)', ascending=True).head(5).to_string(float_format="%.4f"))
    
    # 统计符号错误数
    sign_errors = (~results['Sign Match']).sum()
    print(f"\n符号预测错误样本数: {sign_errors} / {len(results)}")

# --- 主程序入口 ---

def main():
    try:
        config = ExperimentConfig()
        processor = DataProcessor(config)
        df = processor.load_and_filter()
        X, R = processor.engineer_features(df)
        
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=config.random_state
        )
        
        R_train = R.iloc[idx_train]
        R_test = R.iloc[idx_test]
        
        # 训练
        trainer = AdmittanceSVM(config)
        trainer.train(X_train, R_train)
        
        # 评估
        evaluate_performance(trainer.best_model, X_test, R_test)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
