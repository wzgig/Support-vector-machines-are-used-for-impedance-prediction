"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'a' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 R_a 分支进行预测。

Author: GitHub Copilot (Refactored for R_a)
Date: 2026-02-13
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'a' 的电阻 R 进行预测的模型。
    
    问题复盘 (Problem Analysis for Branch 'a'):
    1. 数据分布 (Data Distribution): R_a 的值主要分布在负数区间 (-20 ~ -250)，但在极少数情况下会出现极端的正值点(R > 1500)。
       这些正值点极差较大，且在物理上下文中可能代表优化异常或噪声，因此在预处理中会被剔除。
    2. 策略选择 (Strategy): 
       - 采用 ArcSinh 变换: 有效压缩 R_a 的动态范围，同时完美保留负数符号信息。
       - 无加权回归 (Unweighted): 由于 R_a 主体范围平稳且同数量级，去除额外的样本权重能提升模型对整体趋势的拟合。
       - 窄范围 Epsilon: 针对 ArcSinh 变换后的数值范围，采用较小的不敏感区以捕捉细节。

Usage:
    python svm_prediction_R_a.py
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
    backup_filenames:Tuple[str, ...] = ("equivalent_circuit_parameters_optimized_Y11.csv",)
    model_save_path: str = "svm_model_ArcSinh_Impedance_a.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "a"
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 策略更新 (2026-02-13 - 针对 Branch 'a' 优化): 
                # 1. C 值范围: 保持适中，防止过拟合
                'regressor__svr__C': [100, 500, 1000, 3000, 5000], 
                
                # 2. Gamma: 控制 RBF 核的影响范围
                'regressor__svr__gamma': ['scale', 0.1, 0.5],
                
                # 3. Epsilon (不敏感区): 
                # 针对 ArcSinh 变换后的 Branch 'a' 数据 (值域主要在 -3 到 -6 之间)，
                # 0.05~0.2 的 epsilon 能有效忽略数据中的微小噪声，避免过度拟合锯齿状波动。
                'regressor__svr__epsilon': [0.05, 0.1, 0.2],
                'regressor__svr__kernel': ['rbf'] 
            }
        
        # 减少搜索次数以快速验证
        self.n_iter_search = 10  # 保持适度的搜索次数
        self.cv_folds = 5        # 5折交叉验证保证评估可靠性

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
        
        # 严格筛选目标分支 (a)
        mask = (df['Branch_Type'] == self.config.target_branch_type) & \
               (df['Branch_ID'] == self.config.target_branch_id)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (R_a 专用) ---
        # 现象: R_a 绝大多数为负值 (R ~ -20...-250)，仅有极个别(2个)正值(R ~ 1500)。
        # 判定: 这些正值大概率为优化算法未收敛或物理上无意义的噪点，应予以剔除以防止干扰回归面。
        if self.config.target_branch_id == 'a':
            mask_valid = df_filtered['R'] < 0
            n_removed = (~mask_valid).sum()
            if n_removed > 0:
                df_filtered = df_filtered[mask_valid]
                logger.warning(f"针对 Branch 'a' 移除了 {n_removed} 个正值异常样本 (R > 0)")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df_filtered)}")
        return df_filtered

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程
        注意：此前我们在这里做目标变换，现在改为在 Pipeline 中使用 TransformedTargetRegressor 做变换。
        这里只返回原始 Target R。
        
        修改注记 (2026-02-13): 
        移除了针对 Branch 'a' 有害的倒数特征 (inv_xi)。
        R_a 在 xi=0 处是连续平滑的，引入 1/xi 会人为制造特征空间的不连续性，干扰 SVM 学习。
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 (保留) ---
        # 阻抗/导纳幅值与相位是预测 R 的强相关特征
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)

        # --- 2. 形状特征 (精简) ---
        # R_a 随 xi 变化通常较为平滑 (可能是二次曲线或单调的)，保留平方项以捕捉非线性。
        # 移除了 inv_xi 等倒数特征，因为它们适用于有极点(Resonance)的分支，不适用于串联电阻。
        X['xi_2'] = X['xi'] ** 2

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
        # 优化: 增加 cache_size 并在极端情况下限制 max_iter 防止死循环
        base_svr = SVR(cache_size=1000, max_iter=50000, tol=1e-3)
        
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
        # 针对 Branch 'a' 的优化 (2026-02-13):
        # 原始 R_a 数据在同一数量级 (-20 ~ -250) 且较为平稳。
        # 之前针对宽动态范围设计的 Sqrt 权重反而可能放大噪声的影响。
        # 既然我们已经剔除了明显的正值离群点，采用均等权重 (weights=None) 能让模型更关注整体的平均拟合效果。
        weights = None
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(R))...")
        logger.info(f"样本加权策略: None (Standard Unweighted)")
        
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
    
    with open("results_arcsinh.txt", "w", encoding="utf-8") as f:
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
