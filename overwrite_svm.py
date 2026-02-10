# -*- coding: utf-8 -*-
import os

NEW_CONTENT = r'''"""
Impedance Prediction using SVM (Log-Impedance Strategy)
===========================================================
基于支持向量机（SVM）的阻抗预测模型 - 对数阻抗域（Log-Impedance Domain）策略

Author: GitHub Copilot (Refactored for Research)
Date: 2026-02-09
Description:
    此脚本实现了一种物理信息增强的机器学习策略，用于预测等效电路参数中的电阻 R。
    
    问题分析 (Problem Analysis):
    直接预测 R (Impedance) 困难，因为 R 的动态范围极大 (200 ~ 1,000,000+)。
    预测 G (Admittance = 1/R) 虽然在数值上稳定，但在 R 很大 (G 接近 0) 时存在严重的误差放大效应 ($dR/dG = -R^2$)。
    
    改进策略 (Improved Strategy):
    采用 "对数阻抗回归" (Log-Space Regression)。
    目标变量设为 y = log10(R)。
    这样可以将不同数量级的 R 压缩到线性区间 (如 2.3 ~ 6.0)，使 SVM 能均衡学习所有量级的特征，
    且 Log 域的均方误差(MSE)在物理上近似对应于相对误差(MAPE)的最小化。

Dependnecies:
    pandas, numpy, scikit-learn, joblib

Usage:
    python svm_prediction_admittance_G.py
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
    # 文件路径配置
    input_filename: str = "equivalent_circuit_parameters_optimized_accurate_Y11.csv"
    backup_filenames:Tuple[str, ...] = ("equivalent_circuit_parameters_optimized_Y11.csv",)
    model_save_path: str = "svm_model_LogR_Impedance.pkl"
    
    # 筛选条件
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "e"
    
    # 特征列定义
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    # 模型超参数搜索空间
    # LogR 域的值域通常在 [2, 7] 之间 (对应 100 到 10M 欧姆)
    # StandardScaler 后变为 N(0,1)，因此常规 C, epsilon 适用
    param_distributions: dict = None
    n_iter_search: int = 15  
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                'regressor__svr__C': [0.1, 1, 10, 100, 500, 1000, 2000],
                'regressor__svr__epsilon': [0.001, 0.01, 0.05, 0.1, 0.2],
                'regressor__svr__gamma': ['scale', 0.01, 0.1, 0.2, 0.5],
                'regressor__svr__kernel': ['rbf'] # R 与物理参数通常高度非线性，linear可能不够
            }

# --- 2. 数据处理模块 ---

class DataProcessor:
    """处理数据加载、清洗和特征工程的类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _find_data_file(self) -> Path:
        """智能查找数据文件（支持多路径回退）"""
        candidates = [
            Path(self.config.input_filename),
            Path("csv_data") / self.config.input_filename,
        ] + [Path(f) for f in self.config.backup_filenames]
        
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"未找到输入数据文件。尝试查找路径: {[str(p) for p in candidates]}")

    def load_and_filter(self) -> pd.DataFrame:
        """加载 CSV 并根据 Branch Type/ID 进行筛选"""
        file_path = self._find_data_file()
        logger.info(f"正在读取数据文件: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 筛选特定分支的数据
        mask = (df['Branch_Type'] == self.config.target_branch_type) & \
               (df['Branch_ID'] == self.config.target_branch_id)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            raise ValueError(f"筛选后的数据集为空! 请检查 Branch_Type={self.config.target_branch_type}, Branch_ID={self.config.target_branch_id}")
            
        logger.info(f"数据加载完成，筛选后样本数: {len(df_filtered)}")
        return df_filtered

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        特征工程与目标变换 (Log-Space)
        
        Returns:
            X (DataFrame): 处理后的特征矩阵
            y_log (Series): 变换后的目标变量 (Log10 R)
            R (Series): 原始目标变量 (电阻 R)
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 物理特征增强 (Physics-Informed Features) ---
        # 计算视在功率幅值 |S|
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        
        # 计算阻抗幅值 |Z| = |V|^2 / |S|
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        
        # 计算相位 Phase = atan2(Q, P)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # 计算导纳幅值 |Y| (辅助特征)
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)

        # --- 多尺度奇异性特征 (Singularity Features) ---
        logger.info("添加倒数特征 (Inverse xi) 以捕捉潜在的谐振极点...")
        X['inv_xi_0.1'] = 1.0 / (X['xi'].abs() + 0.1)
        X['inv_xi_0.5'] = 1.0 / (X['xi'].abs() + 0.5)

        # --- 目标变换: R -> Log10(R) ---
        # 策略: 学习 y = Log10(R)。这样 MSE 损失函数对应于相对误差。
        R_values = df['R']
        
        # 极小值保护: 防止 log(0) 或 log(负数)，取绝对值加 epsilon
        # 物理上 R 应该是正的，但防止数值噪声
        R_safe = np.abs(R_values) + 1e-6
        y_log = np.log10(R_safe)
        
        self._log_statistics(R_values, y_log)
        
        return X, y_log, R_values

    def _log_statistics(self, R: pd.Series, y_log: pd.Series):
        """打印统计信息以验证变换效果"""
        logger.info("-" * 40)
        logger.info(f"原始 R 统计 (Min/Max/Skew): {R.min():.2e} / {R.max():.2e} / {R.skew():.2f}")
        logger.info(f"变换 LogR 统计 (Min/Max/Skew): {y_log.min():.2f} / {y_log.max():.2f} / {y_log.skew():.2f}")
        logger.info("说明: LogR 的分布通常比原始 R 更接近正态分布，且消除了数量级差异。")
        logger.info("-" * 40)

# --- 3. 模型训练模块 ---

class AdmittanceSVM:
    """SVM 模型封装"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_model = None
        self.search_results = None

    def build_pipeline(self) -> TransformedTargetRegressor:
        """构建机器学习流水线"""
        # Log 域已经很好，但再加一个 StandardScaler 归一化到 N(0,1) 通常对 SVM 更好
        base_pipeline = Pipeline([
            ('scaler_X', RobustScaler()), 
            ('svr', SVR())
        ])
        
        model = TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=StandardScaler() 
        )
        return model

    def train(self, X_train, y_train):
        """执行超参数搜索与训练"""
        model = self.build_pipeline()
        
        search = RandomizedSearchCV(
            model,
            param_distributions=self.config.param_distributions,
            n_iter=self.config.n_iter_search,
            cv=self.config.cv_folds,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1,
            random_state=self.config.random_state
        )
        
        logger.info(f"开始超参数搜索 (Target: LogR)...")
        start_time = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")
        logger.info(f"最佳参数: {search.best_params_}")
        
        self.best_model = search.best_estimator_
        self.search_results = search

    def save(self):
        """保存模型"""
        if self.best_model:
            joblib.dump(self.best_model, self.config.model_save_path)
            logger.info(f"模型已保存至: {self.config.model_save_path}")

# --- 4. 评估工具 ---

def evaluate_performance(model, X_test, y_test_log, R_test_original):
    """
    双域评估：
    1. 学习域 (Log Domain): 评估模型拟合能力
    2. 物理域 (Real Domain R): 评估实际应用精度
    """
    logger.info("\n=== 模型评估报告 ===")
    
    # 1. 预测 LogR
    y_pred_log = model.predict(X_test)
    
    # 计算 Log 域指标
    r2_log = r2_score(y_test_log, y_pred_log)
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    logger.info(f"[Learning Domain (LogR)] R2: {r2_log:.4f} | MSE: {mse_log:.4f}")
    
    # 2. 还原回 R (R = 10^y)
    R_pred = 10 ** y_pred_log
    
    # 计算 R 域指标
    r2_R = r2_score(R_test_original, R_pred)
    
    diff = np.abs(R_test_original - R_pred)
    mape = np.mean(diff / (np.abs(R_test_original) + 1e-9)) * 100
    
    logger.info(f"[Physical Domain (R)]    R2: {r2_R:.4f} | MAPE: {mape:.2f}%")
    
    analyze_samples(R_test_original, R_pred, y_test_log, y_pred_log)

def analyze_samples(R_true, R_pred, Log_true, Log_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True R': R_true.values,
        'Pred R': R_pred,
        'Diff R (Abs)': np.abs(R_true.values - R_pred),
        'Rel Err %': np.abs(R_true.values - R_pred) / (np.abs(R_true.values) + 1e-9) * 100,
        'True LogR': Log_true.values,
        'Pred LogR': Log_pred
    })
    
    print("\n--- 最差预测样本 Top 5 (按 R 绝对误差) ---")
    print(results.sort_values('Diff R (Abs)', ascending=False).head(5).to_string(float_format="%.4f"))
    
    print("\n--- 最佳预测样本 Top 5 (按 R 绝对误差) ---")
    print(results.sort_values('Diff R (Abs)', ascending=True).head(5).to_string(float_format="%.4f"))

# --- 主程序入口 ---

def main():
    try:
        # 1. 初始化配置
        config = ExperimentConfig()
        
        # 2. 数据准备
        processor = DataProcessor(config)
        df = processor.load_and_filter()
        X, y_log, R_original = processor.engineer_features(df)
        
        # 数据集划分 (We split indices to keep R_original aligned)
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=config.random_state
        )
        
        y_log_train = y_log.iloc[idx_train]
        y_log_test = y_log.iloc[idx_test]
        R_test_real = R_original.iloc[idx_test] 
        
        # 3. 模型训练
        trainer = AdmittanceSVM(config)
        trainer.train(X_train, y_log_train)
        
        # 4. 评估与保存
        evaluate_performance(trainer.best_model, X_test, y_log_test, R_test_real)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

TARGET_FILE = "svm_prediction_admittance_G.py"

try:
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(NEW_CONTENT)
    print(f"Successfully updated {TARGET_FILE}")
except Exception as e:
    print(f"Error: {e}")
