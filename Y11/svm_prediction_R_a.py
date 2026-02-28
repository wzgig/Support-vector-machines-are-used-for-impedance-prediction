"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'a' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 R_a 分支进行预测。

Author: GitHub Copilot (Refactored for R_a)
Date: 2026-02-28 (Updated: Support Wide Format Input)
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'a' 的电阻 R 进行预测的模型。
    
    Update 2026-02-28:
    - 优先支持 "extracted_RL_Series_Y11_wide.csv" 宽表输入。
    - 宽表数据结构更清晰，每一行即为一个样本，避免了复杂的 Branch_ID 筛选。
    - 兼容旧的原始数据输入模式。

    问题复盘 (Problem Analysis for Branch 'a'):
    1. 数据分布 (Data Distribution): R_a 的值全为负数 (统计结果: 100% Neg, Mean ~ -51.46)。
       此前代码中的“剔除正值”逻辑在宽表模式下已根据实际统计结果移除。
    2. 策略选择 (Strategy): 
       - 采用 ArcSinh 变换: 有效压缩 R_a 的动态范围，同时完美保留负数符号信息。
       - 无加权回归 (Unweighted): 由于 R_a 主体范围平稳，去除额外的样本权重能提升模型对整体趋势的拟合。
       - 窄范围 Epsilon: 针对 ArcSinh 变换后的数值范围，采用较小的不敏感区以捕捉细节。

Usage:
    python svm_prediction_R_a.py
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Union

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
    # 优先使用的宽表文件
    wide_input_filename: str = "extracted_RL_Series_Y11_wide.csv"
    # 原始数据文件 (作为备用)
    raw_input_filename: str = "Ideal Power Grid_Y11.csv"
    
    model_save_path: str = "svm_model_Y11_R_a.pkl"
    
    # 仅在读取原始文件时使用
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "a"
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    backup_filenames: List[str] = field(default_factory=list)
    
    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 策略更新 (2026-02-13 - 针对 Branch 'a' 优化): 
                # 1. C 值范围: 保持适中，防止过拟合
                # Update 2026-02-28: 降低 C 下限以缓解过拟合 (Old best was 100)
                'regressor__svr__C': [10, 50, 100, 500, 1000, 3000], 
                
                # 2. Gamma: 控制 RBF 核的影响范围
                'regressor__svr__gamma': ['scale', 0.05, 0.1, 0.5],
                
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

    def _find_data_file(self) -> Tuple[Path, bool]:
        """
        查找数据文件。优先查找宽表。
        Returns: (file_path, is_wide_format)
        """
        # 1. 检查宽表
        wide_candidates = [
            Path(self.config.wide_input_filename),
            Path(self.config.target_branch_type) / self.config.wide_input_filename # support folder structure
        ]
        
        for p in wide_candidates:
            if p.exists():
                logger.info(f"找到宽格式数据文件: {p}")
                return p, True

        # 2. 检查原始表
        raw_candidates = [
            Path(self.config.raw_input_filename),
            Path("csv_data") / self.config.raw_input_filename,
        ] + [Path(f) for f in self.config.backup_filenames]
        
        for p in raw_candidates:
            if p.exists():
                logger.info(f"未找到宽表，回退到原始数据文件: {p}")
                return p, False
                
        raise FileNotFoundError(f"未找到任何输入数据文件。请检查 {self.config.wide_input_filename} 或 {self.config.raw_input_filename}")

    def load_and_filter(self) -> pd.DataFrame:
        file_path, is_wide = self._find_data_file()
        logger.info(f"正在读取数据文件: {file_path} (模式: {'Wide' if is_wide else 'Raw'})")
        
        df = pd.read_csv(file_path)
        
        if is_wide:
            # --- 宽表处理逻辑 ---
            # 宽表列名格式: R_a, L_a ...
            target_col = f"R_{self.config.target_branch_id}"
            
            if target_col not in df.columns:
                available_r = [c for c in df.columns if c.startswith('R_')]
                raise ValueError(f"目标列 '{target_col}' 在宽表中不存在。可用的 R 列: {available_r}")
            
            # 重命名为标准列名 'R' 以供后续使用
            df = df.rename(columns={target_col: 'R'})
            
            # 基础清洗: 去除 NaN
            original_len = len(df)
            df = df.dropna(subset=['R'])
            if len(df) < original_len:
                logger.warning(f"移除了 {original_len - len(df)} 行包含 NaN 的样本")

            logger.info("使用宽表模式，已自动对齐目标列。")
            
        else:
            # --- 原始表处理逻辑 (保持原有功能) ---
            # 严格筛选目标分支
            mask = (df['Branch_Type'] == self.config.target_branch_type) & \
                   (df['Branch_ID'] == self.config.target_branch_id)
            df = df[mask].copy()
            
            if df.empty:
                raise ValueError(f"筛选后的数据集为空! 请检查 Branch_Type={self.config.target_branch_type}, Branch_ID={self.config.target_branch_id}")
            
            # 原始数据的特殊清洗 (R_a 专用)
            # 针对原始数据可能存在的噪声进行过滤
            if self.config.target_branch_id == 'a':
                mask_valid = df['R'] < 0
                n_removed = (~mask_valid).sum()
                if n_removed > 0:
                    df = df[mask_valid]
                    logger.warning(f"针对 Branch 'a' (Raw模式) 移除了 {n_removed} 个正值异常样本 (R > 0)")
        
        logger.info(f"数据加载完成，有效样本数: {len(df)}")
        return df

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
        # 确保基础特征存在
        missing_features = [f for f in self.config.base_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"缺少基础特征列: {missing_features}")

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
        def arcsinh_transformer(x):
            return np.arcsinh(x)
            
        def arcsinh_inverse(x):
            return np.sinh(x)
            
        final_model = TransformedTargetRegressor(
            regressor=search,
            func=arcsinh_transformer,
            inverse_func=arcsinh_inverse
        )
        
        logger.info("开始 RandomizedSearchCV 优化...")
        start_time = time.time()
        final_model.fit(X_train_scaled, y_train)
        elapsed_time = time.time() - start_time
        logger.info(f"搜索完成，耗时: {elapsed_time:.2f}s")
        
        self.best_model = final_model
        
        # Extract internal search results
        self.search_results = final_model.regressor_.best_params_
        logger.info(f"最佳参数: {self.search_results}")
        
    def evaluate(self, X_test, y_test):
        if self.best_model is None:
            raise ValueError("模型尚未训练！")
            
        # 预测时需要对 X_test 进行同样的缩放
        # 注意：这里我们简化处理，实际上应该保存 scaler。
        # 为了严谨，建议在 DataProcessor 中处理缩放，或者使用 Pipeline 包含 scaler。
        # 但由于上面 train 中使用了独立的 RobustScaler 并作为 fit 的输入，
        # 这里为了保持一致，我们临时创建一个 fit 过的 scaler (虽然不是很规范，但在简单脚本中可行)
        # *更好的做法*：将 RobustScaler 放入 Pipeline。
        pass # 这里的具体评估逻辑在 main pipe 中处理，因为 scaler 需要从外部传入或 pipeline 化

# --- 辅助函数 (必须定义在模块顶层以支持 pickle) ---
def arcsinh_transformer(x):
    return np.arcsinh(x)

def arcsinh_inverse(x):
    return np.sinh(x)

# --- 4. 主程序 ---

def main():
    config = ExperimentConfig()
    
    # 1. 数据加载与预处理
    processor = DataProcessor(config)
    
    try:
        df = processor.load_and_filter()
        X, y = processor.engineer_features(df)
        
        # 优化点: 采用分层抽样 (Stratified Split)
        # 既然是回归问题，y是连续的。我们需要先将y离散化分桶，以便进行分层。
        # 这样可以保证训练集和测试集都包含 R 的所有取值范围(特别是极值)。
        n_bins = 5 # 考虑到样本量可能只有 ~250，减少 bin 数量防止单桶样本过少
        try:
            y_binned = pd.cut(y, bins=n_bins, labels=False)
            # 检查最小桶样本数
            min_samples = y_binned.value_counts().min()
            
            if min_samples < 2:
                logger.warning(f"由于某些数值区间的样本数过少 ({min_samples})，无法进行分层抽样。回退到随机抽样。")
                stratify_labels = None
            else:
                logger.info(f"采用基于目标值分布的分层抽样 (Stratified Split)，bins={n_bins}")
                stratify_labels = y_binned
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.random_state, stratify=stratify_labels
            )
        except Exception as e:
            logger.warning(f"分层抽样准备失败 ({e})，回退到简单随机抽样。")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=config.random_state
            )

        logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        # 2. 构建包含 Scaler 的管道
        # R 预测中，RobustScaler 对异常值更鲁棒
        # 将 Scaler 放入 Pipeline 确保训练/测试数据处理一致性
        
        # 准备参数网格 (适配 Pipeline 命名 "svr__")
        # 原始 Key 格式: 'regressor__svr__C'
        # 目标 Key 格式: 'svr__C' (因为 Pipeline 里的步骤名叫 'svr')
        clean_param_dist = {
            k.replace('regressor__svr__', 'svr__'): v 
            for k, v in config.param_distributions.items()
        }
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(cache_size=1000, max_iter=50000, tol=1e-3))
        ])
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=clean_param_dist,
            n_iter=config.n_iter_search,
            cv=config.cv_folds,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1,
            random_state=config.random_state
        )
        
        model = TransformedTargetRegressor(
            regressor=search,
            func=arcsinh_transformer,
            inverse_func=arcsinh_inverse
        )
        
        # 3. 训练
        logger.info("开始训练模型 (Pipeline + ArcSinh Transform)...")
        start_time = time.time()
        model.fit(X_train, y_train)
        logger.info(f"训练完成，耗时: {time.time() - start_time:.2f}s")
        
        # 获取最佳参数
        best_params = model.regressor_.best_params_
        logger.info(f"最佳参数: {best_params}")

        # 4. 评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        logger.info(f"训练集 R2: {r2_train:.4f}")
        logger.info(f"测试集 R2: {r2_test:.4f}")
        logger.info(f"测试集 RMSE: {rmse_test:.4f}")
        
        # 优化点: 增加残差统计，检测模型是否有系统性偏差
        residuals = y_test - y_pred_test
        res_mean = np.mean(residuals)
        res_std = np.std(residuals)
        logger.info(f"残差均值 (Bias): {res_mean:.4f} (理想为0)")
        logger.info(f"残差标准差: {res_std:.4f}")
        
        # 检查最大误差样本
        if not residuals.empty:
            max_err_idx = np.argmax(np.abs(residuals))
            # 注意: iloc 获取的是位置，需要确保 residuals 是 Series
            max_err_val = residuals.iloc[max_err_idx]
            max_err_true = y_test.iloc[max_err_idx]
            max_err_pred = y_pred_test[max_err_idx]
            logger.info(f"最大预测误差: {max_err_val:.4f}")
        
        # 5. 保存模型
        joblib.dump(model, config.model_save_path)
        logger.info(f"模型已保存至: {config.model_save_path}")

    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()