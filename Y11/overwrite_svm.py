
import os

target_file = r"e:\ruanjian\GitHubDesktop\Support-vector-machines-are-used-for-impedance-prediction\Y11\svm_prediction_R_a.py"

new_content = r'''"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'a' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 R_a 分支进行预测。

Author: GitHub Copilot (Refactored for R_a)
Date: 2026-03-01 (Updated: Batch Processing Integration)
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'a' 的电阻 R 进行预测的模型。
    
    Update 2026-03-01:
    - 集成 batch_processing.py 生成的 Train/Test 数据集。
    - 严格的数据清洗策略：
        1. 剔除 R >= -0.1 的物理无效值 (正值和近零值)。
        2. 剔除 R < -500 的极端异常值 (基于 IQR 统计分析 -110，稍微放宽以容纳可能的非正态尾部)。
    
    Data Analysis Summary:
    - Training Set: 存在极个别正值 (2个) 和极端负值 (-1e14)，需严格过滤。
    - Testing Set: 范围稳定在 [-100, -10]，无明显异常。

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
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
    # 指定训练集和测试集文件路径
    # 注意: 这里假设脚本在 Y11 目录下运行，或者能找到相对路径
    train_file: str = "Ideal_Power_Grid_Train_Processed_Y11_RL_Wide.csv"
    test_file: str = "Ideal_Power_Grid_Test_Processed_Y11_RL_Wide.csv"
    
    model_save_path: str = "svm_model_Y11_R_a.pkl"
    
    target_branch_id: str = "a"  # 我们预测的目标分支后缀
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 15  # 适度搜索次数
    cv_folds: int = 5
    random_state: int = 42
    
    # 数据清洗阈值
    val_min_threshold: float = -500.0  # 下限: 剔除 -1e14 这种极端值
    val_max_threshold: float = -0.1    # 上限: 剔除 正值 和 0

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # C: 正则化参数。
                'regressor__svr__C': [10, 50, 100, 300, 500, 1000], 
                
                # Gamma: RBF 核系数。
                'regressor__svr__gamma': ['scale', 0.01, 0.05, 0.1, 0.2],
                
                # Epsilon: 不敏感损失区的宽度。
                'regressor__svr__epsilon': [0.01, 0.05, 0.1, 0.2],
                'regressor__svr__kernel': ['rbf'] 
            }

# --- 2. 数据处理模块 ---
class DataProcessor:
    """处理数据加载、清洗和特征工程的类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _load_single_file(self, filename: str, dataset_name: str) -> pd.DataFrame:
        """加载并清洗单个文件"""
        # 尝试直接路径
        file_path = Path(filename)
        if not file_path.exists():
             # 尝试在当前目录查找 (假设在根目录运行)
            file_path = Path("Y11") / filename
            if not file_path.exists():
                # 尝试绝对路径 (如果传入的是文件名)
                print(f"[Warn] Cannot find {filename} in local or Y11 subdir.")
                
        if not file_path.exists():
            raise FileNotFoundError(f"未找到{dataset_name}文件: {filename}")
        
        logger.info(f"正在读取 {dataset_name}: {file_path}")
        df = pd.read_csv(file_path)
        
        target_col = f"R_{self.config.target_branch_id}"
        if target_col not in df.columns:
            # 可能是列名没有前缀? 检查一下
            if 'R' in df.columns and len(df.columns) < 15: # 简单检查
                 logger.warning(f"未找到 {target_col}, 尝试使用 'R'")
                 target_col = 'R'
            else:
                 raise ValueError(f"列 {target_col} 在 {dataset_name} 中不存在！Available: {df.columns[:5]}")
            
        # 重命名目标列
        df = df.rename(columns={target_col: 'R'})
        
        # --- 严格数据清洗 ---
        initial_count = len(df)
        
        # 1. 去除 NaN (以及必须存在的特征列)
        cols_to_check = ['R'] + list(self.config.base_features)
        df = df.dropna(subset=cols_to_check)
        
        # 2. 物理约束: R < 0 (针对 Branch 'a')
        # 根据统计，R_a 应该全是负值。正值往往是拟合失败或数值误差。
        valid_sign_mask = df['R'] < self.config.val_max_threshold
        
        # 3. 统计学约束: 去除极端负异常值 (如 -1e14)
        valid_range_mask = df['R'] > self.config.val_min_threshold
        
        final_mask = valid_sign_mask & valid_range_mask
        df_clean = df[final_mask].copy()
        
        removed_count = initial_count - len(df_clean)
        
        n_pos = (~valid_sign_mask).sum()
        n_extreme = (~valid_range_mask).sum()
        
        logger.info(
            f"[{dataset_name}] 原始: {initial_count}, 清洗后: {len(df_clean)}, "
            f"移除: {removed_count} (Pos/Zero: {n_pos}, Extreme Neg: {n_extreme})"
        )
            
        return df_clean

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载训练集和测试集"""
        train_df = self._load_single_file(self.config.train_file, "训练集")
        test_df = self._load_single_file(self.config.test_file, "测试集")
        return train_df, test_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """特征工程"""
        X = df[list(self.config.base_features)].copy()
        
        # --- 特征增强 ---
        # 1. 电气特征
        # 增加小量防止除零
        S_mag = np.sqrt(X['P']**2 + X['Q']**2) + 1e-9
        X['Z_mag'] = (X['V']**2) / S_mag
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)
        
        # 相位 (rad)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # 2. 形状特征
        X['xi_sq'] = X['xi'] ** 2
        
        y = df['R']
        return X, y

# --- 辅助函数 (需在顶层) ---
def arcsinh_transformer(x):
    return np.arcsinh(x)

def arcsinh_inverse(x):
    return np.sinh(x)

# --- 3. 主流程 ---

def main():
    config = ExperimentConfig()
    processor = DataProcessor(config)
    
    try:
        # 1. 加载数据
        train_df, test_df = processor.load_data()
        
        # 2. 准备特征
        X_train, y_train = processor.prepare_features(train_df)
        X_test, y_test = processor.prepare_features(test_df)
        
        logger.info(f"最终用于训练的样本数: {len(X_train)}")
        logger.info(f"最终用于测试的样本数: {len(X_test)}")
        
        # 3. 构建 Pipeline
        # TransformedTargetRegressor 内部使用 regressor 参数
        # 我们需要在 regressor 内部放入 RandomizedSearchCV
        # 而 RandomizedSearchCV 内部再放入 Pipeline
        
        # A. 内部 Pipeline (Scaler + SVR)
        # 注意: 参数名前缀必须匹配 Pipeline 中的步骤名
        inner_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('svr', SVR(cache_size=1000, max_iter=50000, tol=1e-3))
        ])
        
        # B. 参数网格转换
        # config.param_distributions 的键是 'regressor__svr__C' 这种形式 (为了兼容旧代码习惯?)
        # 这里我们需要将其转换为 inner_pipeline 能识别的格式 'svr__C'
        clean_param_dist = {
            k.replace('regressor__svr__', 'svr__'): v 
            for k, v in config.param_distributions.items()
        }
        
        # C. 搜索对象
        search = RandomizedSearchCV(
            inner_pipeline,
            param_distributions=clean_param_dist,
            n_iter=config.n_iter_search,
            cv=config.cv_folds,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=1,
            random_state=config.random_state
        )
        
        # D. 最终模型 (目标变换)
        model = TransformedTargetRegressor(
            regressor=search,
            func=arcsinh_transformer,
            inverse_func=arcsinh_inverse
        )
        
        logger.info("开始训练 (ArcSinh Transformed SVR)...")
        start_time = time.time()
        model.fit(X_train, y_train)
        logger.info(f"训练耗时: {time.time() - start_time:.2f}s")
        
        # 获取最佳参数 (有些复杂，因为包裹了多层)
        try:
            best_params = model.regressor_.best_params_
            logger.info(f"最佳参数: {best_params}")
        except:
            pass
        
        # 4. 评估
        logger.info("--- 模型评估 ---")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # R2 Score
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # RMSE
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # MAPE (Mean Absolute Percentage Error)
        # 过滤掉分母为0的情况 (虽然清洗过 R < -0.1)
        non_zero_mask = np.abs(y_test) > 1e-6
        mape_test = np.mean(np.abs((y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / y_test[non_zero_mask])) * 100
        
        logger.info(f"Train R2: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
        logger.info(f"Test  R2: {r2_test:.4f} | RMSE: {rmse_test:.4f}")
        logger.info(f"Test MAPE: {mape_test:.2f}%")
        
        # 残差分析
        residuals = y_test - y_pred_test
        logger.info(f"残差均值 (Bias): {np.mean(residuals):.4f}")
        logger.info(f"残差标准差: {np.std(residuals):.4f}")
        
        # 5. 保存模型
        joblib.dump(model, config.model_save_path)
        logger.info(f"模型保存成功: {config.model_save_path}")

    except Exception as e:
        logger.error(f"运行时发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()
'''

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print(f"Overwrote {target_file}")
