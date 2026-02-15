"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'L_f' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 L_f 分支进行预测。

Author: GitHub Copilot (Modified based on R_f framework)
Date: 2026-02-15
Description:
    此脚本针对 extracted_RL_Series_Y11_wide.csv 中的 L_f 分支进行 SVR 建模与预测。
    
    问题复盘 (Problem Analysis for Branch 'L_f'):
    1. 数据分布 (Data Distribution): 
       - L_f 数据特征为负值 (Range: -25.21 ~ -4.53)。
       - 数据呈现左偏态。
       - 物理上在此等效电路模型中表现为负电感特性（或容性效应）。
       
    2. 策略选择 (Strategy): 
       - 清洗策略: 强制 L_f < 0，设置合理的负值范围 (如 -100 ~ -0.1)，排除 0 或 正值异常。
       - 目标变换 (ArcSinh): 对负偏态分布有效。L_f 在 [-25, -4] 区间变换后约为 [-3.9, -2.1]。
         ArcSinh 变换在处理负值且偏离 0 的数据时，能很好地保留相对关系。
       - 特征工程: 同步引入 1/xi, Z_mag 等物理特征。

Usage:
    python svm_prediction_L_f.py
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    model_save_path: str = "svm_model_ArcSinh_Impedance_L_f.pkl"
    result_save_path: str = "results_arcsinh_L_f.txt"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "f" # 目标分支为 f
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 30
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 针对 Branch 'L_f' 优化: 
                # L_f (abs ~ 4-25) 量级较小
                'regressor__svr__C': [100, 500, 1000, 2000, 5000], 
                
                # Gamma: 适中偏大
                'regressor__svr__gamma': ['scale', 0.01, 0.1, 0.5],
                
                # Epsilon: 在 ArcSinh 变换域 (值域 delta ~ 2.0) 中，0.01 精度足够
                'regressor__svr__epsilon': [0.001, 0.005, 0.01, 0.02, 0.05],
                'regressor__svr__kernel': ['rbf'] 
            }

# --- 2. 数据处理模块 ---
class DataProcessor:
    """处理数据加载、清洗和特征工程的类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _find_data_file(self) -> Path:
        candidates = [
            Path(self.config.input_filename),
            Path(__file__).parent / self.config.input_filename,
            Path("E:/ruanjian/GitHubDesktop/Support-vector-machines-are-used-for-impedance-prediction") / self.config.input_filename
        ] + [Path(f) for f in self.config.backup_filenames]
        
        for path in candidates:
            if path.exists():
                logger.info(f"找到数据文件: {path}")
                return path
        raise FileNotFoundError(f"未找到输入数据文件。尝试查找路径: {[str(p) for p in candidates]}")

    def load_and_filter(self) -> pd.DataFrame:
        file_path = self._find_data_file()
        logger.info(f"正在读取数据文件: {file_path}")
        
        df = pd.read_csv(file_path)
        
        target_col = f"L_{self.config.target_branch_id}"
        
        # 兼容处理: 如果是 wide 表，查找 L_{id} 列
        if target_col in df.columns:
            logger.info(f"检测到宽表格式 (Wide Format)，提取 {target_col} 列...")
            df = df.copy()
            df['target'] = df[target_col] # 将目标列重命名为 target 以匹配后续逻辑
            
            # 过滤掉非数值或缺失值
            df = df.dropna(subset=['target'])
             
        else:
            # Long format logic (legacy or for backup files)
            mask = (df['Branch_Type'] == self.config.target_branch_type) & \
                   (df['Branch_ID'] == self.config.target_branch_id)
            df = df[mask].copy()
            if 'L' in df.columns:
                df['target'] = df['L']
            else:
                 raise ValueError("未找到目标列 (L_f 或 L)")
            
        if df.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (L_f 专用) ---
        # 现象: L_f 为负值 (Range: -25 ~ -4)。
        # 策略: 移除 0 或正值，允许负值。
        min_limit = -1000.0 # 宽松下限
        max_limit = -0.1     # 必须为显著负值
        
        mask_range = (df['target'] >= min_limit) & (df['target'] <= max_limit)
        n_removed_range = (~mask_range).sum()
        
        if n_removed_range > 0:
            df = df[mask_range]
            logger.warning(f"根据阈值 (L >= {min_limit} and L <= {max_limit}) 移除了 {n_removed_range} 个异常样本")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df)}")
        logger.info(f"L_f 统计: Min={df['target'].min():.2f}, Max={df['target'].max():.2f}, Mean={df['target'].mean():.2f}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程 - 同步 R_c/R_d 的策略
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # --- 2. 形状特征 (Singularity Handling) ---
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9) 
        
        # 平方项保留，对应对称性
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)

        y_values = df['target']
        return X, y_values

# --- 3. 模型训练模块 ---

class ImpedanceSVM:
    """SVM 模型封装"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_model = None
        self.search_results = None

    def train(self, X_train, y_train):
        # 1. Manual Scaling (RobustScaler 适合有偏态分布的数据)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 2. Configure SVR and Search
        clean_param_dist = {
            k.replace('regressor__svr__', ''): v 
            for k, v in self.config.param_distributions.items()
        }
        
        # Base estimator is SVR 
        base_svr = SVR(cache_size=1000, max_iter=500000, tol=1e-4)
        
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
        # ArcSinh 变换: 对负值也有效 (单调递增)，能压缩 large magnitude (left tail)
        ttr_model = TransformedTargetRegressor(
            regressor=search,
            func=np.arcsinh,
            inverse_func=np.sinh,
            check_inverse=True
        )
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(L_f))...")
        
        start_time = time.time()
        
        # Fit TTR
        ttr_model.fit(X_train_scaled, y_train)
        
        elapsed = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {elapsed:.2f}秒")
        best_search = ttr_model.regressor_
        logger.info(f"最佳参数: {best_search.best_params_}")
        logger.info(f"最佳 CV Score (Neg MSE): {best_search.best_score_:.4e}")
        
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

def evaluate_performance(model, X_test, y_test_true, save_path="results_arcsinh_L_f.txt"):
    """
    评估 (都在物理域 L 进行)
    """
    logger.info("\n=== 模型评估报告 ===")
    
    y_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(y_test_true, y_pred)
    
    # 2. MAPE (L_f 远离 0)
    diff = np.abs(y_test_true - y_pred)
    mape = np.mean(diff / (np.abs(y_test_true) + 1e-3)) * 100
    
    # 3. RMSE & MAE
    mse = mean_squared_error(y_test_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true, y_pred)
    
    log_msg = f"[Physical Domain L_f] R2: {r2:.4f}\n[Physical Domain L_f] RMSE: {rmse:.4f}\n[Physical Domain L_f] MAE: {mae:.4f}\n[Physical Domain L_f] MAPE: {mape:.2f}%"
    logger.info(log_msg)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True L': y_test_true.values,
            'Pred L': y_pred,
            'Diff L (Abs)': np.abs(y_test_true.values - y_pred),
            'Rel Err %': np.abs(y_test_true.values - y_pred) / (np.abs(y_test_true.values) + 1e-3) * 100
        })
        results['Sign Match'] = np.sign(results['True L']) == np.sign(results['Pred L'])
        
        f.write("\n--- 最差预测样本 Top 10 ---\n")
        f.write(results.sort_values('Diff L (Abs)', ascending=False).head(10).to_string() + "\n")
        
        f.write("\n--- 最佳预测样本 Top 10 ---\n")
        f.write(results.sort_values('Diff L (Abs)', ascending=True).head(10).to_string() + "\n")
        
        f.write("\n--- 符号错误 ---\n")
        f.write(f"Count: {(~results['Sign Match']).sum()} / {len(results)}\n")
        
    analyze_samples(y_test_true, y_pred)

def analyze_samples(y_true, y_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True L': y_true.values,
        'Pred L': y_pred,
        'Diff L (Abs)': np.abs(y_true.values - y_pred),
        'Rel Err %': np.abs(y_true.values - y_pred) / (np.abs(y_true.values) + 1e-3) * 100
    })
    
    results['Sign Match'] = np.sign(results['True L']) == np.sign(results['Pred L'])
    
    print("\n--- 最差预测样本 Top 5 (按 L 绝对误差) ---")
    print(results.sort_values('Diff L (Abs)', ascending=False).head(5).to_string(float_format="%.4f"))
    
    print("\n--- 最佳预测样本 Top 5 (按 L 绝对误差) ---")
    print(results.sort_values('Diff L (Abs)', ascending=True).head(5).to_string(float_format="%.4f"))
    
    sign_errors = (~results['Sign Match']).sum()
    print(f"\n符号预测错误样本数: {sign_errors} / {len(results)}")

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
        trainer = ImpedanceSVM(config)
        trainer.train(X_train, L_train)
        
        # 评估
        evaluate_performance(trainer.best_model, X_test, L_test, save_path=config.result_save_path)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
