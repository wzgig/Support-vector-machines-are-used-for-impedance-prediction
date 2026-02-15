"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'e' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 R_e 分支进行预测。

Author: GitHub Copilot
Date: 2026-02-14
Description:
    针对 R_e 分支的特殊物理特性进行适配，同时保持与 R_d 分支代码框架一致。
    
    Processing Keep (针对 e 支路的特殊处理):
    1.  清洗策略: 允许负值，范围限制在 [-90000, 90000]。
    2.  特征工程: 引入 P/xi, Q/xi, V/xi 等显式倒数特征。
    3.  模型参数: 使用 StandardScaler，更大的 C 值和迭代次数。

Usage:
    python svm_prediction_R_e.py
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
    backup_filenames: Tuple[str, ...] = ("extracted_RL_Series_Y11_wide.csv",)
    model_save_path: str = "svm_model_ArcSinh_Impedance_e.pkl"
    result_save_path: str = "results_arcsinh_e.txt"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "e" # 目标分支为 e
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 50  # 增加搜索次数
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 针对 R_e 优化: C 范围扩大，偏向大值以拟合陡峭变化
                'regressor__svr__C': [100, 500, 1000, 5000, 10000, 20000], 
                # Gamma: R_e 对 xi 的变化非常敏感
                'regressor__svr__gamma': ['scale', 'auto', 0.01, 0.05, 0.1, 0.2],
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
            
        # --- 自动清洗逻辑 (R_e 专用) ---
        # 现象: R_e 包含正负大数值，且在奇点附近极大。
        # 策略: 允许负值，设定范围限制 [-40000, 40000] (基于 99% 分位数 ~35k 分析)
        min_limit = -40000.0
        max_limit = 40000.0
        
        mask_range = (df['R'] >= min_limit) & (df['R'] <= max_limit)
        n_removed_range = (~mask_range).sum()
        
        if n_removed_range > 0:
            df = df[mask_range]
            logger.warning(f"根据阈值 ({min_limit} <= R <= {max_limit}) 移除了 {n_removed_range} 个极值样本")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df)}")
        logger.info(f"R_e 统计: Min={df['R'].min():.2f}, Max={df['R'].max():.2f}, Mean={df['R'].mean():.2f}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程 - 增强版 (针对 R_e 奇异性)
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # --- 2. 形状特征 (Singularity Handling) ---
        # 关键: R_e 对 xi 的倒数极其敏感 (xi -> 0, R_e -> inf)
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9) 
        
        # 二阶项
        X['xi_2'] = X['xi'] ** 2
        X['inv_xi_2'] = 1.0 / (X['xi']**2 + 1e-9)
        
        # --- 3. 交互特征 (针对 R_e 增强) ---
        # 显式引入反比关系 P/xi, Q/xi, V/xi
        X['P_div_xi'] = X['P'] * X['inv_xi']
        X['Q_div_xi'] = X['Q'] * X['inv_xi']
        X['V_div_xi'] = X['V'] * X['inv_xi']

        R_values = df['R']
        return X, R_values

# --- 3. 模型训练模块 ---

class ImpedanceSVM:
    """SVM 模型封装"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_model = None
        self.search_results = None

    def train(self, X_train, y_train):
        # 1. Scaling: 使用 StandardScaler (R_e 特有，为了更好的收敛性)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 2. Configure SVR and Search
        clean_param_dist = {
            k.replace('regressor__svr__', ''): v 
            for k, v in self.config.param_distributions.items()
        }
        
        # Base estimator
        # 增加 cache_size 和 tol 优化大数值训练，max_iter 增大
        base_svr = SVR(cache_size=2000, max_iter=2000000, tol=1e-4)
        
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
        
        # 3. TransformedTargetRegressor with ArcSinh
        ttr_model = TransformedTargetRegressor(
            regressor=search,
            func=np.arcsinh,
            inverse_func=np.sinh,
            check_inverse=True
        )
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(R_e))...")
        
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

def evaluate_performance(model, X_test, R_test_true, save_path="results_arcsinh_e.txt"):
    """
    评估
    """
    logger.info("\n=== 模型评估报告 ===")
    
    R_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(R_test_true, R_pred)
    
    # 2. MAPE (处理 R_e 穿过 0 点的情况，分母加 1.0)
    diff = np.abs(R_test_true - R_pred)
    mape = np.mean(diff / (np.abs(R_test_true) + 1.0)) * 100
    
    # 3. RMSE & MAE
    mse = mean_squared_error(R_test_true, R_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(R_test_true, R_pred)
    
    log_msg = f"[Physical R_e] R2: {r2:.4f}\n[Physical R_e] RMSE: {rmse:.4f}\n[Physical R_e] MAE: {mae:.4f}\n[Physical R_e] MAPE (adj): {mape:.2f}%"
    logger.info(log_msg)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True R': R_test_true.values,
            'Pred R': R_pred,
            'Diff R (Abs)': np.abs(R_test_true.values - R_pred),
            'Rel Err %': np.abs(R_test_true.values - R_pred) / (np.abs(R_test_true.values) + 1.0) * 100
        })
        results['Sign Match'] = np.sign(results['True R']) == np.sign(results['Pred R'])
        
        f.write("\n--- 最差预测样本 Top 10 ---\n")
        f.write(results.sort_values('Diff R (Abs)', ascending=False).head(10).to_string() + "\n")
        
        f.write("\n--- 最佳预测样本 Top 10 ---\n")
        f.write(results.sort_values('Diff R (Abs)', ascending=True).head(10).to_string() + "\n")
        
        f.write("\n--- 符号错误 ---\n")
        f.write(f"Count: {(~results['Sign Match']).sum()} / {len(results)}\n")
        
    analyze_samples(R_test_true, R_pred)

def analyze_samples(R_true, R_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True R': R_true.values,
        'Pred R': R_pred,
        'Diff R (Abs)': np.abs(R_true.values - R_pred),
        'Rel Err %': np.abs(R_true.values - R_pred) / (np.abs(R_true.values) + 1.0) * 100
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
        
        # 按照 R_d 的方式分割 (虽然我们这里不需要 indices 做什么，但保持结构对齐)
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=config.random_state
        )
        
        R_train = R.iloc[idx_train]
        R_test = R.iloc[idx_test]
        
        # 训练
        trainer = ImpedanceSVM(config)
        trainer.train(X_train, R_train)
        
        # 评估
        evaluate_performance(trainer.best_model, X_test, R_test, save_path=config.result_save_path)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
