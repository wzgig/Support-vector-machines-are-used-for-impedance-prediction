"""
Impedance Prediction using SVM (ArcSinh Strategy) - Branch 'L_b' Specialized
==========================================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）
专门针对 RL_Series 拓扑中的 L_b 分支进行预测。

Author: GitHub Copilot (Refactored for L_b)
Date: 2026-02-14
Description:
    此脚本实现了一种针对 RL 串联电路中 Branch 'b' 的电感 L_b 进行预测的模型。
    
    问题复盘 (Problem Analysis for Branch 'L_b'):
    1. 数据分布 (Data Distribution): 
       - L_b 的数值范围跨度较大且呈现高度不对称性（例如从 -173 到 +0.8）。
       - 存在显著的奇异性：当 xi 接近 0 或 Q 接近 0 时，L_b 可能出现大幅度数值波动。
       - 与 R_b 类似，L_b 与 1/xi 存在很强的物理关联性 (Inverse Relationship)。
       
    2. 策略选择 (Strategy): 
       - 目标变换 (ArcSinh): Essential! 既然 L_b 跨越正负且包含奇异值 (-173 到 0.8)，
         ArcSinh 能将线性增长压缩为对数增长，同时无缝处理零点和负数，非常适合此类奇异数据。
       - 特征工程: 必须引入倒数特征 (1/xi)，因为物理上这似乎是一个反比关系。
       - 鲁棒缩放: 使用 RobustScaler 处理输入特征，抵抗极端值对缩放的影响。

Usage:
    python svm_prediction_L_b.py
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
    input_filename: str = "equivalent_circuit_parameters_optimized_accurate_Y11.csv" # 切换至宽表以获取正确的 L_b 数据
    backup_filenames:Tuple[str, ...] = ("extracted_RL_Series_Y11_wide.csv",)
    model_save_path: str = "svm_model_ArcSinh_Impedance_L_b.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "b" # 目标分支为 b
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 策略更新 (2026-02-14 - 针对 Branch 'L_b' 优化): 
                # 1. C 值: L_b 的动态范围 (-173 ~ 0.8) 比 R_b 小，但仍需要较大的 C 来拟合奇异点。
                'regressor__svr__C': [100, 1000, 5000, 10000, 20000], 
                
                # 2. Gamma: 数据变化剧烈，gamma 可能需要较高以捕捉局部特征
                'regressor__svr__gamma': ['scale', 0.1, 1.0],
                
                # 3. Epsilon: 在 ArcSinh 变换后，-173 变为 -5.8，0.8 变为 0.7。
                # 范围缩小了。Epsilon 应该比较小，适应对数空间。
                'regressor__svr__epsilon': [0.01, 0.05, 0.1],
                'regressor__svr__kernel': ['rbf'] 
            }
        
        # 减少搜索次数以快速验证
        self.n_iter_search = 15  # 保持适度的搜索次数
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
        
        # 兼容处理: 如果是 wide 表，没有 Branch_ID 列，但有 L_b 列。
        # 如果是 long 表，有 Branch_ID 列。
        if 'L_b' in df.columns and self.config.target_branch_id == 'b':
            # Wide format detected
            logger.info("检测到宽表格式 (Wide Format)，提取 L_b 列...")
            df = df.copy()
            df['target'] = df['L_b'] # 将目标列重命名为 target (L) 以匹配后续逻辑
            
            # 过滤掉非数值或缺失值
            df = df.dropna(subset=['target'])
             
        else:
            # Long format logic (legacy or for backup files) - 这里假设 long 表用 'L' 列表示电感
            # 但实际上 equivalent_circuit... 可能结构不同，需保持谨慎。
            # 为了专注于 wide 表处理 L_b，这里仅保留基本兼容性。
            logger.warning("未检测到 L_b 列，尝试使用通用逻辑（可能不适用）...")
            if 'Branch_Type' in df.columns:
                mask = (df['Branch_Type'] == self.config.target_branch_type) & \
                       (df['Branch_ID'] == self.config.target_branch_id)
                df = df[mask].copy()
                df['target'] = df['L'] # 假设 long 表有名为 L 的列
            else:
                 raise ValueError("无法识别数据格式，缺少 L_b 列")

        if df.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 自动清洗逻辑 (L_b 专用) ---
        # 现象: L_b 在某些条件下 (如 xi -> 0, Q -> 0) 有较大的负值 (-173)。
        # 策略: 设定阈值移除极端物理异常值。考虑到 -173 仍可能是物理模型拟合的结果，
        # 我们放宽阈值至 [-500, 500]，仅移除极端数值错误。
        range_limit = 500
        mask_range = (df['target'] >= -range_limit) & (df['target'] <= range_limit)
        n_removed_range = (~mask_range).sum()
        
        if n_removed_range > 0:
            df = df[mask_range]
            logger.warning(f"根据阈值 [{-range_limit}, {range_limit}] 移除了 {n_removed_range} 个极端样本")
        
        logger.info(f"数据加载完成，筛选后样本数: {len(df)}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程 - 针对 L_b 优化
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 1. 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        
        # --- 2. 形状特征 (Singularity Handling) ---
        # L_b 与 xi 反比 (1/xi 行为)。
        # 引入 1/xi 特征对线性化 L_b 的行为至关重要。
        # 加一个小 epsilon 防止 xi=0 精确除零
        X['inv_xi'] = 1.0 / (X['xi'] + 1e-9) 
        
        # 平方项保留，对应对称性或其他偶次效应
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
        # 针对 Branch 'b' 的优化 (2026-02-13):
        # 均等权重 (Unweighted) 通常稳健。
        weights = None
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(L_b))...")
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

def evaluate_performance(model, X_test, y_test_true):
    """
    评估 (都在物理域 L_b 进行，因为模型会自动反变换)
    """
    logger.info("\n=== 模型评估报告 ===")
    
    # model is the Pipeline([scaler, ttr])
    # predict calls transform -> ttr.predict -> search.predict -> inverse_func
    y_pred = model.predict(X_test)
    
    # 1. R2
    r2 = r2_score(y_test_true, y_pred)
    
    # 2. MAPE (处理 y_true 为 0 的情况)
    diff = np.abs(y_test_true - y_pred)
    # 对于 L_b ~ 0.01 - 100，1e-3 足够小。
    mape = np.mean(diff / (np.abs(y_test_true) + 1e-3)) * 100
    
    log_msg = f"[Physical Domain L_b] R2: {r2:.4f}\n[Physical Domain L_b] MAPE: {mape:.2f}%"
    logger.info(log_msg)
    
    output_res_file = "results_arcsinh_L_b.txt"
    with open(output_res_file, "w", encoding="utf-8") as f:
        f.write(log_msg + "\n")
        
        results = pd.DataFrame({
            'True L_b': y_test_true.values,
            'Pred L_b': y_pred,
            'Diff L_b (Abs)': np.abs(y_test_true.values - y_pred),
            'Rel Err %': np.abs(y_test_true.values - y_pred) / (np.abs(y_test_true.values) + 1e-3) * 100
        })
        results['Sign Match'] = np.sign(results['True L_b']) == np.sign(results['Pred L_b'])
        
        f.write("\n--- 最差预测样本 Top 5 ---\n")
        f.write(results.sort_values('Diff L_b (Abs)', ascending=False).head(5).to_string() + "\n")
        
        f.write("\n--- 符号错误 ---\n")
        f.write(f"Count: {(~results['Sign Match']).sum()} / {len(results)}\n")
        
    analyze_samples(y_test_true, y_pred)

def analyze_samples(y_true, y_pred):
    """详细的样本误差分析"""
    results = pd.DataFrame({
        'True L_b': y_true.values,
        'Pred L_b': y_pred,
        'Diff L_b (Abs)': np.abs(y_true.values - y_pred),
        'Rel Err %': np.abs(y_true.values - y_pred) / (np.abs(y_true.values) + 1e-3) * 100
    })
    
    # 添加符号检查
    results['Sign Match'] = np.sign(results['True L_b']) == np.sign(results['Pred L_b'])
    
    print("\n--- 最差预测样本 Top 5 (按 L_b 绝对误差) ---")
    print(results.sort_values('Diff L_b (Abs)', ascending=False).head(5).to_string(float_format="%.4f"))
    
    print("\n--- 最佳预测样本 Top 5 (按 L_b 绝对误差) ---")
    print(results.sort_values('Diff L_b (Abs)', ascending=True).head(5).to_string(float_format="%.4f"))
    
    # 统计符号错误数
    sign_errors = (~results['Sign Match']).sum()
    print(f"\n符号预测错误样本数: {sign_errors} / {len(results)}")

# --- 主程序入口 ---

def main():
    try:
        config = ExperimentConfig()
        processor = DataProcessor(config)
        df = processor.load_and_filter()
        X, y = processor.engineer_features(df)
        
        indices = np.arange(len(X))
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, indices, test_size=0.2, random_state=config.random_state
        )
        
        y_train = y.iloc[idx_train]
        y_test = y.iloc[idx_test]
        
        # 训练
        trainer = AdmittanceSVM(config)
        trainer.train(X_train, y_train)
        
        # 评估
        evaluate_performance(trainer.best_model, X_test, y_test)
        trainer.save()
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
