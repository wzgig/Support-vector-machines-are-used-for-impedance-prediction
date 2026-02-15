"""
Impedance Prediction using SVM (ArcSinh Strategy)
===========================================================
基于支持向量机（SVM）的阻抗预测模型 - 反双曲正弦变换（ArcSinh Strategy）

Author: GitHub Copilot (Refactored for Research)
Date: 2026-02-09
Description:
    此脚本实现了一种针对"宽动态范围且含正负值"数据的回归策略。
    
    问题复盘 (Problem Review):
    1. G-Strategy (导纳): 虽精度高，但在 R 趋于无穷（G趋于0）时，微小误差会导致 R 的符号翻转或数值爆炸。
    2. Log-Strategy (对数): Log 只能处理正数，被迫取绝对值 `log(|R|)`，导致丢失 R 的符号信息（R 存在负值），
       造成涉及符号的样本预测完全错误。
    
    最终策略 (Final Strategy): ArcSinh (反双曲正弦)
    Target: y = arcsinh(R)
    
    优势:
    1. **压缩范围**: 类似 Log，将 10^6 压缩为 ~14.5，将 100 压缩为 ~5.3，利于 SVM 学习。
    2. **保留符号**: arcsinh(-x) = -arcsinh(x)。完美解决负阻抗预测问题。
    3. **零点连续**: 在 0 附近近似线性，避免了 Log(0) 的奇异性。

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
    model_save_path: str = "svm_model_ArcSinh_Impedance_e.pkl"
    
    target_branch_type: str = "RL_Series"
    target_branch_id: str = "e"  # 即使是 c 分支，截断策略也是安全的
    
    # 【核心简化策略】: 物理截断阈值
    # 将 R 限制在 [-clip, +clip] 范围内。
    # 理由: R > 20000 在电路中已近似开路。区分 20k 和 1M 对模型拟合是巨大的负担，但物理意义极小。
    # 这将极大稳定 Branch 'e' 的奇异点训练，并防止预测值爆炸。
    clipping_threshold: float = 20000.0 
    
    base_features: Tuple[str, ...] = ('P', 'Q', 'V', 'xi')
    
    param_distributions: dict = None
    n_iter_search: int = 15
    cv_folds: int = 5
    random_state: int = 42

    def __post_init__(self):
        if self.param_distributions is None:
            self.param_distributions = {
                # 策略更新 (2026-02-10 - 极简稳定版): 
                # 配合数值截断，我们不需要极端的 C。
                'regressor__svr__C': [100, 500, 1000, 2000], 
                
                # 针对截断后的平滑数据，常规 gamma 即可
                'regressor__svr__gamma': ['scale', 0.1, 0.5],
                
                # 放宽 epsilon 以获得稀疏解，提速训练
                'regressor__svr__epsilon': [0.05, 0.1, 0.2],
                'regressor__svr__kernel': ['rbf'] 
            }
        
        # 极速搜索配置
        self.n_iter_search = 8
        self.cv_folds = 3

# --- 2. 数据处理模块 ---

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
        
        mask = (df['Branch_Type'] == self.config.target_branch_type) & \
               (df['Branch_ID'] == self.config.target_branch_id)
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            raise ValueError(f"筛选后的数据集为空!")
            
        # --- 物理截断 (Clipping) ---
        # 针对 Branch 'e' 等包含奇点的分支，将 R 限制在合理物理范围内。
        limit = self.config.clipping_threshold
        count_clipped = ((df_filtered['R'] > limit) | (df_filtered['R'] < -limit)).sum()
        if count_clipped > 0:
            logger.info(f"应用数值截断: 将 {count_clipped} 个样本的 R 限制在 [-{limit}, +{limit}] 之间。")
            df_filtered['R'] = df_filtered['R'].clip(lower=-limit, upper=limit)
            
        logger.info(f"数据加载完成，筛选后样本数: {len(df_filtered)}")
        return df_filtered

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程
        注意：此前我们在这里做目标变换，现在改为在 Pipeline 中使用 TransformedTargetRegressor 做变换。
        这里只返回原始 Target R。
        """
        logger.info("开始特征工程 (Feature Engineering)...")
        X = df[list(self.config.base_features)].copy()
        
        # --- 物理特征增强 ---
        S_mag = np.sqrt(X['P']**2 + X['Q']**2)
        X['Z_mag'] = X['V']**2 / (S_mag + 1e-9)
        X['Phase'] = np.arctan2(X['Q'], X['P'])
        X['Y_mag'] = 1.0 / (X['Z_mag'] + 1e-9)

        # --- 奇异性特征 ---
        logger.info("添加倒数特征...")
        # 改进：不仅仅使用绝对值倒数（偶函数），增加带符号的倒数特征（奇函数）
        # Branch 'e' 在 xi=0 附近表现出 R 从负无穷跳变到正无穷（或反之）的奇点特性。
        # ArcSinh(R) 在此处会类似 ArcSinh(1/x) ~ -ln(x)，仍然保留符号变化。
        # 提供显式的带符号倒数特征能极大帮助 SVM 拟合这种突变。
        epsilon = 1e-3 # 防止除零
        X['inv_xi_signed'] = np.sign(X['xi']) / (X['xi'].abs() + epsilon)
        
        # 保留原有的偶对称特征，用于辅助拟合幅值的对称性
        X['inv_xi_0.1'] = 1.0 / (X['xi'].abs() + 0.1)
        X['inv_xi_0.5'] = 1.0 / (X['xi'].abs() + 0.5)
        # 增加高阶特征以应对复杂的谐振形状
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
        
        # 4. Calculate Sample Weights (Crucial Step)
        # 现状分析: ArcSinh 将 1e6 压缩为 14.5, 将 100 压缩为 5.3。
        # 如果没有权重，模型会认为预测错那 1e6 (Error~9) 和预测错 100 (Error~1) 的代价差不多。
        # 实际上前者的物理误差是毁灭性的。
        
        # 策略: 使用"部分还原"物理量级的权重。
        # log10(R) 权重 (1~6) 依然不够，这里使用 sqrt(|R|) 权重。
        # R=100 -> w=10; R=1,000,000 -> w=1000.
        # 100倍的权重差迫使 SVM 必须优先拟合那些尖峰。
        weights = np.sqrt(1.0 + np.abs(y_train))
        weights = np.clip(weights, 1.0, 2000.0) # 截断防止极端值主导
        
        logger.info(f"开始超参数搜索 (Target: ArcSinh(R))...")
        logger.info(f"应用强力样本加权 (Sqrt Strategy): Max Weight={weights.max():.2f}")
        
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
    
    # 额外统计: 在有效范围内的误差 (忽略那些本来就被截断的大值误差)
    # 如果 True R 很大且 Pred R 也很大，那么物理上是准确的，即使 diff 很大。
    evaluation_limit = 20000
    mask_valid = (np.abs(results['True R']) < evaluation_limit)
    if mask_valid.sum() > 0:
        mape_valid = results.loc[mask_valid, 'Rel Err %'].mean()
        print(f"\n[有效物理范围 < {evaluation_limit}] MAPE: {mape_valid:.2f}%")
        
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
