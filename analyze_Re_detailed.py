import pandas as pd
import numpy as np
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def analyze_re_detailed():
    print("正在加载数据...")
    try:
        df = pd.read_csv('extracted_RL_Series_Y11_wide.csv')
    except FileNotFoundError:
        print("错误：找不到文件 extracted_RL_Series_Y11_wide.csv")
        return

    # 1. 基础统计分析
    print("\nXXX R_e 基础统计分析 XXX")
    re_stats = df['R_e'].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(re_stats)
    print(f"Skewness (偏度): {df['R_e'].skew():.4f}")
    print(f"Kurtosis (峰度): {df['R_e'].kurt():.4f}")

    # 2. 检查缺失值和无穷大
    print("\nXXX 数据完整性检查 XXX")
    print(f"缺失值数量: \n{df[['R_e', 'P', 'Q', 'V', 'xi']].isnull().sum()}")
    print(f"无穷大值数量: {np.isinf(df['R_e']).sum()}")

    # 3. 相关性分析 (Pearson & Spearman)
    print("\nXXX 相关性分析 XXX")
    # 构造周期性特征
    df['cos_xi'] = np.cos(np.radians(df['xi'])) # 假设 xi 是角度，如果不是需确认
    # 既然之前的报告提到 xi 是相角，通常单位可能是度或弧度。看之前的数据 xi 有 -10000md，猜测可能是 millidegrees?
    # 查看附件数据 iX01__...xi-10000md.csv，对应的 xi 列是 -10.0。如果是 -10000md (millidegrees)，那 -10.0 就是度。
    # 让我们假设单位是度 (degrees)。

    df['xi_rad'] = np.radians(df['xi'])
    df['cos_xi'] = np.cos(df['xi_rad'])
    df['sin_xi'] = np.sin(df['xi_rad'])

    cols_to_corr = ['R_e', 'P', 'Q', 'V', 'xi', 'cos_xi', 'sin_xi']
    
    print("\n--- Pearson 相关系数 (线性相关) ---")
    print(df[cols_to_corr].corr(method='pearson')['R_e'].sort_values(ascending=False))

    print("\n--- Spearman 相关系数 (单调相关，抗离群值) ---")
    print(df[cols_to_corr].corr(method='spearman')['R_e'].sort_values(ascending=False))

    # 4. 极端值分析
    print("\nXXX 极端值/尾部分析 XXX")
    q01 = df['R_e'].quantile(0.01)
    q99 = df['R_e'].quantile(0.99)
    outliers = df[(df['R_e'] < q01) | (df['R_e'] > q99)]
    print(f"1% - 99% 分位数范围: [{q01:.2f}, {q99:.2f}]")
    print(f"在此范围之外的数据量: {len(outliers)}")
    print("极端值的 R_e 均值 vs 整体均值:")
    print(f"极端值均值: {outliers['R_e'].mean():.2f}")
    print(f"剔除极端值后的均值: {df[(df['R_e'] >= q01) & (df['R_e'] <= q99)]['R_e'].mean():.2f}")

    # 极端值主要集中在哪些工况？
    print("\n极端值主要分布的工况特征 (Top 5 均值):")
    print(outliers[['P', 'V', 'Q']].mean())

    # 5. 特征变换尝试
    print("\nXXX 特征变换测试 XXX")
    # 对数变换 (处理负数: sign(x) * log(|x|+1))
    df['Re_log_modulus'] = np.sign(df['R_e']) * np.log1p(np.abs(df['R_e']))
    # Arcsinh 变换
    df['Re_arcsinh'] = np.arcsinh(df['R_e'])

    print("变换后与输入特征的 Pearson 相关性 (Top 3):")
    print("\n原始 R_e:")
    print(df[cols_to_corr].corr()['R_e'].abs().sort_values(ascending=False).head(4))
    
    print("\nLog-Modulus 变换后:")
    cols_log = ['Re_log_modulus', 'P', 'Q', 'V', 'xi', 'cos_xi', 'sin_xi']
    print(df[cols_log].corr()['Re_log_modulus'].abs().sort_values(ascending=False).head(4))

    print("\nArcsinh 变换后:")
    cols_asinh = ['Re_arcsinh', 'P', 'Q', 'V', 'xi', 'cos_xi', 'sin_xi']
    print(df[cols_asinh].corr()['Re_arcsinh'].abs().sort_values(ascending=False).head(4))

if __name__ == "__main__":
    analyze_re_detailed()
