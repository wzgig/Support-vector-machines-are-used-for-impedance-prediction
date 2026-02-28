import pandas as pd
import os
import sys
import traceback

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================
# 输入文件映射：Element名称 -> 文件路径
INPUT_FILE_MAP = {
    "Y11": "Ideal Power Grid_Y11.csv",
    "Y12": "Ideal Power Grid_Y12.csv",
    "Y21": "Ideal Power Grid_Y21.csv",
    "Y22": "Ideal Power Grid_Y22.csv"
}

# 必需的列名
REQUIRED_COLS = [
    'Branch_Type', 'Branch_ID', 'R', 'L', 'P', 'Q', 'V', 'xi', 'Element', 'Filename'
]

# 输出文件筛选列
OUTPUT_SELECTED_COLS = [
    'Element', 'Branch_ID', 'P', 'Q', 'V', 'xi', 'R', 'L', 'Filename'
]

# 用于唯一标识样本的索引列 (用于宽格式转换)
INDEX_COLS = ['Filename', 'Element', 'P', 'Q', 'V', 'xi']

# ==============================================================================
# 功能函数 (Functions)
# ==============================================================================

def load_and_validate_data(file_path):
    """
    读取 CSV 文件并验证必需的列是否存在。
    """
    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' not found. Skipping.")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"  Error: Missing columns {missing_cols} in {file_path}. Skipping.")
        return None
    
    return df

def save_long_format(df, target_element):
    """
    保存原始的长格式 (Long Format) 数据。
    """
    output_filename = f"extracted_RL_Series_{target_element}.csv"
    df.to_csv(output_filename, index=False)
    print(f"  -> Saved {len(df)} rows to: {output_filename} (Long Format)")
    return df

def save_wide_format(df, target_element):
    """
    将数据转换为宽格式 (Wide Format) 并保存。
    Branch_ID (a, b...) 将被转为列后缀 (R_a, L_a, R_b, L_b...)
    """
    try:
        # 使用 pivot_table 进行透视
        # index: 保持不变的列
        # columns: 要变成列头的列 (Branch_ID)
        # values: 要填充的值 (R, L)
        wide_df = df.pivot_table(
            index=INDEX_COLS, 
            columns='Branch_ID', 
            values=['R', 'L'],
            aggfunc='first'  # 理论上每个文件每个分支只有一个值
        )
        
        # 此时列是 MultiIndex，例如 ('R', 'a'), ('L', 'a')
        # 我们将其展平为 R_a_temp, L_a_temp 格式便于处理
        # 注意：pivot后列顺序可能混乱，这里我们手动重构列名
        
        # 获取所有的 Branch_ID 并排序
        branch_ids = sorted(df['Branch_ID'].unique())
        
        # 暂时展平列名
        wide_df.columns = [f"{val}_{bid}" for val, bid in wide_df.columns]
        wide_df = wide_df.reset_index()
        
        # 构建期望的列顺序: R_a, R_b... 然后 L_a, L_b...
        r_cols = [f"R_{bid}" for bid in branch_ids]
        l_cols = [f"L_{bid}" for bid in branch_ids]
        
        # 组合最终列名 (仅包含实际存在的列)
        final_cols = INDEX_COLS + \
                     [c for c in r_cols if c in wide_df.columns] + \
                     [c for c in l_cols if c in wide_df.columns]
        
        # 应用列排序
        wide_df = wide_df[final_cols]
        
        # 保存
        output_filename_wide = f"extracted_RL_Series_{target_element}_wide.csv"
        wide_df.to_csv(output_filename_wide, index=False)
        print(f"  -> Saved {len(wide_df)} unique cases to: {output_filename_wide} (Wide Format)")
        
    except Exception as e:
        print(f"  Error creating wide format for {target_element}: {e}")
        # 不中断程序，仅打印错误

def print_statistics(df, target_element):
    """
    打印分组统计信息 (按 Branch_ID)。
    """
    print(f"\n  --- Statistical Analysis for {target_element} (by Branch_ID) ---")
    
    branch_ids = sorted(df['Branch_ID'].unique())
    print(f"  Found Branch IDs: {', '.join(branch_ids)}")
    
    stats_list = []
    
    for bid in branch_ids:
        subset = df[df['Branch_ID'] == bid]
        
        # 计算 R 的统计量
        r_stats = subset['R'].agg(['mean', 'std', 'skew'])
        r_neg = (subset['R'] < 0).sum()
        r_neg_pct = (r_neg / len(subset)) * 100
        
        # 计算 L 的统计量
        l_stats = subset['L'].agg(['mean', 'std', 'skew'])
        l_neg = (subset['L'] < 0).sum()
        l_neg_pct = (l_neg / len(subset)) * 100
        
        stats_list.append({
            'Branch_ID': bid,
            'Count': len(subset),
            'R_Mean': r_stats['mean'],
            'R_Std': r_stats['std'],
            'R_Skew': r_stats['skew'],
            'R_Neg_%': r_neg_pct,
            'L_Mean': l_stats['mean'],
            'L_Std': l_stats['std'],
            'L_Skew': l_stats['skew'],
            'L_Neg_%': l_neg_pct
        })
        
    # 打印格式化表格
    if stats_list:
        summary_df = pd.DataFrame(stats_list)
        # 设置显示格式
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\n  Summary Table:")
        print(summary_df.to_string(index=False))
        
        # 简单的极值检查
        print("\n  Extreme Value Check (Top 3 Abs Max):")
        for bid in branch_ids:
            subset = df[df['Branch_ID'] == bid]
            if not subset.empty:
                # 获取绝对值最大的前3个R值
                top_indices = subset['R'].abs().nlargest(3).index
                max_r = subset.loc[top_indices, 'R'].values
                print(f"    Branch {bid}: Max |R| = {max_r}")

def analyze_and_extract(target_element, file_path):
    """
    处理单个文件的完整流程：读取 -> 筛选 -> 保存 -> 统计
    """
    print(f"\n{'='*60}")
    print(f"Processing: {target_element} ({file_path})")
    print(f"{'='*60}")

    try:
        # 1. 读取与验证
        df = load_and_validate_data(file_path)
        if df is None:
            return

        # 2. 筛选 RL_Series 分支
        mask = (df['Branch_Type'] == 'RL_Series')
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            print(f"  No 'RL_Series' records found in {file_path}.")
            return

        # 3. 选择列
        output_df = filtered_df[OUTPUT_SELECTED_COLS]
        
        # 4. 保存文件 (长格式)
        save_long_format(output_df, target_element)

        # 5. 保存文件 (宽格式)
        save_wide_format(output_df, target_element)

        # 6. 统计分析
        print_statistics(output_df, target_element)

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        traceback.print_exc()

def main():
    print("Starting batch extraction for ALL RL_Series branches...")
    
    # 遍历配置字典进行处理
    for element, file_path in INPUT_FILE_MAP.items():
        analyze_and_extract(element, file_path)
    
    print("\nBatch extraction complete.")

if __name__ == "__main__":
    main()