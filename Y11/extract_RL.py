import pandas as pd
import os
import sys
import traceback

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================
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
    """读取 CSV 文件并验证必需的列是否存在。"""
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return None

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"[Error] Missing columns {missing_cols} in {file_path}. Skipping.")
        return None
    
    return df

def save_wide_format(df, target_element, output_base_name):
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
        # 我们将其展平为 R_a, L_a 格式
        
        # 获取所有的 Branch_ID 并排序
        branch_ids = sorted(df['Branch_ID'].unique())
        
        # 展平列名
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
        
        # 保存宽表
        output_filename = output_base_name.replace(".csv", "_RL_Wide.csv")
        wide_df.to_csv(output_filename, index=False)
        print(f"    -> Saved Wide Format: {output_filename} ({len(wide_df)} unique cases)")
        
        return wide_df
        
    except Exception as e:
        print(f"    [Error] Creating wide format for {target_element}: {e}")
        return None

def print_statistics(df, target_element):
    """打印详细的分组统计信息 (按 Branch_ID)。"""
    print(f"\n{'='*70}")
    print(f"  DATA ANALYSIS FOR: {target_element}")
    print(f"{'='*70}")
    
    branch_ids = sorted(df['Branch_ID'].unique())
    print(f"  Identified RL Branches: {', '.join(branch_ids)}")
    
    stats_list = []
    
    for bid in branch_ids:
        subset = df[df['Branch_ID'] == bid]
        
        # 计算 R 和 L 的统计量
        desc_r = subset['R'].describe()
        desc_l = subset['L'].describe()
        
        # 负值检查
        r_neg_count = (subset['R'] < 0).sum()
        l_neg_count = (subset['L'] < 0).sum()
        
        stats_list.append({
            'Branch': bid,
            'Count': int(desc_r['count']),
            'R_Mean': desc_r['mean'],
            'R_Std': desc_r['std'],
            'R_Min': desc_r['min'],
            'R_Max': desc_r['max'],
            'R_Neg(%)': (r_neg_count / len(subset)) * 100,
            'L_Mean': desc_l['mean'],
            'L_Std': desc_l['std'],
            'L_Min': desc_l['min'],
            'L_Max': desc_l['max'],
            'L_Neg(%)': (l_neg_count / len(subset)) * 100
        })
        
    if stats_list:
        summary_df = pd.DataFrame(stats_list)
        
        # 格式化输出
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: '%.4g' % x)
        
        print("\n  [Summary Statistics Table]")
        print(summary_df.to_string(index=False))
        
        print("\n  [Anomaly Detection (Top 3 Max |R|)]")
        for bid in branch_ids:
            subset = df[df['Branch_ID'] == bid]
            if not subset.empty:
                # 获取绝对值最大的前3个R值对应的记录
                top_indices = subset['R'].abs().nlargest(3).index
                max_records = subset.loc[top_indices, ['R', 'L', 'P', 'Q', 'V', 'xi']]
                
                print(f"    Branch {bid}:")
                for _, row in max_records.iterrows():
                    print(f"      R={row['R']:.4g}, L={row['L']:.4g} @ (P={row['P']}, Q={row['Q']}, V={row['V']}, xi={row['xi']})")

def process_file_map(file_map):
    """
    批量处理传入的文件字典
    file_map: { "Y11": "path/to/Y11.csv", ... }
    """
    print("\n" + "#"*70)
    print("STARTING RL PARAMETER EXTRACTION & ANALYSIS".center(70))
    print("#"*70)
    
    for element, file_path in file_map.items():
        print(f"\n>>> Processing Element: {element}")
        print(f"    Input File: {file_path}")

        try:
            # 1. 读取与验证
            df = load_and_validate_data(file_path)
            if df is None:
                continue

            # 2. 筛选 RL_Series 分支
            # 注意：batch_processing 输出的 CSV 可能包含多种 Branch_Type
            # 我们只关心 Branch_Type == 'RL_Series'
            mask = (df['Branch_Type'] == 'RL_Series')
            filtered_df = df[mask].copy()

            if filtered_df.empty:
                print(f"    [Info] No 'RL_Series' records found. Skipping.")
                continue

            # 3. 选择列
            output_df = filtered_df[OUTPUT_SELECTED_COLS]

            # 4. 只保存宽格式 (符合用户需求)
            save_wide_format(output_df, element, file_path)

            # 5. 打印详细统计
            print_statistics(output_df, element)

        except Exception as e:
            print(f"    [Error] Processing {element}: {e}")
            traceback.print_exc()
            
    print("\n" + "#"*70)
    print("EXTRACTION COMPLETE".center(70))
    print("#"*70 + "\n")

# 兼容直接运行测试
if __name__ == "__main__":
    # 默认测试配置
    TEST_MAP = {
        "Y11": "Ideal_Power_Grid_Train_Processed_Y11.csv",
        # "Y12": "Ideal Power Grid_Y12.csv",
    }
    process_file_map(TEST_MAP)
