
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_extract(file_path, target_element):
    """
    处理单个文件：提取所有 RL_Series 分支数据，保存 CSV，并打印统计信息
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {target_element} ({file_path})")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(file_path)
        
        # 1. 基础检查
        required_cols = ['Branch_Type', 'Branch_ID', 'R', 'L', 'P', 'Q', 'V', 'xi', 'Element', 'Filename']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"  Error: Missing columns {missing_cols}. Skipping.")
            return

        # 2. 筛选所有 RL_Series 分支
        # 目标：提取 Branch_Type == 'RL_Series' 的所有行，保留 Branch_ID
        mask = (df['Branch_Type'] == 'RL_Series')
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            print(f"  No 'RL_Series' records found in {file_path}.")
            return

        # 3. 选择列
        selected_cols = ['Element', 'Branch_ID', 'P', 'Q', 'V', 'xi', 'R', 'L', 'Filename']
        output_df = filtered_df[selected_cols]
        
        # 4. 保存文件 (每个 Element 一个文件) - 原始长格式 (Long Format)
        output_filename = f"extracted_RL_Series_{target_element}.csv"
        output_df.to_csv(output_filename, index=False)
        print(f"  -> Saved {len(output_df)} rows to: {output_filename} (Long Format)")

        # ---------------------------------------------------------
        # NEW: 转换为宽格式 (Wide Format) 并保存
        # 目标：将 Branch_ID 转为列，生成 R_a, L_a, R_b, L_b...
        # ---------------------------------------------------------
        
        # 定义唯一标识一个样本的列
        index_cols = ['Filename', 'Element', 'P', 'Q', 'V', 'xi']
        
        # 使用 pivot_table 进行透视
        # index: 保持不变的列
        # columns: 要变成列头的列 (Branch_ID)
        # values: 要填充的值 (R, L)
        try:
            wide_df = output_df.pivot_table(
                index=index_cols, 
                columns='Branch_ID', 
                values=['R', 'L'],
                aggfunc='first'  # 理论上每个文件每个分支只有一个值
            )
            
            # 此时列是 MultiIndex，例如 ('R', 'a'), ('L', 'a')
            # 我们将其展平为 R_a, L_a
            wide_df.columns = [f"{val}_{bid}" for val, bid in wide_df.columns]
            
            # 重置索引，让 Filename, P, Q 等回到列中
            wide_df = wide_df.reset_index()
            
            # 重新排序列，符合用户要求的 "R_a...所有R, L_a...所有L"
            # 1. 基础列
            base_cols = index_cols
            
            # 2. 获取所有的 Branch_ID
            branch_ids = sorted(output_df['Branch_ID'].unique())
            
            # 3. 构建 R 列和 L 列的列表
            r_cols = [f"R_{bid}" for bid in branch_ids]
            l_cols = [f"L_{bid}" for bid in branch_ids]
            
            # 4. 合并所有列名 (确保只包含存在的列)
            final_cols = base_cols + [c for c in r_cols if c in wide_df.columns] + [c for c in l_cols if c in wide_df.columns]
            
            # 5. 应用列排序
            wide_df = wide_df[final_cols]
            
            # 保存宽格式文件
            output_filename_wide = f"extracted_RL_Series_{target_element}_wide.csv"
            wide_df.to_csv(output_filename_wide, index=False)
            print(f"  -> Saved {len(wide_df)} unique cases to: {output_filename_wide} (Wide Format)")
            
        except Exception as e:
            print(f"  Error creating wide format for {target_element}: {e}")
            # 继续执行后续统计分析，不因宽格式转换失败而中断

        # 5. 分组统计分析 (按 Branch_ID)
        print(f"\n  --- Statistical Analysis for {target_element} (by Branch_ID) ---")
        
        # 获取所有存在的 Branch_ID
        branch_ids = output_df['Branch_ID'].unique()
        branch_ids.sort()
        
        print(f"  Found Branch IDs: {', '.join(branch_ids)}")
        
        # 针对每个 Branch_ID 进行统计
        stats_list = []
        for bid in branch_ids:
            subset = output_df[output_df['Branch_ID'] == bid]
            
            # R 统计
            r_stats = subset['R'].describe()
            r_skew = subset['R'].skew()
            r_neg = (subset['R'] < 0).sum()
            r_neg_pct = (r_neg / len(subset)) * 100
            
            # L 统计
            l_stats = subset['L'].describe()
            l_skew = subset['L'].skew()
            l_neg = (subset['L'] < 0).sum()
            l_neg_pct = (l_neg / len(subset)) * 100
            
            stats_list.append({
                'Branch_ID': bid,
                'Count': len(subset),
                'R_Mean': r_stats['mean'],
                'R_Std': r_stats['std'],
                'R_Skew': r_skew,
                'R_Neg_%': r_neg_pct,
                'L_Mean': l_stats['mean'],
                'L_Std': l_stats['std'],
                'L_Skew': l_skew,
                'L_Neg_%': l_neg_pct
            })

        # 打印格式化表格
        stats_df = pd.DataFrame(stats_list)
        # 设置显示格式
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n  Summary Table:")
        print(stats_df.to_string(index=False))
        
        # 简单的分布检查 (可选)
        print("\n  Extreme Value Check (Top 3 Abs Max):")
        for bid in branch_ids:
            subset = output_df[output_df['Branch_ID'] == bid]
            max_r = subset.loc[subset['R'].abs().nlargest(3).index, 'R'].values
            print(f"    Branch {bid}: Max |R| = {max_r}")

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 配置输入文件映射
    input_map = {
        "Y11": "equivalent_circuit_parameters_optimized_accurate_Y11.csv",
        "Y12": "equivalent_circuit_parameters_optimized_accurate_Y12.csv",
        "Y21": "equivalent_circuit_parameters_optimized_accurate_Y21.csv",
        "Y22": "equivalent_circuit_parameters_optimized_accurate_Y22.csv"
    }

    print("Starting batch extraction for ALL RL_Series branches...")
    
    for element, file_path in input_map.items():
        analyze_and_extract(file_path, element)
    
    print("\nBatch extraction complete.")

if __name__ == "__main__":
    main()
