import os
import glob
import re
import pandas as pd
import numpy as np
import VF
import concurrent.futures
import time
import extract_RL  # 引入后续处理模块

# --- 配置 ---
# 可配置多个任务一次性处理
DATASETS = [
    {
        "name": "Training Set",
        "input_dir": os.path.join("Ideal Power Grid_Train", "Frequency_Response_Data"),
        "output_file": "Ideal_Power_Grid_Train_Processed.csv"
    },
    {
        "name": "Testing Set",
        "input_dir": os.path.join("Ideal Power Grid_Test", "Frequency_Response_Data"),
        "output_file": "Ideal_Power_Grid_Test_Processed.csv"
    }
]

# 目标元素
ELEMENTS = ["Y11", "Y12", "Y21", "Y22"]

# 正则匹配模式
FILENAME_PATTERN = re.compile(
    r"iP(\d+)_iV(\d+)_iQ(\d+)_iX(\d+)__P([+-]?\d+m)_Q([+-]?\d+m)_V([+-]?\d+m)_xi([+-]?\d+md)"
)

def parse_milli_token(tok: str) -> float:
    """将 '-300m' 解析为 -0.3"""
    tok = tok.strip()
    if tok.endswith("md"):
        tok = tok[:-2]
    elif tok.endswith("m"):
        tok = tok[:-1]
    return float(tok) / 1000.0

def process_single_file(fpath):
    """
    单个文件处理函数，用于并行调用
    返回: (成功列表, 失败信息)
    """
    basename = os.path.basename(fpath)
    file_results = []
    
    # 1. 解析文件名元数据
    match = FILENAME_PATTERN.search(basename)
    conditions = {}
    if match:
        conditions = {
            'iP': match.group(1),
            'iV': match.group(2),
            'iQ': match.group(3),
            'iX': match.group(4),
            'P':  parse_milli_token(match.group(5)),
            'Q':  parse_milli_token(match.group(6)),
            'V':  parse_milli_token(match.group(7)),
            'xi': parse_milli_token(match.group(8)),
        }
    else:
        conditions = {k: np.nan for k in ['iP', 'iV', 'iQ', 'iX', 'P', 'Q', 'V', 'xi']}

    try:
        df = pd.read_csv(fpath)
        if df.empty:
            return [], f"{basename}: Empty file"
        
        # 频率向量 (复数s)
        if 'Frequency_Hz' not in df.columns:
            return [], f"{basename}: Missing 'Frequency_Hz' column"
            
        freq_hz = df['Frequency_Hz'].values
        if np.isnan(freq_hz).any():
             return [], f"{basename}: Frequency_Hz contains NaN"

        s_vec = 1j * 2 * np.pi * freq_hz
        
        errors = []

        for elem in ELEMENTS:
            # 预定义 base_info 的一部分，用于错误记录
            error_base_info = {
                    'Filename': basename,
                    **conditions,
                    'Element': elem,
            }
            
            try:
                real_col = f"{elem}_Real"
                imag_col = f"{elem}_Imag"
                
                if real_col not in df.columns or imag_col not in df.columns:
                    # [新增] 记录缺失列的错误行
                    err_row = error_base_info.copy()
                    err_row.update({'Branch_Type': 'Missing_Columns', 'Branch_ID': 'Error'})
                    file_results.append(err_row)
                    continue
                
                # 2. 构建复数响应
                real_vals = df[real_col].values
                imag_vals = df[imag_col].values
                
                if np.isnan(real_vals).any() or np.isnan(imag_vals).any() or np.isinf(real_vals).any() or np.isinf(imag_vals).any():
                    errors.append(f"{elem}: Data contains NaN/Inf")
                    # [新增] 记录数据无效的错误行
                    err_row = error_base_info.copy()
                    err_row.update({'Branch_Type': 'Data_Invalid_NaN_Inf', 'Branch_ID': 'Error'})
                    file_results.append(err_row)
                    continue

                f_vec = real_vals + 1j * imag_vals
                
                # 4. 执行矢量拟合
                poles, residues, d, h, metrics = VF.vectfit_find_best_order(
                    f_vec, s_vec, 
                    min_poles=3, max_poles=3, step=1,
                    # max_iter=100, tol=1e-6,
                    target_error=1e-5, 
                    weighting_policy='inv_mag', # <--- 优雅的接口调用
                    silent=True
                )
                
                # [优化] 5. 检查无源性 (Passivity Check)
                is_passive, min_real, viol_freq = VF.check_passivity(s_vec, poles, residues, d, h)

                # 6. 系统参数提取
                analyzer = VF.SystemAnalyzer()
                analyzer.load_fitting_result(poles, residues, d, h)
                
                # 基础元数据 (对每一行都重复，方便后续 Pandas 分析)
                base_info = {
                    'Filename': basename,
                    **conditions,
                    'Element': elem,
                    'RMS_Rel_Error': metrics['rms_rel'],
                    'Max_Rel_Error': metrics['max_rel'],
                    'Order': len(poles),
                    'Is_Passive': is_passive,       # 新增指标
                    'Min_Real_Part': min_real       # 新增指标
                }

                # 收集电路参数
                extracted_any = False

                # 并联 RC
                if analyzer.output_data['rc_params']:
                    extracted_any = True
                    row = base_info.copy()
                    row.update({
                        'Branch_Type': 'RC_Parallel',
                        'Branch_ID': 'Parallel',
                        'R': analyzer.output_data['rc_params']['R'],
                        'C': analyzer.output_data['rc_params']['C']
                    })
                    file_results.append(row)
                
                # 串联 RL
                if analyzer.output_data['rl_params']:
                    extracted_any = True
                    for item in analyzer.output_data['rl_params']:
                        row = base_info.copy()
                        row.update({
                            'Branch_Type': 'RL_Series',
                            'Branch_ID': item['id'],
                            'R': item['R'],
                            'L': item['L']
                        })
                        file_results.append(row)

                # 串联 RLC
                if analyzer.output_data['rlc_params']:
                    extracted_any = True
                    for item in analyzer.output_data['rlc_params']:
                        row = base_info.copy()
                        row.update({
                            'Branch_Type': 'RLC_Series',
                            'Branch_ID': item['id'],
                            'R': item['R'],
                            'L': item['L'],
                            'C': item['C']
                        })
                        file_results.append(row)

                # [修复] 如果未提取到任何标准电路拓扑，记录一条原始拟合结果，防止文件丢失
                if not extracted_any:
                    row = base_info.copy()
                    row.update({
                        'Branch_Type': 'No_Topology_Match', 
                        'Branch_ID': 'None',
                    })
                    file_results.append(row)
            except Exception as elem_e:
                errors.append(f"{elem}: {str(elem_e)}")
                # [新增] 发生异常时记录错误行
                err_row = error_base_info.copy()
                err_row.update({
                    'Branch_Type': 'Processing_Exception', 
                    'Branch_ID': 'Error',
                })
                file_results.append(err_row)

        return file_results, "; ".join(errors) if errors else None

    except Exception as e:
        # [修改] 全局解析失败也返回一行记录，防止文件完全丢失
        err_row = {
            'Filename': basename,
            **conditions,
            'Element': 'All',
            'Branch_Type': 'File_Read_Error',
            'Branch_ID': 'Error',
            'R': str(e) # 将错误信息借放在 R 列或其他备注列
        }
        return [err_row], f"{basename}: {str(e)}"

def run_batch(config):
    dataset_name = config["name"]
    input_dir    = config["input_dir"]
    output_file  = config["output_file"]

    print(f"\n[{dataset_name}] 正在处理...")
    print(f"  输入目录: {input_dir}")
    print(f"  输出文件: {output_file}")

    if not os.path.isdir(input_dir):
        print(f"  [跳过] 目录不存在: {input_dir}")
        return

    # 查找所有 CSV
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    total_files = len(csv_files)
    print(f"找到 {total_files} 个 CSV 文件，准备开始并行处理...")

    all_data = []
    errors = []
    
    start_time = time.time()

    # 使用多进程可以避开 Python GIL，利用多核 CPU
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {executor.submit(process_single_file, f): f for f in csv_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res, err = future.result()
            if res:
                all_data.extend(res)
            if err:
                errors.append(err)
            
            # 进度打印
            if (i + 1) % 50 == 0:
                print(f"    进度: {i + 1}/{total_files} ({(i + 1)/total_files*100:.1f}%)")

    end_time = time.time()
    print(f"  处理完成，耗时: {end_time - start_time:.2f} 秒")

    if errors:
        print(f"  出现 {len(errors)} 个错误:")
        for e in errors[:5]: # 只打印前5个错误
            print(f"    - {e}")
        if len(errors) > 5:
            print("    ... (更多错误见日志)")

    # 保存结果
    if all_data:
        print("  正在保存结果文件...")
        full_df = pd.DataFrame(all_data)
        
        # 定义列顺序
        cols_order = ['Filename', 'iP', 'iV', 'iQ', 'iX', 'P', 'Q', 'V', 'xi', 
                      'Element', 'Is_Passive', 'Min_Real_Part', # 放在显眼位置
                      'Branch_Type', 'Branch_ID', 
                      'R', 'L', 'C', 
                      'RMS_Rel_Error', 'Max_Rel_Error', 'Order']
        
        # 确保所有列存在（填补 NaN）
        for c in cols_order:
            if c not in full_df.columns:
                full_df[c] = None
                
        # 用于收集生成的文件路径，传递给 extract_RL
        generated_files_map = {}

        # 分文件保存 (按 Element 拆分)
        for elem in ELEMENTS:
            elem_df = full_df[full_df['Element'] == elem].copy()
            if not elem_df.empty:
                # 重新排序列
                elem_df = elem_df[cols_order]
                # 构建输出文件名
                fname = output_file.replace(".csv", f"_{elem}.csv")
                
                elem_df.to_csv(fname, index=False)
                print(f"    -> 已保存: {fname} ({len(elem_df)} 行)")
                
                # 记录路径
                generated_files_map[elem] = fname
        
        # === 衔接 extract_RL 流程 ===
        if generated_files_map:
            print(f"\n  [Post-Processing] Chain Execution -> extract_RL.py")
            extract_RL.process_file_map(generated_files_map)

    else:
        print("  未生成任何有效数据。")

if __name__ == "__main__":
    # Windows 下使用 multiprocess 必须放在 if __name__ == "__main__": 下
    print("=== 开始批量处理所有数据集 ===")
    for ds in DATASETS:
        run_batch(ds)
    print("\n=== 所有任务完成 ===")
