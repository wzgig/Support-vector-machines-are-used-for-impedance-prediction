"""
Generate Admittance Transfer Function for MATLAB
================================================
读取 Step2 生成的阻抗预测结果 CSV 文件，
根据等效电路参数回推导纳传递函数 Y(s) 的完整模型，用于 MATLAB 绘图（Bode 图等）。
结果保存为 .mat 文件。

对应关系:
  Y(s) = sum( r_i / (s - p_i) ) + d + s * h

1. RL Series Branch (branch a-f):
   Z = R + sL  =>  Y = 1 / (R + sL) = (1/L) / (s + R/L)
   => Pole p = -R/L
   => Residue r = 1/L

2. RC Parallel Branch:
   Y = 1/R + sC
   => Constant d = 1/R
   => Slope h = C

3. RLC Series Branch with Controlled Source:
   Y = (2*c_r * s + b) / (s^2 - 2αs + |p|^2)
   => Pole p = α + jβ
   => Residue k = c_r + jc_i
   转换公式:
   α = -R / (2*L)
   β = sqrt( 1/(LC) - α^2 )
   c_r = 1 / (2*L)
   b = g_m / (LC)
   c_i = -(b + 2*c_r*α) / (2*β)
   
Usage:
    Run this script. It looks for 'predicted_impedance_results.csv' in the current directory.
"""

import os
import pandas as pd
import numpy as np
import scipy.io as sio

# ================= Configuration =================
# 输入文件 (Step 2 生成的 CSV)
# 默认在 Step2 目录下
INPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Step2", "predicted_impedance_results.csv")

# 如果找不到，尝试当前目录
if not os.path.exists(INPUT_CSV_PATH):
    INPUT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predicted_impedance_results.csv")

# 输出文件夹 (Output Directory)
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(INPUT_CSV_PATH), "matlab_models")
# =================================================

def generate_matlab_data():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"[错误] 找不到输入文件: {INPUT_CSV_PATH}")
        print("请先运行 Step2/predict_impedance_console.py 生成预测结果。")
        return

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR_PATH):
        os.makedirs(OUTPUT_DIR_PATH)
        print(f"创建输出目录: {OUTPUT_DIR_PATH}")

    print(f"正在读取文件: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        print(f"[错误] 读取 CSV 失败: {e}")
        return

    # 检查必要列, 根据实际CSV动态检查
    # 如果有RLC列，需要检查 R, L, C, g_m 等
    # 目前假设您的 CSV 是 Step2 生成的，主要包含 RL_Series 和 RC_Parallel
    # 如果需要处理 RLC，请确保 CSV 中包含相关列
    
    # 按工况分组处理
    grouped = df.groupby('Case_Name')
    
    print("-" * 60)
    print(f"{'Case Name':<20} | {'Poles':<10} | {'D':<10} | {'H':<10}")
    print("-" * 60)

    for case_name, group in grouped:
        poles = []
        residues = []
        d_val = 0.0
        h_val = 0.0
        
        # 提取工况参数 (P, Q, V, xi) - 取该组第一行的值即可
        # 注意: CSV 中的 xi 通常是角度 (degrees)，用户要求输出为弧度 (radians)
        xi_deg = float(group.iloc[0]['xi'])
        xi_rad = xi_deg * np.pi / 180.0
        
        condition_info = {
            'P': float(group.iloc[0]['P']),
            'Q': float(group.iloc[0]['Q']),
            'V': float(group.iloc[0]['V']),
            'xi': xi_rad  # 转换为弧度
        }

        for _, row in group.iterrows():
            branch_type = str(row['Branch_Type'])
            
            # --- 1. RL Series Branch ---
            if 'RL_Series' in branch_type:
                try:
                    R = float(row['R'])
                    L = float(row['L'])
                    
                    if L != 0:
                        # p = -R/L, r = 1/L
                        p = -R / L
                        r = 1.0 / L
                        poles.append(p)
                        residues.append(r)
                    else:
                        # L=0 退化为纯电阻 G=1/R (归入 d)
                        if R != 0:
                            d_val += 1.0 / R
                except (ValueError, TypeError):
                    continue

            # --- 2. RC Parallel Branch ---
            elif 'RC_Parallel' in branch_type:
                try:
                    # Y = 1/R + sC
                    # d += 1/R
                    # h += C
                    R = float(row['R']) if pd.notna(row['R']) and row['R'] != '' else None
                    C = float(row['C']) if pd.notna(row['C']) and row['C'] != '' else None
                    
                    if R is not None and R != 0:
                        d_val += 1.0 / R
                    
                    if C is not None:
                        h_val += C
                        
                except (ValueError, TypeError):
                    continue
            
            # --- 3. RLC Series Branch with Controlled Source (Placeholder) ---
            # 如果您的 CSV 将来包含 RLC 数据，这里是处理逻辑
            # 需要 CSV 有 g_m, b 等列，或者 R, L, C 和受控源参数
            # 假设 row 有 'g_m' 或者是通过 b 计算的
            elif 'RLC' in branch_type:
                try:
                    # 获取电路参数
                    R = float(row['R'])
                    L = float(row['L'])
                    C = float(row['C'])
                    # g_m 或 b，取决于 CSV 存储内容，这里假设存的是 g_m
                    g_m = float(row['g_m']) if 'g_m' in row and pd.notna(row['g_m']) else 0.0
                    
                    # 1. 计算极点 p = α ± jβ
                    # α = -R / (2L)
                    alpha = -R / (2*L)
                    
                    # 判别式 delta = 1/(LC) - α^2
                    delta = 1.0/(L*C) - alpha**2
                    
                    if delta >= 0:
                        beta = np.sqrt(delta)
                        p_real = alpha
                        p_imag = beta # 取正部
                    else:
                        # 过阻尼情况，实际上变成两个实极点
                        beta = np.sqrt(abs(delta)) * 1j 
                        
                    # 2. 计算留数 k = c_r ± jc_i
                    # c_r = 1 / (2L)
                    c_r = 1.0 / (2*L)
                    
                    # b = g_m / (LC)
                    b = g_m / (L*C)
                    
                    # c_i = -(b + 2*c_r*α) / (2β)
                    # 注意如果 beta 为虚数这里的计算和共轭关系依然适用
                    if beta != 0:
                        c_i = -(b + 2*c_r*alpha) / (2*beta)
                    else:
                        c_i = 0 
                        
                    # 构造极点和留数对
                    # P1 = α + jβ, R1 = c_r + jc_i
                    # P2 = α - jβ, R2 = c_r - jc_i
                    
                    p1 = alpha + 1j * beta
                    r1 = c_r + 1j * c_i
                    
                    poles.append(p1)
                    poles.append(p1.conjugate())
                    residues.append(r1)
                    residues.append(r1.conjugate())
                    
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    print(f"RLC Error: {e}")
                    continue
        
        # 构造 MATLAB 结构体所需的数据
        # poles 和 residues 必须以复数形式存储，且是一维列向量
        poles_arr = np.array(poles, dtype=np.complex128).reshape(-1, 1)
        residues_arr = np.array(residues, dtype=np.complex128).reshape(-1, 1)
        
        # 构造传递函数分子分母系数 (Transfer Function Numerator/Denominator)
        # H(s) = num(s) / den(s)
        # 通过 residue() 函数的逆运算： [num, den] = residue(r, p, k)
        # 但 scipy.signal.invres 可以做这个
        # 为了让 MATLAB 能直接画伯德图，我们最好也计算出 num 和 den
        
        # 使用 scipy.signal.invres 计算传递函数系数
        # Y(s) = residues / (s - poles) + k(常数项)
        # 注意：这里的 k 是我们的 d，而 s*h 项无法直接用传递函数表示（除非分母阶数比分子大2? 不，是分子比分母阶数高1）
        # 标准传递函数是 proper 或 strictly proper 的。
        # 如果有 s*h 项，说明并非严格 proper。
        # 在 MATLAB 中 tf([num], [den]) 可以表示不严格 proper 的，只要 length(num) <= length(den) + 1
        
        # 我们手动构造 num, den
        # 因为 invres 的 k 参数只支持常数项，不支持线性项 s*h
        # 如果 h != 0，则 Y(s) 有一个 s 的项，我们可以把它看作 Y(s) = ( num_proper(s) / den(s) ) + h*s
        # 合并通分: Y(s) = ( num_proper(s) + h*s * den(s) ) / den(s)
        
        from scipy import signal
        
        # 1. 计算 Proper 部分 num_p / den
        # 只有当 poles 非空时才计算，否则如果没有极点（只有 R/C 并联），则 num/den 很简单
        if len(poles) > 0:
            # SciPy 的 invres 要求 r, p 是 1D array
            # k 必须是 rank-1 array? invres return b, a. 
            # 我们用 signal.invres(r, p, k) -> b, a
            # 注意：scipy 的 k 对应常数项 d
            num_proper, den = signal.invres(residues_arr.flatten(), poles_arr.flatten(), np.array([d_val]))
            
            # num_proper 和 den 是实系数数组（如果 poles/residues 是共轭对）
            num_proper = np.real(num_proper)
            den = np.real(den)
            
            # 2. 加入 h*s 项
            # Y(s) = num_proper/den + h*s = (num_proper + h*s*den) / den
            if abs(h_val) > 1e-15:
                # h*s*den 相当于 den 的系数左移一位（乘以 s），再乘以 h
                term_h = np.convolve(den, [h_val, 0]) # [h, 0] represents h*s + 0
                
                # 相加 num_proper 和 term_h
                # 需要对齐长度
                len_n = len(num_proper)
                len_h = len(term_h)
                max_len = max(len_n, len_h)
                
                num_total = np.zeros(max_len)
                # 右对齐相加 (低次幂在右，如果系数是降幂排列)
                # scipy/numpy poly 是降幂排列 (s^n, s^n-1, ..., s^0)
                
                # num_proper 放入
                num_total[max_len-len_n:] += num_proper
                # term_h 放入
                num_total[max_len-len_h:] += term_h
            else:
                num_total = num_proper
            
        else:
            # 没有极点，只有 d + sh
            # Y(s) = d + s*h
            # den = [1]
            # num = [h, d]
            den = np.array([1.0])
            if abs(h_val) > 1e-15:
                num_total = np.array([h_val, d_val])
            else:
                num_total = np.array([d_val])
        
        # 构造要保存的数据字典
        case_data = {
            'case_name': case_name,
            'P': condition_info['P'],
            'Q': condition_info['Q'],
            'V': condition_info['V'],
            'xi': condition_info['xi'],
            'poles': poles_arr,
            'residues': residues_arr,
            'd': d_val,
            'h': h_val,
            'num': num_total, # 分子多项式系数（降幂）
            'den': den        # 分母多项式系数（降幂）
        }
        
        # 构造文件名: CaseName_P_..._Q_..._V_..._xi_... .mat (xi 为弧度)
        # 为了避免文件名中出现非法字符或过长，可以做一定的格式化
        # 例如保留几位小数
        filename = f"{case_name}_P_{condition_info['P']:.2f}_Q_{condition_info['Q']:.2f}_V_{condition_info['V']:.2f}_xi_{condition_info['xi']:.4f}.mat"
        # 替换文件名中可能存在的非法字符 (虽然 P,Q,V,xi 应该都是数字)
        filename = filename.replace(" ", "_")
        
        save_path = os.path.join(OUTPUT_DIR_PATH, filename)
        
        # 保存单个 .mat 文件
        sio.savemat(save_path, {'admittance_model': case_data})
        
        print(f"{case_name:<20} | {len(poles):<10} | {d_val:.2e}   | {h_val:.2e}")
        # print(f"  Saved to: {filename}")

    print("-" * 60)
    print(f"成功! 所有数据已保存至文件夹: {OUTPUT_DIR_PATH}")
    print("每个工况保存为一个单独的 .mat 文件，文件名为: CaseName_P_..._Q_..._V_..._xi_(rad).mat")
    print("在 MATLAB 中使用: data = load('matlab_models/your_file.mat');")
    print("                 model = data.admittance_model;")
    print("绘图示例: tf_sys = tf(model.num, model.den); bode(tf_sys);")

if __name__ == "__main__":
    generate_matlab_data()
