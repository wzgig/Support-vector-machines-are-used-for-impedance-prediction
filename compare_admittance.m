clear;
clc;

% 1. 加载预测的导纳模型数据 (包含工况 P, Q, V, xi)
pred_data_file = 'predicted_admittance_transfer_function.mat';
if ~exist(pred_data_file, 'file')
    error(['找不到预测数据文件: ', pred_data_file]);
end
load(pred_data_file, 'admittance_models'); % 结构体数组

% 2. 运行 PMSG 线性化脚本 (修改后的版本)
disp('正在运行 PMSG 线性化计算 (真实值)...');
run('PMSGlinearization.m'); 
% 假设 PMSGlinearization.m 运行后会在工作区留下 linearization_results

% 3. 准备绘图
num_cases = length(admittance_models);
plot_indices = 1:min(num_cases, 9); % 最多画前9个

% 初始化误差存储
rmse_mag_all = zeros(length(plot_indices), 1);
rmse_phase_all = zeros(length(plot_indices), 1);

% 创建两个独立的图形窗口：一个用于幅值，一个用于相位
fig_mag = figure('Name', 'Admittance Magnitude Comparison', 'Color', 'w', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
fig_phase = figure('Name', 'Admittance Phase Comparison', 'Color', 'w', 'NumberTitle', 'off', 'Position', [150, 150, 1200, 800]);

for k = 1:length(plot_indices)
    idx = plot_indices(k);
    model_pred = admittance_models(idx);
    
    % --- A. 获取预测的 Y11 ---
    sys_pred = tf(model_pred.num, model_pred.den);
    
    % --- B. 获取实际计算的 Y11 ---
    if exist('linearization_results', 'var') && idx <= length(linearization_results)
        sys_actual_sys = linearization_results(idx).sys_y11; 
    else
        warning('未找到索引 %d 的线性化结果，跳过绘图', idx);
        continue;
    end
    
    % --- C. 获取数据点 ---
    % 使用预测模型的频率范围或自动范围
    [mag_pred, phase_pred, w] = bode(sys_pred);
    [mag_act, phase_act] = bode(sys_actual_sys, w); % 使用相同的频率向量 w
    
    f = w / (2*pi);
    
    % 数据处理
    mag_pred_db = 20*log10(squeeze(mag_pred));
    mag_act_db = 20*log10(squeeze(mag_act));
    
    phase_pred_deg = squeeze(phase_pred);
    phase_act_deg = squeeze(phase_act);
    
    % 相位平滑处理 (unwrap) - 可选，视情况而定，bode通常已经处理较好，但在某些极点处可能跳变
    % 这里我们直接使用输出值，如果差异巨大可能涉及 360 度
    
    % --- D. 计算误差 (RMSE) ---
    % 简单的 RMSE 计算
    rmse_mag = sqrt(mean((mag_pred_db - mag_act_db).^2));
    rmse_phase = sqrt(mean((phase_pred_deg - phase_act_deg).^2));
    
    rmse_mag_all(k) = rmse_mag;
    rmse_phase_all(k) = rmse_phase;
    
    % --- E. 绘制幅值图 ---
    figure(fig_mag);
    subplot(3, 3, k);
    semilogx(f, mag_pred_db, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Predicted'); hold on;
    semilogx(f, mag_act_db, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Actual');
    grid on;
    % xlabel('Frequency (Hz)', 'FontSize', 9); % 仅在底部显示标签以保持整洁？或者每个都显示
    xlabel('Frequency (Hz)', 'FontSize', 9);
    ylabel('Magnitude (dB)', 'FontSize', 9);
    title(sprintf('Case %d: P=%.2f, Q=%.2f\nRMSE=%.4f dB', idx, model_pred.P, model_pred.Q, rmse_mag), 'FontSize', 10, 'Interpreter', 'none');
    if k == 1
        legend('Location', 'best');
    end
    
    % --- F. 绘制相位图 ---
    figure(fig_phase);
    subplot(3, 3, k);
    semilogx(f, phase_pred_deg, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Predicted'); hold on;
    semilogx(f, phase_act_deg, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Actual');
    grid on;
    xlabel('Frequency (Hz)', 'FontSize', 9);
    ylabel('Phase (deg)', 'FontSize', 9);
    title(sprintf('Case %d: P=%.2f, Q=%.2f\nRMSE=%.4f deg', idx, model_pred.P, model_pred.Q, rmse_phase), 'FontSize', 10, 'Interpreter', 'none');
    if k == 1
        legend('Location', 'best');
    end
end

% 打印总体误差统计
fprintf('\n--- 误差统计 ---\n');
fprintf('平均幅值 RMSE: %.4f dB\n', mean(rmse_mag_all));
fprintf('平均相位 RMSE: %.4f deg\n', mean(rmse_phase_all));

disp('对比绘图完成。');
