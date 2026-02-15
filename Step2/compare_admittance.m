clear;
clc;
close all; % Close old figures

% 1. Run PMSG linearization script
% This script has been updated to:
% - Look for 'matlab_models' directory
% - Iterate through all .mat files found there
% - Compute linearization for each
% - Store results in 'linearization_results' struct array
disp('Running PMSGlinearization.m to process all cases...');
try
    run('PMSGlinearization.m');
catch ME
    error('Error running PMSGlinearization.m: %s', ME.message);
end

if ~exist('linearization_results', 'var') || isempty(linearization_results)
    error('linearization_results not found or empty. Please check PMSGlinearization.m.');
end

% 2. Iterate through results and plot
num_cases = length(linearization_results);
fprintf('Found %d cases. Generating plots...\n', num_cases);

for k = 1:num_cases
    result = linearization_results(k);
    case_name = result.case_name;
    
    % Get predicted model (saved in result.pred_model by PMSGlinearization.m)
    if ~isfield(result, 'pred_model')
        warning('Case %s missing pred_model. Skipping.', case_name);
        continue;
    end
    model_pred = result.pred_model;
    
    % Get actual model (linearization result)
    sys_actual_sys = result.sys_y11;
    
    fprintf('Plotting Case %d/%d: %s\n', k, num_cases, case_name);

    % --- A. Construct Predicted Transfer Function ---
    % num and den might be cell arrays or arrays in the loaded struct
    num = double(model_pred.num);
    den = double(model_pred.den);
    sys_pred = tf(num, den);

    % --- B. Frequency Response ---
    % Frequency range (1Hz to 100kHz)
    w_min = 2*pi*1e-1; 
    w_max = 2*pi*1e5;
    w = logspace(log10(w_min), log10(w_max), 500); 
    f = w / (2*pi);

    [resp_pred] = freqresp(sys_pred, w);
    [resp_act]  = freqresp(sys_actual_sys, w);

    resp_pred = squeeze(resp_pred);
    resp_act  = squeeze(resp_act);

    % --- C. Data Processing ---
    % Magnitude (dB)
    mag_pred_db = 20*log10(abs(resp_pred));
    mag_act_db  = 20*log10(abs(resp_act));

    % Phase (deg)
    phase_pred_deg = angle(resp_pred) * (180/pi);
    phase_act_deg  = angle(resp_act) * (180/pi);

    % Calculate RMSE
    rmse_mag   = sqrt(mean((mag_pred_db - mag_act_db).^2));
    % Phase difference (unwrap/angle difference)
    diff_phase = angle(resp_pred ./ resp_act) * (180/pi);
    rmse_phase = sqrt(mean(diff_phase.^2));
    
    % Parameters for title
    P_val = model_pred.P;
    Q_val = model_pred.Q;
    V_val = model_pred.V;
    xi_val = model_pred.xi;

    % Construct detailed title string
    case_info_str = sprintf('%s (P=%.2f, Q=%.2f, V=%.2f, xi=%.4f rad)', case_name, P_val, Q_val, V_val, xi_val);

    % --- D. Plotting ---
    font_name = 'Times New Roman';
    label_font_size = 10;
    axis_font_size = 9;
    fig_width = 18;  % cm
    fig_height = 8;  % cm

    % Colors
    color_act  = [0.85, 0.32, 0.10]; % Orange (Actual)
    color_pred = [0.00, 0.45, 0.74]; % Blue (Predicted)
    line_style_act  = '--';
    line_style_pred = '-';
    line_width_act  = 1.5;
    line_width_pred = 1.5;

    fig = figure('Units', 'centimeters', 'Position', [2 + k*0.5, 2 + k*0.5, fig_width, fig_height], ...
                 'Name', ['Admittance Comparison - ' case_name], 'Color', 'w', 'NumberTitle', 'on');

    % Add a super title for the whole figure with case details
    % checking if sgtitle exists (R2018b+)
    if exist('sgtitle', 'file')
        sgtitle(case_info_str, 'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 11);
    else
        % Fallback for older MATLAB versions: use annotation
        annotation('textbox', [0, 0.9, 1, 0.1], 'String', case_info_str, 'EdgeColor', 'none', ...
                   'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 11, 'Interpreter', 'none');
    end

    % === Magnitude Plot ===
    subplot(1, 2, 1);
    p1 = semilogx(f, mag_act_db, 'LineStyle', line_style_act, 'Color', color_act, 'LineWidth', line_width_act); hold on;
    p2 = semilogx(f, mag_pred_db, 'LineStyle', line_style_pred, 'Color', color_pred, 'LineWidth', line_width_pred);

    grid on;
    set(gca, 'GridAlpha', 0.15, 'MinorGridAlpha', 0.1);
    xlabel('Frequency (Hz)', 'FontName', font_name, 'FontSize', label_font_size);
    ylabel('Magnitude (dB)', 'FontName', font_name, 'FontSize', label_font_size);
    xlim([min(f), max(f)]);

    % Title with RMSE only
    title(sprintf('Magnitude (RMSE=%.3f dB)', rmse_mag), ...
          'FontName', font_name, 'FontSize', label_font_size, 'FontWeight', 'bold');
          
    set(gca, 'FontName', font_name, 'FontSize', axis_font_size, 'Box', 'on', 'LineWidth', 1);

    % Legend
    if k == 1
        legend([p2, p1], {'SVM Predicted', 'Analytical Model'}, 'Location', 'southwest', ...
               'FontSize', 8, 'FontName', font_name, 'EdgeColor', 'none', 'Color', 'none'); 
    end

    % === Phase Plot ===
    subplot(1, 2, 2);
    semilogx(f, phase_act_deg, 'LineStyle', line_style_act, 'Color', color_act, 'LineWidth', line_width_act); hold on;
    semilogx(f, phase_pred_deg, 'LineStyle', line_style_pred, 'Color', color_pred, 'LineWidth', line_width_pred);

    grid on;
    set(gca, 'GridAlpha', 0.15, 'MinorGridAlpha', 0.1);
    xlabel('Frequency (Hz)', 'FontName', font_name, 'FontSize', label_font_size);
    ylabel('Phase (deg)', 'FontName', font_name, 'FontSize', label_font_size);
    xlim([min(f), max(f)]);
    ylim([-200, 200]); 
    yticks(-180:90:180);

    % Title with RMSE only
    title(sprintf('Phase (RMSE=%.3f deg)', rmse_phase), ...
          'FontName', font_name, 'FontSize', label_font_size, 'FontWeight', 'bold');
          
    set(gca, 'FontName', font_name, 'FontSize', axis_font_size, 'Box', 'on', 'LineWidth', 1);

    % Print stats to console
    fprintf('  - File: %s\n  - Mag RMSE: %.4f dB\n  - Phase RMSE: %.4f deg\n', case_name, rmse_mag, rmse_phase);
end

disp('All plots generated.');
