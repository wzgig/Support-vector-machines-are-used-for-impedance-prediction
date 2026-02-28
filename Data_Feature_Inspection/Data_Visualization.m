%% R_e 数据特征可视化（按 PVQxi 工况；xi 为电网相角）
clear; clc; close all;

%% 1) 读入数据
filePath = "extracted_RL_Series_Y11_wide.csv";  % <- 改成你的文件路径
T = readtable(filePath);

mustHave = {'R_e','P','V','Q','xi'};
for k = 1:numel(mustHave)
    assert(any(strcmp(T.Properties.VariableNames, mustHave{k})), ...
        "表格中找不到列名 '%s'", mustHave{k});
end

% 取出
Re = T.R_e;
P  = T.P;
V  = T.V;
Q  = T.Q;
xi = T.xi;

% 清理
valid = isfinite(Re) & isfinite(P) & isfinite(V) & isfinite(Q) & isfinite(xi);
T  = T(valid,:);
Re = T.R_e; P=T.P; V=T.V; Q=T.Q; xi=T.xi;

fprintf("样本数 N = %d\n", numel(Re));
fprintf("R_e: mean=%.6g, std=%.6g, min=%.6g, max=%.6g\n", mean(Re), std(Re), min(Re), max(Re));

%% 2) xi 作为相角：单位与归一化
% 如果你的 xi 是"度"，改成 "deg"
xi_unit = "rad";  % "rad" 或 "deg"

if xi_unit == "deg"
    xi_rad = deg2rad(xi);
else
    xi_rad = xi;
end

% 归一化到 [-pi, pi)
xi_wrap = wrapToPi(xi_rad);

% 同时也给一个 [0, 2pi) 版本（画极坐标/分箱可能更顺手）
xi_0_2pi = mod(xi_rad, 2*pi);

%% 3) 主要多图面板：分布 + 序列 + 相角关系
figure('Name','R_e 总览（含相角特征）','Color','w','Position',[50 50 1450 850]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% (a) 分布：Histogram + KDE
nexttile;
histogram(Re, 'Normalization','pdf', 'NumBins', 40);
hold on;
try
    [f,xx] = ksdensity(Re);
    plot(xx, f, 'LineWidth', 1.8);
catch
end
grid on; xlabel('R_e'); ylabel('PDF');
title('分布：Histogram (+KDE)');

% (b) 箱线图
nexttile;
boxchart(Re);
grid on; ylabel('R_e');
title('箱线图（Boxplot）');

% (c) Q-Q 图
nexttile;
qqplot(Re); grid on;
title('Q-Q 图（正态性）');

% (d) 序列图（按样本顺序）
nexttile;
plot(Re,'-'); grid on;
xlabel('样本索引'); ylabel('R_e');
title('序列图（按样本顺序）');

% (e) R_e vs 相角（散点，使用 wrap 后的角度）
nexttile;
scatter(xi_wrap, Re, 10, 'filled', 'MarkerFaceAlpha', 0.25);
grid on;
xlabel('\xi (wrapped to [-\pi,\pi))'); ylabel('R_e');
title('关系：R_e vs 相角 \xi（散点）');

% (f) 相角分箱趋势：均值/中位数（更容易看周期）
nexttile;
nbins = 36; % 相角分箱数：36=每10度(约)；可改 24/72 等
edges = linspace(-pi, pi, nbins+1);
binId = discretize(xi_wrap, edges);

binCenter = (edges(1:end-1)+edges(2:end))/2;
meanRe_bin = nan(size(binCenter));
medRe_bin  = nan(size(binCenter));
cnt_bin    = zeros(size(binCenter));

for i = 1:numel(binCenter)
    m = (binId==i);
    cnt_bin(i) = nnz(m);
    if cnt_bin(i) > 0
        meanRe_bin(i) = mean(Re(m));
        medRe_bin(i)  = median(Re(m));
    end
end

plot(binCenter, meanRe_bin, '-o', 'LineWidth',1.2); hold on;
plot(binCenter, medRe_bin,  '-s', 'LineWidth',1.2);
grid on;
xlabel('\xi bin center (rad)'); ylabel('R_e');
title(sprintf('相角分箱趋势（%d bins）', nbins));
legend('mean(R_e)','median(R_e)','Location','best');

%% 4) （可选但很有用）极坐标：R_e 随相角的"环形"视图
figure('Name','极坐标：mean(R_e) vs 相角','Color','w','Position',[80 80 700 650]);
polarplot([binCenter binCenter(1)], [meanRe_bin meanRe_bin(1)], '-o', 'LineWidth', 1.5);
title('极坐标：mean(R_e) 随相角变化（分箱）');

%% 5) 工况分组：PVQxi 作为"一组工况"
uniqRatio = numel(unique(xi_wrap)) / numel(xi_wrap);
fprintf("\nxi 唯一值占比 = %.3f（越小越离散）\n", uniqRatio);

% --- A) 严格 PVQxi 分组（离散工况适用）
if uniqRatio < 0.2
    grpPVQxi = "P="+string(P) + "|V="+string(V) + "|Q="+string(Q) + "|xi="+string(round(xi_wrap,4));
    figure('Name','R_e 按 (P,V,Q,xi) 分组箱线图','Color','w','Position',[100 100 1500 600]);
    boxchart(categorical(grpPVQxi), Re);
    grid on; xlabel('工况组 (P,V,Q,xi)'); ylabel('R_e');
    title('R_e 分组对比：按 (P,V,Q,xi) 组合');
    xtickangle(30);
else
    fprintf("提示：xi 看起来更像连续角度，严格 PVQxi 分组会导致组数过多，已跳过该图。\n");
end

% --- B) 推荐：按 (P,V,Q) 分组，在组内观察 R_e 与相角
grpPVQ = "P="+string(P) + "|V="+string(V) + "|Q="+string(Q);
[G, grpNames] = findgroups(grpPVQ);

figure('Name','按 (P,V,Q) 分组：组内 R_e~xi','Color','w','Position',[120 120 1500 800]);
tiledlayout('flow','Padding','compact','TileSpacing','compact');

K = 12;
grpCount = splitapply(@numel, Re, G);
[~, ord] = sort(grpCount, 'descend');
showIdx = ord(1:min(K, numel(ord)));

for ii = 1:numel(showIdx)
    g = showIdx(ii);
    m = (G==g);

    nexttile;
    scatter(xi_wrap(m), Re(m), 10, 'filled', 'MarkerFaceAlpha', 0.25);
    grid on;
    xlabel('\xi (wrapped)'); ylabel('R_e');
    title(sprintf('%s (n=%d)', grpNames(g), nnz(m)));

    % 组内相角分箱均值（修复原来的索引错误）
    edges2 = linspace(-pi, pi, nbins+1);
    binId2 = discretize(xi_wrap(m), edges2);
    bc2 = (edges2(1:end-1)+edges2(2:end))/2;
    mean2 = nan(size(bc2));
    for jj = 1:numel(bc2)
        mm = (binId2==jj);
        if any(mm)
            mean2(jj) = mean(Re(m));
            % 更严格：只取该组该bin
            mean2(jj) = mean(Re(m).*(binId2==jj),'omitnan'); % 兼容性差，下面用安全写法替代
        end
    end
    % 上面那行可能因版本不支持 omitnan 或逻辑乘法导致问题，统一采用安全写法：
    mean2(:) = NaN;
    Re_g = Re(m);
    for jj = 1:numel(bc2)
        mm = (binId2==jj);
        if any(mm)
            mean2(jj) = mean(Re_g(mm));
        end
    end
    hold on;
    plot(bc2, mean2, '-','LineWidth',1.2);
end

%% 6) 异常值（IQR）+ 标注
Q1 = quantile(Re, 0.25);
Q3 = quantile(Re, 0.75);
IQRv = Q3 - Q1;
lowThr  = Q1 - 1.5*IQRv;
highThr = Q3 + 1.5*IQRv;
isOut = (Re < lowThr) | (Re > highThr);

fprintf("\nIQR异常值阈值：low=%.6g, high=%.6g\n", lowThr, highThr);
fprintf("异常值数量：%d (%.2f%%)\n", nnz(isOut), 100*nnz(isOut)/numel(Re));

figure('Name','异常值标注（IQR）','Color','w','Position',[140 140 1200 450]);
plot(Re,'-'); hold on;
plot(find(isOut), Re(isOut), 'o', 'MarkerSize',6, 'LineWidth',1.5);
yline(lowThr,'--'); yline(highThr,'--');
grid on; xlabel('样本索引'); ylabel('R_e');
title('异常值标注（IQR）');
legend('R_e','Outliers','lowThr','highThr','Location','best');

%% ===================== 7) 四变量独立效应分析（P/V/Q/xi -> R_e） =====================
% xi 为相角 -> 用 cos/sin 表达周期性
doGAM = true;

cos_xi = cos(xi_wrap);
sin_xi = sin(xi_wrap);

Tbl = table(Re, P, V, Q, cos_xi, sin_xi, ...
    'VariableNames', {'Re','P','V','Q','cos_xi','sin_xi'});

% A) OLS
lm = fitlm(Tbl, 'Re ~ P + V + Q + cos_xi + sin_xi');

% A2) 标准化 OLS（修复：保持变量名，避免 varfun 改名）
TblZ = Tbl;
varsZ = {'Re','P','V','Q','cos_xi','sin_xi'};
for k = 1:numel(varsZ)
    vn = varsZ{k};
    x = TblZ.(vn);
    sx = std(x);
    if sx == 0 || ~isfinite(sx)
        TblZ.(vn) = zeros(size(x));
    else
        TblZ.(vn) = (x - mean(x)) ./ sx;
    end
end
lmZ = fitlm(TblZ, 'Re ~ P + V + Q + cos_xi + sin_xi');

coefTable  = lm.Coefficients;
coefTableZ = lmZ.Coefficients;

% Partial R^2
SSE_full = lm.SSE;
vars = {'P','V','Q','cos_xi','sin_xi'};
partialR2 = nan(numel(vars),1);
for i = 1:numel(vars)
    formula_reduced = "Re ~ " + strjoin(setdiff(vars, vars{i}), " + ");
    lm_red = fitlm(Tbl, formula_reduced);
    SSE_red = lm_red.SSE;
    partialR2(i) = (SSE_red - SSE_full) / SSE_red;
end

indepEffectTable = table( ...
    vars', ...
    coefTable.Estimate(2:end), ...
    coefTableZ.Estimate(2:end), ...
    coefTable.pValue(2:end), ...
    partialR2, ...
    'VariableNames', {'Variable','Coef','StdCoef','pValue','PartialR2'});

indepEffectTable = sortrows(indepEffectTable, 'PartialR2', 'descend');

disp("=== 独立效应（OLS + 相角谐波）===");
disp(indepEffectTable);
fprintf("OLS 模型 R^2 = %.4f, Adjusted R^2 = %.4f\n", lm.Rsquared.Ordinary, lm.Rsquared.Adjusted);

figure('Name','独立效应强弱：标准化系数（OLS）','Color','w','Position',[160 160 900 450]);
bar(categorical(indepEffectTable.Variable), indepEffectTable.StdCoef);
grid on; xlabel('变量'); ylabel('标准化系数 StdCoef');
title('控制其它变量后：各变量对 R_e 的独立影响强弱（OLS 线性近似）');

figure('Name','独立解释度：Partial R^2（OLS）','Color','w','Position',[180 180 900 450]);
bar(categorical(indepEffectTable.Variable), indepEffectTable.PartialR2);
grid on; xlabel('变量'); ylabel('Partial R^2');
title('控制其它变量后：各变量独立解释度（Partial R^2, OLS）');

% B) 稳健回归（应对异常值）
lmRob = [];
try
    lmRob = fitlm(Tbl, 'Re ~ P + V + Q + cos_xi + sin_xi', 'RobustOpts','on');
    fprintf("fitlm 稳健回归：R^2=%.4f, AdjR^2=%.4f\n", lmRob.Rsquared.Ordinary, lmRob.Rsquared.Adjusted);
catch
    warning("当前版本 fitlm 不支持 RobustOpts 或工具箱缺失，跳过 lmRob。");
end

% robustfit（更通用）
X = [Tbl.P, Tbl.V, Tbl.Q, Tbl.cos_xi, Tbl.sin_xi];
y = Tbl.Re;
[bRob, statsRob] = robustfit(X, y);
robNames = {'Intercept','P','V','Q','cos_xi','sin_xi'}';
robCoef  = bRob;
robP     = statsRob.p;
robTable = table(robNames, robCoef, robP, 'VariableNames',{'Term','Coef','pValue'});
disp("=== robustfit 系数与显著性（抗异常值）===");
disp(robTable);

% C) GAM 非线性（可选）
gam = [];
if doGAM
    try
        gam = fitrgam(Tbl, 'Re ~ P + V + Q + cos_xi + sin_xi');
        figure('Name','GAM 非线性独立效应曲线','Color','w','Position',[200 200 1200 700]);
        tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
        for i = 1:numel(vars)
            nexttile;
            plotPartialDependence(gam, vars{i});
            grid on;
            title("GAM: " + vars{i} + " 的独立效应");
        end
    catch ME
        warning(ME.identifier, '%s', ME.message);
        gam = [];
        doGAM = false;
    end
end

%% ===================== 8) 生成 Markdown 报告并保存 =====================
reportPath = "R_e_feature_report.md";
exportFigures = true;
figDir = "report_figs";

if exportFigures
    if ~exist(figDir, "dir"); mkdir(figDir); end
end

N = numel(Re);

% 基本统计
Re_mean = mean(Re);
Re_std  = std(Re);
Re_min  = min(Re);
Re_max  = max(Re);
Re_med  = median(Re);
Re_iqr  = iqr(Re);
Re_skew = skewness(Re);
Re_kurt = kurtosis(Re);

% 分位数
q01 = quantile(Re,0.01); q05 = quantile(Re,0.05);
q25 = quantile(Re,0.25); q75 = quantile(Re,0.75);
q95 = quantile(Re,0.95); q99 = quantile(Re,0.99);

outN = nnz(isOut);

% xi 圆统计
mu = atan2(mean(sin(xi_wrap)), mean(cos(xi_wrap)));
Rbar = hypot(mean(cos(xi_wrap)), mean(sin(xi_wrap)));

% 一阶谐波相关
corr_c1 = corr(Re, cos_xi, 'Rows','pairwise');
corr_s1 = corr(Re, sin_xi, 'Rows','pairwise');

% 分箱峰谷差
tmp = meanRe_bin(isfinite(meanRe_bin));
if isempty(tmp), mean_peak2peak = NaN;
else, mean_peak2peak = max(tmp) - min(tmp);
end

% PVQ 分组统计
[G2, grpNames2] = findgroups(grpPVQ);
numGroups = numel(grpNames2);
grpCount2 = splitapply(@numel, Re, G2);
grpMean2  = splitapply(@mean,  Re, G2);
grpStd2   = splitapply(@std,   Re, G2);
grpMed2   = splitapply(@median,Re, G2);

TopK = min(10, numGroups);
[~, ord2] = sort(grpCount2, 'descend');
topIdx = ord2(1:TopK);

% 导出图
figFiles = strings(0);
if exportFigures
    figs = findobj('Type','figure');
    figs = flipud(figs);
    maxExport = min(8, numel(figs));
    for i = 1:maxExport
        f = figs(i);
        safeName = "fig_" + string(i);
        pngPath = fullfile(figDir, safeName + ".png");
        try
            exportgraphics(f, pngPath, 'Resolution', 180);
            figFiles(end+1) = pngPath; %#ok<AGROW>
        catch
            try
                print(f, pngPath, "-dpng", "-r180");
                figFiles(end+1) = pngPath; %#ok<AGROW>
            catch
            end
        end
    end
end

% 写 Markdown
fid = fopen(reportPath, "w");
assert(fid>0, "无法创建报告文件：%s", reportPath);
cleanupObj = onCleanup(@() fclose(fid));

ts = datetime("now","TimeZone","local","Format","yyyy-MM-dd HH:mm:ss");

fprintf(fid, "# R_e 数据特征分析报告\n\n");
fprintf(fid, "- 生成时间：%s\n", string(ts));
fprintf(fid, "- 数据文件：`%s`\n", filePath);
fprintf(fid, "- 样本数：`%d`\n\n", N);

fprintf(fid, "## 1. 变量与工况说明\n\n");
fprintf(fid, "- 工况定义：`(P, V, Q, xi)` 为一组工况。\n");
fprintf(fid, "- `xi`：电网相角（作为周期变量处理，并 wrap 到 `[-π, π)`）。\n\n");

fprintf(fid, "## 2. R_e 基本统计特征\n\n");
fprintf(fid, "| 指标 | 数值 |\n|---:|---:|\n");
fprintf(fid, "| mean | %.6g |\n", Re_mean);
fprintf(fid, "| std  | %.6g |\n", Re_std);
fprintf(fid, "| min  | %.6g |\n", Re_min);
fprintf(fid, "| 1%% quantile | %.6g |\n", q01);
fprintf(fid, "| 5%% quantile | %.6g |\n", q05);
fprintf(fid, "| 25%% quantile (Q1) | %.6g |\n", q25);
fprintf(fid, "| median | %.6g |\n", Re_med);
fprintf(fid, "| 75%% quantile (Q3) | %.6g |\n", q75);
fprintf(fid, "| 95%% quantile | %.6g |\n", q95);
fprintf(fid, "| 99%% quantile | %.6g |\n", q99);
fprintf(fid, "| max  | %.6g |\n", Re_max);
fprintf(fid, "| IQR  | %.6g |\n", Re_iqr);
fprintf(fid, "| skewness | %.6g |\n", Re_skew);
fprintf(fid, "| kurtosis | %.6g |\n", Re_kurt);
fprintf(fid, "\n");

fprintf(fid, "## 3. 异常值分析（IQR 规则）\n\n");
fprintf(fid, "- 阈值：`low = %.6g`，`high = %.6g`\n", lowThr, highThr);
fprintf(fid, "- 异常值数量：`%d`（占比 `%.2f%%`）\n\n", outN, 100*outN/N);

fprintf(fid, "### 3.1 异常值与稳健建模说明\n\n");
fprintf(fid, "- 本数据 `R_e` 极值范围很大（min/max 远超 IQR 阈值），异常值占比约 `%.2f%%`。\n", 100*outN/N);
fprintf(fid, "- 在异常值比例较高时：\n");
fprintf(fid, "  - 普通最小二乘（OLS）的系数可能被极端点拉偏；\n");
fprintf(fid, "  - 因此本报告同时给出 **稳健回归（Robust Regression）** 作为对照，它会降低异常点权重，使‘独立效应’更稳定。\n\n");

fprintf(fid, "## 4. 相角 xi（圆统计）与 R_e 的关系\n\n");
fprintf(fid, "- 圆均值方向 `mu = %.6g rad`\n", mu);
fprintf(fid, "- 集中度 `R = %.6g`（越接近1越集中）\n\n", Rbar);

fprintf(fid, "### 4.1 一阶谐波相关\n\n");
fprintf(fid, "- corr(`R_e`, cos(xi)) = `%.4f`\n", corr_c1);
fprintf(fid, "- corr(`R_e`, sin(xi)) = `%.4f`\n\n", corr_s1);

fprintf(fid, "### 4.2 相角分箱趋势\n\n");
fprintf(fid, "- 分箱数：`%d`\n", nbins);
fprintf(fid, "- `mean(R_e)` 峰谷差（peak-to-peak）：`%.6g`\n\n", mean_peak2peak);

fprintf(fid, "## 5. 工况分组差异（按 P,V,Q 分组）\n\n");
fprintf(fid, "- `(P,V,Q)` 组数量：`%d`\n\n", numGroups);

fprintf(fid, "### 5.1 样本数最多的 Top %d 组摘要\n\n", TopK);
fprintf(fid, "| Rank | 组 (P,V,Q) | n | mean(R_e) | std(R_e) | median(R_e) |\n");
fprintf(fid, "|---:|---|---:|---:|---:|---:|\n");
for k = 1:TopK
    g = topIdx(k);
    fprintf(fid, "| %d | %s | %d | %.6g | %.6g | %.6g |\n", ...
        k, grpNames2(g), grpCount2(g), grpMean2(g), grpStd2(g), grpMed2(g));
end
fprintf(fid, "\n");

fprintf(fid, "## 6. 四变量独立效应分析（控制其它变量后）\n\n");
fprintf(fid, "模型：`R_e ~ P + V + Q + cos(xi) + sin(xi)`（xi 用 cos/sin 表达周期性，避免角度断点）\n\n");
fprintf(fid, "- OLS 模型 R² = `%.4f`，Adjusted R² = `%.4f`\n\n", lm.Rsquared.Ordinary, lm.Rsquared.Adjusted);

fprintf(fid, "| 变量 | Coef | StdCoef | p-value | PartialR2 |\n");
fprintf(fid, "|---|---:|---:|---:|---:|\n");
for i = 1:height(indepEffectTable)
    fprintf(fid, "| %s | %.6g | %.6g | %.3g | %.6g |\n", ...
        indepEffectTable.Variable{i}, ...
        indepEffectTable.Coef(i), ...
        indepEffectTable.StdCoef(i), ...
        indepEffectTable.pValue(i), ...
        indepEffectTable.PartialR2(i));
end
fprintf(fid, "\n");

fprintf(fid, "### 6.1 解读建议\n\n");
fprintf(fid, "- `StdCoef` 绝对值越大，表示在**线性近似**下独立影响越强。\n");
fprintf(fid, "- `PartialR2` 越大，表示该变量在控制其它变量后能独立解释更多 `R_e` 方差。\n");
fprintf(fid, "- `p-value < 0.05` 常用作统计显著的经验阈值（需结合工程意义解释）。\n\n");

fprintf(fid, "### 6.2 稳健回归对照（robustfit）\n\n");
fprintf(fid, "| Term | Coef | p-value |\n|---|---:|---:|\n");
for i = 1:height(robTable)
    fprintf(fid, "| %s | %.6g | %.3g |\n", robTable.Term{i}, robTable.Coef(i), robTable.pValue(i));
end
fprintf(fid, "\n");
fprintf(fid, "- 若稳健回归与 OLS 在系数符号/显著性上差异明显，说明 OLS 可能受到异常值影响，建议以稳健结果为主。\n\n");

if ~isempty(gam)
    fprintf(fid, "### 6.3 非线性独立效应（GAM）\n\n");
    fprintf(fid, "- 已拟合 GAM 并输出各变量的 partial dependence 曲线（用于观察非线性独立效应形状）。\n\n");
else
    fprintf(fid, "### 6.3 非线性独立效应（GAM）\n\n");
    fprintf(fid, "- 本次未生成 GAM（可能因版本/工具箱缺失或拟合失败）。需要时可将 `doGAM=false/true` 调整并重试。\n\n");
end

fprintf(fid, "## 7. 结论要点（自动生成）\n\n");

if outN/N > 0.05
    outText = "异常值比例偏高，建议检查数据来源或采用稳健建模/异常点处理。";
elseif outN/N > 0.01
    outText = "存在少量异常值，建模时可考虑稳健损失或异常点处理。";
else
    outText = "异常值比例较低，整体较稳定。";
end

if abs(corr_c1) > 0.3 || abs(corr_s1) > 0.3 || (isfinite(mean_peak2peak) && mean_peak2peak > 0.1*Re_std)
    angleText = "R_e 与相角存在较明显关联（可能有周期成分）；建议分 (P,V,Q) 工况做谐波/傅里叶回归量化幅值与相位。";
else
    angleText = "全局看 R_e 与相角一阶谐波关联不强，但仍建议在分工况后复核局部相角依赖。";
end

if abs(Re_skew) > 1
    skewText = "分布偏态较明显，必要时可考虑对 R_e 做变换（log/Box-Cox）或使用分位数/非参数方法。";
else
    skewText = "分布偏态不算极端，可结合稳健统计结果给出工程解释。";
end

fprintf(fid, "- %s\n", outText);
fprintf(fid, "- %s\n", angleText);
fprintf(fid, "- %s\n\n", skewText);

if exportFigures && ~isempty(figFiles)
    fprintf(fid, "## 8. 图表（自动导出）\n\n");
    for i = 1:numel(figFiles)
        relPath = strrep(figFiles(i), "\", "/");
        fprintf(fid, "### 图 %d\n\n", i);
        fprintf(fid, "![Figure %d](%s)\n\n", i, relPath);
    end
else
    fprintf(fid, "## 8. 图表\n\n（本次未导出图像文件。如需导出，请设置 `exportFigures = true`。）\n\n");
end

fprintf(fid, "---\n");
fprintf(fid, "报告文件已保存为：`%s`\n", reportPath);

fprintf("✅ Markdown 报告已生成：%s\n", reportPath);
if exportFigures
    fprintf("✅ 图像已导出到目录：%s\n", figDir);
end
