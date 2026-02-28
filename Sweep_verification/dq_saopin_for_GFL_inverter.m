% %% clc; clear; tic
% % 逆变器 dq 阻抗扫频程序（可读性改进版，不改变功能/方法）
% 
% clc;
% clear;
% tic;
% 
% %% =========================
% % 1) 基本参数
% %% =========================
% Ts      = 5e-6;
% t_inj   = 7.5;
% 
% f_begin = 1e0;
% f_end   = 1e3;
% Point   = 40;
% 
% % 扫描频率点（与你原代码一致：logspace 后 floor）
% hp0 = floor(logspace(log10(f_begin), log10(f_end), Point));   % 扫描点频率
% hpn = length(hp0) / 5;                                       % 循环次数（每次注入 5 个频点）
% 
% %% =========================
% % 2) 扰动幅值配置（保持原方法：0.01/0.03/... * funp 或 funn）
% %% =========================
% ampScale = [0.01 0.03 0.05 0.07 0.09];  % 幅值越大越精确，但过大会影响稳态工作点
% 
% % 用于生成 Ap1~Ap5 / An1~An5（保持变量名，避免影响模型/脚本依赖）
% setInjectionAmps = @(funp, funn) assignin('base', 'dummy', []); %#ok<NASGU>
% % 上面这行只是为了提示：我们下面用明确赋值来保证变量名存在，避免你模型依赖工作区变量时出问题。
% 
% %% =========================
% % 3) 主循环：每轮 5 个频点
% %% =========================
% for n0 = 1:hpn
% 
%     %% ---------- 3.1 本轮注入频点 fc0（5 点） ----------
%     hp1 = hp0(n0);
%     hp2 = hp0(n0 + hpn);
%     hp3 = hp0(n0 + 2*hpn);
%     hp4 = hp0(n0 + 3*hpn);
%     hp5 = hp0(n0 + 4*hpn);
%     fc0 = [hp1, hp2, hp3, hp4, hp5];   % 本轮的 5 个频点
%     disp(fc0);
% 
%     %% ---------- 3.2 正序注入：funp=50, funn=0 ----------
%     funp = 50;
%     funn = 0;
% 
%     % 保持你原来的 Ap1~Ap5 / An1~An5 变量名与计算方式
%     Ap1 = ampScale(1)*funp;  Ap2 = ampScale(2)*funp;  Ap3 = ampScale(3)*funp;  Ap4 = ampScale(4)*funp;  Ap5 = ampScale(5)*funp;
%     An1 = ampScale(1)*funn;  An2 = ampScale(2)*funn;  An3 = ampScale(3)*funn;  An4 = ampScale(4)*funn;  An5 = ampScale(5)*funn;
% 
%     % 仿真运行（保持不变）
%     sim('GFL_inverter.slx');
% 
%     % FFT 读取：对本轮 5 个频点逐个读取（保持：n=fc0(x)+1 + FFTread_Vp）
%     for x = 1:length(fc0)
%         n = fc0(x) + 1; %#ok<NASGU>  % 外部脚本可能依赖 n
%         FFTread_Vp;                 % 你原注释：算出一次
%     end
% 
%     %% ---------- 3.3 负序注入：funp=0, funn=50 ----------
%     funp = 0;
%     funn = 50;
% 
%     Ap1 = ampScale(1)*funp;  Ap2 = ampScale(2)*funp;  Ap3 = ampScale(3)*funp;  Ap4 = ampScale(4)*funp;  Ap5 = ampScale(5)*funp;
%     An1 = ampScale(1)*funn;  An2 = ampScale(2)*funn;  An3 = ampScale(3)*funn;  An4 = ampScale(4)*funn;  An5 = ampScale(5)*funn;
% 
%     sim('GFL_inverter.slx');
% 
%     for x = 1:length(fc0)
%         n = fc0(x) + 1; %#ok<NASGU>
%         FFTread_Vn;
%     end
% 
%     %% =========================
%     % 4) 阻抗/导纳计算（不改变：Zdq = -vdqj/idqj；Ydq = inv(Zdq)）
%     %% =========================
%     len = length(fc0);
% 
%     Zdq = zeros(2, 2, len);
%     Ydq = zeros(2, 2, len);
% 
%     % 预分配 GM/PM（阻抗 + 导纳），避免动态增长
%     GM_Zdd_scan_1 = zeros(1, len); PM_Zdd_scan_1 = zeros(1, len);
%     GM_Zdq_scan_1 = zeros(1, len); PM_Zdq_scan_1 = zeros(1, len);
%     GM_Zqd_scan_1 = zeros(1, len); PM_Zqd_scan_1 = zeros(1, len);
%     GM_Zqq_scan_1 = zeros(1, len); PM_Zqq_scan_1 = zeros(1, len);
% 
%     GM_Ydd_scan_1 = zeros(1, len); PM_Ydd_scan_1 = zeros(1, len);
%     GM_Ydq_scan_1 = zeros(1, len); PM_Ydq_scan_1 = zeros(1, len);
%     GM_Yqd_scan_1 = zeros(1, len); PM_Yqd_scan_1 = zeros(1, len);
%     GM_Yqq_scan_1 = zeros(1, len); PM_Yqq_scan_1 = zeros(1, len);
% 
%     for x = 1:len
%         % 组装电压/电流扰动矩阵（保持原变量名）
%         vdqj = [vdjp(x) vdjn(x); vqjp(x) vqjn(x)];
%         idqj = [idjp(x) idjn(x); iqjp(x) iqjn(x)];
% 
%         % 阻抗矩阵
%         Zdq(:, :, x) = -vdqj / idqj;
% 
%         % 导纳矩阵（Z 的逆）
%         Ydq(:, :, x) = inv(Zdq(:, :, x));
% 
%         % 阻抗 GM/PM
%         GM_Zdd_scan_1(x) = GM(Zdq(1,1,x));  PM_Zdd_scan_1(x) = PM(Zdq(1,1,x));
%         GM_Zdq_scan_1(x) = GM(Zdq(1,2,x));  PM_Zdq_scan_1(x) = PM(Zdq(1,2,x));
%         GM_Zqd_scan_1(x) = GM(Zdq(2,1,x));  PM_Zqd_scan_1(x) = PM(Zdq(2,1,x));
%         GM_Zqq_scan_1(x) = GM(Zdq(2,2,x));  PM_Zqq_scan_1(x) = PM(Zdq(2,2,x));
% 
%         % 导纳 GM/PM
%         GM_Ydd_scan_1(x) = GM(Ydq(1,1,x));  PM_Ydd_scan_1(x) = PM(Ydq(1,1,x));
%         GM_Ydq_scan_1(x) = GM(Ydq(1,2,x));  PM_Ydq_scan_1(x) = PM(Ydq(1,2,x));
%         GM_Yqd_scan_1(x) = GM(Ydq(2,1,x));  PM_Yqd_scan_1(x) = PM(Ydq(2,1,x));
%         GM_Yqq_scan_1(x) = GM(Ydq(2,2,x));  PM_Yqq_scan_1(x) = PM(Ydq(2,2,x));
%     end
% 
%     %% =========================
%     % 5) 绘图（不改变图的含义，只把重复代码结构化）
%     %% =========================
% 
%     % ---- 阻抗绘图：figure(1~4) ----
%     Z_GM = {GM_Zdd_scan_1, GM_Zdq_scan_1, GM_Zqd_scan_1, GM_Zqq_scan_1};
%     Z_PM = {PM_Zdd_scan_1, PM_Zdq_scan_1, PM_Zqd_scan_1, PM_Zqq_scan_1};
% 
%     for k = 1:4
%         figure(k);
%         subplot(2,1,1);
%         semilogx(fc0, Z_GM{k}, 'r+', 'linewidth', 1); hold on;
%         subplot(2,1,2);
%         semilogx(fc0, Z_PM{k}, 'r+', 'linewidth', 1); hold on;
%     end
% 
%     % ---- 导纳绘图：figure(5~8) ----
%     Y_GM = {GM_Ydd_scan_1, GM_Ydq_scan_1, GM_Yqd_scan_1, GM_Yqq_scan_1};
%     Y_PM = {PM_Ydd_scan_1, PM_Ydq_scan_1, PM_Yqd_scan_1, PM_Yqq_scan_1};
%     yName = {'Y_{dd}', 'Y_{dq}', 'Y_{qd}', 'Y_{qq}'};
% 
%     for k = 1:4
%         figure(4+k);
%         subplot(2,1,1);
%         semilogx(fc0, Y_GM{k}, 'b+', 'linewidth', 1); hold on;
%         title([yName{k} ' 幅频特性']);
% 
%         subplot(2,1,2);
%         semilogx(fc0, Y_PM{k}, 'b+', 'linewidth', 1); hold on;
%         title([yName{k} ' 相频特性']);
%     end
% 
% end
% 
% fprintf('Run Time（Second）：');
% toc
% fprintf('***************END***************\n');


% %% run_dq_scan_single_point.m
% % 逆变器 dq 阻抗/导纳 扫频（逐频点注入版本）
% % -------------------------------------------------------------------------
% % 【调试建议】
% % 1) 先把 Point 设小（比如 5~10），确认每个频点 Z/Y 是否合理；
% % 2) 再逐步增大 Point；
% % 3) 每次 FFT 结果异常时，优先检查：
% %    - 注入频率 f 是否与采样窗、FFT点数匹配（避免严重泄漏）
% %    - 注入幅值是否过大扰动了稳态
% %    - FFTread_Vp/Vn 中取样区间是否正确（是否避开了暂态）
% % -------------------------------------------------------------------------
% 
% clc; clear;
% tic;
% 
% %% ========== 1. 扫频参数 ==========
% Ts      = 5e-6;      %#ok<NASGU>  % 如果模型中用 workspace 变量 Ts，这里保留
% t_inj   = 7.5;       %#ok<NASGU>  % 如果模型中用 workspace 变量 t_inj，这里保留
% 
% f_begin = 1e0;
% f_end   = 1e3;
% Point   = 40;
% 
% % 注意：floor(logspace()) 会造成重复频点（尤其在低频段）
% % 【调试建议】如果发现某些频点重复导致覆盖数据，可改用 unique()
% f_list = floor(logspace(log10(f_begin), log10(f_end), Point));
% f_list = unique(f_list, 'stable');
% Nf = numel(f_list);
% 
% fprintf('Total frequency points = %d\n', Nf);
% 
% %% ========== 2. 注入幅值策略 ==========
% % 你原来的逻辑：funp=50 或 funn=50，然后 A=0.01~0.09 * fun
% % 逐频点注入时建议：用一个固定幅值（或随频率变化），更容易对比各点结果
% % 【建议】先用较小幅值（例如 0.5 ~ 2），确认不会破坏稳态，再逐步调大
% injAmp = 30;
% 
% % 如果你仍想保留"多档幅值"来提高精度，可以扩展为数组多次仿真
% % injAmpList = [0.5 1.0 2.0];
% 
% %% ========== 3. 结果预分配 ==========
% % Z/Y: 2x2xNf
% Zdq = complex(zeros(2,2,Nf));
% Ydq = complex(zeros(2,2,Nf));
% 
% % GM/PM：按分量分别存（你原代码是逐元素算 GM/PM）
% GMZ = zeros(2,2,Nf);  PMZ = zeros(2,2,Nf);
% GMY = zeros(2,2,Nf);  PMY = zeros(2,2,Nf);
% 
% %% ========== 4. 主循环：逐频点注入 ==========
% modelName = 'GFL_inverter.slx';
% 
% for k = 1:Nf
%     f = f_list(k);
% 
%     % ---------------- 正序注入 ----------------
%     % 【关键】把"注入频率/幅值"写到 workspace，供模型读取
%     % 你原代码是 hp1~hp5 + Ap1~Ap5，这里改为单点：hp1 + Ap1（其余置0）
%     setInjectionSinglePoint(f, injAmp, 'pos');   % 正序注入
%     sim(modelName);
% 
%     % 读取正序注入下的频域量（由 FFTread_Vp 写入 workspace）
%     % 【调试建议】如果 FFTread_Vp 依赖变量 n=fc0+1，这里也给它
%     n = f + 1; %#ok<NASGU>
%     FFTread_Vp;
% 
%     % ---------------- 负序注入 ----------------
%     setInjectionSinglePoint(f, injAmp, 'neg');   % 负序注入
%     sim(modelName);
% 
%     n = f + 1; %#ok<NASGU>
%     FFTread_Vn;
% 
%     % ---------------- 计算 Z 和 Y ----------------
%     % 这里假设 FFTread_Vp/Vn 最终给出：
%     % vdjp(k)/vdjn(k)/vqjp(k)/vqjn(k)/idjp(k)/idjn(k)/iqjp(k)/iqjn(k)
%     %
%     % 逐频点注入的"更干净"做法：FFTread_* 直接输出标量 vdjp,vdjn,...
%     % 如果你现在仍是数组形式，这里兼容两种写法：
%     [vdqj, idqj] = buildVIdqForThisPoint(k);
% 
%     % Z = -V / I（你的原定义）
%     Zdq(:,:,k) = -vdqj / idqj;
% 
%     % Y = inv(Z)
%     % 【调试建议】如果 Z 在某些频点接近奇异，inv() 会放大噪声
%     % 可改用 Y = pinv(Z) 或做条件数判断
%     Ydq(:,:,k) = inv(Zdq(:,:,k));
% 
%     % GM/PM（逐元素）
%     for r = 1:2
%         for c = 1:2
%             GMZ(r,c,k) = GM(Zdq(r,c,k));
%             PMZ(r,c,k) = PM(Zdq(r,c,k));
%             GMY(r,c,k) = GM(Ydq(r,c,k));
%             PMY(r,c,k) = PM(Ydq(r,c,k));
%         end
%     end
% 
%     fprintf('Done: %d/%d, f=%g Hz\n', k, Nf, f);
% end
% 
% %% ========== 5. 绘图（统一） ==========
% plotGM_PM(f_list, GMZ, PMZ, 'Z');   % 阻抗
% plotGM_PM(f_list, GMY, PMY, 'Y');   % 导纳
% 
% fprintf('Run Time (Second): ');
% toc
% fprintf('***************END***************\n');
% 
% %% ======================= 局部函数 =======================
% function setInjectionSinglePoint(f, A, seqType)
% % setInjectionSinglePoint
% % 把"单频点注入"的参数写入 base workspace，供 Simulink 模型读取
% %
% % 【你需要在模型中配合修改】
% % - 模型应读取 hp1,Ap1,An1 等变量（或你也可以改成更清晰的变量名）
% % - 这里选择只启用第1路：hp1 + Ap1/An1，其他路全部清零
% 
%     % 全部先清零（保证模型不会残留上一次的注入）
%     hp1 = f; hp2=0; hp3=0; hp4=0; hp5=0; %#ok<NASGU>
%     Ap1 = 0; Ap2=0; Ap3=0; Ap4=0; Ap5=0; %#ok<NASGU>
%     An1 = 0; An2=0; An3=0; An4=0; An5=0; %#ok<NASGU>
% 
%     switch lower(seqType)
%         case 'pos'
%             Ap1 = A; %#ok<NASGU>
%         case 'neg'
%             An1 = A; %#ok<NASGU>
%         otherwise
%             error('seqType must be "pos" or "neg".');
%     end
% 
%     % 写入 base workspace（Simulink 通常从 base workspace 取参）
%     assignin('base','hp1',hp1); assignin('base','hp2',hp2);
%     assignin('base','hp3',hp3); assignin('base','hp4',hp4); assignin('base','hp5',hp5);
% 
%     assignin('base','Ap1',Ap1); assignin('base','Ap2',Ap2);
%     assignin('base','Ap3',Ap3); assignin('base','Ap4',Ap4); assignin('base','Ap5',Ap5);
% 
%     assignin('base','An1',An1); assignin('base','An2',An2);
%     assignin('base','An3',An3); assignin('base','An4',An4); assignin('base','An5',An5);
% end
% 
% function [vdqj, idqj] = buildVIdqForThisPoint(k)
% % buildVIdqForThisPoint
% % 从 workspace 组装当前频点的 vdqj/idqj
% %
% % 兼容两种情况：
% % 1) FFTread_* 生成的是数组：vdjp(k), vdjn(k)...
% % 2) FFTread_* 生成的是标量：vdjp, vdjn...
% %
% % 【调试建议】
% % - 如果你能修改 FFTread_Vp/Vn，让它输出标量（对应当前频点），
% %   那么这里会更简单，也更不容易"索引错位"。
% 
%     vars = evalin('base','whos');
% 
%     hasArrayForm = any(strcmp({vars.name}, 'vdjp')) && ~isscalar(evalin('base','vdjp'));
% 
%     if hasArrayForm
%         vdjp = evalin('base','vdjp'); vdjn = evalin('base','vdjn');
%         vqjp = evalin('base','vqjp'); vqjn = evalin('base','vqjn');
%         idjp = evalin('base','idjp'); idjn = evalin('base','idjn');
%         iqjp = evalin('base','iqjp'); iqjn = evalin('base','iqjn');
% 
%         vdqj = [vdjp(k) vdjn(k); vqjp(k) vqjn(k)];
%         idqj = [idjp(k) idjn(k); iqjp(k) iqjn(k)];
%     else
%         vdjp = evalin('base','vdjp'); vdjn = evalin('base','vdjn');
%         vqjp = evalin('base','vqjp'); vqjn = evalin('base','vqjn');
%         idjp = evalin('base','idjp'); idjn = evalin('base','idjn');
%         iqjp = evalin('base','iqjp'); iqjn = evalin('base','iqjn');
% 
%         vdqj = [vdjp vdjn; vqjp vqjn];
%         idqj = [idjp idjn; iqjp iqjn];
%     end
% end
% 
% function plotGM_PM(f_list, GMm, PMm, tag)
% % plotGM_PM
% % 把 2x2 的 GM/PM 统一绘图（四个分量各一张图，图中含 GM/PM 两子图）
% %
% % tag='Z' 或 'Y' 用来区分阻抗/导纳
% 
%     names = {'dd','dq','qd','qq'};
%     idxMap = {[1 1],[1 2],[2 1],[2 2]};
% 
%     for i = 1:4
%         rc = idxMap{i};
%         r = rc(1); c = rc(2);
% 
%         figure('Name', sprintf('%s_{%s}', tag, names{i}));
%         subplot(2,1,1);
%         semilogx(f_list, squeeze(GMm(r,c,:)), '+', 'linewidth', 1);
%         grid on;
%         title(sprintf('%s_{%s} GM', tag, names{i}));
%         xlabel('Frequency (Hz)'); ylabel('GM');
% 
%         subplot(2,1,2);
%         semilogx(f_list, squeeze(PMm(r,c,:)), '+', 'linewidth', 1);
%         grid on;
%         title(sprintf('%s_{%s} PM', tag, names{i}));
%         xlabel('Frequency (Hz)'); ylabel('PM');
%     end
% end


%% 逆变器 dq 阻抗/导纳扫频（单频点注入版：每次仿真只注入 1 个频率）
clc; clear; tic;

%% =========================
% 1) 基本参数
%% =========================
Ts      = 5e-6;
t_inj   = 7.5;

f_begin = 1e0;
f_end   = 1e3;
Point   = 40;

hp0 = floor(logspace(log10(f_begin), log10(f_end), Point));   % 扫频点
hp0 = unique(hp0, 'stable');
Nf  = length(hp0);
fprintf('Total frequency points = %d\n', Nf);
%% =========================
% 2) 扰动幅值配置（仍沿用你的比例：0.01/0.03/...）
%    单频点注入：只用第一个幅值档（对应 Ap1/An1），其余设为 0
%% =========================
ampScale = [0.01 0.03 0.05 0.07 0.09];
amp1     = ampScale(1);  % 单频注入使用 Ap1/An1 的比例（保持你原方法的一档）

%% =========================
% 3) 预分配存储（整段扫频的结果）
%% =========================
Zdq_all = zeros(2,2,Nf);
Ydq_all = zeros(2,2,Nf);

GM_Zdd = zeros(1,Nf); PM_Zdd = zeros(1,Nf);
GM_Zdq = zeros(1,Nf); PM_Zdq = zeros(1,Nf);
GM_Zqd = zeros(1,Nf); PM_Zqd = zeros(1,Nf);
GM_Zqq = zeros(1,Nf); PM_Zqq = zeros(1,Nf);

GM_Ydd = zeros(1,Nf); PM_Ydd = zeros(1,Nf);
GM_Ydq = zeros(1,Nf); PM_Ydq = zeros(1,Nf);
GM_Yqd = zeros(1,Nf); PM_Yqd = zeros(1,Nf);
GM_Yqq = zeros(1,Nf); PM_Yqq = zeros(1,Nf);

%% =========================
% 4) 主循环：逐频点注入（每次仿真只有 1 个频率）
%% =========================
for k = 1:Nf
    f = hp0(k);          % 当前注入频率
    fc0 = f;             % 保留 fc0 变量（兼容你原流程）
    disp(['Inject f = ', num2str(f), ' Hz']);

    % 为兼容模型/脚本接口：仍定义 hp1~hp5
    % 但只让第一个通道幅值非零，其余通道幅值为 0，实现"单频点注入"
    hp1 = f; hp2 = f; hp3 = f; hp4 = f; hp5 = f;

    %% ---------- 4.1 正序注入：funp=50, funn=0 ----------
    funp = 50;
    funn = 0;

    % 单频点注入：只用 Ap1 / An1，其余幅值为 0
    Ap1 = amp1*funp;  Ap2 = 0; Ap3 = 0; Ap4 = 0; Ap5 = 0;
    An1 = amp1*funn;  An2 = 0; An3 = 0; An4 = 0; An5 = 0;

    sim('GFL_inverter.slx');

    % 保持你原来的 FFT 调用方式（仍使用 n = f + 1）
    for x = 1:length(fc0) %#ok<FORFLG>  % length(fc0)=1
        n = fc0(x) + 1; %#ok<NASGU>
        FFTread_Vp;
    end

    %% ---------- 4.2 负序注入：funp=0, funn=50 ----------
    funp = 0;
    funn = 50;

    Ap1 = amp1*funp;  Ap2 = 0; Ap3 = 0; Ap4 = 0; Ap5 = 0;
    An1 = amp1*funn;  An2 = 0; An3 = 0; An4 = 0; An5 = 0;

    sim('GFL_inverter.slx');

    for x = 1:length(fc0) %#ok<FORFLG>
        n = fc0(x) + 1; %#ok<NASGU>
        FFTread_Vn;
    end

    %% ---------- 4.3 组装矩阵并计算 Zdq / Ydq（单点结果放入 k） ----------
    % 注意：FFTread_* 原先可能输出数组（vdjp(1)、vdjn(1)...）
    % 单频点下我们取 index=1，并存入全局第 k 个频点位置
    vdqj = [vdjp(1) vdjn(1); vqjp(1) vqjn(1)];
    idqj = [idjp(1) idjn(1); iqjp(1) iqjn(1)];

    Zdq = -vdqj / idqj;
    Ydq = inv(Zdq);

    Zdq_all(:,:,k) = Zdq;
    Ydq_all(:,:,k) = Ydq;

    % GM/PM：阻抗
    GM_Zdd(k) = GM(Zdq(1,1));  PM_Zdd(k) = PM(Zdq(1,1));
    GM_Zdq(k) = GM(Zdq(1,2));  PM_Zdq(k) = PM(Zdq(1,2));
    GM_Zqd(k) = GM(Zdq(2,1));  PM_Zqd(k) = PM(Zdq(2,1));
    GM_Zqq(k) = GM(Zdq(2,2));  PM_Zqq(k) = PM(Zdq(2,2));

    % GM/PM：导纳
    GM_Ydd(k) = GM(Ydq(1,1));  PM_Ydd(k) = PM(Ydq(1,1));
    GM_Ydq(k) = GM(Ydq(1,2));  PM_Ydq(k) = PM(Ydq(1,2));
    GM_Yqd(k) = GM(Ydq(2,1));  PM_Yqd(k) = PM(Ydq(2,1));
    GM_Yqq(k) = GM(Ydq(2,2));  PM_Yqq(k) = PM(Ydq(2,2));
end

%% =========================
% 5) 统一绘图（整段扫频）
%% =========================
fAxis = hp0;

% 阻抗：figure(1~4)
Z_GM = {GM_Zdd, GM_Zdq, GM_Zqd, GM_Zqq};
Z_PM = {PM_Zdd, PM_Zdq, PM_Zqd, PM_Zqq};

for fig = 1:4
    figure(fig);
    subplot(2,1,1); semilogx(fAxis, Z_GM{fig}, 'r+','linewidth',1); grid on;
    subplot(2,1,2); semilogx(fAxis, Z_PM{fig}, 'r+','linewidth',1); grid on;
end

% 导纳：figure(5~8)
Y_GM = {GM_Ydd, GM_Ydq, GM_Yqd, GM_Yqq};
Y_PM = {PM_Ydd, PM_Ydq, PM_Yqd, PM_Yqq};
yName = {'Y_{dd}','Y_{dq}','Y_{qd}','Y_{qq}'};

for i = 1:4
    figure(4+i);
    subplot(2,1,1); semilogx(fAxis, Y_GM{i}, 'b+','linewidth',1); grid on;
    title([yName{i},' 幅频特性']);
    subplot(2,1,2); semilogx(fAxis, Y_PM{i}, 'b+','linewidth',1); grid on;
    title([yName{i},' 相频特性']);
end

fprintf('Run Time（Second）：');
toc
fprintf('***************END***************\n');
