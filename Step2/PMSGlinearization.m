tic

Path_root_Results = 'your_root_small';

if ~exist(Path_root_Results, 'dir')
    mkdir(Path_root_Results);
    disp(['Folder ', Path_root_Results, ' created.']);
else
    disp(['Folder ', char(Path_root_Results), ' already exists.']);
end

% Get list of .mat files
models_dir = 'matlab_models';
if ~exist(models_dir, 'dir')
    error('Cannot find directory %s. Please run generate_transfer_function_for_matlab.py first.', models_dir);
end

file_list = dir(fullfile(models_dir, '*.mat'));
if isempty(file_list)
    error('No .mat files found in %s.', models_dir);
end

% Initialize results struct
linearization_results = struct('case_name', {}, 'sys', {}, 'sys_y11', {}, 'pred_model', {});

% Loop through each file
for k = 1:length(file_list)
    filename = file_list(k).name;
    full_path = fullfile(models_dir, filename);
    
    % Load data
    data = load(full_path);
    if ~isfield(data, 'admittance_model')
        warning('File %s does not contain admittance_model. Skipping.', filename);
        continue;
    end
    
    this_case = data.admittance_model;
    
    % Extract operating point
    P = this_case.P;
    Q = this_case.Q;
    V = this_case.V;
    xi = this_case.xi; % Already in radians from Python script
    case_name = this_case.case_name;
    
    fprintf('Processing Case %d/%d: %s (P=%.3f, Q=%.3f, V=%.3f, xi=%.3f rad)\\n', ...
            k, length(file_list), case_name, P, Q, V, xi);

f    = 60;
Fnom    = f;
Np  = 48;
w_g = 2*pi*f;
Sbase = 2e6;
Vbase = 690;
Ibase = Sbase/(Vbase);
wbase = w_g;
Zbase = Vbase^2/Sbase;
Lbase = Zbase/w_g;%��л�ֵ
Cbase = 1/(wbase*Zbase);%���ݻ�ֵ
Tbase = Sbase/(w_g/Np);
U_dcref = 1150;
ugq_ref = 0;

%% ---- ������ѹ�Ħ�-������ϵ��ʾ�����ʲ�������v_alpha��v_betaȡֵ��ͬ������ϵ��������p,q�����У� ----
v_ab    = V * exp(1i*xi);
v_alpha = real(v_ab);  % ��ѹ�������
v_beta  = imag(v_ab);  % ��ѹ�������

%% ---- ��p-q�������i_alpha��i_beta���ȹ��� => ����3/2ϵ���� ----
% p = v_alpha*i_alpha + v_beta*i_beta
% q = v_beta*i_alpha  - v_alpha*i_beta

v2 = v_alpha^2 + v_beta^2;   % ����V

if v2 < 1e-12
    error('��ѹ��ֵ��С���޷���������ο�ֵ��');
end

% ��ӳ�乫ʽ���ȹ��ʣ���
% [i_alpha; i_beta] = 1/v2 * [ v_alpha  v_beta;
%                              v_beta  -v_alpha] * [P; Q]
i_vec   = 1 / v2 * [ v_alpha,  v_beta;
                     v_beta,  -v_alpha ] * [P; Q];

i_alpha = i_vec(1);
i_beta  = i_vec(2);
i_ab    = i_alpha + 1i*i_beta;

R_g = 0.05;
L_g = 7e-3;
C_dc = 1e-3;

Psi_f = 3.88889;%ac           % ת�Ӵ��� 
L_sd = 1.8e-3;%ac         % d���� 
L_sq = 1.8e-3;%ac         % q���� 
R_s = 0.0026;%ac          % ���ӵ���
beta = 0;                 % ����ǣ���λ���ȣ��̶�ֵ
pitch = beta;
J = 35000;%ac
D_m = 0.078;%ac           % ������ϵ��
R_t = 36.6;
% rho = 1.12;
v_w = 12;

Kp_Id_stator = 1;
Ki_Id_stator = 12;
Kp_Iq_stator = 1;
Ki_Iq_stator = 12;
Kp_Speed = 100;
Ki_Speed = 220;
Kp_Udc = 1;
Ki_Udc = 10;
Kp_Id_grid = 1;
Ki_Id_grid = 15;
Kp_Iq_grid = 1;
Ki_Iq_grid = 15;

Kp_PLL = 250;
Ki_PLL = 3200;
T_d = 1/6000;

syms PLL_int thetapll
%% ========== ����任��GSC�ĵ�ѹ�۲⣩ ==========
% PLL
theta_g = thetapll;         % ����PLL���Ƶ�ͬ����
% ����任���󣬽�d-q����ת��Ϊ��������ϵ�����ڲ�������ƣ�
Tg_alphabeta_dq = [cos(theta_g) sin(theta_g); -sin(theta_g) cos(theta_g)];
vg_dq_real = Vbase * Tg_alphabeta_dq * [v_alpha; v_beta];  % GSC�е�ʵ�ʵ�ѹ
% �任������ϵĵ�ѹֵ��ȡ
v_g_d = vg_dq_real(1);          % ����d���ѹ
v_g_q = vg_dq_real(2);         % ����q���ѹ

%% ========== xi��� ==========
eqnq = Ki_PLL*(v_g_q/Vbase - ugq_ref) == 0;
[pllg] = solve(eqnq, thetapll);
% === �ӽ���ѡȡ�� xi ��ӽ����Ǹ�������ֻȡʵ���� ===
pllg_v   = vpa(pllg);             % �����ֵ����/ת�ɿɱȽ���ʽ
pllg_r   = real(pllg_v);          % ���и����⣬ֻȡʵ��
xi_v = vpa(xi);
[~, idx] = min(abs(pllg_r - xi_v));
thetapll = pllg_r(idx);
disp(thetapll)
PLL_int = 0;

% v_g_d = round(eval(v_g_d));
% v_g_q = round(eval(v_g_q));

syms omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int...
    u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int;
syms omega_best i_gqref

Tg_dq_alphabeta = [cos(theta_g) -sin(theta_g); sin(theta_g) cos(theta_g)];
ig_dq_alphabeta = (1/Ibase)*Tg_dq_alphabeta*[i_gd;i_gq];

i_galpha = ig_dq_alphabeta(1);
i_gbeta = ig_dq_alphabeta(2);

n_ref = omega_best*30/pi;
ugq_ref = 0;

n = omega_m * 30/pi;
omega_e = omega_m*Np;

i_sd = (Psi_sd - Psi_f)/L_sd;
i_sq = Psi_sq/L_sq;

Te = Np*(Psi_f*i_sq);

u1 = 0.5*pi;
u2 = 1.225;
lambda = (R_t*omega_m)/(v_w);
lambda_i = 1/(1/(lambda+0.08*beta)-0.035/(beta^3+1));
Cp = 0.51763*(116/lambda_i-0.4*beta-5)*exp(-21/lambda_i)+0.006795*lambda;
P_m = u1*u2*(R_t^2)*(v_w^3)*Cp;
T_m = (-1)*(P_m)/omega_m;

% dy1 = n_ref - n;
% dx1 = i_sdref - i_sd;
% dx2 = i_sqref - i_sq;
i_sdref = 0;
i_sqref = Kp_Speed*(n_ref - n) + Ki_Speed*Speed_int;
u_sqref = Kp_Iq_stator*(i_sqref - i_sq)+Ki_Iq_stator*Iq_stator_int+omega_e*Psi_f+omega_e*L_sd*i_sd;
u_sdref = Kp_Id_stator*(i_sdref - i_sd)+Ki_Id_stator*Id_stator_int-omega_e*L_sq*i_sq;

P_s = (u_sd * i_sd + u_sq * i_sq);%(2-17)
Q_s = (u_sq * i_sd - u_sd * i_sq);%(2-18)
P_e = Te * omega_m;

i_gdref = Kp_Udc*(U_dc - U_dcref) + Ki_Udc*Udc_int;

u_gd = - Kp_Id_grid*(i_gdref - i_gd) - Ki_Id_grid*Id_grid_int - R_g*i_gd + v_g_d+i_gq*w_g*L_g;
u_gq = - Kp_Iq_grid*(i_gqref - i_gq) - Ki_Iq_grid*Iq_grid_int - R_g*i_gq + v_g_q-i_gd*w_g*L_g;

P_dc = (u_gd* i_gd + u_gq * i_gq);%(2-17)
P_g = (v_g_d * i_gd + v_g_q * i_gq);%(2-17)
Q_g = (v_g_q * i_gd - v_g_d * i_gq);%(2-18)

eqn1 = (Te - T_m - D_m*omega_m) / J == 0;
eqn2 = u_sq - R_s*i_sq - omega_e*Psi_sd == 0;
eqn3 = u_sd - R_s*i_sd + omega_e*Psi_sq == 0;
eqn4 = i_sdref - i_sd == 0;
eqn5 = i_sqref - i_sq == 0;
eqn6 = n_ref - n == 0;
eqn7 = (u_sdref - u_sd)/T_d == 0;
eqn8 = (u_sqref - u_sq)/T_d == 0;
eqn9 = (P_s - P_dc)/(C_dc*U_dc) == 0;
eqn10 = U_dc - U_dcref == 0;
eqn11 = (v_g_d - u_gd - R_g*i_gd + w_g*L_g*i_gq)/L_g == 0;
eqn12 = (v_g_q - u_gq - R_g*i_gq - w_g*L_g*i_gd)/L_g == 0;
eqn13 = i_gdref - i_gd == 0;
eqn14 = i_gqref - i_gq == 0;
eqn15 = i_galpha == i_alpha;
eqn16 = i_gbeta == i_beta;


[omega_m, Psi_sq, Psi_sd, Id_stator_int, Iq_stator_int, Speed_int,...
    u_sd, u_sq, U_dc, Udc_int, i_gd, i_gq, Id_grid_int, Iq_grid_int,...
    omega_best, i_gqref]...
                      = vpasolve(eval([eqn1,eqn2,eqn3,eqn4,eqn5,eqn6, ...
                      eqn7,eqn8,eqn9,eqn10,eqn11,eqn12,eqn13,eqn14, ...
                      eqn15,eqn16]),...
                      [omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int ...
                      u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int ...
                      omega_best i_gqref],...
                      [-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf; ...
                      -inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf;-inf,inf; ...
                      -inf,inf;-inf,inf]);

inistate = [omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int...
    u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int PLL_int thetapll]';

omega_best = getRealIfSmallImag(omega_best);
i_gqref = getRealIfSmallImag(i_gqref);

state_size = size(inistate,1);

inistate2 = [omega_m Psi_sq Psi_sd Id_stator_int Iq_stator_int Speed_int...
    u_sd u_sq U_dc Udc_int i_gd i_gq Id_grid_int Iq_grid_int PLL_int thetapll]';


x_ss = inistate2;  % ��̬״̬
u_ss = [v_alpha, v_beta, omega_best, i_gqref];           % ��̬����
% y_ss = yy;

epsilon = 1e-7;
n = length(x_ss);  % ״̬ά��
m = length(u_ss);  % ����ά��
p = 2;  % ���ά��

% ��ʼ������
A = zeros(n, n);
B = zeros(n, m);
C = zeros(p, n);
D = zeros(p, m);

[dxdt0, y0] = PMSG_fun(0, x_ss, u_ss);

for i = 1:n
    hi = 1e-6 * max(1, abs(x_ss(i)));
    xp = x_ss; xm = x_ss;
    xp(i) = xp(i) + hi;
    xm(i) = xm(i) - hi;

    [fp, yp] = PMSG_fun(0, xp, u_ss);
    [fm, ym] = PMSG_fun(0, xm, u_ss);

    A(:,i) = (fp - fm)/(2*hi);
    C(:,i) = (yp - ym)/(2*hi);
end

for j = 1:m
    hj = 1e-6 * max(1, abs(u_ss(j)));
    up = u_ss; um = u_ss;
    up(j) = up(j) + hj;
    um(j) = um(j) - hj;

    [fp, yp] = PMSG_fun(0, x_ss, up);
    [fm, ym] = PMSG_fun(0, x_ss, um);

    B(:,j) = (fp - fm)/(2*hj);
    D(:,j) = (yp - ym)/(2*hj);
end

sys = ss(A,B,C,D);

% --- Calculate Y11 (Admittance) ---
    sys_y11 = sys(1,1); 
    
    % Save to struct
    linearization_results(k).case_name = case_name;
    linearization_results(k).sys = sys;
    linearization_results(k).sys_y11 = sys_y11;
    linearization_results(k).pred_model = this_case; % Also save the loaded prediction model for comparison later
    
    disp(['Linearization for ', case_name, ' complete.']);
end % End of loop

disp('====================================================')
disp('All cases processed.')
disp('Results stored in linearization_results variable.')

% Save all results to a single .mat file in the output directory
save_path = fullfile(Path_root_Results, 'all_linearization_results.mat');
save(save_path, 'linearization_results');
fprintf('Results saved to: %s\n', save_path);
toc


