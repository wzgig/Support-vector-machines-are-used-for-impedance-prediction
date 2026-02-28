
%%%%%%%%%%%%%%%%%%%%  vd   %%%%%%%%%%%%


FFTDATAa = power_fftscope(vdq); %% 生成FFTDATA结构体
FFTDATAa.input = 1; %% 输入变量数
FFTDATAa.signal = 1; %% 通道选择
FFTDATAa.startTime = 1; %% 起始时刻
FFTDATAa.cycles = 50; %% 参与FFT的周期数
FFTDATAa.fundamental = 50; %% 基波周期
FFTDATAa.maxFrequency = 100000; %% FFT最大频率
FFTDATAa_update = power_fftscope(FFTDATAa);

vd2M=FFTDATAa_update.mag(n);
vd2P=FFTDATAa_update.phase(n);
vdjp(x)=vd2M*exp(1i*vd2P*pi/180);

%%%%%%%%%%%%%%%%%%%%  vq   %%%%%%%%%%%%
FFTDATAa = power_fftscope(vdq); %% 生成FFTDATA结构体
FFTDATAa.input = 2; %% 输入变量数
FFTDATAa.signal = 1; %% 通道选择
FFTDATAa.startTime = 1; %% 起始时刻
FFTDATAa.cycles = 50; %% 参与FFT的周期数
FFTDATAa.fundamental = 50; %% 基波周期
FFTDATAa.maxFrequency = 100000; %% FFT最大频率
FFTDATAa_update = power_fftscope(FFTDATAa);

vq2M=FFTDATAa_update.mag(n);
vq2P=FFTDATAa_update.phase(n);
vqjp(x)=vq2M*exp(1i*vq2P*pi/180);





%***********************************************************
%%%%%%%%%%%%%%%%%%%%  id   %%%%%%%%%%%%
FFTDATAa = power_fftscope(idq); %% 生成FFTDATA结构体
FFTDATAa.input = 1; %% 输入变量数
FFTDATAa.signal = 1; %% 通道选择
FFTDATAa.startTime = 1; %% 起始时刻
FFTDATAa.cycles = 50; %% 参与FFT的周期数
FFTDATAa.fundamental = 50; %% 基波周期
FFTDATAa.maxFrequency = 100000; %% FFT最大频率
FFTDATAa_update = power_fftscope(FFTDATAa);

id2M=FFTDATAa_update.mag(n);
id2P=FFTDATAa_update.phase(n);
idjp(x)=id2M*exp(1i*id2P*pi/180);

%%%%%%%%%%%%%%%%%%%%  iq   %%%%%%%%%%%%

FFTDATAa = power_fftscope(idq); %% 生成FFTDATA结构体
FFTDATAa.input = 2; %% 输入变量数
FFTDATAa.signal = 1; %% 通道选择
FFTDATAa.startTime = 1; %% 起始时刻
FFTDATAa.cycles = 50; %% 参与FFT的周期数
FFTDATAa.fundamental = 50; %% 基波周期
FFTDATAa.maxFrequency = 100000; %% FFT最大频率
FFTDATAa_update = power_fftscope(FFTDATAa);

iq2M=FFTDATAa_update.mag(n);
iq2P=FFTDATAa_update.phase(n);
iqjp(x)=iq2M*exp(1i*iq2P*pi/180);


