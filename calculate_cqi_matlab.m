%% CQI计算和可视化脚本 (MATLAB版本)
% 功能：从合并后的CSI矩阵计算CQI，并生成各种可视化图表
% 参考：DeepMIMO-5GNR项目
% 
% 使用方法：
%   1. 确保已安装必要的工具箱（Statistics and Machine Learning Toolbox用于t-SNE）
%   2. 修改下面的参数配置部分
%   3. 运行脚本: calculate_cqi_matlab

clear all
close all
clc

%% ==================== 参数配置 ====================
% 输入文件路径（合并后的CSI文件，.mat格式）
% 如果.mat文件已包含场景信息，则info_file和env_image_file可以为空
input_csi_file = 'C:\Users\GreyGoo\Desktop\Sem\data\01450.mat';

% 位置信息文件路径（Info.npy，可选，如果.mat文件已包含则不需要）
info_file='';
% 环境图像文件路径（environment.png，可选，如果.mat文件已包含则不需要）
env_image_file = '';  % 留空表示从.mat文件中读取

% 输出文件名前缀（不含扩展名）
output_prefix = 'losangeles_adCSI_right_01450';  % 例如: scenario2_00032_BS0

% 系统参数
Nt = [1, 8, 4];  % BS天线配置 [x, y, z]
Nr = [1, 1, 1];  % UE天线配置 [x, y, z]
Nt_total = Nt(1) * Nt(2) * Nt(3);  % 发射天线总数
Nr_total = Nr(1) * Nr(2) * Nr(3);  % 接收天线总数
sampledCarriers = 52;  % 采样子载波数
BWGHz = 0.00936;  % 带宽 (GHz)
subcarrier_spacing = 15e3;  % 子载波间隔 (Hz)

% 噪声和功率参数
k_B = 1.38e-23;  % 玻尔兹曼常数 (J/K)
T = 290;  % 系统温度 (K)
noise_figure = 5;  % 噪声系数 (dB)
total_power_dBm = 0;  % 总发射功率 (dBm)

% 子带参数
subband_size = 4;  % 每个子带包含的子载波数

% 3GPP标准 SINR-to-CQI 映射表 (dB)
SINR_to_CQI_mapping = [-6.3, -5.8, -1.4, 3.9, 5.3, 8.1, 9.8, ...
    11.7, 13.6, 15.8, 18.8, 21.4, 23.6, 28.2, 32.0];

% 可视化选项
enable_plots = true;  % 是否生成图表
enable_tsne = true;  % 是否生成t-SNE图（需要Statistics and Machine Learning Toolbox）
enable_csi_visualization = true;  % 是否可视化CSI矩阵
num_sample_users_for_plot = 2;  % 用于CSI可视化的示例用户数量

%% ==================== 读取数据 ====================
fprintf('正在读取CSI数据...\n');

% 检查文件是否存在
if ~exist(input_csi_file, 'file')
    error('CSI文件不存在: %s\n请先运行com.py生成合并文件，并使用convert_npy_to_mat.py转换为.mat格式', input_csi_file);
end

% 读取.mat文件
[~, ~, ext] = fileparts(input_csi_file);
if strcmpi(ext, '.mat')
    % 直接读取.mat文件
    fprintf('读取.mat文件: %s\n', input_csi_file);
    loaded_data = load(input_csi_file);
    
    % 查找CSI数据变量（可能是csi_data或其他名称）
    var_names = fieldnames(loaded_data);
    csi_var_found = false;
    for i = 1:length(var_names)
        var_name = var_names{i};
        var_data = loaded_data.(var_name);
        if isnumeric(var_data) && length(size(var_data)) == 4 && size(var_data, 1) == 2
            csi_data = var_data;
            csi_var_name = var_name;
            csi_var_found = true;
            fprintf('找到CSI数据变量: %s\n', var_name);
            break;
        end
    end
    
    if ~csi_var_found
        error('无法在.mat文件中找到CSI数据（期望4维数组，第1维为2）');
    end
    
    % 检查是否包含场景信息
    if isfield(loaded_data, 'scenario_info')
        scenario_info_from_file = loaded_data.scenario_info;
        fprintf('✓ 检测到.mat文件中包含场景信息\n');
        
        % 从场景信息中提取位置信息
        if isfield(scenario_info_from_file, 'img_size')
            img_size = scenario_info_from_file.img_size(:)';  % 确保是行向量
        end
        if isfield(scenario_info_from_file, 'bs_loc')
            bs_loc = scenario_info_from_file.bs_loc(:)';  % 确保是行向量
        end
        if isfield(scenario_info_from_file, 'ue_locs')
            ue_locs = scenario_info_from_file.ue_locs;
        end
        if isfield(scenario_info_from_file, 'environment_image')
            env_image_array = scenario_info_from_file.environment_image;
            fprintf('✓ 检测到环境图像数据\n');
        end
        
        % 提取其他元数据
        if isfield(scenario_info_from_file, 'env_id')
            env_id = scenario_info_from_file.env_id;
        end
        if isfield(scenario_info_from_file, 'carrier_freq')
            carrier_freq = scenario_info_from_file.carrier_freq;
        end
        if isfield(scenario_info_from_file, 'scenario')
            scenario = scenario_info_from_file.scenario;
        end
    else
        fprintf('提示: .mat文件中未包含场景信息，将尝试从外部文件读取\n');
    end
    
else
    % 尝试读取.npy文件（如果安装了npy-matlab工具）
    try
        csi_data = readNPY(input_csi_file);
        fprintf('成功读取.npy文件\n');
    catch
        error('请使用convert_npy_to_mat.py将.npy文件转换为.mat格式');
    end
end

% 检查维度
if length(size(csi_data)) ~= 4 || size(csi_data, 1) ~= 2
    error('CSI数据维度不正确。期望: (2, num_users, Nt*Nr, sampledCarriers)，实际: %s', mat2str(size(csi_data)));
end

num_users = size(csi_data, 2);
antenna_pairs = size(csi_data, 3);
num_subcarriers = size(csi_data, 4);

fprintf('CSI数据维度: %s\n', mat2str(size(csi_data)));
fprintf('用户数量: %d\n', num_users);
fprintf('天线对数量: %d (期望: %d)\n', antenna_pairs, Nt_total * Nr_total);
fprintf('子载波数: %d (期望: %d)\n', num_subcarriers, sampledCarriers);

% 重构复数CSI矩阵
% 形状: (num_users, Nt*Nr, sampledCarriers)
H_complex = csi_data(1, :, :, :) + 1j * csi_data(2, :, :, :);
H_complex = squeeze(H_complex);  % 移除单例维度
% 重新排列为: (Nt*Nr, sampledCarriers, num_users)
H_complex = permute(H_complex, [2, 3, 1]);

fprintf('复数CSI矩阵维度: %s\n', mat2str(size(H_complex)));

%% ==================== 读取位置信息 ====================
fprintf('\n正在读取位置信息...\n');

% 初始化变量（如果.mat文件中没有场景信息，则使用默认值）
if ~exist('bs_loc', 'var')
    bs_loc = [0, 0];
end
if ~exist('ue_locs', 'var')
    ue_locs = zeros(num_users, 2);
end
if ~exist('img_size', 'var')
    img_size = [0, 0];
end
if ~exist('env_image_array', 'var')
    env_image_array = [];
end

% 如果.mat文件中没有场景信息，尝试从外部文件读取
if isempty(info_file) || ~exist('scenario_info_from_file', 'var')
    % 尝试从外部文件读取
    if ~isempty(info_file) && exist(info_file, 'file')
    try
        % 尝试读取.npy文件
        if endsWith(info_file, '.npy')
            try
                info_data = readNPY(info_file);
                fprintf('成功读取.npy位置信息文件\n');
            catch
                % 如果readNPY不可用，尝试转换为.mat后读取
                mat_file = strrep(info_file, '.npy', '.mat');
                if exist(mat_file, 'file')
                    load(mat_file);
                    if exist('Info', 'var')
                        info_data = Info;
                    elseif exist('info_data', 'var')
                        % 变量名已经是info_data
                    else
                        error('无法识别.mat文件中的变量名');
                    end
                    fprintf('成功读取.mat位置信息文件\n');
                else
                    error('无法读取.npy文件，且对应的.mat文件不存在');
                end
            end
        else
            % 直接读取.mat文件
            load(info_file);
            if exist('Info', 'var')
                info_data = Info;
            elseif exist('info_data', 'var')
                % 变量名已经是info_data
            else
                error('无法识别.mat文件中的变量名');
            end
            fprintf('成功读取位置信息文件\n');
        end
        
        % 解析位置信息（根据scenario_2的格式）
        % scenario_2格式: 前2个是图像尺寸，接下来是BS坐标(2个)，然后是UE坐标(10000*2)
        if length(info_data) >= 4
            img_size = floor(info_data(1:2));
            bs_loc = info_data(3:4);  % BS位置 (x, y)
            
            % 提取UE位置
            if length(info_data) > 4
                ue_data = info_data(5:end);
                num_ue_in_file = length(ue_data) / 2;
                ue_locs_file = reshape(ue_data, [], 2);  % UE位置 (num_ue_in_file, 2)
                
                % 如果UE数量不匹配，截断或填充
                if size(ue_locs_file, 1) > num_users
                    ue_locs = ue_locs_file(1:num_users, :);
                    fprintf('警告: 位置信息中的UE数量(%d)多于CSI数据中的用户数量(%d)，已截断\n', ...
                        size(ue_locs_file, 1), num_users);
                elseif size(ue_locs_file, 1) < num_users
                    ue_locs(1:size(ue_locs_file, 1), :) = ue_locs_file;
                    fprintf('警告: 位置信息中的UE数量(%d)少于CSI数据中的用户数量(%d)，已用零填充\n', ...
                        size(ue_locs_file, 1), num_users);
                else
                    ue_locs = ue_locs_file;
                end
            end
            
            fprintf('图像尺寸: [%.0f, %.0f]\n', img_size(1), img_size(2));
            fprintf('BS位置: (%.2f, %.2f)\n', bs_loc(1), bs_loc(2));
            if any(ue_locs(:) ~= 0)
                fprintf('UE位置范围: X[%.2f, %.2f], Y[%.2f, %.2f]\n', ...
                    min(ue_locs(:,1)), max(ue_locs(:,1)), ...
                    min(ue_locs(:,2)), max(ue_locs(:,2)));
            else
                fprintf('警告: UE位置信息全为零\n');
            end
        else
            warning('位置信息文件格式不正确，将使用默认值');
        end
        
    catch ME
        warning('读取位置信息失败: %s\n将使用默认值', ME.message);
        bs_loc = [0, 0];
        ue_locs = zeros(num_users, 2);
    end
    else
        if ~isempty(info_file)
            warning('位置信息文件不存在: %s\n将使用默认值', info_file);
        else
            fprintf('提示: 未指定位置信息文件，使用.mat文件中的信息或默认值\n');
        end
    end
else
    fprintf('使用.mat文件中的位置信息\n');
end

% 读取环境图像（如果.mat文件中没有）
if isempty(env_image_array) && ~isempty(env_image_file) && exist(env_image_file, 'file')
    fprintf('从外部文件读取环境图像: %s\n', env_image_file);
    try
        env_image_array = imread(env_image_file);
        fprintf('成功读取环境图像，尺寸: %s\n', mat2str(size(env_image_array)));
    catch ME
        warning('无法读取环境图像: %s', ME.message);
        env_image_array = [];
    end
elseif ~isempty(env_image_array)
    fprintf('使用.mat文件中的环境图像，尺寸: %s\n', mat2str(size(env_image_array)));
else
    fprintf('提示: 未找到环境图像，2D位置图将不显示背景\n');
end

%% ==================== 计算噪声功率 ====================
fprintf('\n计算系统参数...\n');

% 计算有效带宽
bandwidth_effective = num_subcarriers * 12 * subcarrier_spacing;  % 假设1个RB=12个子载波

% 计算噪声功率
N0 = k_B * T;  % 热噪声密度 (W/Hz)
N0_dB = 10 * log10(N0) + 30;  % 转换为 dBm/Hz
noise_power_dBm = N0_dB + 10 * log10(bandwidth_effective) + noise_figure;
noise_power = 10^((noise_power_dBm - 30) / 10);  % 转换为线性单位 (W)

% 计算每个子载波的发射功率
total_power_linear = 10^((total_power_dBm - 30) / 10);  % 转换为线性单位 (W)
power_per_subcarrier = total_power_linear / num_subcarriers;

fprintf('有效带宽: %.2f MHz\n', bandwidth_effective / 1e6);
fprintf('噪声功率: %.2e W (%.2f dBm)\n', noise_power, noise_power_dBm);
fprintf('每子载波功率: %.2e W (%.2f dBm)\n', power_per_subcarrier, ...
    10*log10(power_per_subcarrier) + 30);

%% ==================== DFT变换（频域到角度-延迟域）====================
fprintf('\n执行DFT变换（频域 -> 角度-延迟域）...\n');

% 保存原始频域H（用于CQI计算，因为CQI基于频域SNR）
H_complex_freq = H_complex;  % (Nt*Nr, sampledCarriers, num_users)

% 初始化DFT后的H（角度-延迟域，用于保存）
H_complex_angle_delay = zeros(size(H_complex));  % (Nt*Nr, sampledCarriers, num_users)

% 生成DFT矩阵
% 对天线维度做DFT：dftmtx(Nt*Nr)
% 对子载波维度做DFT：dftmtx(sampledCarriers)
DFT_antenna = dftmtx(Nt_total);  % (Nt*Nr, Nt*Nr)
DFT_subcarrier = dftmtx(sampledCarriers);  % (sampledCarriers, sampledCarriers)

fprintf('DFT矩阵尺寸: 天线维度 %s, 子载波维度 %s\n', ...
    mat2str(size(DFT_antenna)), mat2str(size(DFT_subcarrier)));

% 对每个用户进行DFT变换
fprintf('处理用户: ');
for user_idx = 1:num_users
    if mod(user_idx, 1000) == 0
        fprintf('%d/%d ', user_idx, num_users);
    end
    
    % 获取该用户的频域H矩阵，形状 (Nt*Nr, sampledCarriers)
    H_user_freq = H_complex_freq(:, :, user_idx);
    
    % 执行DFT变换：H_angle_delay = DFT_antenna * H_freq * DFT_subcarrier
    % 注意：MATLAB中矩阵乘法是列优先，所以顺序正确
    H_user_angle_delay = DFT_antenna * H_user_freq * DFT_subcarrier;
    
    % 保存DFT后的H
    H_complex_angle_delay(:, :, user_idx) = H_user_angle_delay;
end
fprintf('\nDFT变换完成！\n');

fprintf('DFT变换后的H矩阵维度: %s\n', mat2str(size(H_complex_angle_delay)));


%% ==================== CQI计算 ====================
fprintf('\n开始计算CQI...\n');

% 计算子带数量
num_subbands = floor(num_subcarriers / subband_size);

% 初始化结果数组
wideband_avg_SNR_dB = zeros(1, num_users);
wideband_CQI_results = zeros(1, num_users);
subband_avg_SNR_dB = zeros(num_subbands, num_users);
subband_CQI_results = zeros(num_subbands, num_users);

% 对每个用户计算CQI
fprintf('处理用户: ');
for user_idx = 1:num_users
    if mod(user_idx, 1000) == 0
        fprintf('%d/%d ', user_idx, num_users);
    end
    
    % 获取该用户的CSI矩阵（使用DFT后的角度-延迟域H进行CQI计算），形状 (Nt*Nr, sampledCarriers)
    H_user = H_complex_angle_delay(:, :, user_idx);
    
    % ========== Wideband CQI 计算 ==========
    sum_SNR_dB = 0;
    for sc_idx = 1:num_subcarriers
        % 提取该子载波的信道向量
        H_sc = H_user(:, sc_idx);  % 形状 (antenna_pairs,)
        
        % MRC波束赋形
        H_norm = norm(H_sc);
        if H_norm < 1e-10
            SNR_dB = -inf;
        else
            v = H_sc / H_norm;  % 波束赋形向量
            received_power = abs(H_sc' * v)^2 * power_per_subcarrier;
            SNR_linear = received_power / noise_power;
            SNR_dB = 10 * log10(SNR_linear);
        end
        sum_SNR_dB = sum_SNR_dB + SNR_dB;
    end
    
    wideband_avg_SNR_dB(user_idx) = sum_SNR_dB / num_subcarriers;
    % 映射到CQI
    wideband_CQI_results(user_idx) = sum(wideband_avg_SNR_dB(user_idx) > SINR_to_CQI_mapping);
    wideband_CQI_results(user_idx) = min(wideband_CQI_results(user_idx), 15);
    
    % ========== Subband CQI 计算 ==========
    for subband_idx = 1:num_subbands
        start_sc = (subband_idx - 1) * subband_size + 1;
        end_sc = subband_idx * subband_size;
        
        sum_SNR_dB_subband = 0;
        for sc_idx = start_sc:end_sc
            H_sc = H_user(:, sc_idx);
            H_norm = norm(H_sc);
            if H_norm < 1e-10
                SNR_dB = -inf;
            else
                v = H_sc / H_norm;
                received_power = abs(H_sc' * v)^2 * power_per_subcarrier;
                SNR_linear = received_power / noise_power;
                SNR_dB = 10 * log10(SNR_linear);
            end
            sum_SNR_dB_subband = sum_SNR_dB_subband + SNR_dB;
        end
        
        subband_avg_SNR_dB(subband_idx, user_idx) = sum_SNR_dB_subband / subband_size;
        % 映射到CQI
        subband_CQI_results(subband_idx, user_idx) = sum(...
            subband_avg_SNR_dB(subband_idx, user_idx) > SINR_to_CQI_mapping);
        subband_CQI_results(subband_idx, user_idx) = min(...
            subband_CQI_results(subband_idx, user_idx), 15);
    end
end
fprintf('\nCQI计算完成！\n');

% 打印统计信息
fprintf('\nCQI统计信息:\n');
fprintf('  Wideband CQI范围: [%d, %d]\n', min(wideband_CQI_results), max(wideband_CQI_results));
fprintf('  Wideband CQI均值: %.2f\n', mean(wideband_CQI_results));
fprintf('  Wideband SNR范围: [%.2f, %.2f] dB\n', ...
    min(wideband_avg_SNR_dB), max(wideband_avg_SNR_dB));
fprintf('  Wideband SNR均值: %.2f dB\n', mean(wideband_avg_SNR_dB));

%% ==================== CSI矩阵可视化 ====================
if enable_plots && enable_csi_visualization
    fprintf('\n生成CSI矩阵可视化...\n');
    
    % 选择几个示例用户进行可视化
    sample_users = round(linspace(1, num_users, num_sample_users_for_plot));
    
    for plot_idx = 1:length(sample_users)
        user_idx = sample_users(plot_idx);
        H_user = H_complex(:, :, user_idx);
        
        figure('Position', [100 + plot_idx*50, 100 + plot_idx*50, 1000, 400]);
        
        % 幅度
        subplot(1, 2, 1);
        imagesc(abs(H_user));
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('CSI Amplitude - User %d (CQI=%d)', user_idx, wideband_CQI_results(user_idx)));
        colormap(gca, 'jet');
        
        % 相位
        subplot(1, 2, 2);
        imagesc(angle(H_user) / pi * 180);
        colorbar;
        xlabel('Subcarrier Index');
        ylabel('Antenna Pair Index');
        title(sprintf('CSI Phase (degrees) - User %d', user_idx));
        colormap(gca, 'hsv');
        
        sgtitle(sprintf('Channel Matrix Visualization - User %d', user_idx), 'FontSize', 14);
    end
end

%% ==================== t-SNE可视化 ====================
if enable_plots && enable_tsne
    fprintf('\n生成t-SNE可视化...\n');
    
    try
        % 准备数据
        num_features = antenna_pairs * num_subcarriers;
        H_reshaped = reshape(H_complex, [num_features, num_users]).';
        H_real = [real(H_reshaped), imag(H_reshaped)];  % 实部和虚部
        
        % t-SNE降维
        rng(42);  % 固定随机种子
        fprintf('执行t-SNE降维（可能需要一些时间）...\n');
        Y = tsne(H_real);
        
        % 按CQI值着色
        cqi_values = wideband_CQI_results;
        unique_cqi = unique(cqi_values);
        num_colors = length(unique_cqi);
        
        % 生成颜色映射
        if num_colors <= 10
            colors = lines(num_colors);
        else
            colors = jet(num_colors);
        end
        
        % 绘制t-SNE图
        figure('Position', [100, 100, 800, 600]);
        gscatter(Y(:,1), Y(:,2), cqi_values, colors, '.', 10);
        title('t-SNE of Channel Matrix H (Colored by CQI)', 'FontSize', 14);
        xlabel('t-SNE Dimension 1', 'FontSize', 12);
        ylabel('t-SNE Dimension 2', 'FontSize', 12);
        grid on;
        legend('Location', 'bestoutside', 'FontSize', 9);
        
    catch ME
        warning('t-SNE可视化失败: %s\n请确保已安装Statistics and Machine Learning Toolbox', ME.message);
    end
end

%% ==================== CDF图 ====================
if enable_plots
    fprintf('\n生成CDF图...\n');
    
    figure('Position', [100, 100, 800, 600]);
    
    % CQI CDF
    subplot(2, 2, 1);
    cdfplot(wideband_CQI_results);
    xlabel('CQI Value');
    ylabel('CDF');
    title('Wideband CQI CDF');
    grid on;
    xlim([0, 16]);
    
    % SNR CDF
    subplot(2, 2, 2);
    cdfplot(wideband_avg_SNR_dB);
    xlabel('SNR (dB)');
    ylabel('CDF');
    title('Wideband SNR CDF');
    grid on;
    
    % Subband CQI分布（箱线图）
    subplot(2, 2, 3);
    boxplot(subband_CQI_results', 'Labels', 1:num_subbands);
    xlabel('Subband Index');
    ylabel('CQI Value');
    title('Subband CQI Distribution');
    grid on;
    
    % CQI直方图
    subplot(2, 2, 4);
    histogram(wideband_CQI_results, 0:16, 'Normalization', 'probability');
    xlabel('CQI Value');
    ylabel('Probability');
    title('Wideband CQI Histogram');
    grid on;
    xlim([0, 16]);
    
    sgtitle('CQI and SNR Statistics', 'FontSize', 14);
end

%% ==================== 2D平面图（用户位置按CQI着色）====================
if enable_plots && ~isempty(ue_locs) && any(ue_locs(:) ~= 0)
    fprintf('\n生成2D位置图...\n');
    
    figure('Position', [100, 100, 1000, 800]);
    hold on;
    
    % 如果有环境图像，先显示作为背景
    if ~isempty(env_image_array)
        % 显示环境图像作为背景
        % 注意：ue_locs可能是归一化坐标，需要映射到图像坐标
        
        % 检查图像数据格式
        img_dims = size(env_image_array);
        if length(img_dims) == 2
            % 灰度图像 (m×n)
            img_for_display = env_image_array;
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        elseif length(img_dims) == 3 && img_dims(3) == 3
            % RGB图像 (m×n×3)
            img_for_display = env_image_array;
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        elseif length(img_dims) == 3 && img_dims(3) == 4
            % RGBA图像，只取前3个通道
            img_for_display = env_image_array(:, :, 1:3);
            actual_img_height = img_dims(1);
            actual_img_width = img_dims(2);
        else
            warning('环境图像格式不支持，跳过背景显示');
            img_for_display = [];
            actual_img_height = [];
            actual_img_width = [];
        end
        
        if ~isempty(img_for_display)
            % 确定使用哪个尺寸：优先使用环境图像的实际尺寸
            % 同时检查是否需要坐标缩放
            need_coord_scaling = false;
            scale_x = 1.0;
            scale_y = 1.0;
            
            % 确保尺寸变量为双精度
            actual_img_width = double(actual_img_width);
            actual_img_height = double(actual_img_height);
            
            if ~isempty(img_size) && all(img_size > 0)
                % 确保img_size为双精度
                img_size = double(img_size);
                
                % 检查img_size是否与环境图像尺寸匹配
                if abs(img_size(1) - actual_img_width) < 5 && abs(img_size(2) - actual_img_height) < 5
                    % 尺寸匹配，使用img_size
                    display_width = img_size(1);
                    display_height = img_size(2);
                    fprintf('使用Info.npy中的图像尺寸: [%.0f, %.0f]\n', display_width, display_height);
                else
                    % 尺寸不匹配，需要坐标缩放
                    display_width = actual_img_width;
                    display_height = actual_img_height;
                    need_coord_scaling = true;
                    scale_x = double(actual_img_width) / double(img_size(1));
                    scale_y = double(actual_img_height) / double(img_size(2));
                    fprintf('检测到尺寸不匹配:\n');
                    fprintf('  Info.npy尺寸: [%.0f, %.0f]\n', img_size(1), img_size(2));
                    fprintf('  实际图像尺寸: [%.0f, %.0f]\n', actual_img_width, actual_img_height);
                    fprintf('  坐标缩放比例: X=%.4f, Y=%.4f\n', scale_x, scale_y);
                end
            else
                % 没有img_size，使用环境图像的实际尺寸
                display_width = actual_img_width;
                display_height = actual_img_height;
                fprintf('使用环境图像实际尺寸: [%.0f, %.0f]\n', display_width, display_height);
            end
            
            % 判断坐标类型并映射
            % 确保坐标数组为双精度类型
            ue_locs = double(ue_locs);
            bs_loc = double(bs_loc);
            
            if max(ue_locs(:)) <= 1 && min(ue_locs(:)) >= 0
                % 归一化坐标，需要映射到图像尺寸
                if need_coord_scaling
                    % 如果尺寸不匹配，归一化坐标应该映射到实际图像尺寸
                    x_coords = ue_locs(:, 1) * display_width;
                    y_coords = ue_locs(:, 2) * display_height;
                    bs_x = bs_loc(1) * display_width;
                    bs_y = bs_loc(2) * display_height;
                    fprintf('坐标映射: 归一化坐标 -> [0, %.0f] × [0, %.0f] (已缩放)\n', ...
                        display_width, display_height);
                else
                    % 尺寸匹配，直接映射
                    x_coords = ue_locs(:, 1) * display_width;
                    y_coords = ue_locs(:, 2) * display_height;
                    bs_x = bs_loc(1) * display_width;
                    bs_y = bs_loc(2) * display_height;
                    fprintf('坐标映射: 归一化坐标 -> [0, %.0f] × [0, %.0f]\n', ...
                        display_width, display_height);
                end
                
                % 显示图像（注意MATLAB的y轴方向）
                if size(img_for_display, 3) == 3
                    % RGB图像
                    image([0, display_width], [0, display_height], flipud(img_for_display));
                else
                    % 灰度图像
                    image([0, display_width], [0, display_height], flipud(img_for_display), 'CDataMapping', 'scaled');
                    colormap(gca, 'gray');
                end
                axis([0, display_width, 0, display_height]);
            else
                % 已经是实际坐标（基于Info.npy的坐标系）
                if need_coord_scaling
                    % 需要将坐标从Info.npy坐标系缩放到实际图像坐标系
                    x_coords = ue_locs(:, 1) * scale_x;
                    y_coords = ue_locs(:, 2) * scale_y;
                    bs_x = bs_loc(1) * scale_x;
                    bs_y = bs_loc(2) * scale_y;
                    fprintf('坐标映射: Info.npy坐标系 [%.0f, %.0f] -> 实际图像坐标系 [%.0f, %.0f]\n', ...
                        img_size(1), img_size(2), actual_img_width, actual_img_height);
                    fprintf('  BS位置: (%.2f, %.2f) -> (%.2f, %.2f)\n', ...
                        bs_loc(1), bs_loc(2), bs_x, bs_y);
                else
                    % 尺寸匹配，直接使用
                    x_coords = ue_locs(:, 1);
                    y_coords = ue_locs(:, 2);
                    bs_x = bs_loc(1);
                    bs_y = bs_loc(2);
                end
                
                % 计算坐标范围
                x_range = [min(x_coords), max(x_coords)];
                y_range = [min(y_coords), max(y_coords)];
                
                % 如果坐标范围有效，显示图像
                if diff(x_range) > 0 && diff(y_range) > 0
                    % 显示图像（使用实际图像尺寸）
                    if size(img_for_display, 3) == 3
                        % RGB图像
                        image([0, display_width], [0, display_height], flipud(img_for_display));
                    else
                        % 灰度图像
                        image([0, display_width], [0, display_height], flipud(img_for_display), 'CDataMapping', 'scaled');
                        colormap(gca, 'gray');
                    end
                    axis([0, display_width, 0, display_height]);
                    fprintf('坐标范围: X[%.2f, %.2f] Y[%.2f, %.2f]\n', ...
                        x_range(1), x_range(2), y_range(1), y_range(2));
                else
                    warning('坐标范围无效，跳过背景图像显示');
                end
            end
            alpha(0.5);  % 设置透明度，使图像半透明
            fprintf('已添加环境图像作为背景\n');
        end
    end
    
    % 如果没有环境图像或图像显示失败，使用原始坐标
    if ~exist('x_coords', 'var') || isempty(x_coords)
        x_coords = ue_locs(:, 1);
        y_coords = ue_locs(:, 2);
        if ~exist('bs_x', 'var')
            bs_x = bs_loc(1);
            bs_y = bs_loc(2);
        end
    end
    
    % 自定义色图（从红到绿渐变）
    custom_colormap = [linspace(1,0,16)', linspace(0,1,16)', zeros(16,1)];
    
    % 绘制用户位置（按CQI值着色）
    scatter_handles = gobjects(16, 1);
    for cqi_level = 1:16
        idx = round(wideband_CQI_results) == (cqi_level - 1);
        if any(idx)
            scatter_handles(cqi_level) = scatter(...
                x_coords(idx), y_coords(idx), 80, ...
                'MarkerFaceColor', custom_colormap(cqi_level,:), ...
                'MarkerEdgeColor', [0.2 0.2 0.2], ...
                'LineWidth', 0.5, ...
                'DisplayName', sprintf('CQI=%d', (cqi_level-1)));
        end
    end
    
    % 绘制基站位置（红色五角星）
    scatter(bs_x, bs_y, 400, 'pentagram', 'k', 'filled', ...
        'MarkerFaceColor', [1 0.2 0.2], ...
        'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Base Station');
    
    % 图表修饰
    colormap(custom_colormap);
    clim([0 15]);
    cbar = colorbar;
    cbar.Label.String = 'CQI Value';
    cbar.Ticks = 0:15;
    cbar.TickLabels = arrayfun(@(x) sprintf('%d',x), 0:15, 'UniformOutput', false);
    
    grid on;
    xlabel('X Coordinate', 'FontSize', 12);
    ylabel('Y Coordinate', 'FontSize', 12);
    title('User Distribution with CQI Coloring', 'FontSize', 14);
    legend('Location', 'eastoutside', 'FontSize', 9);
    axis equal;
    axis tight;
    
    hold off;
end

%% ==================== 保存结果 ====================
fprintf('\n保存结果...\n');

% 准备场景信息结构体
% 从文件路径提取环境ID（如果可能）
env_id = '';
if contains(input_csi_file, filesep)
    path_parts = strsplit(input_csi_file, filesep);
    for i = 1:length(path_parts)
        if length(path_parts{i}) == 5 && all(isstrprop(path_parts{i}, 'digit'))
            env_id = path_parts{i};
            break;
        end
    end
end

% 从文件路径提取载波频率（如果可能）
carrier_freq = '';
if contains(input_csi_file, filesep)
    path_parts = strsplit(input_csi_file, filesep);
    for i = 1:length(path_parts)
        if contains(path_parts{i}, '_2_6_') || contains(path_parts{i}, '_28_0_') || ...
           contains(path_parts{i}, '_60_0_') || contains(path_parts{i}, '_100_0_')
            % 尝试提取频率
            parts = strsplit(path_parts{i}, '_');
            for j = 1:length(parts)-1
                if strcmp(parts{j}, '2') && strcmp(parts{j+1}, '6')
                    carrier_freq = '2_6';
                    break;
                elseif strcmp(parts{j}, '28') && strcmp(parts{j+1}, '0')
                    carrier_freq = '28_0';
                    break;
                elseif strcmp(parts{j}, '60') && strcmp(parts{j+1}, '0')
                    carrier_freq = '60_0';
                    break;
                elseif strcmp(parts{j}, '100') && strcmp(parts{j+1}, '0')
                    carrier_freq = '100_0';
                    break;
                end
            end
            if ~isempty(carrier_freq)
                break;
            end
        end
    end
end

% 创建场景信息结构体
scenario_info = struct();
scenario_info.img_size = img_size;  % 图像尺寸 [width, height]
scenario_info.bs_loc = bs_loc;  % 基站位置 [x, y]
scenario_info.ue_locs = ue_locs;  % 用户位置 (num_users, 2)
scenario_info.scenario = 2;  % 场景编号（根据实际情况修改）
scenario_info.env_id = env_id;  % 环境ID（如 '00032'）
scenario_info.carrier_freq = carrier_freq;  % 载波频率（如 '2_6'）
scenario_info.Nt = Nt;  % BS天线配置
scenario_info.Nr = Nr;  % UE天线配置
scenario_info.Nt_total = Nt_total;  % BS天线总数
scenario_info.Nr_total = Nr_total;  % UE天线总数
scenario_info.sampledCarriers = sampledCarriers;  % 采样子载波数
scenario_info.BWGHz = BWGHz;  % 带宽 (GHz)
scenario_info.subcarrier_spacing = subcarrier_spacing;  % 子载波间隔 (Hz)
scenario_info.num_users = num_users;  % 用户数量
scenario_info.num_subbands = num_subbands;  % 子带数量
scenario_info.subband_size = subband_size;  % 子带大小

fprintf('场景信息:\n');
fprintf('  环境ID: %s\n', env_id);
fprintf('  载波频率: %s\n', carrier_freq);
fprintf('  场景编号: %d\n', scenario_info.scenario);
fprintf('  用户数量: %d\n', num_users);

%% ==================== 准备兼容Python数据加载器的数据格式 ====================
fprintf('\n准备Python兼容的数据格式...\n');


% 重新排列DFT后的H: (num_users, Nt*Nr, sampledCarriers)
H_reshaped = permute(H_complex_angle_delay, [3, 1, 2]);  % 从 (Nt*Nr, sampledCarriers, num_users) 到 (num_users, Nt*Nr, sampledCarriers)

% 分离实部和虚部
H_real = real(H_reshaped);  % (num_users, Nt*Nr, sampledCarriers)
H_imag = imag(H_reshaped);  % (num_users, Nt*Nr, sampledCarriers)

% 将实部和虚部堆叠为4D数组，形状为 (num_users, Nt*Nr, sampledCarriers, 2)
% 最后一个维度: [real, imag]
% 创建一个包含 'real' 和 'imag' 字段的结构体
H_final_angle_delay = struct('real', H_real, 'imag', H_imag);

fprintf('H数据格式转换完成（使用DFT后的角度-延迟域数据）:\n');
fprintf('  DFT后H形状: %s\n', mat2str(size(H_complex_angle_delay)));
fprintf('  转换后类型: struct with fields ''real'' and ''imag''\n');
    fprintf('  H_final_angle_delay.real 形状: %s\n', mat2str(size(H_final_angle_delay.real)));
    fprintf('  H_final_angle_delay.imag 形状: %s\n', mat2str(size(H_final_angle_delay.imag)));


wideband_CQI_results_original = wideband_CQI_results;
subband_CQI_results_original = subband_CQI_results;


wideband_CQI_results_save = wideband_CQI_results';  % 强制转置为列向量 (num_users, 1)

% subband_CQI_results当前形状: (num_subbands, num_users)
% Python期望: (num_users, num_subbands)，需要转置
if size(subband_CQI_results, 1) == num_subbands && size(subband_CQI_results, 2) == num_users
    subband_CQI_results_save = subband_CQI_results';  % 转置为 (num_users, num_subbands)
else
    subband_CQI_results_save = subband_CQI_results;  % 如果已经是正确形状，直接使用
end

fprintf('CQI数据格式:\n');
fprintf('  Wideband CQI形状: %s (Python期望: (%d,))\n', ...
    mat2str(size(wideband_CQI_results_save)), num_users);
fprintf('  Subband CQI形状: %s (Python期望: (%d, %d))\n', ...
    mat2str(size(subband_CQI_results_save)), num_users, num_subbands);

%% ==================== 保存Python兼容格式的文件 ====================


output_file_wide = sprintf('%s_wide.mat', output_prefix);
output_file_wide_python = sprintf('losangeles_adCSI_right_wide.mat');


wideband_CQI_results = wideband_CQI_results_save;  % 确保是列向量
save(output_file_wide, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
fprintf('\n已保存: %s\n', output_file_wide);
fprintf('  包含变量: H_final_angle_delay, wideband_CQI_results, scenario_info\n');


[output_dir, ~, ~] = fileparts(output_file_wide);
if ~isempty(output_dir) && exist(output_dir, 'dir')
    output_file_wide_python_full = fullfile(output_dir, output_file_wide_python);
    save(output_file_wide_python_full, 'H_final_angle_delay', 'wideband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存(Python兼容): %s\n', output_file_wide_python_full);
end


output_file_sub = sprintf('%s_sub.mat', output_prefix);
output_file_sub_python = sprintf('losangeles_adCSI_right_sub.mat');


subband_CQI_results = subband_CQI_results_save;  % 临时赋值以匹配Python期望的变量名
save(output_file_sub, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
fprintf('已保存: %s\n', output_file_sub);
fprintf('  包含变量: H_final_angle_delay, subband_CQI_results, scenario_info\n');


if ~isempty(output_dir) && exist(output_dir, 'dir')
    output_file_sub_python_full = fullfile(output_dir, output_file_sub_python);
    save(output_file_sub_python_full, 'H_final_angle_delay', 'subband_CQI_results', 'scenario_info', '-v7.3');
    fprintf('已保存(Python兼容): %s\n', output_file_sub_python_full);
end

output_file_full = sprintf('%s_full.mat', output_prefix);
save(output_file_full, 'H_complex_freq', 'H_complex_angle_delay', 'H_final_angle_delay', ...
    'wideband_CQI_results_original', 'subband_CQI_results_original', ...
    'wideband_avg_SNR_dB', 'subband_avg_SNR_dB', 'ue_locs', 'bs_loc', ...
    'scenario_info', 'num_users', 'num_subbands', 'subband_size', '-v7.3');
fprintf('已保存: %s\n', output_file_full);
fprintf('  包含变量: H_complex_freq (频域), H_complex_angle_delay (DFT后), H_final_angle_delay (4D数组),\n');
fprintf('           wideband_CQI_results_original, subband_CQI_results_original,\n');
fprintf('           wideband_avg_SNR_dB, subband_avg_SNR_dB, ue_locs, bs_loc,\n');
fprintf('           scenario_info, num_users, num_subbands, subband_size\n');

fprintf('\n所有处理完成！\n');
fprintf('\n提示: Python数据加载器期望的文件名为 losangeles_adCSI_right_{wide|sub}.mat\n');
fprintf('      如果文件名不匹配，请将生成的文件重命名或复制到数据目录中。\n');

