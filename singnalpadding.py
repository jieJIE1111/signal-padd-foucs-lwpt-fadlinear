# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, medfilt, spectrogram
import pandas as pd
import os
# Try importing Holt from exponential_smoothing
try:
    from statsmodels.tsa.holtwinters import Holt
except ImportError:
    print("Warning: statsmodels.tsa.holtwinters.Holt not found. Trend extrapolation will use linear method.")
    Holt = None # Set Holt to None if import fails
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
# Import AutoRegResults to access residuals if needed (though fit results usually have them)
from statsmodels.tsa.ar_model import AutoRegResults
import pywt # 确保已安装 pip install PyWavelets
from scipy.interpolate import interp1d
from numba import jit
import traceback # 用于打印错误信息
import math # Added for ceil

# --- 配置 ---
# 文件输出目录 (请确保这是你希望的基础目录)
output_dir = "/mnt/c/Users/shiki/Desktop/ocean engineering" # <--- 请修改为你的实际路径
# 新建的保存结果的子文件夹名称 (包含最终 C0/C1 强制的标识)
# Updated folder name to reflect the change (Proposed method now always uses AR path)
results_folder_name = "xrsyfinal_with_comparisons_v4_final_C0C1_enforced_local_metrics_proposed_always_AR"
# 最终结果保存目录
results_dir = os.path.join(output_dir, results_folder_name)
# 确保结果目录存在
os.makedirs(results_dir, exist_ok=True)
# 图像DPI
IMG_DPI = 600
# 方差阈值，用于检测信号末端是否平稳
STABLE_END_VARIANCE_THRESHOLD = 1e-10
# 用于检测平稳性的窗口大小
STABLE_END_WINDOW = 50
# AR阶数选择中检查根稳定性的阈值 (略小于1)
STABILITY_ROOT_THRESHOLD = 0.999
# 保存填充后数据的文件名 (包含最终 C0/C1 强制的标识)
# Updated filename to reflect the change
PADDED_DATA_FILENAME = "proposed_method_v4_always_AR_C0C1_enforced_padded_data_local_metrics.csv"
# 本地统计分析窗口
LOCAL_STATS_WINDOW = 100 # Window size for local mean/std calculation (used in extend_signal_base)
# AR Noise Injection Parameters
DEFAULT_AR_NOISE_FACTOR = 0.15 # Factor to scale residual stddev for added noise (0 = no noise)
# AR Clamping Factor
DEFAULT_AR_CLAMPING_FACTOR = 2.5 # Factor for amplitude limit (e.g., 2.5 * local_std)
# 新增: 局部窗口对比大小
LOCAL_COMPARISON_WINDOW_SIZE = 100
# 新增: 局部窗口对比指标的键列表 (移动到全局作用域)
LOCAL_METRIC_KEYS = ['local_mean_diff', 'local_variance_ratio_100', 'local_rms_diff_100',
                     'local_centroid_diff_100', 'local_spectrum_diff_100', 'local_kl_divergence_100']


# --- 辅助函数 (calculate_wavelet_packet_energy, welch_psd, select_best_order, fast_ar_predict_optimized) ---
# (这些函数保持不变)

def calculate_wavelet_packet_energy(signal, wavelet='db4', level=4):
    """计算小波包分解后的频带能量"""
    try:
        min_len_for_level = 2**(level+1) # 基础检查
        if len(signal) < min_len_for_level:
             max_possible_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet))
             if max_possible_level < level and max_possible_level > 0:
                 level = max_possible_level
             else:
                 if max_possible_level <= 0:
                      # 对于非常短的信号，降低level尝试，或者返回NaN
                      if len(signal) >= 2**(1+1): # 至少能做 level 1
                          level = 1
                      else:
                          # raise ValueError(f"Signal length {len(signal)} too short for any WPT level.")
                          num_bands = 2**level if level > 0 else 1 # 返回期望的band数，值为NaN
                          return np.full(num_bands, np.nan)
                 else:
                      level = max_possible_level

        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = wp.get_level(level, 'freq') # 'freq' 顺序节点
        energy = [np.sum(np.abs(node.data)**2) for node in nodes] # 使用能量（平方和）
        total_energy = np.sum(energy)
        if total_energy < 1e-10: # 避免除零
            num_bands = 2**level if level > 0 else 1
            return np.zeros(num_bands) # 返回正确数量的零
        return np.array(energy) / total_energy
    except Exception as e:
        print(f"Error in calculate_wavelet_packet_energy: {e}")
        num_bands = 2**level if level > 0 else 16 # 默认16
        return np.full(num_bands, np.nan) # 返回NaN表示失败

def welch_psd(signal, fs=1.0, nperseg=None, noverlap=None):
    """使用 Welch 方法计算功率谱密度"""
    sig_len = len(signal);
    if sig_len < 2: return np.array([0]), np.array([1e-10]) # 需要至少2点

    if nperseg is None:
        # 根据信号长度自适应选择 nperseg
        if sig_len > 1024: nperseg = 512
        elif sig_len > 256: nperseg = 256
        elif sig_len > 64: nperseg = 64
        else: nperseg = max(8, sig_len // 4) # 保证至少为8或长度的1/4
    nperseg = min(nperseg, sig_len) # nperseg 不能超过信号长度
    nperseg = max(2, nperseg) # nperseg 至少为 2

    if noverlap is None:
        noverlap = nperseg // 2
    # 确保 noverlap 合法
    noverlap = min(noverlap, nperseg - 1)
    noverlap = max(0, noverlap)

    try:
        # window='hann' 是常用的选择
        freq, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        # 对 PSD 值进行 clip，避免出现负数或零导致后续 log 计算问题
        return freq, np.clip(psd, 1e-10, None)
    except ValueError as e:
        # 如果 welch 出错（例如 nperseg > sig_len 等），返回默认值
        # print(f"Welch PSD calculation failed: {e}. Returning default.") # 可选的调试信息
        return np.array([0]), np.array([1e-10])


def select_best_order(signal_demeaned, max_order, stability_threshold=STABILITY_ROOT_THRESHOLD):
    """使用 AIC 选择最佳 AR 模型阶数，并进行严格稳定性检查"""
    best_aic, best_order = float('inf'), 5 # 默认阶数 5
    sig_len = len(signal_demeaned)
    # 允许的最大阶数不能超过信号长度减去一个小量 (防止过拟合或矩阵奇异)
    max_allowable_order = max(1, sig_len - 10)
    current_max_order = min(max_order, max_allowable_order)

    # 如果允许的最大阶数太小，直接返回默认值
    if current_max_order < 5:
        return 5 # 默认阶数

    selected_order = 5 # 初始化默认

    # --- 尝试使用 statsmodels 的 ar_select_order (如果信号足够长，效率更高) ---
    # 要求信号长度至少是最大阶数的两倍，并且大于某个阈值（例如30）
    if sig_len > max(2 * current_max_order, 30):
        try:
            # trend='n' 表示无趋势 (因为我们输入的是去均值信号)
            # ic='aic' 使用 AIC 准则
            sel = ar_select_order(signal_demeaned, maxlag=current_max_order, ic='aic', old_names=False, trend='n')
            # 获取选定的阶数 (lags 可能为空，也可能包含多个值，取最后一个)
            potential_order = sel.ar_lags[-1] if sel.ar_lags else 5
            potential_order = max(1, potential_order) # 确保阶数至少为1

            # 检查选定阶数的稳定性
            if potential_order > 0 and potential_order <= current_max_order:
                model_instance_sel = AutoReg(signal_demeaned, lags=potential_order, old_names=False)
                model_fit_sel = model_instance_sel.fit()
                roots_sel = model_fit_sel.roots
                # 检查根的模是否都小于阈值
                if len(roots_sel) == 0 or np.max(np.abs(roots_sel)) < stability_threshold:
                    # print(f"Selected order {potential_order} by ar_select_order and passed stability check.") # 调试信息
                    return potential_order # 如果稳定，直接返回此阶数
        except Exception as e_sel:
            # 如果 ar_select_order 失败，则继续下面的迭代检查
            # print(f"ar_select_order failed: {e_sel}. Falling back to iteration.") # 调试信息
            pass

    # --- 迭代 AIC 检查 (作为备选，或当信号太短时) ---
    # 从默认阶数 5 开始检查到 current_max_order
    for order in range(5, current_max_order + 1):
        try:
            model_instance = AutoReg(signal_demeaned, lags=order, old_names=False)
            try:
                 # 尝试拟合模型
                 model_fit = model_instance.fit()
            except (np.linalg.LinAlgError, ValueError):
                 # 如果拟合失败（如矩阵奇异），跳过此阶数
                 continue

            # 获取模型的根
            roots = model_fit.roots
            is_stable = True
            # 检查稳定性
            if len(roots) > 0:
                if np.max(np.abs(roots)) >= stability_threshold:
                    is_stable = False
                    # print(f"Order {order} is unstable (max root: {np.max(np.abs(roots)):.4f}). Skipping.") # 调试信息
                    continue # 跳过不稳定模型

            # 如果模型稳定，获取 AIC
            current_aic = model_fit.aic
            # 如果当前 AIC 比已找到的最佳 AIC 更好
            if is_stable and current_aic < best_aic:
                best_aic = current_aic
                selected_order = order
                # print(f"Found new best stable order: {selected_order} with AIC: {best_aic:.2f}") # 调试信息
        except Exception as e:
            # 捕获其他可能的错误，跳过此阶数
            # print(f"Error processing order {order}: {e}") # 调试信息
            continue

    # print(f"Final selected order (iteration): {selected_order}") # 调试信息
    return selected_order


@jit(nopython=True)
def fast_ar_predict_optimized(params, last_samples, pad_length, local_std,
                              noise_std=0.0, clamping_factor=2.5):
    """使用 Numba 加速的 AR 预测 (优化版: 可选噪声, 可调幅度限制)"""
    extension = np.zeros(pad_length)
    samples = last_samples.copy() # 操作副本，避免修改原始数据
    n_params = len(params)
    # AR 模型的阶数是参数数量减 1 (减去截距项)
    order = n_params - 1
    # 第一个参数是截距 (均值项)
    intercept = params[0]

    # 计算振幅限制 (基于局部标准差)
    amplitude_limit = clamping_factor * local_std
    # 防止限制过小或为零
    if amplitude_limit < 1e-9:
        amplitude_limit = 1e-9 # 设置一个非常小的下限

    for i in range(pad_length):
        # --- 预测步骤 ---
        next_sample = intercept # 从截距开始
        len_samples = len(samples)
        # AR(p) 模型需要 p 个历史样本来预测下一个点
        # 这里使用 min(order, len_samples) 是为了处理开始阶段样本不足 order 个的情况 (虽然调用时通常保证足够)
        current_order = min(order, len_samples)
        # AR 模型预测: y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p}
        # params[1] 对应 phi_1, params[order] 对应 phi_p
        # samples[-1] 对应 y_{t-1}, samples[-current_order] 对应 y_{t-current_order}
        for j in range(current_order):
            # params索引从1开始，对应滞后项系数
            # samples索引从后往前取：len_samples - 1 - j
            next_sample += params[j + 1] * samples[len_samples - 1 - j]

        # --- 添加噪声 (在钳位之前) ---
        # 只有当噪声标准差大于一个很小的值时才添加
        if noise_std > 1e-12:
             added_noise = np.random.normal(0.0, noise_std)
             next_sample += added_noise

        # --- 应用幅度限制 (Clamping) ---
        # 限制预测值的绝对值不超过计算出的 amplitude_limit
        if np.abs(next_sample) > amplitude_limit:
            # 如果超过限制，将其设置为限制值，并保持原始符号
            next_sample = np.sign(next_sample) * amplitude_limit

        # 将预测值存入结果数组
        extension[i] = next_sample

        # --- 高效更新样本缓冲区 (用于下一次预测) ---
        # 如果样本缓冲区不为空
        if len_samples > 0:
            # 将所有样本向前移动一个位置 (丢弃最旧的样本)
            for k in range(len_samples - 1):
                samples[k] = samples[k+1]
            # 在缓冲区末尾添加新的预测值
            samples[-1] = next_sample

    return extension

# --- extend_signal 的基础函数 (v4 - C0/C1 在最后强制执行) ---
def extend_signal_base(signal, pad_length, max_order=100,
                        overlap=50, variance_threshold=STABLE_END_VARIANCE_THRESHOLD,
                        stable_end_window=STABLE_END_WINDOW,
                        local_stats_window=LOCAL_STATS_WINDOW,
                        apply_transition=True, apply_postprocessing=True,
                        force_ar_path=False, # This parameter is kept for flexibility but Proposed_Method_v4 will always set it to True
                        use_ar_in_notrend=True,
                        # --- AR 噪声和钳位参数 ---
                        add_ar_noise=True,
                        ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                        ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR):
    """信号填充的基础函数 (v4 - 最终强制 C0/C1)"""
    original_length = len(signal)

    # 处理极短信号
    if original_length < 5:
        mean_val = np.mean(signal) if original_length > 0 else 0
        safe_pad_length = max(0, pad_length)
        if safe_pad_length == 0 and original_length > 0: return signal.copy()
        elif original_length == 0: return np.full(safe_pad_length, mean_val)
        else: return np.concatenate([signal, np.full(safe_pad_length, mean_val)])

    # 处理 pad_length <= 0
    if pad_length <= 0:
        return signal.copy()

    # --- 1. 信号特性和局部统计 ---
    local_analysis_window = min(original_length, local_stats_window)
    local_segment = signal[-local_analysis_window:]
    signal_local_mean = np.mean(local_segment) if local_analysis_window > 0 else (signal[-1] if original_length > 0 else 0)
    signal_local_demeaned = local_segment - signal_local_mean
    local_std = np.std(signal_local_demeaned) if local_analysis_window > 1 else 0
    # 如果局部标准差非常小，尝试使用全局标准差，或者设定一个最小默认值
    if local_std < 1e-9:
        overall_std = np.std(signal - np.mean(signal)) if original_length > 1 else 0
        local_std = max(1e-9, overall_std if overall_std > 1e-9 else 1e-9) # 保证 local_std > 0
    # 准备用于 AR 模型的去均值信号 (使用局部均值)
    signal_demeaned_for_ar = signal - signal_local_mean

    # --- 稳定末端检测 ---
    is_stable_end = False
    stable_window = min(stable_end_window, original_length)
    end_variance = np.inf
    if stable_window > 1:
        end_variance = np.var(signal[-stable_window:])
        if end_variance < variance_threshold:
            is_stable_end = True

    # --- 稳定末端处理 ---
    if is_stable_end:
        mean_padding_value = signal_local_mean # 使用局部均值填充
        extension = np.full(pad_length, mean_padding_value)
        full_signal = np.concatenate([signal, extension])
        # 确保最终长度正确 (以防万一)
        expected_length = original_length + pad_length
        if len(full_signal) != expected_length:
             if len(full_signal) > expected_length: full_signal = full_signal[:expected_length]
             else: diff = expected_length - len(full_signal); full_signal = np.concatenate([full_signal, np.full(diff, mean_padding_value)])
        # print(f"Stable end detected (variance={end_variance:.2e}). Using mean padding.") # 调试信息
        return full_signal # 直接返回，不再进行后续处理

    # --- 2. 趋势检测 (Now potentially skipped if force_ar_path=True) ---
    has_trend = False
    # 仅在不强制使用 AR 路径时进行趋势检测
    if not force_ar_path:
        trend_segment_len = min(local_stats_window, original_length) # 使用局部统计窗口长度
        if trend_segment_len > 10: # 需要足够长度才能可靠检测趋势
            end_segment = signal[-trend_segment_len:]
            x = np.arange(trend_segment_len)
            try:
                # 拟合线性趋势
                slope, intercept = np.polyfit(x, end_segment, 1)
                # 判断趋势是否显著：斜率乘以段长度（总变化量）是否大于局部标准差的一半
                if np.abs(slope) * trend_segment_len > 0.5 * local_std :
                    has_trend = True
                    # print(f"Trend detected (slope={slope:.4f}, scaled_change={np.abs(slope) * trend_segment_len:.4f}, threshold={0.5 * local_std:.4f}).") # 调试
            except (np.linalg.LinAlgError, ValueError):
                # 拟合失败则认为无趋势
                has_trend = False

    extension = np.array([]) # 初始化 extension

    # --- 3. 预测分支 (生成初始 extension) ---
    # Condition modified: If force_ar_path is True, this 'if' block is skipped.
    if has_trend and not force_ar_path:
        # --- 3a. 趋势外插 ---
        # print("Using Trend Extrapolation path.") # 调试
        try:
            # 优先尝试 Holt-Winters (如果可用且安装了 statsmodels)
            if Holt is not None:
                # 使用阻尼趋势 (damped_trend=True) 更稳健
                holt_model = Holt(end_segment, exponential=False, damped_trend=True).fit(optimized=True)
                extension = holt_model.forecast(pad_length)
                # 检查 Holt 结果是否有效
                if np.isnan(extension).any() or np.isinf(extension).any(): raise ValueError("Holt forecast resulted in NaN/Inf")
            else:
                 # 如果 Holt 不可用，抛出错误以进入备选方法
                 raise ValueError("Holt method not available.")
        except Exception as e_holt:
            # print(f"Holt trend extrapolation failed ({e_holt}). Falling back to damped linear extrapolation.") # 调试
            # Holt 失败或不可用，使用带阻尼的线性外插作为备选
            try:
                x = np.arange(trend_segment_len)
                slope, intercept = np.polyfit(x, end_segment, 1) # 再次拟合线性趋势
                x_ext = np.arange(trend_segment_len, trend_segment_len + pad_length)
                linear_extrapolation = intercept + slope * x_ext
                # 添加阻尼项，使外插值逐渐衰减回局部均值
                # decay_rate 控制衰减速度，使其在 pad_length 长度内大致衰减 e^-3 (~5%)
                decay_rate = 3.0 / max(1, pad_length) # 避免 pad_length 为 0
                damping_weights = np.exp(-decay_rate * np.arange(pad_length))
                # 最终外插值 = 线性外插 * 衰减权重 + 局部均值 * (1 - 衰减权重)
                extension = linear_extrapolation * damping_weights + signal_local_mean * (1 - damping_weights)
            except (np.linalg.LinAlgError, ValueError) as e_linear:
                 # 如果线性拟合也失败，则使用末值填充作为最终备选
                 print(f"Warning: Linear trend fallback failed ({e_linear}). Using last value padding for trend case.") # 警告
                 last_val = signal[-1] if original_length > 0 else 0
                 extension = np.full(pad_length, last_val)
    else:
        # --- 3b. 无趋势 或 强制 AR ---
        # This path is now taken if force_ar_path=True OR if no trend was detected (and force_ar_path=False).
        # print("Using AR or Mean Padding path.") # 调试
        # 如果设置了 use_ar_in_notrend=True 或 force_ar_path=True，则使用 AR 模型
        # The condition simplifies slightly if force_ar_path is always true for the proposed method.
        if use_ar_in_notrend or force_ar_path:
            # print("Attempting AR model.") # 调试
            try:
                # 选择最佳 AR 阶数
                actual_order = select_best_order(signal_demeaned_for_ar, max_order)
                # 再次检查阶数是否合法 (不能超过可用样本数)
                if actual_order >= len(signal_demeaned_for_ar): actual_order = max(1, len(signal_demeaned_for_ar) - 2)
                if actual_order <= 0: raise ValueError(f"Signal length ({len(signal_demeaned_for_ar)}) too short for AR after order selection.")

                # print(f"Selected AR order: {actual_order}") # 调试
                # 训练 AR 模型 (使用去均值的信号)
                model = AutoReg(signal_demeaned_for_ar, lags=actual_order, old_names=False)
                model_fit = model.fit(cov_type='nonrobust') # 使用非鲁棒协方差估计更快
                ar_params = model_fit.params # 获取模型参数 (包括截距)

                # 计算用于添加噪声的标准差 (基于模型残差)
                noise_std_pred = 0.0
                if add_ar_noise and ar_noise_factor > 0:
                    try:
                        residuals = model_fit.resid # 获取残差
                        residual_std = np.nanstd(residuals) # 计算残差标准差 (忽略 NaN)
                        if residual_std > 1e-9: noise_std_pred = residual_std * ar_noise_factor
                        # print(f"AR noise std: {noise_std_pred:.4f} (residual_std={residual_std:.4f}, factor={ar_noise_factor})") # 调试
                    except Exception as e_resid:
                        # print(f"Could not get residuals for noise calculation: {e_resid}") # 调试
                        pass # 忽略残差计算错误

                # 准备用于预测的最后 N 个样本 (N=阶数)
                if len(signal_demeaned_for_ar) < actual_order:
                    raise ValueError(f"Not enough samples ({len(signal_demeaned_for_ar)}) for AR order {actual_order} prediction start.")
                last_samples_for_pred = signal_demeaned_for_ar[-actual_order:].copy()

                # print(f"Predicting {pad_length} steps using AR({actual_order})...") # 调试
                # 使用 Numba 优化的函数进行预测
                extension_demeaned = fast_ar_predict_optimized(
                    ar_params, last_samples_for_pred, pad_length,
                    local_std, noise_std=noise_std_pred, clamping_factor=ar_clamping_factor)

                # 将预测结果加上之前减去的局部均值
                extension = extension_demeaned + signal_local_mean

            except Exception as e_ar:
                # AR 模型失败，回退到简单方法
                print(f"AR model failed: {e_ar}. Falling back to linear interpolation to local mean.") # 警告
                # 使用从最后一个值线性插值到局部均值的方法
                last_val = signal[-1] if original_length > 0 else 0
                if pad_length > 0: extension = np.linspace(last_val, signal_local_mean, pad_length)
                else: extension = np.array([])
        else:
            # This case is only reachable if force_ar_path=False AND use_ar_in_notrend=False AND has_trend=False
            # It remains for the Ablation_AR_vs_Mean_NoTrend_v4 experiment.
            # print("Using local mean padding (no trend, AR disabled).") # 调试
            mean_padding_value = signal_local_mean
            extension = np.full(pad_length, mean_padding_value)

    # --- 3.5. 确保初始 Extension 长度正确 ---
    # 在后续处理前，必须确保 extension 数组的长度正好是 pad_length
    if len(extension) != pad_length:
         print(f"Warning: Initial extension length mismatch ({len(extension)} vs {pad_length}). Adjusting.") # 警告
         if len(extension) > pad_length:
             extension = extension[:pad_length] # 截断
         elif pad_length > 0:
             # 如果 extension 太短，用其最后一个值或局部均值填充剩余部分
             diff = pad_length - len(extension)
             last_val_ext = extension[-1] if len(extension) > 0 else signal_local_mean
             extension = np.concatenate([extension, np.full(diff, last_val_ext)])

    # --- 4. 边界过渡平滑 ---
    # (此步骤会修改 extension 的前 overlap_points 个点)
    # 条件：应用过渡、信号和填充段有足够长度、重叠点数大于1
    if apply_transition and original_length > 1 and pad_length > 0 and len(extension) > 0:
        # overlap 是基础重叠点数，但实际点数会根据信号末端稳定性和可用长度调整
        # 动态调整重叠窗口大小：更不稳定的末端（方差大）使用稍大的重叠窗口
        base_overlap = min(overlap, original_length // 2, pad_length // 2, 75) # 基础值，不超过75或可用长度的一半
        # 计算方差与局部标准差平方的比值，作为不稳定性的度量
        variance_ratio = end_variance / (local_std**2 + 1e-9) if local_std > 1e-9 else np.inf
        # 根据方差比调整重叠因子 (对数关系，避免极端值)
        overlap_factor = np.clip(1.0 + 0.5 * np.log1p(variance_ratio * 5), 0.5, 1.5) # 因子在 0.5 到 1.5 之间
        # 计算实际重叠点数
        overlap_points = int(np.clip(base_overlap * overlap_factor, 10, overlap)) # 保证至少10点，不超过原始 overlap 参数
        # 再次确保不超过可用长度的一半
        overlap_points = min(overlap_points, original_length // 2, pad_length // 2)
        overlap_points = max(2, overlap_points) # 保证至少2点才能进行平滑

        # print(f"Applying transition: overlap_points={overlap_points} (base={base_overlap}, factor={overlap_factor:.2f})") # 调试

        # 执行平滑操作
        if overlap_points > 1 and overlap_points <= len(extension):
            # 使用 cos 形状的权重函数，实现平滑过渡
            t = np.linspace(0, 1, overlap_points)
            weights = 0.5 * (1 - np.cos(np.pi * t)) # 权重从 0 平滑增加到 1
            # 获取信号末端和扩展段开始的相应片段
            signal_end_segment = signal[-overlap_points:]
            extension_start_segment = extension[:overlap_points].copy() # 复制一份以防修改影响计算
            # 加权平均混合两个片段
            blend_zone = signal_end_segment * (1 - weights) + extension_start_segment * weights
            # 将混合结果写回 extension 的开始部分
            extension[:overlap_points] = blend_zone # <--- 这会覆盖 extension 的开头

    # --- 5. 后处理 ---
    # (此步骤也会修改 extension)
    if apply_postprocessing and len(extension) > 3: # 需要至少4个点才能应用滤波器
        # 5a. 中值滤波 (去除突刺噪声)
        # print("Applying post-processing: Median filter...") # 调试
        try:
            # 使用 3 点中值滤波器
            extension = medfilt(extension, kernel_size=3)
        except Exception as e_med: print(f"Median filter failed: {e_med}") # 警告

        # 5b. 条件低通滤波 (Conditional Low-pass Filtering)
        # 目的是平滑可能由 AR 模型引入的高频噪声，但仅在必要时应用
        apply_lpf = False
        try:
            # --- LPF 条件检查逻辑 ---
            # 比较扩展段开始部分与信号末端的高频能量比例
            ext_demeaned = extension - np.mean(extension) # 对当前 extension 去均值
            seg_len_ext = min(len(ext_demeaned), 256) # 检查窗口长度
            if seg_len_ext > 10: # 需要足够长度计算 PSD
                 f_ext, psd_ext = welch_psd(ext_demeaned[:seg_len_ext], fs=1.0, nperseg=max(8, seg_len_ext // 2))
                 # 计算高频能量 (后半部分 PSD)
                 hf_energy_ext = np.sum(psd_ext[len(psd_ext)//2:]) if len(psd_ext) > 1 else 0
                 total_energy_ext = np.sum(psd_ext) + 1e-10 # 避免除零
                 hf_ratio_ext = hf_energy_ext / total_energy_ext # 高频能量占比

                 # 获取信号末端的局部去均值片段 (之前已计算)
                 sig_end_demeaned = signal_local_demeaned # local_segment - signal_local_mean
                 seg_len_sig = len(sig_end_demeaned)
                 if seg_len_sig > 10:
                      f_sig_end, psd_sig_end = welch_psd(sig_end_demeaned, fs=1.0, nperseg=max(8, seg_len_sig // 2))
                      hf_energy_sig_end = np.sum(psd_sig_end[len(psd_sig_end)//2:]) if len(psd_sig_end) > 1 else 0
                      total_energy_sig_end = np.sum(psd_sig_end) + 1e-10
                      hf_ratio_sig_end = hf_energy_sig_end / total_energy_sig_end

                      # 条件：扩展段高频能量占比显著高于信号末端，且本身占比不低
                      if hf_ratio_ext > hf_ratio_sig_end * 1.5 and hf_ratio_ext > 0.1:
                          apply_lpf = True
                          # print(f"Condition met for LPF: hf_ratio_ext={hf_ratio_ext:.3f}, hf_ratio_sig_end={hf_ratio_sig_end:.3f}") # 调试

            if apply_lpf:
                # print("Applying post-processing: Conditional Low-pass filter...") # 调试
                # --- LPF 应用逻辑 ---
                # 动态确定截止频率：基于原始信号整体的频谱质心
                overall_signal_demeaned = signal - np.mean(signal)
                f_orig_pp, psd_orig_pp = welch_psd(overall_signal_demeaned, fs=1.0)
                cutoff = 0.1 # 默认截止频率
                if np.sum(psd_orig_pp) > 1e-9: # 确保 PSD 有效
                     centroid = np.sum(f_orig_pp * psd_orig_pp) / np.sum(psd_orig_pp) # 计算频谱质心
                     # 将截止频率设为质心的 1.5 倍，并限制在合理范围 [0.05, 0.4]
                     cutoff = np.clip(centroid * 1.5, 0.05, 0.4)

                filter_order = 2 # 使用 2 阶巴特沃斯滤波器
                nyquist = 0.5 # 奈奎斯特频率 (fs=1.0)
                cutoff = min(cutoff, nyquist - 1e-6) # 确保截止频率低于奈奎斯特频率
                # print(f"LPF cutoff frequency: {cutoff:.3f}") # 调试

                # 应用滤波器 (仅当截止频率有效且扩展段足够长时)
                if cutoff > 0 and len(extension) > filter_order * 3 :
                     # 设计 Butterworth 低通滤波器
                     b, a = butter(filter_order, cutoff, btype='low', fs=1.0)
                     # 使用 filtfilt 进行零相位滤波
                     extension = filtfilt(b, a, extension) # 应用 LPF
        except Exception as e_lpf: print(f"Post-processing LPF failed: {e_lpf}") # 警告


    # --- 6. 最终组合、长度检查和 C0/C1 强制执行 ---
    expected_length = original_length + pad_length

    # 在最终强制 C0/C1 之前，再次确保 extension 长度正确
    if len(extension) != pad_length:
        print(f"Warning: Extension length mismatch ({len(extension)} vs {pad_length}) before final C0/C1. Adjusting.") # 警告
        if len(extension) > pad_length: extension = extension[:pad_length]
        elif pad_length > 0:
            diff = pad_length - len(extension)
            last_val_ext = extension[-1] if len(extension) > 0 else signal_local_mean
            extension = np.concatenate([extension, np.full(diff, last_val_ext)])

    # --- *** 最终 C0/C1 强制执行 *** ---
    # 在所有预测、平滑、滤波步骤之后进行，以保证 C0/C1 连续性指标严格为零
    if pad_length > 0 and original_length > 0:
        # print("DEBUG: Enforcing final C0/C1 continuity...") # 调试
        # 强制 C0 (值连续)
        # 将扩展段的第一个点直接设置为原始信号的最后一个点
        extension[0] = signal[-1]

        # 如果可能，强制 C1 (梯度连续)
        # 需要扩展段至少有 2 个点，原始信号至少有 2 个点才能计算梯度
        if pad_length >= 2 and original_length >= 2:
            # 计算原始信号末端的梯度 (后向差分)
            gradient_end = signal[-1] - signal[-2]
            # 设置扩展段的第二个点，使得 extension[1] - extension[0] = gradient_end
            extension[1] = extension[0] + gradient_end
            # print(f"DEBUG: Final C0 enforced (value={extension[0]:.4f}). Final C1 enforced (target grad={gradient_end:.4f}, ext[1]={extension[1]:.4f}).") # 调试
        # else:
            # print("DEBUG: C1 not enforced (pad_length < 2 or original_length < 2).") # 调试

    # --- 拼接 ---
    full_signal = np.concatenate([signal, extension])

    # --- 最终长度检查 (鲁棒性) ---
    # 确保拼接后的信号长度符合预期
    if len(full_signal) != expected_length:
         print(f"Warning: Final signal length mismatch ({len(full_signal)} vs {expected_length}) after concat. Adjusting.") # 警告
         if len(full_signal) > expected_length:
             full_signal = full_signal[:expected_length] # 截断
         else:
             # 如果长度不足，用最后一个值填充
             diff = expected_length - len(full_signal)
             last_val_full = full_signal[-1] if len(full_signal) > 0 else signal_local_mean
             full_signal = np.concatenate([full_signal, np.full(diff, last_val_full)])

    return full_signal


# --- 定义各个消融实验对应的调用函数 (v4 - 都基于新的 base 函数) ---
# 这些函数现在都隐式地使用了最终的 C0/C1 强制逻辑

# MODIFIED: Proposed method now *always* uses the AR path by setting force_ar_path=True
def extend_signal_proposed_v4(signal, pad_length, **kwargs):
    """Proposed Method (v4 - Modified): Always uses AR path + Full features + AR Noise + Tuned Clamping + Final C0/C1"""
    return extend_signal_base(signal, pad_length,
                              apply_transition=True,
                              apply_postprocessing=True,
                              force_ar_path=True, # <-- CHANGED TO TRUE
                              use_ar_in_notrend=True, # This becomes less relevant as trend path is skipped
                              add_ar_noise=True,
                              ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                              ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR,
                              **kwargs)

# Kept for potential reference, but behavior will be same as Proposed_Method_v4 now.
def extend_signal_proposed_v3_legacy(signal, pad_length, **kwargs):
    """Legacy Proposed Method (v3 logic - C0/C1 attempted before smoothing)"""
    print("Warning: Calling 'extend_signal_proposed_v3_legacy' but using v4 base function. Results will be identical to v4 (always AR). For true comparison, implement v3 base separately.")
    return extend_signal_base(signal, pad_length, # 实际调用的是 v4 base
                              apply_transition=True, apply_postprocessing=True,
                              force_ar_path=True, # <-- Reflecting the change in v4 here too for consistency if used
                              use_ar_in_notrend=True,
                              add_ar_noise=True, ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                              ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR, **kwargs)


def extend_signal_no_transition_v4(signal, pad_length, **kwargs):
    """Ablation (v4): No boundary transition smoothing (C0/C1 still enforced last)"""
    # This still uses the base function, which *could* follow the trend path if force_ar_path was False.
    # However, to compare apples-to-apples with the *modified* Proposed_Method_v4,
    # we should ideally also force AR path here. But keeping the original ablation logic for now.
    # If the intent is to compare against the *new* proposed method, all ablations should also have force_ar_path=True.
    # For now, leaving it as is, but note the comparison point has changed.
    return extend_signal_base(signal, pad_length,
                              apply_transition=False, # <-- Ablation target
                              apply_postprocessing=True,
                              force_ar_path=False, # <-- Original Ablation logic (doesn't force AR)
                              use_ar_in_notrend=True,
                              add_ar_noise=True, ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                              ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR, **kwargs)

def extend_signal_no_postprocessing_v4(signal, pad_length, **kwargs):
    """Ablation (v4): No post-processing (median/LPF) (C0/C1 still enforced last)"""
    return extend_signal_base(signal, pad_length,
                              apply_transition=True,
                              apply_postprocessing=False, # <-- Ablation target
                              force_ar_path=False, # <-- Original Ablation logic
                              use_ar_in_notrend=True,
                              add_ar_noise=True, ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                              ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR, **kwargs)

# REMOVED: This function is no longer needed as the experiment is removed.
# def extend_signal_no_trend_handling_v4(signal, pad_length, **kwargs):
#     """Ablation (v4): Force AR path (ignore trend) (C0/C1 still enforced last)"""
#     return extend_signal_base(signal, pad_length, apply_transition=True, apply_postprocessing=True, force_ar_path=True, use_ar_in_notrend=True, add_ar_noise=True, ar_noise_factor=DEFAULT_AR_NOISE_FACTOR, ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR, **kwargs)

def extend_signal_ar_vs_mean_in_notrend_v4(signal, pad_length, **kwargs):
    """Ablation (v4): Use local mean padding instead of AR when no trend (C0/C1 still enforced last)"""
    # This experiment explicitly tests the use_ar_in_notrend=False path when force_ar_path=False.
    # It remains relevant as a comparison.
    return extend_signal_base(signal, pad_length,
                              apply_transition=True,
                              apply_postprocessing=True,
                              force_ar_path=False, # <-- Necessary for this specific ablation
                              use_ar_in_notrend=False, # <-- Ablation target
                              add_ar_noise=True, # These have no effect if AR isn't used
                              ar_noise_factor=DEFAULT_AR_NOISE_FACTOR,
                              ar_clamping_factor=DEFAULT_AR_CLAMPING_FACTOR, **kwargs)


# --- Baseline padding methods (不变) ---
# 注意：这些简单方法不使用 extend_signal_base，因此没有最终的 C0/C1 强制。
def extend_signal_zero_padding(signal, pad_length, **kwargs):
    """对比: 零填充"""
    if pad_length <= 0: return signal.copy()
    extension = np.zeros(pad_length)
    return np.concatenate([signal, extension])

def extend_signal_mean_padding(signal, pad_length, **kwargs):
    """对比: 均值填充 (使用整个信号的均值)"""
    if pad_length <= 0: return signal.copy()
    mean_val = np.mean(signal) if len(signal) > 0 else 0
    extension = np.full(pad_length, mean_val)
    return np.concatenate([signal, extension])

def extend_signal_last_value_padding(signal, pad_length, **kwargs):
    """对比: 末值填充 (最近邻)"""
    if pad_length <= 0: return signal.copy()
    last_val = signal[-1] if len(signal) > 0 else 0
    extension = np.full(pad_length, last_val)
    return np.concatenate([signal, extension])

def extend_signal_symmetric_padding(signal, pad_length, **kwargs):
    """对比: 对称/镜像填充 (使用 numpy.pad)"""
    if pad_length <= 0: return signal.copy()
    effective_pad_width = max(0, int(pad_length))
    try:
        if len(signal) == 0: return np.zeros(effective_pad_width)
        # 使用 reflect_type='even' 实现对称填充
        padded = np.pad(signal, (0, effective_pad_width), mode='symmetric')
        # 确保返回正确长度
        return padded[:len(signal) + pad_length]
    except ValueError as e:
        # 如果信号太短无法对称填充，回退到末值填充
        print(f"Warning: Symmetric padding failed ('{e}'). Falling back to last value padding.")
        return extend_signal_last_value_padding(signal, pad_length)

def extend_signal_periodic_padding(signal, pad_length, **kwargs):
    """对比: 周期填充 (使用 numpy.pad)"""
    if pad_length <= 0: return signal.copy()
    effective_pad_width = max(0, int(pad_length))
    try:
        if len(signal) == 0: return np.zeros(effective_pad_width)
        # mode='wrap' 实现周期填充
        padded = np.pad(signal, (0, effective_pad_width), mode='wrap')
        # 确保返回正确长度
        return padded[:len(signal) + pad_length]
    except ValueError as e:
        # 如果信号太短无法周期填充，回退到末值填充
        print(f"Warning: Periodic padding failed ('{e}'). Falling back to last value padding.")
        return extend_signal_last_value_padding(signal, pad_length)


# --- 度量计算函数 (calculate_boundary_discontinuity, calculate_boundary_wavelet_metrics, calculate_local_stats_ratio) ---
# (这些函数保持不变)
def calculate_boundary_discontinuity(signal, extension):
    """计算 C0 (值跳变) 和 C1 (梯度跳变) 不连续性"""
    metrics = {'c0_jump': np.nan, 'c1_jump': np.nan}
    # 需要信号至少1点，扩展段至少1点才能计算 C0
    if len(signal) < 1 or len(extension) < 1: return metrics # C0 至少需要各 1 点
    # 需要信号至少2点，扩展段至少2点才能计算 C1
    if len(signal) < 2 or len(extension) < 2:
        try:
           metrics['c0_jump'] = np.abs(extension[0] - signal[-1])
        except IndexError: pass
        return metrics # C1 无法计算，但 C0 可能可以计算

    try:
        # C0 跳变: 扩展段第一个点 - 原始信号最后一个点
        c0_jump = np.abs(extension[0] - signal[-1])

        # C1 跳变: 边界两侧的梯度差
        grad_signal_end = signal[-1] - signal[-2]       # 后向差分
        grad_extension_start = extension[1] - extension[0] # 前向差分
        c1_jump = np.abs(grad_extension_start - grad_signal_end)

        metrics['c0_jump'] = c0_jump
        metrics['c1_jump'] = c1_jump
    except IndexError: pass # 以防万一
    except Exception as e: print(f"Error calculating boundary discontinuity: {e}")
    return metrics

def calculate_boundary_wavelet_metrics(padded_signal, original_len, wavelet='db4', level=4, window_size=50):
    """计算边界附近的小波系数最大值和能量"""
    metrics = {'boundary_wavelet_max': np.nan, 'boundary_wavelet_energy': np.nan}
    # 检查输入有效性
    if len(padded_signal) < 10 or original_len <= 0 or original_len >= len(padded_signal): return metrics

    try:
        # 确定合适的小波分解层数
        max_level_possible = pywt.dwt_max_level(len(padded_signal), pywt.Wavelet(wavelet))
        current_level = min(level, max_level_possible)
        if current_level <= 0: return metrics # 无法进行分解

        # 进行小波分解
        coeffs = pywt.wavedec(padded_signal, wavelet=wavelet, level=current_level, mode='symmetric')

        boundary_idx = original_len # 边界在原始信号末尾
        max_coeff_abs = 0.0 # 初始化边界附近最大系数绝对值
        total_energy_boundary = 0.0 # 初始化边界附近总能量
        total_coeffs_count_boundary = 0 # 初始化边界附近系数总数

        # 迭代细节系数层 (从 level 1 到 current_level)
        # coeffs[0] 是近似系数，coeffs[1] 是 level=current_level 的细节系数... coeffs[current_level] 是 level=1 的细节系数
        for i in range(1, current_level + 1):
            detail_coeffs = coeffs[i] # 获取第 i 层的细节系数 (对应 level = current_level - i + 1)
            if detail_coeffs is None or len(detail_coeffs) == 0: continue

            # 计算当前层级下，边界索引和窗口大小对应的缩放后位置
            # 注意：wavedec 返回的系数长度与层级有关，近似为 len(signal) / 2^level
            # 我们需要将原始信号中的 boundary_idx 和 window_size 映射到当前系数层
            # 简单近似：scale_factor = 2**(current_level - i + 1)
            scale_factor = 2**(current_level - i + 1) # 实际层级是 current_level - i + 1
            boundary_idx_scaled = boundary_idx // scale_factor # 边界在当前系数层的大致位置
            window_scaled = max(1, window_size // scale_factor) # 窗口在当前系数层的大小 (至少为1)

            # 确定在当前系数层中检查的窗口范围
            start_idx = max(0, boundary_idx_scaled - window_scaled)
            end_idx = min(len(detail_coeffs), boundary_idx_scaled + window_scaled + 1) # +1 包含右边界

            if start_idx < end_idx: # 确保窗口有效
                boundary_window_coeffs = detail_coeffs[start_idx:end_idx]
                if len(boundary_window_coeffs) > 0:
                     # 更新最大系数绝对值
                     max_coeff_abs = max(max_coeff_abs, np.max(np.abs(boundary_window_coeffs)))
                     # 累加能量 (系数平方和)
                     total_energy_boundary += np.sum(boundary_window_coeffs**2)
                     # 累加系数数量
                     total_coeffs_count_boundary += len(boundary_window_coeffs)

        # 如果在边界附近找到了系数
        if total_coeffs_count_boundary > 0:
             metrics['boundary_wavelet_max'] = max_coeff_abs
             metrics['boundary_wavelet_energy'] = total_energy_boundary
    except Exception as e: print(f"Error calculating wavelet boundary metrics: {e}")
    return metrics


def calculate_local_stats_ratio(signal, extension, window_size=50):
    """计算扩展段开始处方差与信号末端方差的比率 (使用 window_size=50)"""
    metrics = {'local_variance_ratio_50': np.nan} # 重命名以区分 100 点窗口
    # 确定实际使用的窗口大小 (不能超过信号和扩展段的长度)
    win = min(window_size, len(signal), len(extension))
    if win < 2: return metrics # 需要至少2点计算方差

    try:
        # 计算信号末端窗口的方差
        var_signal_end = np.var(signal[-win:])
        # 计算扩展段开始窗口的方差
        var_extension_start = np.var(extension[:win])

        # 处理信号末端方差接近零的情况
        if var_signal_end < 1e-12:
            # 如果信号末端方差为零，扩展段方差也为零，则比率为 1
            # 如果信号末端方差为零，扩展段方差不为零，则比率为无穷大
            metrics['local_variance_ratio_50'] = 1.0 if var_extension_start < 1e-12 else np.inf
        else:
            # 正常计算比率
            metrics['local_variance_ratio_50'] = var_extension_start / var_signal_end
    except Exception as e: print(f"Error calculating local stats ratio (50pt): {e}")
    return metrics

# --- 绘图与分析函数 (修改版：使用全局 LOCAL_METRIC_KEYS) ---
def plot_and_analyze_padding(signal, padded, experiment_name, col_name, results_dir):
    """生成所有可视化图表并计算所有指标 (包括新增的局部窗口对比指标)"""
    original_len = len(signal)
    padded_len = len(padded) # 使用实际填充后的长度

    extension = np.array([])
    pad_length = 0
    if padded_len > original_len:
        extension = padded[original_len:]
        pad_length = len(extension)

    # 初始化指标字典
    metrics = {'experiment': experiment_name, 'column': col_name, 'original_length': original_len, 'padded_length': padded_len}
    orig_wpe_trunc, pad_wpe_trunc = None, None # 用于 WPE 绘图

    # --- 指标计算 ---
    try:
        # 确保使用正确的长度进行计算
        padded_for_metrics = padded[:padded_len]

        # 计算去均值信号
        signal_mean = np.mean(signal) if original_len > 0 else 0
        padded_mean = np.mean(padded_for_metrics) if padded_len > 0 else 0
        signal_demeaned = signal - signal_mean
        padded_demeaned = padded_for_metrics - padded_mean

        # --- 1. 全局指标 ---
        # RMS 差异
        orig_rms = np.sqrt(np.mean(signal_demeaned**2)) if original_len > 1 else 0
        pad_rms = np.sqrt(np.mean(padded_demeaned**2)) if padded_len > 1 else 0
        metrics['global_rms_diff'] = np.abs(pad_rms - orig_rms) / (orig_rms + 1e-10)

        # 波峰因子差异
        peak_orig = np.max(np.abs(signal_demeaned)) if original_len > 0 else 0
        peak_pad = np.max(np.abs(padded_demeaned)) if padded_len > 0 else 0
        crest_orig = peak_orig / (orig_rms + 1e-10) if orig_rms > 1e-10 else 0
        crest_pad = peak_pad / (pad_rms + 1e-10) if pad_rms > 1e-10 else 0
        metrics['global_crest_diff'] = np.abs(crest_pad - crest_orig) / (crest_orig + 1e-10)

        # 每样本能量差异
        orig_energy_per_sample = np.mean(signal**2) if original_len > 0 else 0
        pad_energy_per_sample = np.mean(padded_for_metrics**2) if padded_len > 0 else 0
        metrics['global_energy_diff'] = np.abs(pad_energy_per_sample - orig_energy_per_sample) / (orig_energy_per_sample + 1e-10)

        # 全局频谱指标 (对去均值信号)
        f_orig, pxx_orig = welch_psd(signal_demeaned, fs=1.0)
        f_pad, pxx_pad = welch_psd(padded_demeaned, fs=1.0)

        # 用总功率归一化 PSD
        total_power_orig = np.trapz(pxx_orig, f_orig) if len(f_orig) > 1 else 0
        total_power_pad = np.trapz(pxx_pad, f_pad) if len(f_pad) > 1 else 0
        pxx_orig_norm_by_power = pxx_orig / (total_power_orig + 1e-12)
        pxx_pad_norm_by_power = pxx_pad / (total_power_pad + 1e-12)

        # 频谱质心差异
        centroid_orig = np.sum(f_orig * pxx_orig_norm_by_power) if len(f_orig) > 1 and total_power_orig > 1e-12 else 0
        centroid_pad = np.sum(f_pad * pxx_pad_norm_by_power) if len(f_pad) > 1 and total_power_pad > 1e-12 else 0
        metrics['global_centroid_diff'] = np.abs(centroid_pad - centroid_orig) / (centroid_orig + 1e-10)

        # 频谱 L2 范数差异 (插值后)
        if len(f_orig) > 1 and len(f_pad) > 1 and total_power_orig > 1e-12 and total_power_pad > 1e-12:
            f_common = np.linspace(max(f_orig[0], f_pad[0]), min(f_orig[-1], f_pad[-1]), 512)
            interp_orig = interp1d(f_orig, pxx_orig_norm_by_power, bounds_error=False, fill_value=0, kind='linear')
            interp_pad = interp1d(f_pad, pxx_pad_norm_by_power, bounds_error=False, fill_value=0, kind='linear')
            pxx_orig_interp = interp_orig(f_common)
            pxx_pad_interp = interp_pad(f_common)
            norm_orig_psd = np.linalg.norm(pxx_orig_interp)
            metrics['global_spectrum_diff'] = np.linalg.norm(pxx_orig_interp - pxx_pad_interp) / (norm_orig_psd + 1e-10)
        else:
            metrics['global_spectrum_diff'] = np.nan

        # 全局小波包能量 KL 散度
        orig_wpe = calculate_wavelet_packet_energy(signal_demeaned, level=4) # 使用默认 level 4
        pad_wpe = calculate_wavelet_packet_energy(padded_demeaned, level=4)
        if isinstance(orig_wpe, np.ndarray) and isinstance(pad_wpe, np.ndarray) and len(orig_wpe)>0 and len(pad_wpe)>0:
             min_len_wpe = min(len(orig_wpe), len(pad_wpe))
             orig_wpe_trunc = orig_wpe[:min_len_wpe] # 截取用于绘图和计算
             pad_wpe_trunc = pad_wpe[:min_len_wpe]
             smooth = 1e-10
             if not np.isnan(orig_wpe_trunc).any() and not np.isnan(pad_wpe_trunc).any():
                 orig_wpe_dist = (orig_wpe_trunc + smooth) / np.sum(orig_wpe_trunc + smooth)
                 pad_wpe_dist = (pad_wpe_trunc + smooth) / np.sum(pad_wpe_trunc + smooth)
                 # KL(P || Q) = sum(P * log(P / Q))
                 kl_div = np.sum(np.where(orig_wpe_dist > 0, orig_wpe_dist * np.log(orig_wpe_dist / (pad_wpe_dist + 1e-20)), 0)) # 增加分母平滑
                 metrics['global_kl_divergence'] = kl_div if np.isfinite(kl_div) else np.nan
             else: metrics['global_kl_divergence'] = np.nan
        else: metrics['global_kl_divergence'] = np.nan


        # --- 2. 边界和整体一致性指标 ---
        if pad_length > 0 and len(extension) > 0:
            metrics.update(calculate_boundary_discontinuity(signal, extension)) # C0, C1 jump
            metrics.update(calculate_boundary_wavelet_metrics(padded_for_metrics, original_len, window_size=50)) # WPE max/energy @ boundary (50pt)
            metrics.update(calculate_local_stats_ratio(signal, extension, window_size=50)) # Variance ratio @ boundary (50pt)
        else: # 无填充
            metrics.update({'c0_jump': 0.0, 'c1_jump': 0.0, 'boundary_wavelet_max': 0.0, 'boundary_wavelet_energy': 0.0, 'local_variance_ratio_50': 1.0})


        # --- 3. 新增：局部窗口对比指标 (Local Window Comparison Metrics) ---
        local_window_size = LOCAL_COMPARISON_WINDOW_SIZE # 使用配置中定义的大小
        metrics['local_window_size'] = local_window_size # 记录窗口大小

        # 检查是否有足够的点进行比较
        can_compare_windows = (original_len >= local_window_size and pad_length >= local_window_size and local_window_size > 1)

        # 使用全局定义的 LOCAL_METRIC_KEYS 初始化为 NaN
        for k in LOCAL_METRIC_KEYS:
            metrics[k] = np.nan

        if can_compare_windows:
            signal_window = signal[-local_window_size:]
            extension_window = extension[:local_window_size]

            try:
                # 1. 统计指标对比
                mean_sig_win = np.mean(signal_window)
                mean_ext_win = np.mean(extension_window)
                metrics['local_mean_diff'] = np.abs(mean_ext_win - mean_sig_win)

                signal_win_var = np.var(signal_window)
                ext_win_var = np.var(extension_window)
                metrics['local_variance_ratio_100'] = ext_win_var / (signal_win_var + 1e-10)

                rms_sig_win = np.sqrt(np.mean(signal_window**2))
                rms_ext_win = np.sqrt(np.mean(extension_window**2))
                metrics['local_rms_diff_100'] = np.abs(rms_ext_win - rms_sig_win) / (rms_sig_win + 1e-10)

                # 2. 频谱指标对比 (对去均值后的窗口)
                signal_win_demeaned = signal_window - mean_sig_win
                ext_win_demeaned = extension_window - mean_ext_win

                # 使用较小的 nperseg 适应短窗口
                nperseg_local = max(8, local_window_size // 8) # 减小 nperseg
                f_sig_win, pxx_sig_win = welch_psd(signal_win_demeaned, fs=1.0, nperseg=nperseg_local)
                f_ext_win, pxx_ext_win = welch_psd(ext_win_demeaned, fs=1.0, nperseg=nperseg_local)

                power_sig_win = np.trapz(pxx_sig_win, f_sig_win) if len(f_sig_win) > 1 else 0
                power_ext_win = np.trapz(pxx_ext_win, f_ext_win) if len(f_ext_win) > 1 else 0
                pxx_sig_win_norm = pxx_sig_win / (power_sig_win + 1e-12)
                pxx_ext_win_norm = pxx_ext_win / (power_ext_win + 1e-12)

                centroid_sig_win = np.sum(f_sig_win * pxx_sig_win_norm) if power_sig_win > 1e-12 else 0
                centroid_ext_win = np.sum(f_ext_win * pxx_ext_win_norm) if power_ext_win > 1e-12 else 0
                metrics['local_centroid_diff_100'] = np.abs(centroid_ext_win - centroid_sig_win) / (centroid_sig_win + 1e-10)

                if len(f_sig_win) > 1 and len(f_ext_win) > 1 and power_sig_win > 1e-12 and power_ext_win > 1e-12:
                    f_common_win = np.linspace(max(f_sig_win[0], f_ext_win[0]), min(f_sig_win[-1], f_ext_win[-1]), 128)
                    interp_sig_win = interp1d(f_sig_win, pxx_sig_win_norm, bounds_error=False, fill_value=0, kind='linear')
                    interp_ext_win = interp1d(f_ext_win, pxx_ext_win_norm, bounds_error=False, fill_value=0, kind='linear')
                    pxx_sig_win_interp = interp_sig_win(f_common_win)
                    pxx_ext_win_interp = interp_ext_win(f_common_win)
                    norm_sig_psd_win = np.linalg.norm(pxx_sig_win_interp)
                    metrics['local_spectrum_diff_100'] = np.linalg.norm(pxx_sig_win_interp - pxx_ext_win_interp) / (norm_sig_psd_win + 1e-10)
                # else: local_spectrum_diff_100 保持 NaN

                # 3. 小波包能量对比 (使用较低 level)
                wpe_level_local = 3 # level 降低以适应短窗口 (2^3=8 bands)
                wpe_sig_win = calculate_wavelet_packet_energy(signal_win_demeaned, level=wpe_level_local)
                wpe_ext_win = calculate_wavelet_packet_energy(ext_win_demeaned, level=wpe_level_local)
                if isinstance(wpe_sig_win, np.ndarray) and isinstance(wpe_ext_win, np.ndarray) and len(wpe_sig_win) > 0 and len(wpe_ext_win) > 0:
                     min_len_wpe_win = min(len(wpe_sig_win), len(wpe_ext_win))
                     wpe_sig_win_trunc = wpe_sig_win[:min_len_wpe_win]
                     wpe_ext_win_trunc = wpe_ext_win[:min_len_wpe_win]
                     smooth = 1e-10
                     if not np.isnan(wpe_sig_win_trunc).any() and not np.isnan(wpe_ext_win_trunc).any():
                         wpe_sig_dist = (wpe_sig_win_trunc + smooth) / np.sum(wpe_sig_win_trunc + smooth)
                         wpe_ext_dist = (wpe_ext_win_trunc + smooth) / np.sum(wpe_ext_win_trunc + smooth)
                         kl_div_win = np.sum(np.where(wpe_sig_dist > 0, wpe_sig_dist * np.log(wpe_sig_dist / (wpe_ext_dist + 1e-20)), 0)) # 增加分母平滑
                         metrics['local_kl_divergence_100'] = kl_div_win if np.isfinite(kl_div_win) else np.nan
                # else: local_kl_divergence_100 保持 NaN

            except Exception as e_local_win:
                print(f"Error calculating local window metrics for {col_name} ({experiment_name}): {e_local_win}")
                # 确保所有局部指标在出错时仍为 NaN (已在 try 块外初始化)
        else:
             # print(f"  Skipping local window metrics for {col_name} ({experiment_name}) due to insufficient length.") # 调试信息
             metrics['warning_local_window'] = "Signal or extension too short for {}-point window comparison.".format(local_window_size)


    except Exception as e:
        print(f"Error calculating metrics for {col_name} ({experiment_name}): {e}")
        print(traceback.format_exc())
        # 确保所有指标在出错时都有占位符 (使用全局 LOCAL_METRIC_KEYS)
        metric_keys_all = ['global_rms_diff', 'global_crest_diff', 'global_energy_diff',
                           'global_centroid_diff', 'global_spectrum_diff', 'global_kl_divergence',
                           'c0_jump', 'c1_jump', 'boundary_wavelet_max', 'boundary_wavelet_energy',
                           'local_variance_ratio_50'] + LOCAL_METRIC_KEYS # 使用全局列表
        for k in metric_keys_all:
            if k not in metrics: metrics[k] = np.nan
        metrics['error'] = str(e)

    # --- 绘图 ---
    plot_padded_len = len(padded) # 使用实际填充长度绘图

    fig = plt.figure(figsize=(18, 28), dpi=IMG_DPI) # 增加高度以容纳更多文本
    plt.subplots_adjust(hspace=0.65, top=0.96, bottom=0.04, left=0.1, right=0.9) # 调整间距
    plot_title = f"Padding Analysis: {col_name} - {experiment_name}"
    fig.suptitle(plot_title, fontsize=16, fontweight='bold')

    # 1. 全局对比图
    ax1 = fig.add_subplot(6, 1, 1)
    ax1.plot(np.arange(original_len), signal, 'b', label='Original', alpha=0.7, linewidth=1)
    ax1.plot(np.arange(plot_padded_len), padded, 'r--', label='Padded', alpha=0.7, linewidth=1)
    if original_len < plot_padded_len:
        ax1.axvline(x=original_len, color='k', linestyle=':', linewidth=1.5, label='Padding Start')
    ax1.set_title('Global Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True); ax1.tick_params(labelsize=10); ax1.set_xlabel("Sample Index", fontsize=10); ax1.set_ylabel("Amplitude", fontsize=10)

    # 2. 边界细节图 (保持不变, 显示 100 点左右)
    ax2 = fig.add_subplot(6, 1, 2)
    boundary_window_plot = min(100, original_len // 4 if original_len > 0 else 10, pad_length // 2 if pad_length > 0 else 10)
    boundary_window_plot = max(10, boundary_window_plot)
    start_idx_orig = max(0, original_len - boundary_window_plot)
    end_idx_pad_plot = min(plot_padded_len, original_len + boundary_window_plot)
    plot_range_orig = np.arange(start_idx_orig, original_len)
    plot_range_pad = np.arange(start_idx_orig, end_idx_pad_plot)

    if len(plot_range_orig) > 0: ax2.plot(plot_range_orig, signal[start_idx_orig:], 'b-', label='Original End', linewidth=1.5)
    if pad_length > 0 and len(plot_range_pad) > (original_len - start_idx_orig): ax2.plot(plot_range_pad, padded[start_idx_orig:end_idx_pad_plot], 'r--', label='Padded Section', linewidth=1)
    if original_len < plot_padded_len: ax2.axvline(x=original_len, color='k', linestyle=':', linewidth=1.5)
    ax2.set_title(f'Boundary Detail (~{boundary_window_plot*2} points)', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True); ax2.tick_params(labelsize=10)
    if len(plot_range_pad) > 0 : ax2.set_xlim(plot_range_pad[0], plot_range_pad[-1])
    ax2.set_xlabel("Sample Index", fontsize=10); ax2.set_ylabel("Amplitude", fontsize=10)

    # 3. 归一化功率谱密度图 (全局)
    ax3 = fig.add_subplot(6, 1, 3)
    if len(f_orig) > 1: ax3.semilogy(f_orig, pxx_orig_norm_by_power, 'b', label='Original PSD (Normalized by Power)', linewidth=1)
    if len(f_pad) > 1: ax3.semilogy(f_pad, pxx_pad_norm_by_power, 'r--', label='Padded PSD (Normalized by Power)', linewidth=1)
    ax3.set_title('Global Normalized Power Spectral Density (Welch)', fontsize=12)
    ax3.set_xlabel('Normalized Frequency (fs=1.0)', fontsize=10); ax3.set_ylabel('Normalized Power / Frequency', fontsize=10)
    ax3.legend(fontsize=10); ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    min_psd_val = 1e-9
    valid_psd_values = np.concatenate([pxx_orig_norm_by_power[pxx_orig_norm_by_power > 1e-15], pxx_pad_norm_by_power[pxx_pad_norm_by_power > 1e-15]])
    if len(valid_psd_values) > 0: min_psd_val = max(1e-9, 0.1 * np.min(valid_psd_values))
    ax3.set_ylim(bottom=min_psd_val); ax3.tick_params(labelsize=10)

    # 4. 频谱图 (全局)
    ax4 = fig.add_subplot(6, 1, 4)
    fs = 1.0
    if plot_padded_len > 512: nperseg_spec = 256
    elif plot_padded_len > 128: nperseg_spec = 128
    elif plot_padded_len > 0: nperseg_spec = max(8, plot_padded_len // 8)
    else: nperseg_spec = 8
    nperseg_spec = min(nperseg_spec, plot_padded_len)
    noverlap_spec = nperseg_spec // 2

    if plot_padded_len > nperseg_spec and nperseg_spec > 0 :
        try:
            f_spec, t_spec, Sxx_spec = spectrogram(padded, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, window='hann')
            Sxx_spec_clipped = np.maximum(Sxx_spec, 1e-10) # 避免 log(0)
            vmax_spec = np.percentile(10*np.log10(Sxx_spec_clipped[Sxx_spec_clipped > 1e-15]), 99.9) if np.any(Sxx_spec_clipped > 1e-15) else 0
            vmin_spec = max(vmax_spec - 60, np.min(10*np.log10(Sxx_spec_clipped[Sxx_spec_clipped > 1e-15])) if np.any(Sxx_spec_clipped > 1e-15) else -80)

            im = ax4.pcolormesh(t_spec * fs, f_spec, 10 * np.log10(Sxx_spec_clipped), shading='gouraud', cmap='viridis', vmin=vmin_spec, vmax=vmax_spec) # t_spec scaled by fs
            fig.colorbar(im, ax=ax4, label='Power/Frequency (dB/Hz)')
            if original_len < plot_padded_len: ax4.axvline(x=original_len, color='w', linestyle=':', linewidth=1.5, label='Padding Start'); ax4.legend(fontsize=10, loc='upper right') # x in samples
            ax4.set_title('Global Spectrogram', fontsize=12); ax4.set_xlabel('Time (samples)', fontsize=10); ax4.set_ylabel('Normalized Frequency', fontsize=10)
            ax4.tick_params(labelsize=10); ax4.set_ylim(bottom=0)
        except Exception as e:
            ax4.text(0.5, 0.5, f"Spectrogram failed: {e}", ha='center', va='center', color='red', transform=ax4.transAxes)
            print(f"Spectrogram failed for {col_name} ({experiment_name}): {e}")
    else:
        ax4.text(0.5, 0.5, "Signal too short for spectrogram", ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Global Spectrogram', fontsize=12); ax4.set_xlabel('Time (samples)', fontsize=10); ax4.set_ylabel('Normalized Frequency', fontsize=10)

    # 5. 小波包能量对比图 (全局)
    ax5 = fig.add_subplot(6, 1, 5)
    plot_wpe = False
    if isinstance(orig_wpe_trunc, np.ndarray) and isinstance(pad_wpe_trunc, np.ndarray) and len(orig_wpe_trunc) > 0:
        plot_wpe = True

    if plot_wpe:
        num_bands = len(orig_wpe_trunc)
        freq_bands = np.arange(num_bands)
        bar_width = 0.35
        smooth = 1e-10
        orig_wpe_dist_plot = (orig_wpe_trunc + smooth) / np.sum(orig_wpe_trunc + smooth)
        pad_wpe_dist_plot = (pad_wpe_trunc + smooth) / np.sum(pad_wpe_trunc + smooth)

        ax5.bar(freq_bands - bar_width/2, orig_wpe_dist_plot, bar_width, label='Original WPE Dist.', color='blue', alpha=0.7)
        ax5.bar(freq_bands + bar_width/2, pad_wpe_dist_plot, bar_width, label='Padded WPE Dist.', color='red', alpha=0.7)
        ax5.set_title(f'Global Wavelet Packet Band Energy Distribution (Level {len(orig_wpe).bit_length()-1 if orig_wpe is not None else "N/A"})', fontsize=12)
        ax5.set_xlabel('Frequency Band Index (Low to High)', fontsize=10); ax5.set_ylabel('Normalized Energy', fontsize=10)
        ax5.legend(fontsize=10); ax5.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax5.tick_params(labelsize=10); ax5.set_xticks(freq_bands)
        if num_bands > 20: ax5.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    else:
        ax5.text(0.5, 0.5, "Global WPE calculation failed or skipped", ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Global Wavelet Packet Band Energy Comparison', fontsize=12); ax5.set_xlabel('Frequency Band Index', fontsize=10); ax5.set_ylabel('Normalized Energy', fontsize=10)


    # 6. 量化指标文本框 (更新结构)
    ax6 = fig.add_subplot(6, 1, 6)
    ax6.axis('off')

    # 构建局部窗口指标文本
    local_metrics_text = f"\nLocal Window ({metrics.get('local_window_size', 'N/A')}pt) Comparison Metrics:\n"
    if 'warning_local_window' in metrics:
         local_metrics_text += f"  ({metrics['warning_local_window']})\n"
    elif can_compare_windows: # Check again if comparison was possible
        # 使用全局 LOCAL_METRIC_KEYS 中的键名
        local_metrics_text += (
            f"  Mean Diff (abs):          {metrics.get(LOCAL_METRIC_KEYS[0], np.nan):.4f}\n"  # local_mean_diff
            f"  Variance Ratio:         {metrics.get(LOCAL_METRIC_KEYS[1], np.nan):.4f}\n"  # local_variance_ratio_100
            f"  RMS Diff Ratio:           {metrics.get(LOCAL_METRIC_KEYS[2], np.nan):.4f}\n"  # local_rms_diff_100
            f"  PSD Centroid Diff Ratio:  {metrics.get(LOCAL_METRIC_KEYS[3], np.nan):.4f}\n"  # local_centroid_diff_100
            f"  PSD L2 Norm Diff Ratio:   {metrics.get(LOCAL_METRIC_KEYS[4], np.nan):.4f}\n"  # local_spectrum_diff_100
            f"  WPE KL Divergence (L{wpe_level_local if 'wpe_level_local' in locals() else '3'}): {metrics.get(LOCAL_METRIC_KEYS[5], np.nan):.4f}" # local_kl_divergence_100
        )
    else: # Should not happen if warning logic is correct, but as fallback
        local_metrics_text += "  (Skipped or Error)\n"


    # 构建完整文本
    metrics_text = (
        f"--- Global Metrics ---\n"
        f"  RMS Diff Ratio:           {metrics.get('global_rms_diff', np.nan):.4f}\n"
        f"  Crest Factor Diff Ratio:  {metrics.get('global_crest_diff', np.nan):.4f}\n"
        f"  Energy/Sample Diff Ratio: {metrics.get('global_energy_diff', np.nan):.4f}\n"
        f"  PSD Centroid Diff Ratio:  {metrics.get('global_centroid_diff', np.nan):.4f}\n"
        f"  PSD L2 Norm Diff Ratio:   {metrics.get('global_spectrum_diff', np.nan):.4f}\n"
        f"  WPE KL Divergence:        {metrics.get('global_kl_divergence', np.nan):.4f}\n\n"
        f"--- Boundary & Overall Consistency Metrics ---\n"
        f"  C0 Jump (Value):          {metrics.get('c0_jump', np.nan):.4g}  <- Should be ~0\n"
        f"  C1 Jump (Gradient):       {metrics.get('c1_jump', np.nan):.4g}  <- Should be ~0\n"
        f"  Wavelet Boundary Max Coeff: {metrics.get('boundary_wavelet_max', np.nan):.4g}\n"
        f"  Wavelet Boundary Energy:    {metrics.get('boundary_wavelet_energy', np.nan):.4g}\n"
        f"  Local Variance Ratio (50pt): {metrics.get('local_variance_ratio_50', np.nan):.4f}"
        f"{local_metrics_text}" # 添加新的局部窗口指标文本
    )
    if 'error' in metrics: metrics_text += f"\n\n  *** Error during processing: {metrics['error']} ***"

    ax6.text(0.05, 0.98, metrics_text, ha='left', va='top', fontsize=8.5, # 稍减小字号
             family='monospace', linespacing=1.6, # 增加行距
             bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.8))

    # 保存图像
    # Use updated folder name in filename path
    filename = os.path.join(results_dir, f"Analysis_{col_name}_{experiment_name}.png")
    try:
        fig.savefig(filename, dpi=IMG_DPI, bbox_inches='tight', format='png')
    except Exception as e: print(f"Error saving plot {filename}: {e}")
    plt.close(fig) # 关闭图形释放内存

    return metrics


# --- 主实验运行函数 (更新实验字典, 调用修改后的分析函数) ---
def run_ablation_experiments(df, exclude_cols=None, verbose=True):
    """处理 DataFrame 中的信号列，运行所有 v4 (最终C0C1强制, Proposed now always AR) 的消融实验和对比实验，并收集结果。
       调用包含局部窗口指标的分析函数。
       同时保存 Proposed_Method_v4 (always AR) 填充后的数据。"""
    exclude_cols = exclude_cols or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_process = [col for col in numeric_cols if col not in exclude_cols]

    if not cols_to_process: print("No numeric columns found to process."); return pd.DataFrame(), None

    max_len = 0; valid_cols = []
    for col in cols_to_process:
         signal_data = df[col].dropna()
         if len(signal_data) >= 10: # 保留最小长度要求
             max_len = max(max_len, len(signal_data)); valid_cols.append(col)
         else: print(f"Skipping column '{col}' due to insufficient data ({len(signal_data)} < 10).")
    cols_to_process = valid_cols

    if not cols_to_process: print("No valid columns with sufficient data found."); return pd.DataFrame(), None

    # 目标长度计算：找到大于等于 max_len 的最小 2 的幂次方
    target_length = 2**int(np.ceil(np.log2(max_len))) if max_len > 0 else 512
    print(f"Max original length: {max_len}, Target length for padding: {target_length}")

    all_metrics_results = []
    proposed_padded_data = {} # 存储新提出的 v4 方法的填充数据

    # --- 实验字典 (v4 - 最终 C0C1 强制, Proposed=Always AR, No_Trend Ablation REMOVED) ---
    experiments = {
        "Proposed_Method_v4": extend_signal_proposed_v4, # Now always uses AR path
        # "Proposed_Method_v3_Legacy": extend_signal_proposed_v3_legacy, # Keep commented unless truly needed
        "Ablation_No_Transition_v4": extend_signal_no_transition_v4,
        "Ablation_No_Postprocessing_v4": extend_signal_no_postprocessing_v4,
        # "Ablation_No_Trend_Handling_v4": extend_signal_no_trend_handling_v4, # <-- REMOVED
        "Ablation_AR_vs_Mean_NoTrend_v4": extend_signal_ar_vs_mean_in_notrend_v4,
        "Compare_Zero_Padding": extend_signal_zero_padding,
        "Compare_Mean_Padding": extend_signal_mean_padding,
        "Compare_LastValue_Padding": extend_signal_last_value_padding,
        "Compare_Symmetric_Padding": extend_signal_symmetric_padding,
        "Compare_Periodic_Padding": extend_signal_periodic_padding,
    }

    num_cols = len(cols_to_process)
    for i, col in enumerate(cols_to_process):
        if verbose: print(f"\n--- Processing Column: {col} ({i+1}/{num_cols}) ---")
        signal = df[col].dropna().values
        original_len = len(signal)

        if original_len == 0:
             if verbose: print(f"  Skipping column '{col}' as it contains no valid data after dropna.")
             continue

        pad_length = target_length - original_len
        current_target_length = target_length

        if pad_length < 0 :
            pad_length = 0; current_target_length = original_len
            if verbose: print(f"  Signal length ({original_len}) >= target ({target_length}). No padding needed for analysis.")
        elif pad_length == 0:
            if verbose: print(f"  Signal length ({original_len}) == target ({target_length}). No padding needed for analysis.")
        else:
            if verbose: print(f"  Original length: {original_len}, Target length: {current_target_length}, Pad length: {pad_length}")


        for exp_name, extend_func in experiments.items():
            if verbose: print(f"  Running Experiment: {exp_name}...")
            try:
                padded_signal = signal.copy() # 默认情况 (无填充或函数内部处理)
                if pad_length > 0 or exp_name in ["Compare_Symmetric_Padding", "Compare_Periodic_Padding"]:
                    # 对需要填充或可能改变长度的基线方法调用函数
                    # 注意: np.pad 基线即使 pad_length=0 也可能因模式返回不同长度，所以调用它们
                    # 其他函数在 pad_length=0 时会直接返回原信号副本
                    padded_signal = extend_func(signal.copy(), pad_length) # 传递副本

                if not isinstance(padded_signal, np.ndarray):
                     error_msg = f"Function {exp_name} returned {type(padded_signal)}, expected numpy array."
                     print(f"  Error: {error_msg}")
                     metrics = {'experiment': exp_name, 'column': col, 'error': error_msg, 'original_length': original_len, 'padded_length': np.nan}
                     # 添加 NaN 指标占位符 (使用全局 LOCAL_METRIC_KEYS)
                     metric_keys_all = ['global_rms_diff', 'global_crest_diff', 'global_energy_diff',
                                        'global_centroid_diff', 'global_spectrum_diff', 'global_kl_divergence',
                                        'c0_jump', 'c1_jump', 'boundary_wavelet_max', 'boundary_wavelet_energy',
                                        'local_variance_ratio_50'] + LOCAL_METRIC_KEYS # 使用全局列表
                     for k in metric_keys_all: metrics[k] = np.nan
                     all_metrics_results.append(metrics)
                     continue

                # 检查最终长度是否符合预期 (对于需要填充的情况)
                # 对于不需要填充的情况 (pad_length <= 0)，预期长度是 original_len
                expected_final_len = current_target_length if pad_length > 0 else original_len
                if len(padded_signal) != expected_final_len:
                    print(f"  Warning: Final padded length mismatch for {col} ({exp_name}). Expected {expected_final_len}, got {len(padded_signal)}. Adjusting before analysis.")
                    # 尝试修复长度
                    if len(padded_signal) > expected_final_len:
                        padded_signal = padded_signal[:expected_final_len]
                    else:
                        diff = expected_final_len - len(padded_signal)
                        last_val = padded_signal[-1] if len(padded_signal) > 0 else (signal[-1] if original_len > 0 else 0)
                        padding_values = np.full(diff, last_val)
                        padded_signal = np.concatenate([padded_signal, padding_values])
                    # 调用分析函数 (传递修复后的信号)
                    # Use updated results_dir path
                    metrics = plot_and_analyze_padding(signal, padded_signal, exp_name, col, results_dir)
                    metrics['warning_length_fixed'] = f'Length mismatch fixed: was {len(padded_signal)} initially, expected {expected_final_len}' # 记录警告
                else:
                    # 长度正确，直接分析
                    # Use updated results_dir path
                    metrics = plot_and_analyze_padding(signal, padded_signal, exp_name, col, results_dir)

                all_metrics_results.append(metrics)

                # 只保存新提出的 v4 (always AR) 方法的填充数据 (确保长度正确)
                if exp_name == "Proposed_Method_v4" and isinstance(padded_signal, np.ndarray) and len(padded_signal) == expected_final_len:
                    # 使用 expected_final_len 创建索引，以处理 pad_length=0 的情况
                    index_range = pd.RangeIndex(start=0, stop=expected_final_len, step=1)
                    proposed_padded_data[col] = pd.Series(padded_signal, index=index_range)

            except Exception as e:
                error_msg = f"Unhandled error processing {col} with {exp_name}: {e}"
                print(f"  Error: {error_msg}")
                print(traceback.format_exc())
                metrics = {'experiment': exp_name, 'column': col, 'error': error_msg, 'original_length': original_len, 'padded_length': np.nan}
                # 使用全局 LOCAL_METRIC_KEYS
                metric_keys_all = ['global_rms_diff', 'global_crest_diff', 'global_energy_diff',
                                   'global_centroid_diff', 'global_spectrum_diff', 'global_kl_divergence',
                                   'c0_jump', 'c1_jump', 'boundary_wavelet_max', 'boundary_wavelet_energy',
                                   'local_variance_ratio_50'] + LOCAL_METRIC_KEYS # 使用全局列表
                for k in metric_keys_all: metrics[k] = np.nan
                all_metrics_results.append(metrics)

    results_df = pd.DataFrame(all_metrics_results)

    # 创建填充数据的 DataFrame
    padded_data_df = None # 初始化为 None
    if proposed_padded_data:
        # 确定填充数据的最终索引 (基于 target_length 或 max_original_len if no padding was needed)
        final_index_len = target_length if max_len < target_length else max_len
        final_index = pd.RangeIndex(start=0, stop=final_index_len, step=1)
        padded_data_df = pd.DataFrame(index=final_index)
        for col_name, series_data in proposed_padded_data.items():
             # 确保所有序列都对齐到最终的 DataFrame 索引
             padded_data_df[col_name] = series_data.reindex(padded_data_df.index)
    # else: padded_data_df 保持 None

    return results_df, padded_data_df


# --- 主执行部分 (更新文件名和报告) ---
def main():
    """主函数 (v4 - Proposed Always AR + 最终 C0C1 强制 + 局部窗口指标)"""
    # 构建完整文件路径
    file_path = os.path.join(output_dir, "voyage_Galveston_to_South_Korea_2022-09-06_2022-11-04.csv")

    # 检查文件是否存在，若不存在则生成样本数据
    if not os.path.exists(file_path):
        print(f"'{file_path}' not found. Generating sample data...")
        # [ 样本数据生成代码 ... ] - Keeping it the same
        data = {'timestamp': pd.date_range(start='2023-01-01', periods=1200, freq='H')}
        time = np.linspace(0, 100, 1200)
        data['signal_smooth'] = np.sin(time * 0.5) + np.random.normal(0, 0.1, 1200)
        data['signal_trend'] = 0.01 * time + np.cos(time * 2) + np.random.normal(0, 0.2, 1200)
        data['signal_noisy'] = np.sin(time * 5) + np.random.normal(0, 0.5, 1200)
        data['signal_short'] = data['signal_smooth'][:30].copy() # Short signal < 100
        data['signal_medium'] = data['signal_trend'][150:350].copy() # Medium length signal
        data['signal_piecewise'] = np.concatenate([np.zeros(600), np.ones(600)*2]) + np.random.normal(0, 0.1, 1200)
        stable_signal = np.sin(time[:1000] * 0.2) + np.random.normal(0, 0.05, 1000)
        stable_end_value = stable_signal[-1]
        stable_end = np.full(200, stable_end_value) + np.random.normal(0, 1e-7, 200) # Stable end
        data['signal_stable_end'] = np.concatenate([stable_signal, stable_end])
        # Add signal with abrupt end
        abrupt_signal = np.sin(time[:850] * 0.8) * (1 + 0.001 * time[:850]) + np.random.normal(0, 0.1, 850)
        data['signal_abrupt'] = abrupt_signal # Ends abruptly
        # Add more generic signals
        num_generic = 12 - len(data.keys()) # Adjust number if needed
        for i in range(num_generic):
            freq = np.random.uniform(0.1, 2.0)
            amp = np.random.uniform(0.5, 1.5)
            noise_level = np.random.uniform(0.05, 0.2)
            data[f'signal_generic_{i+len(data.keys())}'] = amp * np.sin(time * freq) + np.random.normal(0, noise_level, 1200)

        df_gen = pd.DataFrame(data)
        print("Introducing NaNs at the end of some generated signals...")
        np.random.seed(42) # for reproducibility
        for col in df_gen.columns:
             if 'timestamp' in df_gen.columns and col == 'timestamp': continue
             if col.startswith('signal') and col not in ['signal_short', 'signal_stable_end', 'signal_abrupt', 'signal_medium']:
                 # Add NaNs to some longer signals
                 if np.random.rand() > 0.4: # Increase chance of NaNs
                     cutoff = np.random.randint(len(df_gen)//2, len(df_gen)-50) # Ensure some data remains
                     df_gen.loc[cutoff:, col] = np.nan
                     print(f"  -> Added NaNs to '{col}' from index {cutoff}")
        try:
            df_gen.to_csv(file_path, index=False)
            print(f"Sample data saved to '{file_path}'")
        except Exception as e:
            print(f"Error saving generated sample data: {e}"); return
        # --- 样本数据生成结束 ---

    # 读取数据
    print(f"\nReading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError: print(f"Error: File not found at {file_path}. Please check the path."); return
    except Exception as e: print(f"Error reading CSV file: {e}"); return

    # 识别时间戳列
    timestamp_col = None
    if not df.empty:
        first_col_name = df.columns[0]
        is_likely_timestamp = False
        try:
            # 尝试转换为日期时间
            pd.to_datetime(df[first_col_name], errors='raise')
            is_likely_timestamp = True
        except (ValueError, TypeError):
             # 如果不是日期时间，检查是否为非数字（可能是ID）
             if not pd.api.types.is_numeric_dtype(df[first_col_name]):
                 is_likely_timestamp = True # 也排除非数字ID列
        except Exception:
             pass # 其他转换错误

        if is_likely_timestamp:
            timestamp_col = first_col_name
            print(f"Assuming '{timestamp_col}' is the timestamp/index column and excluding it from processing.")
        else:
             print(f"First column '{first_col_name}' is likely numeric data. Including it in processing.")

    exclude_list = [timestamp_col] if timestamp_col is not None else []

    # 使用全局定义的 v4 + local metrics + always AR 文件夹和文件名
    print(f"\nRunning Finalized (v4, Proposed=Always AR + Local Metrics) ablation & comparison experiments...")
    # Use updated results_dir path
    print(f"Results will be saved in: '{results_dir}'")

    # --- 运行实验并获取结果 ---
    metrics_df, padded_data_df = run_ablation_experiments(df, exclude_cols=exclude_list, verbose=True)

    # --- 保存指标摘要 ---
    if metrics_df is not None and not metrics_df.empty:
        # Use updated results_dir path and filename
        metrics_filename = os.path.join(results_dir, "final_metrics_summary_v4_always_AR_local_metrics.csv")
        try:
            # 导出前排序，方便查看
            metrics_df_sorted = metrics_df.sort_values(by=['column', 'experiment'])
            metrics_df_sorted.to_csv(metrics_filename, index=False, float_format='%.6g') # 使用通用格式保存
            print(f"\nMetrics summary saved to: {metrics_filename}")
        except Exception as e: print(f"\nError saving metrics summary: {e}")

        # --- 打印摘要统计 ---
        print("\n=== Final Experiment Summary (v4 - Proposed Always AR + Final C0C1 + Local Metrics) ===")
        processed_cols = metrics_df['column'].unique()
        print(f"Processed {len(processed_cols)} columns: {', '.join(processed_cols)}")

        # 定义期望的打印顺序 (v4 实验 - REMOVED No_Trend_Handling)
        experiments_definition_v4_updated = {
            "Proposed_Method_v4": None, # Now Always AR
            "Ablation_No_Transition_v4": None,
            "Ablation_No_Postprocessing_v4": None,
            # "Ablation_No_Trend_Handling_v4": None, # Removed
            "Ablation_AR_vs_Mean_NoTrend_v4": None,
            "Compare_Zero_Padding": None, "Compare_Mean_Padding": None,
            "Compare_LastValue_Padding": None, "Compare_Symmetric_Padding": None,
            "Compare_Periodic_Padding": None,
        }
        experiments_run_count = len(metrics_df['experiment'].unique())
        expected_exp_count = len(experiments_definition_v4_updated)
        print(f"Ran {experiments_run_count} experiments per column (expected: {expected_exp_count}).")
        print(f"Total results collected: {len(metrics_df)}")

        print("\nAverage Metrics per Experiment (excluding NaNs):")
        try:
             # 选择所有数值类型的列进行平均，排除长度和窗口大小
             numeric_cols_for_avg = metrics_df.select_dtypes(include=np.number).columns
             numeric_cols_for_avg = [col for col in numeric_cols_for_avg if col not in ['original_length', 'padded_length', 'local_window_size']]

             # 计算平均值 (Pandas >= 1.5 默认 numeric_only=False, < 1.5 可能需要指定)
             try:
                  avg_metrics = metrics_df.groupby('experiment')[numeric_cols_for_avg].mean()
             except TypeError: # 处理旧版 pandas 可能出现的 TypeError
                  avg_metrics = metrics_df.groupby('experiment')[numeric_cols_for_avg].mean(numeric_only=True)


             # 重新索引以控制打印顺序 (using updated definition)
             experiments_dict_keys_v4_upd = list(experiments_definition_v4_updated.keys())
             desired_order_v4_upd = [exp for exp in experiments_dict_keys_v4_upd if exp.startswith("Proposed")] + \
                                    [exp for exp in experiments_dict_keys_v4_upd if exp.startswith("Ablation")] + \
                                    [exp for exp in experiments_dict_keys_v4_upd if exp.startswith("Compare")]
             # 确保只包含实际运行的实验
             avg_metrics = avg_metrics.reindex([idx for idx in desired_order_v4_upd if idx in avg_metrics.index], axis=0)

             # 格式化输出，提高可读性
             pd.set_option('display.float_format', '{:.4g}'.format) # 通用格式
             pd.set_option('display.max_columns', None) # 显示所有列
             pd.set_option('display.width', 1000) # 增加显示宽度
             print(avg_metrics.round(5).to_markdown(numalign="left", stralign="left"))
             pd.reset_option('display.float_format')
             pd.reset_option('display.max_columns')
             pd.reset_option('display.width')

        except KeyError as e: print(f"Could not calculate average metrics due to missing column: {e}")
        except Exception as e: print(f"Could not calculate or sort average metrics: {e}"); print(traceback.format_exc())
    else:
        print("\nNo metrics were generated.")

    # --- 保存 Proposed Method (v4 - Always AR) 的填充数据 ---
    if padded_data_df is not None and not padded_data_df.empty:
        # Use updated results_dir path and PADDED_DATA_FILENAME
        padded_data_filepath = os.path.join(results_dir, PADDED_DATA_FILENAME)
        try:
            padded_data_df.to_csv(padded_data_filepath, index=True, index_label='sample_index', float_format='%.8f') # 保存更高精度
            print(f"\nPadded data using Proposed_Method_v4 (always AR, final C0C1 + local metrics) saved to: {padded_data_filepath}")
        except Exception as e: print(f"\nError saving padded data: {e}")
    else:
        print("\nNo padded data was generated by the Proposed_Method_v4 to save (or an error occurred).")


    print("\nProcessing finished.")

# --- 入口点 ---
if __name__ == "__main__":
    # 设置 NumPy 随机种子以提高可复现性 (主要影响 AR 噪声和样本数据生成)
    np.random.seed(2024)
    main()