import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqAdaptiveDecomp(nn.Module):
    def __init__(self, seq_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels

        # 频域注意力参数
        self.freq_attn = nn.Parameter(torch.ones(channels, seq_len // 2 + 1))

        # 改进的动态窗口生成网络
        self.window_mlp = nn.Sequential(
            nn.Linear(seq_len // 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 注册汉宁窗
        self.register_buffer('hanning', torch.hann_window(seq_len))

    def get_dynamic_window(self, x):
        """生成各通道的动态窗口大小"""
        B, L, C = x.shape
        # 应用汉宁窗
        x_windowed = x * self.hanning.view(1, -1, 1)

        # 计算频谱
        x_fft = torch.fft.rfft(x_windowed, dim=1)  # [B, Freq, C]
        # 正确处理维度顺序
        mag = torch.abs(x_fft).permute(1, 2, 0).mean(dim=-1)  # [Freq, C]

        # 批量处理所有通道
        window_ratios = self.window_mlp(mag.t())  # [C, 1]
        window_sizes = window_ratios * self.seq_len

        return torch.clamp(window_sizes, 10, self.seq_len // 2).int()

    def adaptive_ma(self, x, window_sizes):
        """改进的自适应移动平均实现"""
        B, L, C = x.shape
        trend = []

        # 批量处理所有样本
        x_reshaped = x.transpose(1, 2).contiguous()  # [B, C, L]

        for c in range(C):
            window = window_sizes[c].item()
            if window % 2 == 0:  # 确保窗口大小为奇数
                window = window + 1
            pad = window // 2  # 使用整除确保填充正确

            # 使用反射填充以更好地处理边界
            padded = F.pad(x_reshaped[:, c:c + 1], (pad, pad), mode='reflect')
            kernel = torch.ones(1, 1, window, device=x.device) / window
            # 使用卷积计算移动平均
            channel_trend = F.conv1d(padded, kernel)

            # 确保输出长度与输入相同
            assert channel_trend.shape[-1] == L, f"Expected length {L}, got {channel_trend.shape[-1]}"
            trend.append(channel_trend)

        trend = torch.cat(trend, dim=1)  # [B, C, L]
        return trend.transpose(1, 2)  # [B, L, C]

    def forward(self, x):
        # 获取动态窗口大小
        window_sizes = self.get_dynamic_window(x)

        # 提取趋势
        trend = self.adaptive_ma(x, window_sizes)

        # 提取和增强季节性
        residual = x - trend
        x_fft = torch.fft.rfft(residual, dim=1)

        # 应用频域注意力
        freq_attn = self.freq_attn.t().unsqueeze(0)  # [1, Freq, C]
        x_fft = x_fft * freq_attn

        # 反变换得到增强后的季节性
        seasonal = torch.fft.irfft(x_fft, n=self.seq_len, dim=1)

        return seasonal, trend


class Model(nn.Module):
    """
    改进的分解线性模型
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        # 使用改进的分解模块
        self.decomp = FreqAdaptiveDecomp(configs.seq_len, configs.enc_in)

        # 预测头
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal, trend = self.decomp(x)

        # 调整维度用于线性层
        seasonal = seasonal.permute(0, 2, 1)  # [B, C, L]
        trend = trend.permute(0, 2, 1)  # [B, C, L]

        # 各分量预测
        if self.individual:
            seasonal_output = torch.zeros([seasonal.size(0), self.channels, self.pred_len],
                                          dtype=seasonal.dtype, device=seasonal.device)
            trend_output = torch.zeros([trend.size(0), self.channels, self.pred_len],
                                       dtype=trend.dtype, device=trend.device)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal)
            trend_output = self.Linear_Trend(trend)

        # 合并结果并调整维度 [B, pred_len, C]
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)