import torch
import torch.nn as nn
import time

class CHROM(nn.Module):
    def __init__(self, fs: float = 30.0, low_cut: float = 0.7, high_cut: float = 2.5):
        """
        Args:
            fs:        采样率（帧率），默认 30 FPS
            low_cut:   带通滤波下限 (Hz)
            high_cut:  带通滤波上限 (Hz)
        """
        super().__init__()
        self.fs = fs
        self.low = low_cut
        self.high = high_cut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (B, H, W, T, 3)
        Returns:
            Tensor, shape (B, T) — 每帧的 rPPG 信号
        """
        start_time = time.localtime()
        B, H, W, T, C = x.shape
        assert C == 3, "input must have 3 channels"

        # 1) 空间平均：在 H×W 上对每一帧取平均
        #    x_mean: (B, T, 3)
        x_mean = x.mean(dim=1).mean(dim=1)  # 先 H 再 W

        # 2) 归一化：对每一帧的三个通道做能量归一化
        #    sum_rgb: (B, T, 1)
        sum_rgb = x_mean.sum(dim=2, keepdim=True) + 1e-8
        Xn = x_mean / sum_rgb

        R = Xn[:, :, 0]
        G = Xn[:, :, 1]
        Bc = Xn[:, :, 2]

        # 3) 计算投影
        #    X_ = 3R - 2G
        #    Y_ = 1.5R + G - 1.5B
        X_ = 3 * R - 2 * G
        Y_ = 1.5 * R + G - 1.5 * Bc

        # 4) 计算 alpha 并合成
        #    alpha = std(X_) / std(Y_)
        #    s = X_ - alpha * Y_
        # 注意维度：std 沿时间维度
        eps = 1e-8
        std_X = torch.std(X_, dim=1, keepdim=True)
        std_Y = torch.std(Y_, dim=1, keepdim=True) + eps
        alpha = std_X / std_Y
        s = X_ - alpha * Y_  # (B, T)

        # 5) 带通滤波（FFT 实现）
        #    对每个 batch 的信号做 rFFT，滤掉 < low 和 > high 的频率分量
        #    返回时做 irFFT
        # freq: (T//2+1,)
        freq = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(s.device)  # 采样间隔 = 1/fs
        # 构造掩码
        mask = (freq >= self.low) & (freq <= self.high)  # (T//2+1,)
        # 对每个样本做滤波
        S = torch.fft.rfft(s, dim=1)  # (B, T//2+1)
        S_filtered = S * mask.unsqueeze(0)
        s_bp = torch.fft.irfft(S_filtered, n=T, dim=1)  # (B, T)


        return s_bp


# 封装成一个“模型”，直接调用 my_model(data) 即可：
def build_rPPG_model(fs=30.0, low_cut=0.7, high_cut=2.5):
    """
    返回一个可直接用于提取 rPPG 波形的模型：
        my_model = build_rPPG_model()
        wave_pr  = my_model(video_tensor)
    """
    return CHROM(fs=fs, low_cut=low_cut, high_cut=high_cut)