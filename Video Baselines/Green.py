import torch
import torch.nn as nn


class Green(nn.Module):
    """
    简易 rPPG – Green 通道算法
    输入:  (B, 64, 64, 256, 3)
    输出:  (B, 256)
    """

    def __init__(self, fs: float = 30.0, low_cut: float = 0.7, high_cut: float = 2.5):
        """
        Args:
            fs:       采样率 (frames per second)
            low_cut:  带通下限 (Hz)
            high_cut: 带通上限 (Hz)
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
            Tensor, shape (B, T)
        """
        B, H, W, T, C = x.shape
        assert C == 3, "expect last dim = 3 (RGB)"

        # 1) 取 Green 通道 & 空间平均  -> (B, T)
        g = x[..., 1]                      # (B, H, W, T)
        g_mean = g.mean(dim=1).mean(dim=1) # (B, T)

        # 2) 去均值 / 归一化（可选，减少渐变光照的影响）
        g_mean = g_mean - g_mean.mean(dim=1, keepdim=True)

        # 3) 频域带通滤波
        freq = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(g_mean.device)
        mask = (freq >= self.low) & (freq <= self.high)

        G = torch.fft.rfft(g_mean, dim=1)   # (B, T//2+1)
        Gf = G * mask.unsqueeze(0)          # 仅保留目标频率
        g_bp = torch.fft.irfft(Gf, n=T, dim=1)

        return g_bp


def build_rPPG_model(fs=30.0, low_cut=0.7, high_cut=2.5):
    """
    返回一个基于 Green 通道的 rPPG 提取模型：
        my_model = build_rPPG_Green_model()
        wave_pr  = my_model(video_tensor)
    """
    return Green(fs=fs, low_cut=low_cut, high_cut=high_cut)

