import torch
import torch.nn as nn

class POS(nn.Module):
    """
    rPPG —— POS 算法实现
    输入： x (B, H, W, T, 3)
    输出： s_bp (B, T)
    """
    def __init__(self, fs: float = 30.0, low_cut: float = 0.7, high_cut: float = 2.5):
        """
        Args:
            fs:       采样率（帧率），默认 30 FPS
            low_cut:  带通滤波下限 (Hz)
            high_cut: 带通滤波上限 (Hz)
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
        assert C == 3, "输入最后一维必须是 RGB 三通道"

        # 1) 空间平均：对 H×W 求均值，得到 (B, T, 3)
        x_mean = x.mean(dim=1).mean(dim=1)  # (B, T, 3)

        # 2) 归一化：减去每帧的均值，并除以均值
        #    防止光照强度漂移
        mean_rgb = x_mean.mean(dim=2, keepdim=True)
        Xn = (x_mean - mean_rgb) / (mean_rgb + 1e-8)  # (B, T, 3)

        # 3) 投影到两个正交平面
        #    P = [[0, 1, -1],
        #         [-2, 1,  1]]
        #    得到两个时序信号 S1, S2
        #    Xn: (B, T, 3) -> permute -> (B, 3, T)
        Xn_t = Xn.permute(0, 2, 1)  # (B, 3, T)
        P = torch.tensor([[0., 1., -1.],
                          [-2., 1.,  1.]], device=x.device)  # (2,3)
        S = torch.matmul(P, Xn_t)  # (B, 2, T)
        S1 = S[:, 0, :]            # (B, T)
        S2 = S[:, 1, :]            # (B, T)

        # 4) 自适应合成
        #    α = std(S1) / std(S2)
        eps = 1e-8
        std1 = torch.std(S1, dim=1, keepdim=True)
        std2 = torch.std(S2, dim=1, keepdim=True) + eps
        alpha = std1 / std2        # (B, 1)
        s = S1 + alpha * S2        # (B, T)

        # 5) 带通滤波：保留 [low, high] Hz
        #    FFT 域掩码方式
        freq = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(x.device)  # (T//2+1,)
        mask = (freq >= self.low) & (freq <= self.high)             # (T//2+1,)
        S_f = torch.fft.rfft(s, dim=1)      # (B, T//2+1)
        S_f = S_f * mask.unsqueeze(0)       # 掩掉带外频率
        s_bp = torch.fft.irfft(S_f, n=T, dim=1)  # (B, T)

        return s_bp


def build_rPPG_model(fs=30.0, low_cut=0.7, high_cut=2.5):
    """
    返回一个 POS rPPG 提取模型：
        my_model = build_rPPG_POS_model()
        wave_pr  = my_model(video_tensor)
    """
    return POS(fs=fs, low_cut=low_cut, high_cut=high_cut)