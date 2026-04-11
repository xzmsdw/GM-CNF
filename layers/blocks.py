import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg
import torch.nn.utils.spectral_norm as spectral_norm

# ==========================================
# 1. 辅助与基础模块
# ==========================================
class ConditionEncoder(nn.Module):
    """工况编码器：提取均值和标准差作为静态工况特征"""
    def __init__(self, c_in, seq_len, embed_dim):
        super(ConditionEncoder, self).__init__()
        self.input_dim = c_in * 2
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, embed_dim)
        )

    def forward(self, c):
        c_mean = torch.mean(c, dim=2) 
        c_std = torch.std(c, dim=2)
        x = torch.cat([c_mean, c_std], dim=1)
        return self.net(x)

class FFTLayer(nn.Module):
    """将时域信号转换为频域对数幅值谱"""
    def __init__(self):
        super(FFTLayer, self).__init__()

    def forward(self, x):
        fft_x = torch.fft.rfft(x, dim=-1, norm='ortho')
        mag_x = torch.abs(fft_x)
        mag_x = torch.log(mag_x + 1e-6)
        if mag_x.size(-1) % 2 != 0:
            mag_x = mag_x[..., :-1]
        return mag_x

class PatchEmbedding(nn.Module):
    """将长序列分块投影，降低计算量同时保留长视野"""
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x))

class InvertibleLinear(nn.Module):
    """流模型组件：可逆线性变换 (LU分解实现通道混合)"""
    def __init__(self, c_in):
        super(InvertibleLinear, self).__init__()
        w_init = np.random.randn(c_in, c_in)
        q, _ = scipy.linalg.qr(w_init)
        p, l, u = scipy.linalg.lu(q.astype(np.float32))
        s = np.diag(u)
        u = np.triu(u, 1)
        
        self.register_buffer('p_matrix', torch.from_numpy(p))
        self.l_mask = torch.tril(torch.ones(c_in, c_in), -1)
        self.l = nn.Parameter(torch.from_numpy(l))
        self.u_mask = torch.triu(torch.ones(c_in, c_in), 1)
        self.u = nn.Parameter(torch.from_numpy(u))
        self.s = nn.Parameter(torch.from_numpy(s))
        self.register_buffer('mask_l', self.l_mask)
        self.register_buffer('mask_u', self.u_mask)

    def calc_weight(self):
        l = self.l * self.mask_l + torch.eye(self.l.size(0), device=self.l.device)
        u = self.u * self.mask_u + torch.diag(self.s)
        return torch.matmul(self.p_matrix, torch.matmul(l, u))

    def forward(self, x, h_c=None, reverse=False):
        if not reverse:
            weight = self.calc_weight()
            z = F.linear(x, weight)
            dlogdet = torch.sum(torch.log(torch.abs(self.s)))
            dlogdet = dlogdet.unsqueeze(0).expand(x.shape[0]) 
            return z, dlogdet
        else:
            weight = self.calc_weight()
            weight_inv = torch.inverse(weight)
            z = F.linear(x, weight_inv)
            return z, 0

# ==========================================
# 2. TCN 核心模块
# ==========================================
class DilatedTCNBlock(nn.Module):
    """
    带膨胀卷积的残差块 (TCN 核心)
    """
    def __init__(self, in_channels, out_channels, dilation, use_sn=True):
        super(DilatedTCNBlock, self).__init__()
        padding = dilation 
        
        # 1. 第一层
        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.conv1 = spectral_norm(conv1) if use_sn else conv1
        self.bn1 = nn.GroupNorm(4, out_channels)
        
        # 2. 第二层
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.conv2 = spectral_norm(conv2) if use_sn else conv2
        self.bn2 = nn.GroupNorm(4, out_channels)
        
        # 3. 捷径层
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut = nn.Sequential(
                spectral_norm(shortcut_conv) if use_sn else shortcut_conv,
                nn.GroupNorm(4, out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class DualTCN1D(nn.Module):
    """
    双流 TCN 网络：时域 (Patch + TCN) & 频域 (FFT + TCN)
    """
    def __init__(self, in_channels=6, base_filters=16, output_dim=128, patch_size=4, use_sn=True):
        super(DualTCN1D, self).__init__()
        
        # === 1. 时域流 (Time Stream) ===
        self.patch_embed = PatchEmbedding(in_channels, base_filters, patch_size)
        
        self.time_stream = nn.Sequential(
            DilatedTCNBlock(base_filters, base_filters * 2, dilation=1, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 2, base_filters * 4, dilation=2, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 4, base_filters * 8, dilation=4, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 8, base_filters * 16, dilation=8, use_sn=use_sn),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.fft_layer = FFTLayer()
        self.freq_stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.LeakyReLU(0.1)
        )
        
        self.freq_stream = nn.Sequential(
            DilatedTCNBlock(base_filters, base_filters * 2, dilation=1, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 2, base_filters * 4, dilation=2, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 4, base_filters * 8, dilation=4, use_sn=use_sn),
            nn.MaxPool1d(2),
            DilatedTCNBlock(base_filters * 8, base_filters * 16, dilation=8, use_sn=use_sn),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # === 3. 融合层 (Fusion) ===
        fusion_dim = (base_filters * 16) * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # 1. 时域流
        x_patch = self.patch_embed(x)
        h_time = self.time_stream(x_patch)
        
        # 2. 频域流
        x_fft = self.fft_layer(x)
        x_fft_stem = self.freq_stem(x_fft)
        h_freq = self.freq_stream(x_fft_stem)
        
        # 3. 融合
        h_cat = torch.cat([h_time, h_freq], dim=1)
        return self.fusion_layer(h_cat)