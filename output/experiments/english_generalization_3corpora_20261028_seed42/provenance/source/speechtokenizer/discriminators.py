import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import typing as tp
import torchaudio
from einops import rearrange
from .modules import NormConv2d   # 自定义的归一化卷积层（支持 weight_norm 或 spectral_norm）

# ---------------------- 常量定义 ----------------------
LRELU_SLOPE = 0.1  # LeakyReLU 的负半轴斜率


# ---------------------- 工具函数 ----------------------
def get_padding(kernel_size, dilation=1):
    """计算 1D 卷积的对称 padding，使得输出长度与输入一致"""
    return int((kernel_size * dilation - dilation) / 2)

def init_weights(m, mean=0.0, std=0.01):
    """初始化卷积层权重为正态分布，用于稳定训练"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# =====================================================
# 一、周期判别器 DiscriminatorP（Period-based）
# =====================================================
class DiscriminatorP(torch.nn.Module):
    """
    基于“周期”特征的子判别器：
    - 将输入波形 reshape 成 [batch, 1, t//period, period] 的 2D 形式，
      这样可以让卷积层捕捉周期结构（如音高周期）。
    - 类似 HiFi-GAN 中的 MultiPeriodDiscriminator。
    """
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        # 选择归一化方式：weight_norm 或 spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # 依次堆叠多层 2D 卷积
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),  # 无 stride
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))  # 输出层

    def forward(self, x):
        """
        输入: [B, 1, T]
        输出:
            x: 判别结果 [B, N]
            fmap: 各卷积层的 feature maps（用于 feature matching loss）
        """
        fmap = []

        # Step1. 若长度不能整除 period，则在右侧 padding 使其对齐
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")  # 使用反射填充
            t = t + n_pad

        # Step2. 将 1D 波形 reshape 成 2D（周期维度）
        x = x.view(b, c, t // self.period, self.period)

        # Step3. 逐层卷积 + LeakyReLU
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        # Step4. 输出层卷积
        x = self.conv_post(x)
        fmap.append(x)

        # Step5. 展平为一维 logits
        x = torch.flatten(x, 1, -1)

        return x, fmap


# =====================================================
# 二、多周期判别器 MultiPeriodDiscriminator
# =====================================================
class MultiPeriodDiscriminator(torch.nn.Module):
    """
    包含多个不同周期 period 的 DiscriminatorP。
    每个子判别器专注于不同周期特征，如周期为2/3/5/7/11的模式。
    """
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        """
        输入: y (真实波形), y_hat (生成波形)
        输出:
            y_d_rs / y_d_gs: 每个判别器的 logits
            fmap_rs / fmap_gs: 对应 feature maps
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# =====================================================
# 三、尺度判别器 DiscriminatorS（Scale-based）
# =====================================================
class DiscriminatorS(torch.nn.Module):
    """
    基于尺度的 1D 卷积判别器：
    - 用不同的 stride 逐步下采样音频。
    - 模仿 HiFi-GAN 的 MultiScaleDiscriminator。
    """
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm #谱归一化
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),  # 初始感受野较小
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),  # 最终聚合
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


# =====================================================
# 四、多尺度判别器 z
# =====================================================
class MultiScaleDiscriminator(torch.nn.Module):
    """
    不同时间尺度下的判别器集合：
    - 第一个原始尺度；
    - 后两个通过平均池化(AvgPool)获得更低采样率；
    这样能让判别器同时关注全局波形和局部细节。
    """
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),  # 原始尺度
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2), #kernel = 4(每4个点取平均), stride=2,padding=2
            AvgPool1d(4, 2, padding=2)  
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            # 后两个判别器使用平均池化降采样后的版本
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# =====================================================
# 五、STFT 频谱判别器 DiscriminatorSTFT
# =====================================================
#给 2D 卷积算“对称 padding”，让输出尺寸在对应维度上尽量和输入相同（或受 stride 控制，而不是因为 kernel 大小乱掉）
def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    """计算 2D 卷积的 padding，使输出尺寸匹配输入"""
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """
    基于时频（STFT）域的子判别器。
    关键思想：
    - 对输入波形做 STFT（复数频谱）；
    - 将实部与虚部分别作为通道拼接；
    - 用 2D 卷积在 (时间, 频率) 上提取特征；
    - 输出用于对抗性训练的 logits。
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 max_filters: int = 1024, filters_scale: int = 1,
                 kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True,
                 norm: str = 'weight_norm', activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()

        # 保存超参
        self.filters = filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)

        # 生成 STFT 变换器 短时傅里叶变化
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window_fn=torch.hann_window, normalized=self.normalized,
            center=False, pad_mode=None, power=None  # 返回复数谱
        )
        #n_fft: 短时傅里叶变换的窗口大小(), hop_length: 每次移动的步长, win_length: 实际用于计算傅里叶变换的窗口大小   window_fn: 窗口函数     center: 是否将信号中心移到窗口中心   pad_mode: 填充模式   power: 返回复数谱的幂次

        # 通道数为实部 + 虚部
        spec_channels = 2 * in_channels  #in_channels默认是1，是单通道音频，经过傅里叶变换之后，会变成复数频谱，所以spec_channels=2
        self.convs = nn.ModuleList()

        # 第一层卷积
        self.convs.append(
            NormConv2d(spec_channels, filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )

        # 后续多层卷积，逐步扩大感受野（随 dilation）
        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                           dilation=(dilation, 1),
                           padding=get_2d_padding(kernel_size, (dilation, 1)), norm=norm)
            )
            in_chs = out_chs

        # 收尾两层
        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        self.convs.append(
            NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                       padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)
        )
        self.conv_post = NormConv2d(out_chs, out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])), norm=norm)

    def forward(self, x: torch.Tensor):
        """
        输入: 波形 [B, 1, T]
        输出:
            logits: 判别结果
            fmap: 特征图集合（用于特征匹配损失）
        """
        fmap = []
        # 计算复数 STFT 频谱
        z = self.spec_transform(x)           # [B, 1, Freq, Time] (复数)
        z = torch.cat([z.real, z.imag], 1)   # 拼接实部和虚部 → [B, 2, Freq, Time]
        z = rearrange(z, 'b c w t -> b c t w')  # 交换维度，方便时序卷积->[]B, 2, Time, Freq]

        # 卷积提取特征
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)

        # 输出层
        z = self.conv_post(z)
        return z, fmap


# =====================================================
# 六、多尺度 STFT 判别器（MS-STFT）
# =====================================================
class MultiScaleSTFTDiscriminator(nn.Module):
    """
    将多个不同 STFT 配置（n_fft / hop / win）组合，
    每个子判别器在不同频率分辨率下判断真假。
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int] = [256, 512, 128, 64, 32],
                 win_lengths: tp.List[int] = [1024, 2048, 512, 256, 128], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)

        # 创建不同尺度的 STFT 判别器
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i],
                              hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        """
        对每个 STFT 判别器分别计算真假波形的 logits 和特征图。
        """
        logits, logits_fake, fmaps, fmaps_fake = [], [], [], []
        for disc in self.discriminators:
            logit, fmap = disc(y)
            logit_fake, fmap_fake = disc(y_hat)
            logits.append(logit)
            logits_fake.append(logit_fake)
            fmaps.append(fmap)
            fmaps_fake.append(fmap_fake)
        return logits, logits_fake, fmaps, fmaps_fake
