import torch
import torchaudio
import matplotlib.pylab as plt

# =========================
# Mel 相关工具
# =========================

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """对幅度谱做对数压缩，避免 0 导致 log 爆掉。"""
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

mel_basis = {}
hann_window = {}
# 用于缓存不同尺度的 MelSpectrogram 实例
_mel_tfms = {}

def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sample_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False
):
    """
    输入:
      y: [B, T] float
    输出:
      mel: [B, num_mels, T_frames]
    兜底:
      - 若 fmax 为 None，则使用 Nyquist = sample_rate / 2
      - 若 fmin 为 None，则使用 0
    """
    assert y.dim() == 2, f"Expected [B, T], got {tuple(y.shape)}"
    if y.dtype != torch.float32:
        y = y.float()

    # --- 关键兜底，修复 float(None) 报错 ---
    if fmax is None:
        fmax = float(sample_rate) / 2.0
    if fmin is None:
        fmin = 0.0
    # -------------------------------------

    key = (
        int(n_fft), int(num_mels), int(hop_size), int(win_size),
        float(fmin), float(fmax), bool(center), str(y.device)
    )
    if key not in _mel_tfms:
        _mel_tfms[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=num_mels,
            center=center,
            power=2.0,
            normalized=False,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        ).to(y.device)

    mel = _mel_tfms[key](y)           # [B, num_mels, T_frames]
    mel = mel.clamp_(min=1e-9).log_() # 动态范围压缩
    return mel

def plot_spectrogram(spectrogram):
    """spectrogram: [num_mels, T] 或 [H, W] 的 numpy/tensor，返回 matplotlib Figure 用于 TB 记录"""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    return fig

# =========================
# 波形/谱重建损失
# =========================

def recon_loss(x, x_hat):
    """
    L1 波形重建损失
    x, x_hat: [B, 1, T]
    """
    length = min(x.size(-1), x_hat.size(-1))
    return torch.nn.functional.l1_loss(x[:, :, :length], x_hat[:, :, :length])

def mel_loss(x, x_hat, **kwargs):
    """
    Mel 频谱 L1 损失（融入多尺度时，用外面传入不同 n_fft/hop/win 的 kwargs）
    x, x_hat: [B, 1, T]
    """
    x_mel = mel_spectrogram(x.squeeze(1), **kwargs)       # [B, M, Tm]
    x_hat_mel = mel_spectrogram(x_hat.squeeze(1), **kwargs)
    length = min(x_mel.size(2), x_hat_mel.size(2))
    return torch.nn.functional.l1_loss(x_mel[:, :, :length], x_hat_mel[:, :, :length])

# =========================
# GAN 损失（HiFiGAN 风格）
# =========================

def feature_loss(fmap_r, fmap_g):
    """判别器特征匹配损失"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """LSGAN 判别器损失"""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss

def adversarial_loss(disc_outputs):
    """LSGAN 生成器对抗损失"""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss

# =========================
# 语义蒸馏损失（关键修复）
# =========================

_COMMON_DIMS = {80, 128, 256, 512, 768, 1024}

def _ensure_BT_D(x: torch.Tensor) -> torch.Tensor:
    """
    把输入统一成 [B, T, D]：
      - 允许 [T, D] / [1, T, D] / [B, D, T] / [B, T, D]
    这样后续就能稳定在 D 维上做 cosine_similarity。
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim == 2:              # [T, D]
        x = x.unsqueeze(0)       # -> [1, T, D]

    # 可能是 [B, D, T]：中间维看起来像常见特征维（768/1024/...），最后一维不是
    if x.ndim == 3 and x.shape[1] in _COMMON_DIMS and x.shape[2] not in _COMMON_DIMS:
        x = x.transpose(1, 2).contiguous()  # -> [B, T, D]

    # 如果是 [1, T, D] 直接保留；[B, T, D] 也直接保留
    return x

def d_axis_distill_loss(feature, target_feature):
    """
    沿“特征维 D”做相似度的蒸馏损失（与论文中 d-axis 语义对齐一致）。
    统一对齐为 [B, T, D]，对齐时间长度，再在 dim=-1 上做 cosine 相似度。
    """
    f = _ensure_BT_D(feature)          # [B, T, D] or [1, T, D]
    t = _ensure_BT_D(target_feature)   # [B, T, D] or [1, T, D]

    # 对齐时间长度
    T = min(f.size(1), t.size(1))
    f = f[:, :T, :]
    t = t[:, :T, :]

    # 在特征维 D 上做余弦相似度 -> [B, T]
    cos = torch.nn.functional.cosine_similarity(f, t, dim=-1)
    # 二值 Logistic 形式的相似度拉近（稳定且对齐论文习惯）
    distill_loss = -torch.log(torch.sigmoid(cos)).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1.0):
    """
    时间轴蒸馏：逐元素 L1 + 在特征维 D 上的 cosine 相似度正则。
    """
    f = _ensure_BT_D(feature)
    t = _ensure_BT_D(target_feature)

    T = min(f.size(1), t.size(1))
    f = f[:, :T, :]
    t = t[:, :T, :]

    l1 = torch.nn.functional.l1_loss(f, t, reduction='mean')
    cos = torch.nn.functional.cosine_similarity(f, t, dim=-1)  # [B, T]
    sim = -torch.log(torch.sigmoid(cos)).mean()
    return l1 + lambda_sim * sim
