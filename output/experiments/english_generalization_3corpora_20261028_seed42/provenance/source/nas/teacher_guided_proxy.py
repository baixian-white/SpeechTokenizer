"""Teacher-guided proxy helpers for Exp1 encoder-side NAS.

The teacher is a frozen pretrained SpeechTokenizer. It is used only as a
pre-RVQ latent anchor and as a frozen quantizer for compatibility diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from speechtokenizer.model import SpeechTokenizer


TEACHER_METRIC_KEYS = [
    "teacher_latent_smooth_l1",
    "teacher_latent_cosine_distance",
    "teacher_temporal_delta_loss",
    "rvq_code_agreement",
    "rvq_code_flip_rate",
    "rvq_quantized_feature_l1",
]


def freeze_teacher_model(model: torch.nn.Module) -> torch.nn.Module:
    """Put the teacher in eval mode and disable all parameter gradients."""

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_teacher_model(config_path: str, checkpoint_path: str, device: torch.device) -> SpeechTokenizer:
    """Load and freeze a SpeechTokenizer teacher model."""

    if not Path(config_path).exists():
        raise FileNotFoundError(f"teacher config not found: {config_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    teacher = SpeechTokenizer.load_from_checkpoint(config_path, checkpoint_path)
    teacher.to(device)
    return freeze_teacher_model(teacher)


def teacher_reference_dict(
    config_path: Optional[str],
    checkpoint_path: Optional[str],
    target: str,
    cache_mode: str,
    enabled: bool,
) -> Dict[str, object]:
    return {
        "enabled": bool(enabled),
        "teacher_config": config_path or "",
        "teacher_checkpoint": checkpoint_path or "",
        "teacher_target": target,
        "teacher_cache_mode": cache_mode,
        "teacher_role": "frozen SpeechTokenizer encoder pre-RVQ latent anchor",
        "teacher_encoder_frozen": bool(enabled),
        "teacher_quantizer_frozen": bool(enabled),
        "teacher_decoder_frozen": bool(enabled),
    }


def _btd(x: torch.Tensor) -> torch.Tensor:
    """Normalize latent tensors to [B, T, D] for feature comparisons."""

    if x.ndim != 3:
        raise ValueError(f"expected 3D tensor, got shape={tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


def _align_latents(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.size(0) != b.size(0):
        raise ValueError(f"batch mismatch: teacher={tuple(a.shape)}, student={tuple(b.shape)}")
    if a.size(1) != b.size(1):
        raise ValueError(f"latent dimension mismatch: teacher={tuple(a.shape)}, student={tuple(b.shape)}")
    length = min(a.size(-1), b.size(-1))
    if length <= 0:
        raise ValueError("cannot compare empty latent tensors")
    return a[..., :length], b[..., :length]


def _align_codes(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"expected code tensors [n_q, B, T], got {tuple(a.shape)} and {tuple(b.shape)}")
    n_q = min(a.size(0), b.size(0))
    batch = min(a.size(1), b.size(1))
    length = min(a.size(2), b.size(2))
    if n_q <= 0 or batch <= 0 or length <= 0:
        raise ValueError("cannot compare empty code tensors")
    return a[:n_q, :batch, :length], b[:n_q, :batch, :length]


def compute_teacher_guided_metrics(
    *,
    teacher_latent: torch.Tensor,
    student_latent: torch.Tensor,
    teacher_codes: torch.Tensor,
    student_codes: torch.Tensor,
    teacher_quantized: torch.Tensor,
    student_quantized: torch.Tensor,
) -> Dict[str, float]:
    """Compute teacher alignment and frozen-RVQ compatibility metrics."""

    teacher_latent, student_latent = _align_latents(teacher_latent, student_latent)
    teacher_btd = _btd(teacher_latent)
    student_btd = _btd(student_latent)

    latent_smooth_l1 = F.smooth_l1_loss(student_btd, teacher_btd).item()
    cosine = F.cosine_similarity(student_btd, teacher_btd, dim=-1).mean().item()
    cosine_distance = 1.0 - cosine

    if teacher_btd.size(1) > 1:
        teacher_delta = teacher_btd[:, 1:, :] - teacher_btd[:, :-1, :]
        student_delta = student_btd[:, 1:, :] - student_btd[:, :-1, :]
        delta_loss = F.smooth_l1_loss(student_delta, teacher_delta).item()
    else:
        delta_loss = 0.0

    teacher_codes, student_codes = _align_codes(teacher_codes, student_codes)
    agreement = (teacher_codes == student_codes).float().mean().item()

    teacher_quantized, student_quantized = _align_latents(teacher_quantized, student_quantized)
    quantized_l1 = F.l1_loss(student_quantized, teacher_quantized).item()

    return {
        "teacher_latent_smooth_l1": float(latent_smooth_l1),
        "teacher_latent_cosine_distance": float(cosine_distance),
        "teacher_temporal_delta_loss": float(delta_loss),
        "rvq_code_agreement": float(agreement),
        "rvq_code_flip_rate": float(1.0 - agreement),
        "rvq_quantized_feature_l1": float(quantized_l1),
    }


def disabled_teacher_metrics() -> Dict[str, str]:
    return {key: "" for key in TEACHER_METRIC_KEYS}
