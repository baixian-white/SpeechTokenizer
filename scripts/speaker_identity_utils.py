"""Speaker identity helpers for Exp22 and the three-user demo.

The default embedding backend is deliberately dependency-light. It uses MFCC
statistics so the evaluation can run in the current project environment, while
keeping the public interface small enough to swap in a stronger frozen speaker
model later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    utterance_count: int


@dataclass(frozen=True)
class SpeakerPrediction:
    predicted_speaker: str
    score: float
    margin: float
    verified: bool
    scores: Dict[str, float]


class SpeakerEmbeddingExtractor:
    """Unified speaker embedding interface for MFCC and frozen ECAPA backends."""

    def __init__(
        self,
        backend: str = "mfcc",
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        device: str = "cpu",
        ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        ecapa_savedir: str | Path = "output/models/speechbrain_spkrec_ecapa_voxceleb",
    ) -> None:
        self.backend = backend.lower()
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.device = str(device)
        self.ecapa_source = str(ecapa_source)
        self.ecapa_savedir = Path(ecapa_savedir)
        self._ecapa = None
        if self.backend not in {"mfcc", "ecapa"}:
            raise ValueError(f"unsupported speaker backend: {backend}")
        if self.backend == "ecapa":
            self._load_ecapa()

    @property
    def description(self) -> str:
        if self.backend == "mfcc":
            return f"MFCC statistics (`n_mfcc={self.n_mfcc}`)"
        return f"SpeechBrain ECAPA (`{self.ecapa_source}`)"

    def _load_ecapa(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except Exception as exc:
            raise RuntimeError(
                "speechbrain is required for --speaker-backend ecapa; install speechbrain first"
            ) from exc
        run_opts = {"device": self.device}
        self._ecapa = EncoderClassifier.from_hparams(
            source=self.ecapa_source,
            savedir=str(self.ecapa_savedir),
            run_opts=run_opts,
            local_strategy=LocalStrategy.COPY,
        )

    def from_path(self, path: str | Path) -> np.ndarray:
        waveform = load_audio_mono(path, sample_rate=self.sample_rate)
        return self.from_waveform(waveform)

    def from_waveform(self, waveform: torch.Tensor | np.ndarray) -> np.ndarray:
        if self.backend == "mfcc":
            return mfcc_embedding_from_waveform(
                waveform,
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
            )
        if isinstance(waveform, np.ndarray):
            wav = torch.from_numpy(waveform.astype(np.float32, copy=False))
        else:
            wav = waveform.detach().cpu().float()
        wav = wav.reshape(1, -1).to(self.device)
        with torch.no_grad():
            emb = self._ecapa.encode_batch(wav).detach().cpu().float().reshape(-1).numpy()
        return l2_normalize(emb.astype(np.float32, copy=False))


def _path_parts(path: str | Path) -> List[str]:
    normalized = str(path).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def parse_speaker_id(path: str | Path, source: str = "auto") -> str:
    """Parse a stable speaker ID from known local corpus layouts."""

    parts = _path_parts(path)
    name = Path(str(path).replace("\\", "/")).name
    stem = Path(name).stem
    source = source.lower()

    if source in {"auto", "vctk"}:
        match = re.match(r"^(p\d+|s\d+)_", name, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        for part in reversed(parts):
            if re.fullmatch(r"p\d+|s\d+", part, flags=re.IGNORECASE):
                return part

    if source in {"auto", "aishell"}:
        for part in reversed(parts):
            if re.fullmatch(r"S\d{4}", part, flags=re.IGNORECASE):
                return part.upper()
        match = re.search(r"(S\d{4})", name, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    if source in {"auto", "librispeech", "libri"}:
        match = re.match(r"^(\d+)-\d+-\d+", stem)
        if match:
            return match.group(1)
        for idx, part in enumerate(parts):
            if part in {"train-clean-100", "test-clean", "test-other"} and idx + 1 < len(parts):
                return parts[idx + 1]

    if len(parts) >= 2:
        return parts[-2]
    return stem


def read_audio_paths(sample_list: str | Path) -> List[Path]:
    rows: List[Path] = []
    with open(sample_list, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            rows.append(Path(raw.split("\t")[0].lstrip("\ufeff").strip()))
    return rows


def scan_audio_root(root: str | Path, patterns: Sequence[str] = ("*.wav", "*.flac")) -> List[Path]:
    root = Path(root)
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(paths)


def group_paths_by_speaker(paths: Iterable[str | Path], source: str = "auto") -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        speaker_id = parse_speaker_id(path, source=source)
        groups.setdefault(speaker_id, []).append(path)
    return {speaker: sorted(items) for speaker, items in sorted(groups.items())}


def filter_speaker_groups(
    groups: Mapping[str, Sequence[Path]],
    min_utterances: int,
    max_speakers: int = 0,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    eligible = [(speaker, list(paths)) for speaker, paths in groups.items() if len(paths) >= min_utterances]
    rng = random.Random(seed)
    eligible.sort(key=lambda item: item[0])
    if max_speakers and len(eligible) > max_speakers:
        eligible = rng.sample(eligible, max_speakers)
        eligible.sort(key=lambda item: item[0])
    return {speaker: sorted(paths) for speaker, paths in eligible}


def split_enrollment_and_test(
    groups: Mapping[str, Sequence[Path]],
    enroll_per_speaker: int,
    test_per_speaker: int,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    enroll: Dict[str, List[Path]] = {}
    test: Dict[str, List[Path]] = {}
    rng = random.Random(seed)
    for speaker_id, paths in sorted(groups.items()):
        shuffled = list(paths)
        rng.shuffle(shuffled)
        need = enroll_per_speaker + test_per_speaker
        if len(shuffled) < need:
            continue
        enroll[speaker_id] = sorted(shuffled[:enroll_per_speaker])
        test[speaker_id] = sorted(shuffled[enroll_per_speaker:need])
    return enroll, test


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    return float(np.dot(aa, bb))


def load_audio_mono(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    try:
        audio, sr = torchaudio.load(str(path))
    except Exception:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = torch.from_numpy(data.T).contiguous()
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.squeeze(0).float()


def mfcc_embedding_from_waveform(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
) -> np.ndarray:
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(waveform.astype(np.float32, copy=False))
    else:
        wav = waveform.detach().cpu().float()
    wav = wav.reshape(-1)
    if wav.numel() < max(1, sample_rate // 10):
        wav = torch.nn.functional.pad(wav, (0, max(1, sample_rate // 10) - wav.numel()))
    wav = wav - wav.mean()
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak.clamp_min(1e-6)
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 64,
            "center": True,
        },
    )
    with torch.no_grad():
        mfcc = transform(wav.unsqueeze(0)).squeeze(0).transpose(0, 1).contiguous()
    mean = mfcc.mean(dim=0)
    std = mfcc.std(dim=0, unbiased=False)
    delta = mfcc[1:] - mfcc[:-1] if mfcc.shape[0] > 1 else torch.zeros_like(mfcc)
    delta_mean = delta.mean(dim=0)
    emb = torch.cat([mean, std, delta_mean], dim=0).numpy().astype(np.float32, copy=False)
    return l2_normalize(emb)


def mfcc_embedding_from_path(path: str | Path, sample_rate: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    return mfcc_embedding_from_waveform(load_audio_mono(path, sample_rate=sample_rate), sample_rate=sample_rate, n_mfcc=n_mfcc)


def enroll_profiles(
    enroll_paths: Mapping[str, Sequence[Path]],
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    extractor: Optional[SpeakerEmbeddingExtractor] = None,
) -> Dict[str, SpeakerProfile]:
    profiles: Dict[str, SpeakerProfile] = {}
    if extractor is None:
        extractor = SpeakerEmbeddingExtractor(backend="mfcc", sample_rate=sample_rate, n_mfcc=n_mfcc)
    for speaker_id, paths in sorted(enroll_paths.items()):
        embeddings = [extractor.from_path(path) for path in paths]
        if not embeddings:
            continue
        centroid = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        profiles[speaker_id] = SpeakerProfile(speaker_id=speaker_id, centroid=centroid, utterance_count=len(embeddings))
    return profiles


def predict_speaker(
    embedding: np.ndarray,
    profiles: Mapping[str, SpeakerProfile],
    threshold: float = 0.65,
) -> SpeakerPrediction:
    if not profiles:
        return SpeakerPrediction("", float("nan"), float("nan"), False, {})
    scores = {
        speaker_id: cosine_similarity(embedding, profile.centroid)
        for speaker_id, profile in sorted(profiles.items())
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else -1.0
    margin = float(best_score - second_score)
    return SpeakerPrediction(
        predicted_speaker=best_id,
        score=float(best_score),
        margin=margin,
        verified=bool(best_score >= threshold),
        scores=scores,
    )


def compute_eer(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan"), float("nan")
    thresholds = np.unique(scores_arr)
    candidates = np.concatenate(([thresholds.max() + 1e-6], thresholds[::-1], [thresholds.min() - 1e-6]))
    best = (float("inf"), float("nan"), float("nan"))
    pos = labels_arr == 1
    neg = labels_arr == 0
    for threshold in candidates:
        accepted = scores_arr >= threshold
        far = float(np.mean(accepted[neg])) if np.any(neg) else 0.0
        frr = float(np.mean(~accepted[pos])) if np.any(pos) else 0.0
        gap = abs(far - frr)
        eer = (far + frr) / 2.0
        if gap < best[0] or (gap == best[0] and eer < best[1]):
            best = (gap, eer, float(threshold))
    return best[1], best[2]


def tar_at_far(labels: Sequence[int], scores: Sequence[float], target_far: float = 0.01) -> Tuple[float, float]:
    labels_arr = np.asarray(labels, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)
    if labels_arr.size == 0 or len(np.unique(labels_arr)) < 2:
        return float("nan"), float("nan")
    thresholds = np.unique(scores_arr)[::-1]
    thresholds = np.concatenate(([scores_arr.max() + 1e-6], thresholds, [scores_arr.min() - 1e-6]))
    pos = labels_arr == 1
    neg = labels_arr == 0
    best_tar = 0.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        accepted = scores_arr >= threshold
        far = float(np.mean(accepted[neg])) if np.any(neg) else 0.0
        tar = float(np.mean(accepted[pos])) if np.any(pos) else 0.0
        if far <= target_far and (tar > best_tar or (tar == best_tar and threshold < best_threshold)):
            best_tar = tar
            best_threshold = float(threshold)
    return float(best_tar), best_threshold


def pairwise_verification_scores(rows: Sequence[Mapping[str, object]]) -> Tuple[List[int], List[float]]:
    labels: List[int] = []
    scores: List[float] = []
    for row in rows:
        same = row.get("speaker_id") == row.get("predicted_speaker")
        if row.get("correct_score") is not None:
            labels.append(1)
            scores.append(float(row["correct_score"]))
        impostor_score = row.get("max_impostor_score")
        if impostor_score is not None:
            labels.append(0)
            scores.append(float(impostor_score))
        elif not same and row.get("score") is not None:
            labels.append(0)
            scores.append(float(row["score"]))
    return labels, scores
