#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Streaming speaker identity helper for the three-user demo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    (
        parent
        for parent in DEMO_DIR.parents
        if (parent / "scripts" / "speaker_identity_utils.py").exists()
    ),
    DEMO_DIR.parents[-1],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.speaker_identity_utils import (  # noqa: E402
    SpeakerEmbeddingExtractor,
    SpeakerPrediction,
    SpeakerProfile,
    enroll_profiles,
    mfcc_embedding_from_waveform,
    predict_speaker,
)


EmbeddingFn = Callable[[np.ndarray, int, int], np.ndarray]


class _SenderState:
    def __init__(self) -> None:
        self.buffer = np.zeros(0, dtype=np.float32)
        self.codes = None
        self.samples_since_eval = 0


def _collect_profile_paths(profile_dir: str | Path) -> dict[str, list[Path]]:
    root = Path(profile_dir)
    if not profile_dir or not root.exists():
        return {}

    patterns = ("*.wav", "*.flac")
    groups: dict[str, list[Path]] = {}
    for speaker_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(speaker_dir.glob(pattern)))
        if paths:
            groups[speaker_dir.name] = paths
    return groups


class StreamingSpeakerIdentifier:
    """Maintain per-sender decoded-audio windows and predict speaker identity."""

    def __init__(
        self,
        profile_dir: str | Path,
        sample_rate: int,
        window_sec: float = 3.0,
        hop_sec: float = 1.0,
        threshold: float = 0.65,
        n_mfcc: int = 40,
        backend: str = "mfcc",
        speaker_device: str = "cpu",
        ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        ecapa_savedir: str | Path = "output/models/speechbrain_spkrec_ecapa_voxceleb",
        profiles: Optional[Mapping[str, SpeakerProfile]] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
        bundle_classifier=None,
        bundle_dir: str | Path | None = None,
        bundle_device: str = 'cpu',
    ) -> None:
        self.profile_dir = Path(profile_dir) if profile_dir else None
        self.sample_rate = int(sample_rate)
        self.window_samples = max(1, int(round(float(window_sec) * self.sample_rate)))
        self.hop_samples = max(1, int(round(float(hop_sec) * self.sample_rate)))
        self.threshold = float(threshold)
        self.n_mfcc = int(n_mfcc)
        self.backend = str(backend).lower()
        self.speaker_device = str(speaker_device)
        self.ecapa_source = str(ecapa_source)
        self.ecapa_savedir = Path(ecapa_savedir)
        self.extractor = None
        self.embedding_fn = embedding_fn
        self.bundle_classifier = bundle_classifier
        if self.bundle_classifier is None and bundle_dir:
            from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle
            self.bundle_classifier = SpeakerClassifierBundle(bundle_dir, device=bundle_device, threshold=threshold)
        self.states: dict[str, _SenderState] = {}

        if self.bundle_classifier is not None:
            self.profiles = {}
        elif profiles is not None:
            self.profiles = dict(profiles)
        else:
            self.extractor = SpeakerEmbeddingExtractor(
                backend=self.backend,
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                device=self.speaker_device,
                ecapa_source=self.ecapa_source,
                ecapa_savedir=self.ecapa_savedir,
            )
            self.profiles = enroll_profiles(
                _collect_profile_paths(profile_dir),
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                extractor=self.extractor,
            )

    @property
    def enabled(self) -> bool:
        return self.bundle_classifier is not None or bool(self.profiles)

    @property
    def speaker_count(self) -> int:
        return self.bundle_classifier.speaker_count if self.bundle_classifier is not None else len(self.profiles)

    def update(self, sender_id: str, pcm: np.ndarray, codes: np.ndarray | None = None) -> Optional[SpeakerPrediction]:
        if not self.enabled:
            return None

        audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None

        state = self.states.setdefault(sender_id, _SenderState())
        state.buffer = np.concatenate([state.buffer, audio])
        if state.buffer.size > self.window_samples:
            state.buffer = state.buffer[-self.window_samples :]
        if codes is not None:
            code_array = np.asarray(codes, dtype=np.int64)
            if code_array.ndim == 3 and code_array.shape[1] == 1:
                code_array = code_array[:, 0, :]
            if code_array.ndim != 2:
                raise ValueError('codes must have shape [L,T] or [L,1,T]')
            if state.codes is None or state.codes.shape[0] != code_array.shape[0]:
                state.codes = code_array
            else:
                state.codes = np.concatenate([state.codes, code_array], axis=1)
            frame_hop = int(getattr(self.bundle_classifier, 'frame_hop_samples', 320))
            max_frames = max(1, int(np.ceil(self.window_samples / float(frame_hop))))
            if state.codes.shape[1] > max_frames:
                state.codes = state.codes[:, -max_frames:]
        state.samples_since_eval += int(audio.size)

        if state.buffer.size < self.window_samples:
            return None
        if state.samples_since_eval < self.hop_samples:
            return None

        state.samples_since_eval = 0
        if self.bundle_classifier is not None:
            return self.bundle_classifier.predict(state.buffer, codes=state.codes)
        if self.embedding_fn is not None:
            embedding = self.embedding_fn(state.buffer, self.sample_rate, self.n_mfcc)
        elif self.extractor is not None:
            embedding = self.extractor.from_waveform(state.buffer)
        else:
            embedding = mfcc_embedding_from_waveform(state.buffer, self.sample_rate, self.n_mfcc)
        return predict_speaker(embedding, self.profiles, threshold=self.threshold)
