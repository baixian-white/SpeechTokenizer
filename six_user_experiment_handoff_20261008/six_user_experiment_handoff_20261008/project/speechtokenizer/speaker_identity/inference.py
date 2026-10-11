import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REQUIRED_BUNDLE_FILES = ('classifier.pt', 'speaker_labels.json', 'preprocessing.json', 'calibration.json', 'model_config.json', 'manifest.json')


@dataclass(frozen=True)
class SpeakerPrediction:
    predicted_speaker: str
    score: float
    margin: float
    verified: bool
    scores: dict


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_export_bundle(bundle_dir, classifier_artifact, labels, preprocessing, calibration, model_config):
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(classifier_artifact), str(root / 'classifier.pt'))
    _write_json(root / 'speaker_labels.json', list(labels))
    _write_json(root / 'preprocessing.json', dict(preprocessing))
    _write_json(root / 'calibration.json', dict(calibration))
    config = dict(model_config)
    config.setdefault('speaker_count', len(labels))
    _write_json(root / 'model_config.json', config)
    hashes = {name: _sha256(root / name) for name in REQUIRED_BUNDLE_FILES if name != 'manifest.json'}
    _write_json(root / 'manifest.json', {'format': 'exp23-speaker-classifier-v1', 'files': hashes})
    return validate_export_bundle(root)


def validate_export_bundle(bundle_dir):
    root = Path(bundle_dir)
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError('missing bundle files: %s' % ', '.join(missing))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest.get('files', {}).items():
        if _sha256(root / name) != expected:
            raise ValueError('bundle hash mismatch: %s' % name)
    labels = json.loads((root / 'speaker_labels.json').read_text(encoding='utf-8'))
    config = json.loads((root / 'model_config.json').read_text(encoding='utf-8'))
    if int(config.get('speaker_count', -1)) != len(labels):
        raise ValueError('speaker_count does not match speaker labels')
    return {'speaker_count': len(labels), 'labels': labels, 'manifest': manifest}


class CodesWaveformLogits(torch.nn.Module):
    def __init__(self, speaker_model):
        super().__init__()
        self.token_encoder = speaker_model.token_encoder
        self.audio_encoder = speaker_model.audio_encoder
        self.fusion = speaker_model.fusion
        self.fusion_head = speaker_model.fusion_head

    def forward(self, codes, waveform):
        frame_mask = torch.ones((codes.shape[0], codes.shape[2]), dtype=torch.bool, device=codes.device)
        layer_mask = torch.ones((codes.shape[0], codes.shape[1]), dtype=torch.bool, device=codes.device)
        token_embedding = self.token_encoder(codes, frame_mask, layer_mask)
        audio_embedding = self.audio_encoder(waveform)
        fusion_embedding, _ = self.fusion(token_embedding, audio_embedding)
        return self.fusion_head(fusion_embedding)


class SpeakerClassifierBundle:
    def __init__(self, bundle_dir, device='cpu', threshold=0.65, candidate_labels=None):
        metadata = validate_export_bundle(bundle_dir)
        self.root = Path(bundle_dir)
        self.labels = metadata['labels']
        self.speaker_count = len(self.labels)
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.candidate_labels = self._normalize_candidate_labels(candidate_labels)
        self._candidate_indices = (
            None
            if self.candidate_labels is None
            else tuple(self.labels.index(label) for label in self.candidate_labels)
        )
        calibration = json.loads((self.root / 'calibration.json').read_text(encoding='utf-8'))
        preprocessing = json.loads((self.root / 'preprocessing.json').read_text(encoding='utf-8'))
        model_config = json.loads((self.root / 'model_config.json').read_text(encoding='utf-8'))
        self.temperature = float(calibration.get('temperature', 1.0))
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        self.input_mode = str(model_config.get('input_mode', 'waveform'))
        self.frame_hop_samples = int(preprocessing.get('frame_hop_samples', 320))
        # Pass a Python file handle instead of a Unicode path string.  On
        # Windows, the TorchScript C++ loader can fail with errno=2 when the
        # extracted project lives below a non-ASCII directory, even though
        # Python can read the file normally.  The file-like form avoids that
        # path conversion while preserving the same serialized model.
        with (self.root / 'classifier.pt').open('rb') as handle:
            self.model = torch.jit.load(handle, map_location=self.device).eval()

    def _normalize_candidate_labels(self, candidate_labels: Iterable[str] | str | None):
        if candidate_labels is None:
            return None
        if isinstance(candidate_labels, str):
            candidate_labels = candidate_labels.split(',')
        normalized = tuple(dict.fromkeys(str(label).strip() for label in candidate_labels if str(label).strip()))
        if not normalized:
            raise ValueError('candidate_labels must contain at least one speaker label')
        unknown = sorted(set(normalized) - set(self.labels))
        if unknown:
            raise ValueError('candidate_labels are not present in the bundle: ' + ', '.join(unknown))
        return normalized

    @property
    def output_speaker_count(self):
        return len(self.candidate_labels) if self.candidate_labels is not None else self.speaker_count

    def predict(self, waveform, codes=None):
        value = torch.as_tensor(waveform, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.inference_mode():
            if self.input_mode == 'codes_waveform':
                if codes is None:
                    raise ValueError('codes are required for codes_waveform classifier bundles')
                code_value = torch.as_tensor(codes, dtype=torch.long, device=self.device)
                if code_value.ndim == 2:
                    code_value = code_value.unsqueeze(0)
                if code_value.ndim != 3:
                    raise ValueError('codes must have shape [L,T] or [B,L,T]')
                logits = self.model(code_value, value)
            else:
                logits = self.model(value)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        scaled_logits = logits[0] / self.temperature
        if self._candidate_indices is not None:
            # A known-roster demo is a closed-set decision among its enrolled
            # participants, while the underlying model remains the 110-class
            # classifier. Non-roster classes are masked only for this output.
            masked_logits = torch.full_like(scaled_logits, float('-inf'))
            indices = torch.as_tensor(self._candidate_indices, dtype=torch.long, device=scaled_logits.device)
            masked_logits[indices] = scaled_logits[indices]
            scaled_logits = masked_logits
        probabilities = torch.softmax(scaled_logits, dim=-1).detach().cpu().numpy()
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        score = float(probabilities[best])
        margin = float(probabilities[best] - probabilities[second]) if second != best else score
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        return SpeakerPrediction(self.labels[best], score, margin, score >= self.threshold, scores)
