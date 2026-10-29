import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, bundle_dir, device='cpu', threshold=0.65):
        metadata = validate_export_bundle(bundle_dir)
        self.root = Path(bundle_dir)
        self.labels = metadata['labels']
        self.speaker_count = len(self.labels)
        self.device = torch.device(device)
        self.threshold = float(threshold)
        calibration = json.loads((self.root / 'calibration.json').read_text(encoding='utf-8'))
        preprocessing = json.loads((self.root / 'preprocessing.json').read_text(encoding='utf-8'))
        model_config = json.loads((self.root / 'model_config.json').read_text(encoding='utf-8'))
        self.temperature = float(calibration.get('temperature', 1.0))
        if self.temperature <= 0:
            raise ValueError('temperature must be positive')
        self.input_mode = str(model_config.get('input_mode', 'waveform'))
        self.frame_hop_samples = int(preprocessing.get('frame_hop_samples', 320))
        self.model = torch.jit.load(str(self.root / 'classifier.pt'), map_location=self.device).eval()

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
        probabilities = torch.softmax(logits[0] / self.temperature, dim=-1).detach().cpu().numpy()
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        score = float(probabilities[best])
        margin = float(probabilities[best] - probabilities[second]) if second != best else score
        scores = {label: float(probabilities[index]) for index, label in enumerate(self.labels)}
        return SpeakerPrediction(self.labels[best], score, margin, score >= self.threshold, scores)
