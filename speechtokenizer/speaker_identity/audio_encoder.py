import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CodecAwareEcapaEncoder(nn.Module):
    def __init__(self, embedding_model, input_dim, ecapa_dim, output_dim=256, feature_extractor=None, normalizer=None, source=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.feature_extractor = feature_extractor
        self.normalizer = normalizer
        self.input_adapter = nn.Identity() if input_dim == ecapa_dim else nn.Linear(input_dim, ecapa_dim)
        self.projection = nn.Linear(ecapa_dim, output_dim)
        self.source = source
        self.configure_stage('A', [])

    @classmethod
    def from_hparams(cls, source, savedir, output_dim=256, device='cpu'):
        try:
            import speechbrain
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as error:
            raise RuntimeError('SpeechBrain is required; MFCC fallback is forbidden') from error
        classifier = EncoderClassifier.from_hparams(source=source, savedir=str(savedir), run_opts={'device': device})
        embedding_model = classifier.mods.embedding_model
        feature_extractor = classifier.mods.compute_features
        normalizer = classifier.mods.mean_var_norm
        ecapa_dim = int(getattr(classifier.hparams, 'lin_neurons', 192))
        model = cls(embedding_model, ecapa_dim, ecapa_dim, output_dim, feature_extractor, normalizer, source)
        model.speechbrain_version = speechbrain.__version__
        return model

    def configure_stage(self, stage, trainable_patterns):
        if stage not in ('A', 'B'):
            raise ValueError('ECAPA stage must be A or B')
        for parameter in self.embedding_model.parameters():
            parameter.requires_grad_(False)
        if stage == 'B':
            for name, parameter in self.named_parameters():
                if name.startswith('embedding_model.') and any(pattern in name for pattern in trainable_patterns):
                    parameter.requires_grad_(True)
        for module in (self.input_adapter, self.projection):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.stage = stage
        self.trainable_patterns = list(trainable_patterns)
        if stage == 'A':
            self.embedding_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'stage', None) == 'A':
            self.embedding_model.eval()
        return self

    def trainable_parameter_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def provenance(self, artifact_paths=()):
        hashes = {}
        for path in artifact_paths:
            value = Path(path)
            digest = hashlib.sha256(value.read_bytes()).hexdigest()
            hashes[str(value)] = digest
        return {
            'source': self.source,
            'speechbrain_version': getattr(self, 'speechbrain_version', None),
            'stage': self.stage,
            'trainable_parameter_names': self.trainable_parameter_names(),
            'artifact_hashes': hashes,
            'preprocessing': {'feature_extractor': type(self.feature_extractor).__name__, 'normalizer': type(self.normalizer).__name__},
        }

    def forward(self, waveform_or_features, lengths=None):
        value = waveform_or_features
        if self.feature_extractor is not None:
            value = self.feature_extractor(value)
        if self.normalizer is not None:
            if lengths is None:
                lengths = torch.ones(value.shape[0], device=value.device)
            value = self.normalizer(value, lengths)
        value = self.input_adapter(value)
        embedding = self.embedding_model(value)
        while embedding.ndim > 2 and embedding.shape[1] == 1:
            embedding = embedding[:, 0]
        if embedding.ndim == 3:
            embedding = embedding.mean(dim=1)
        return F.normalize(self.projection(embedding), dim=-1)
