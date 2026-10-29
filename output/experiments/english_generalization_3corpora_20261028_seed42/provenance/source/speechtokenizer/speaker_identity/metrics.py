import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def assert_evaluation_allowed(split, allow_test_evaluation=False):
    if split == 'test' and not allow_test_evaluation:
        raise PermissionError('test evaluation requires --allow-test-evaluation')


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(validation_logits, validation_labels):
    logits = np.asarray(validation_logits, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.int64)
    candidates = np.exp(np.linspace(np.log(0.05), np.log(10.0), 400))
    losses = [log_loss(labels, _softmax(logits / value), labels=np.arange(logits.shape[1])) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def classification_metrics(logits, labels, ece_bins=15):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = _softmax(logits)
    predictions = probabilities.argmax(axis=1)
    top_count = min(5, logits.shape[1])
    top_indices = np.argpartition(logits, -top_count, axis=1)[:, -top_count:]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    per_speaker = {}
    for speaker in np.unique(labels):
        mask = labels == speaker
        per_speaker[str(int(speaker))] = float(correct[mask].mean())
    values = sorted(per_speaker.values())
    worst_count = max(1, int(np.ceil(len(values) * 0.1)))
    return {
        'top1': float(accuracy_score(labels, predictions)),
        'top5': float(np.mean([label in row for label, row in zip(labels, top_indices)])),
        'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'nll': float(log_loss(labels, probabilities, labels=np.arange(logits.shape[1]))),
        'ece_15bin': float(ece),
        'per_speaker_accuracy': per_speaker,
        'worst_decile_accuracy': float(np.mean(values[:worst_count])),
    }


def hierarchical_bootstrap_accuracy(rows, replicates=2000, seed=42):
    if not rows:
        raise ValueError('rows must not be empty')
    rng = np.random.default_rng(seed)
    by_speaker = {}
    for row in rows:
        by_speaker.setdefault(row['speaker_id'], []).append(row)
    speakers = list(by_speaker)
    estimates = []
    for _ in range(replicates):
        values = []
        for speaker in rng.choice(speakers, size=len(speakers), replace=True):
            items = by_speaker[speaker]
            sampled = rng.choice(items, size=len(items), replace=True)
            values.extend(float(item['correct']) for item in sampled)
        estimates.append(float(np.mean(values)))
    point = float(np.mean([float(row['correct']) for row in rows]))
    return {'point_estimate': point, 'ci_low': float(np.percentile(estimates, 2.5)), 'ci_high': float(np.percentile(estimates, 97.5))}
