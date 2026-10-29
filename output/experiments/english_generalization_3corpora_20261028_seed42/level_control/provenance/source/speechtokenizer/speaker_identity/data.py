from __future__ import annotations

import hashlib
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Optional


AUDIO_EXTENSIONS = {".flac", ".wav"}
MICROPHONE_SUFFIX = re.compile(r"_mic([12])$")
CANONICAL_SPEAKER_COUNT = 110
TEXT_CONTROLLED_SPEAKER_COUNT = 109
TEXT_CONTROLLED_EXCLUDED_SPEAKERS = frozenset({"p315"})


@dataclass(frozen=True)
class SpeakerSample:
    speaker_id: str
    utterance_group: str
    audio_path: Path
    transcript: Optional[str]
    transcript_hash: Optional[str]


def normalize_utterance_group(audio_path):
    return MICROPHONE_SUFFIX.sub("", Path(audio_path).stem)


def _microphone_name(audio_path):
    match = MICROPHONE_SUFFIX.search(Path(audio_path).stem)
    return None if match is None else f"mic{match.group(1)}"


def discover_audio_groups(audio_root, microphone="mic1"):
    audio_root = Path(audio_root)
    if not audio_root.is_dir():
        raise ValueError(f"audio root does not exist: {audio_root}")
    if microphone is not None:
        microphone = str(microphone).lower().lstrip("_")
        if microphone not in {"mic1", "mic2"}:
            raise ValueError(f"unsupported microphone: {microphone}")

    selected = {}
    for speaker_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        for audio_path in sorted(path for path in speaker_dir.iterdir() if path.is_file()):
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if microphone is not None and _microphone_name(audio_path) != microphone:
                continue
            utterance_group = normalize_utterance_group(audio_path)
            key = (speaker_dir.name, utterance_group)
            candidate = SpeakerSample(
                speaker_id=speaker_dir.name,
                utterance_group=utterance_group,
                audio_path=audio_path,
                transcript=None,
                transcript_hash=None,
            )
            previous = selected.get(key)
            if previous is None or str(candidate.audio_path) < str(previous.audio_path):
                selected[key] = candidate
    return [selected[key] for key in sorted(selected)]


def normalize_transcript(transcript):
    return " ".join(str(transcript).lower().split())


def transcript_sha256(transcript):
    normalized = normalize_transcript(transcript)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_path_component(value, label):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or bool(PureWindowsPath(value).drive)
    ):
        raise ValueError(f"unsafe {label}: {value!r}")


def _load_exact_transcripts(samples, transcript_root):
    transcript_root = Path(transcript_root).resolve(strict=False)
    samples = list(samples)
    for sample in samples:
        _validate_path_component(sample.speaker_id, "speaker_id")
        _validate_path_component(sample.utterance_group, "utterance_group")
    resolved = []
    for sample in _collapse_samples(samples):
        transcript_path = (
            transcript_root / sample.speaker_id / f"{sample.utterance_group}.txt"
        ).resolve(strict=False)
        try:
            transcript_path.relative_to(transcript_root)
        except ValueError as error:
            raise ValueError(f"unsafe transcript path: {transcript_path}") from error
        if not transcript_path.is_file():
            raise ValueError(
                f"missing exact transcript for {sample.speaker_id}/{sample.utterance_group}: "
                f"{transcript_path}"
            )
        transcript = normalize_transcript(transcript_path.read_text(encoding="utf-8"))
        if not transcript:
            raise ValueError(f"empty exact transcript: {transcript_path}")
        resolved.append(
            replace(
                sample,
                transcript=transcript,
                transcript_hash=transcript_sha256(transcript),
            )
        )
    return resolved


def _collapse_samples(samples):
    selected = {}
    for sample in samples:
        audio_path = Path(sample.audio_path)
        utterance_group = normalize_utterance_group(Path(sample.utterance_group))
        normalized = replace(
            sample,
            utterance_group=utterance_group,
            audio_path=audio_path,
        )
        key = (normalized.speaker_id, normalized.utterance_group)
        previous = selected.get(key)
        if previous is None or str(normalized.audio_path) < str(previous.audio_path):
            selected[key] = normalized
    return [selected[key] for key in sorted(selected)]


def _normalize_and_validate_unique_audio_paths(samples):
    normalized_samples = []
    seen_paths = {}
    for sample in list(samples):
        raw_path = Path(sample.audio_path).expanduser()
        try:
            normalized_path = raw_path.resolve(strict=False)
        except OSError:
            normalized_path = raw_path.absolute()
        path_key = os.path.normcase(str(normalized_path))
        if path_key in seen_paths:
            previous = seen_paths[path_key]
            raise ValueError(
                f"duplicate audio path for {previous.speaker_id}/{previous.utterance_group} "
                f"and {sample.speaker_id}/{sample.utterance_group}: {normalized_path}"
            )
        seen_paths[path_key] = sample
        normalized_samples.append(replace(sample, audio_path=normalized_path))
    return normalized_samples


def _validate_audio_samples(samples):
    for sample in samples:
        suffix = sample.audio_path.suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"unsupported audio extension for {sample.speaker_id}/"
                f"{sample.utterance_group}: {sample.audio_path}"
            )
        try:
            with sample.audio_path.open("rb") as audio_file:
                header = audio_file.read(12)
        except FileNotFoundError as error:
            raise ValueError(
                f"missing audio for {sample.speaker_id}/{sample.utterance_group}: "
                f"{sample.audio_path}"
            ) from error
        except (PermissionError, OSError) as error:
            raise ValueError(
                f"audio read/permission error for {sample.speaker_id}/"
                f"{sample.utterance_group}: {sample.audio_path}"
            ) from error
        if suffix == ".flac" and not header.startswith(b"fLaC"):
            raise ValueError(
                f"invalid FLAC signature for {sample.speaker_id}/"
                f"{sample.utterance_group}: {sample.audio_path}"
            )
        if suffix == ".wav" and not (
            header.startswith(b"RIFF") and header[8:12] == b"WAVE"
        ):
            raise ValueError(
                f"invalid WAV signature for {sample.speaker_id}/"
                f"{sample.utterance_group}: {sample.audio_path}"
            )


def _sample_dict(sample):
    return {
        "speaker_id": sample.speaker_id,
        "utterance_group": sample.utterance_group,
        "audio_path": str(sample.audio_path),
        "transcript": sample.transcript,
        "transcript_hash": sample.transcript_hash,
    }


def _group_by_speaker(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.speaker_id].append(sample)
    return {speaker: sorted(values, key=lambda item: item.utterance_group) for speaker, values in sorted(grouped.items())}


def _validate_speaker_set(by_speaker, expected_speaker_count, required_speakers, split_name):
    speakers = set(by_speaker)
    if len(speakers) != expected_speaker_count:
        raise ValueError(
            f"{split_name} requires exactly {expected_speaker_count} speakers; "
            f"found {len(speakers)}"
        )
    missing = sorted(set(required_speakers) - speakers)
    if missing:
        raise ValueError(f"{split_name} missing required speakers: {missing}")


def _validate_split_counts(train_count, validation_count, test_count):
    counts = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    for name, count in counts.items():
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} count must be a non-negative int")
    if sum(counts.values()) <= 0:
        raise ValueError("split counts must sum to more than zero")


def _select_exact(values, count, seed):
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda item: item.utterance_group)


def _assert_group_disjoint(splits):
    group_sets = {
        name: {(sample.speaker_id, sample.utterance_group) for sample in values}
        for name, values in splits.items()
    }
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise RuntimeError(f"utterance groups cross {left}/{right}: {sorted(overlap)[:3]}")


def build_canonical_split(
    samples_or_audio_root,
    train_count=80,
    validation_count=15,
    test_count=15,
    seed=42,
    microphone="mic1",
):
    if not isinstance(samples_or_audio_root, (str, Path)):
        raise TypeError("build_canonical_split requires an audio root Path or str")
    samples = discover_audio_groups(samples_or_audio_root, microphone=microphone)
    return _build_canonical_split_for_samples(
        samples,
        train_count,
        validation_count,
        test_count,
        seed=seed,
        expected_speaker_count=CANONICAL_SPEAKER_COUNT,
        required_speakers={"s5"},
    )


def _build_canonical_split_for_samples(
    samples,
    train_count,
    validation_count,
    test_count,
    seed,
    expected_speaker_count,
    required_speakers,
):
    _validate_split_counts(train_count, validation_count, test_count)
    samples = _normalize_and_validate_unique_audio_paths(samples)
    samples = _collapse_samples(samples)
    if not samples:
        raise ValueError("canonical split has no audio samples")
    by_speaker = _group_by_speaker(samples)
    _validate_speaker_set(
        by_speaker,
        expected_speaker_count,
        required_speakers,
        "canonical split",
    )
    _validate_audio_samples(samples)
    required = train_count + validation_count + test_count
    available = {speaker: len(values) for speaker, values in by_speaker.items()}
    splits = {"train": [], "validation": [], "test": []}

    for speaker, values in by_speaker.items():
        if len(values) < required:
            raise ValueError(
                f"speaker {speaker} shortage: requires {required} groups, found {len(values)}"
            )
        shuffled = list(values)
        random.Random(f"{seed}:{speaker}").shuffle(shuffled)
        boundaries = (train_count, train_count + validation_count, required)
        splits["train"].extend(shuffled[:boundaries[0]])
        splits["validation"].extend(shuffled[boundaries[0]:boundaries[1]])
        splits["test"].extend(shuffled[boundaries[1]:boundaries[2]])

    _assert_group_disjoint(splits)
    for values in splits.values():
        values.sort(key=lambda item: (item.speaker_id, item.utterance_group))
    speakers = sorted(by_speaker)
    return {
        "protocol": {
            "name": "exp23_canonical_110_class",
            "speaker_count": CANONICAL_SPEAKER_COUNT,
            "actual_speaker_count": len(speakers),
            "expected_speaker_count": expected_speaker_count,
            "required_speakers": sorted(required_speakers),
            "audio_validation": "container_signature",
        },
        "seed": seed,
        "speakers": speakers,
        "excluded_speakers": [],
        "splits": {name: [_sample_dict(item) for item in values] for name, values in splits.items()},
        "counts": {
            "per_speaker": {"train": train_count, "validation": validation_count, "test": test_count},
            "total": {name: len(values) for name, values in splits.items()},
            "available_by_speaker": available,
            "minimum_available_groups": min(available.values()),
        },
    }


def _prepare_in_memory_text_samples(samples):
    prepared = []
    for sample in _collapse_samples(samples):
        if sample.transcript is None:
            raise ValueError(f"missing transcript for {sample.speaker_id}/{sample.utterance_group}")
        transcript = normalize_transcript(sample.transcript)
        if not transcript:
            raise ValueError(f"empty transcript for {sample.speaker_id}/{sample.utterance_group}")
        prepared.append(replace(sample, transcript=transcript, transcript_hash=transcript_sha256(transcript)))
    return prepared


def build_text_controlled_split(
    samples_or_audio_root,
    train_count=80,
    validation_count=10,
    test_count=10,
    seed=1,
    transcript_root=None,
    microphone="mic1",
):
    if not isinstance(samples_or_audio_root, (str, Path)):
        raise TypeError("build_text_controlled_split requires an audio root Path or str")
    if transcript_root is None:
        raise ValueError("transcript_root is required when building from audio_root")
    original_samples = discover_audio_groups(samples_or_audio_root, microphone=microphone)
    original_speakers = {sample.speaker_id for sample in original_samples}
    if "p315" not in original_speakers:
        raise ValueError("text-controlled split requires p315 before exclusion")
    eligible = [
        sample
        for sample in original_samples
        if sample.speaker_id not in TEXT_CONTROLLED_EXCLUDED_SPEAKERS
    ]
    eligible = _normalize_and_validate_unique_audio_paths(eligible)
    _validate_speaker_set(
        _group_by_speaker(eligible),
        TEXT_CONTROLLED_SPEAKER_COUNT,
        set(),
        "text-controlled split",
    )
    _validate_audio_samples(eligible)
    samples = _load_exact_transcripts(eligible, transcript_root)
    return _build_text_controlled_split_for_samples(
        samples,
        train_count,
        validation_count,
        test_count,
        seed=seed,
        expected_speaker_count=TEXT_CONTROLLED_SPEAKER_COUNT,
    )


def _build_text_controlled_split_for_samples(
    samples,
    train_count,
    validation_count,
    test_count,
    seed,
    expected_speaker_count,
):
    _validate_split_counts(train_count, validation_count, test_count)
    samples = _normalize_and_validate_unique_audio_paths(samples)
    samples = _prepare_in_memory_text_samples(samples)
    if not samples:
        raise ValueError("text-controlled split has no eligible samples")
    by_speaker = _group_by_speaker(samples)
    _validate_speaker_set(
        by_speaker,
        expected_speaker_count,
        set(),
        "text-controlled split",
    )

    transcript_to_hash = {sample.transcript: sample.transcript_hash for sample in samples}
    transcripts = sorted(transcript_to_hash)
    random.Random(seed).shuffle(transcripts)
    train_boundary = int(len(transcripts) * 0.8)
    validation_boundary = train_boundary + int(len(transcripts) * 0.1)
    hash_split = {}
    for transcript in transcripts[:train_boundary]:
        hash_split[transcript_to_hash[transcript]] = "train"
    for transcript in transcripts[train_boundary:validation_boundary]:
        hash_split[transcript_to_hash[transcript]] = "validation"
    for transcript in transcripts[validation_boundary:]:
        hash_split[transcript_to_hash[transcript]] = "test"

    available_by_speaker = {}
    splits = {"train": [], "validation": [], "test": []}
    requested = {"train": train_count, "validation": validation_count, "test": test_count}
    for speaker, speaker_samples in by_speaker.items():
        buckets = {"train": [], "validation": [], "test": []}
        for sample in speaker_samples:
            buckets[hash_split[sample.transcript_hash]].append(sample)
        available_by_speaker[speaker] = {name: len(values) for name, values in buckets.items()}
        for split_name, count in requested.items():
            if len(buckets[split_name]) < count:
                raise ValueError(
                    f"speaker {speaker} text split shortage in {split_name}: "
                    f"requires {count}, found {len(buckets[split_name])}"
                )
            splits[split_name].extend(
                _select_exact(buckets[split_name], count, f"{seed}:{speaker}:{split_name}")
            )

    _assert_group_disjoint(splits)
    split_hashes = {
        name: sorted({sample.transcript_hash for sample in values})
        for name, values in splits.items()
    }
    hash_sets = {name: set(values) for name, values in split_hashes.items()}
    if hash_sets["train"] & hash_sets["validation"]:
        raise RuntimeError("transcript hashes cross train/validation")
    if hash_sets["train"] & hash_sets["test"]:
        raise RuntimeError("transcript hashes cross train/test")
    if hash_sets["validation"] & hash_sets["test"]:
        raise RuntimeError("transcript hashes cross validation/test")
    for values in splits.values():
        values.sort(key=lambda item: (item.speaker_id, item.utterance_group))
    speakers = sorted(by_speaker)
    minimum_available = {
        name: min(counts[name] for counts in available_by_speaker.values())
        for name in requested
    }
    return {
        "protocol": {
            "name": "exp23_text_controlled_109_class",
            "speaker_count": TEXT_CONTROLLED_SPEAKER_COUNT,
            "canonical_speaker_count": CANONICAL_SPEAKER_COUNT,
            "actual_speaker_count": len(speakers),
            "expected_speaker_count": expected_speaker_count,
            "audio_validation": "container_signature",
        },
        "seed": seed,
        "speakers": speakers,
        "excluded_speakers": sorted(TEXT_CONTROLLED_EXCLUDED_SPEAKERS),
        "splits": {name: [_sample_dict(item) for item in values] for name, values in splits.items()},
        "transcript_hashes": split_hashes,
        "counts": {
            "per_speaker": requested,
            "total": {name: len(values) for name, values in splits.items()},
            "available_by_speaker": available_by_speaker,
            "minimum_available_by_split": minimum_available,
            "global_unique_transcript_hashes": len(transcripts),
        },
    }
