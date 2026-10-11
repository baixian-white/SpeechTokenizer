from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal


SAMPLE_RATE = 16_000
CROP_SECONDS = 3.0
N_FFT = 1_024
WIN_LENGTH = 1_024
HOP_LENGTH = 240
N_MELS = 80
LAYER = 3
PERTURBATION = "dropout-high"


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(frequency_mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(frequency_mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_edges = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, frequencies.size), dtype=np.float64)

    for index in range(n_mels):
        left, center, right = hz_edges[index : index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))

    # Match the Slaney area normalization used by the evaluation transform.
    filters *= (2.0 / np.maximum(hz_edges[2:] - hz_edges[:-2], np.finfo(float).eps))[:, None]
    return filters


def log_mel_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    _, _, stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False,
    )
    power = np.abs(stft) ** 2
    mel_power = mel_filterbank(sample_rate, N_FFT, N_MELS) @ power
    return np.log(np.maximum(mel_power, 1e-9))


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz: {path}")
    return audio, sample_rate


def highest_energy_crop(audio: np.ndarray, sample_rate: int, seconds: float) -> tuple[int, int]:
    window = int(round(seconds * sample_rate))
    if audio.size <= window:
        return 0, audio.size

    squared = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    window_energy = cumulative[window:] - cumulative[:-window]
    start = int(np.argmax(window_energy))
    return start, start + window


def select_representative_sample(metrics_path: Path, samples_root: Path) -> tuple[str, float, float]:
    metrics = pd.read_csv(
        metrics_path,
        usecols=["model", "sample_id", "L", "channel", "stoi"],
        low_memory=False,
    )
    subset = metrics[(metrics["L"] == LAYER) & (metrics["channel"] == PERTURBATION)]
    paired = subset.pivot(index="sample_id", columns="model", values="stoi").dropna()
    paired["delta"] = paired["lca"] - paired["base"]
    full_median = float(paired["delta"].median())

    saved_ids = sorted(path.stem for path in (samples_root / "base" / PERTURBATION / f"L{LAYER}").glob("*.wav"))
    saved = paired.loc[paired.index.intersection(saved_ids)].copy()
    if saved.empty:
        raise RuntimeError("No saved reconstruction samples overlap the metric table.")

    saved["distance"] = np.abs(saved["delta"] - full_median)
    selected_id = str(saved.sort_values(["distance"], kind="stable").index[0])
    selected_delta = float(saved.loc[selected_id, "delta"])
    return selected_id, full_median, selected_delta


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    experiment_root = (
        repo_root
        / "output"
        / "experiments"
        / "exp8_librispeech_test_full_perturb_20260606"
        / "eval_perturb_full_lca_nosave"
        / "test-clean"
    )
    metrics_path = experiment_root / "metrics" / "base_vs_lca_perturb_results.csv"
    samples_root = experiment_root / "samples"

    sample_id, full_median, selected_delta = select_representative_sample(metrics_path, samples_root)
    speaker_id, chapter_id, _ = sample_id.split("-", maxsplit=2)
    source_path = (
        repo_root
        / "data"
        / "SpeechPretrain"
        / "LibriSpeech"
        / "test-clean"
        / speaker_id
        / chapter_id
        / f"{sample_id}.flac"
    )

    rows = [
        ("Source speech", source_path),
        ("SCIT\nclean", samples_root / "base" / "clean" / f"L{LAYER}" / f"{sample_id}.wav"),
        ("SCIT-LCA\nclean", samples_root / "lca" / "clean" / f"L{LAYER}" / f"{sample_id}.wav"),
        (
            "SCIT\n10% index erasure",
            samples_root / "base" / PERTURBATION / f"L{LAYER}" / f"{sample_id}.wav",
        ),
        (
            "SCIT-LCA\n10% index erasure",
            samples_root / "lca" / PERTURBATION / f"L{LAYER}" / f"{sample_id}.wav",
        ),
    ]

    loaded = [(label, *load_mono(path), path) for label, path in rows]
    source_audio = loaded[0][1]
    crop_start, crop_end = highest_energy_crop(source_audio, SAMPLE_RATE, CROP_SECONDS)
    crop_duration = (crop_end - crop_start) / SAMPLE_RATE

    cropped_rows: list[tuple[str, np.ndarray, np.ndarray, Path]] = []
    for label, audio, sample_rate, path in loaded:
        if audio.size < crop_end:
            raise ValueError(f"Audio is shorter than the common crop: {path}")
        crop = audio[crop_start:crop_end]
        cropped_rows.append((label, crop, log_mel_spectrogram(crop, sample_rate), path))

    amplitude_limit = 1.05 * max(float(np.max(np.abs(crop))) for _, crop, _, _ in cropped_rows)
    spectrogram_max = max(float(np.max(spec)) for _, _, spec, _ in cropped_rows)
    spectrogram_min = spectrogram_max - 18.5
    time_axis = np.arange(crop_end - crop_start) / SAMPLE_RATE

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.65,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(7.25, 7.6), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=4,
        left=0.075,
        right=0.925,
        top=0.965,
        bottom=0.075,
        wspace=0.32,
        hspace=0.30,
    )

    block_positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (1, slice(0, 2)),
        (1, slice(2, 4)),
        (2, slice(1, 3)),
    ]
    waveform_color = "#2B6F92"
    image = None
    for block_index, ((grid_row, grid_columns), (label, crop, spectrogram, _)) in enumerate(
        zip(block_positions, cropped_rows)
    ):
        block = grid[grid_row, grid_columns].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=(0.72, 1.28),
            hspace=0.06,
        )
        waveform_ax = fig.add_subplot(block[0, 0])
        spectrogram_ax = fig.add_subplot(block[1, 0])

        waveform_ax.plot(time_axis, crop, color=waveform_color, linewidth=0.55, rasterized=True)
        waveform_ax.axhline(0.0, color="#7C8790", linewidth=0.35, alpha=0.65)
        waveform_ax.set_xlim(0.0, crop_duration)
        waveform_ax.set_ylim(-amplitude_limit, amplitude_limit)
        waveform_ax.set_yticks([-0.5, 0.0, 0.5])
        condition_title = label.replace("\n", " — ")
        waveform_ax.set_title(
            condition_title,
            loc="left",
            pad=4,
            fontsize=9.0,
            fontweight="bold",
            color="#1F2933",
        )
        waveform_ax.text(
            0.985,
            0.86,
            "Waveform",
            transform=waveform_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color="#53606C",
        )
        waveform_ax.set_xticklabels([])

        image = spectrogram_ax.imshow(
            spectrogram,
            origin="lower",
            aspect="auto",
            extent=(0.0, crop_duration, 0.0, SAMPLE_RATE / 2_000.0),
            cmap="magma",
            vmin=spectrogram_min,
            vmax=spectrogram_max,
            interpolation="nearest",
        )
        spectrogram_ax.set_xlim(0.0, crop_duration)
        spectrogram_ax.set_ylim(0.0, SAMPLE_RATE / 2_000.0)
        spectrogram_ax.set_yticks([0, 2, 4, 6, 8])
        spectrogram_ax.set_xlabel("Time (s)")
        spectrogram_ax.set_ylabel("Frequency (kHz)")
        spectrogram_ax.text(
            0.985,
            0.94,
            "Log-mel spectrogram",
            transform=spectrogram_ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            color="white",
            bbox={"boxstyle": "square,pad=0.18", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.72},
        )

        for axis in (waveform_ax, spectrogram_ax):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(length=2.5, width=0.55, color="#59636E")

    if image is not None:
        colorbar_ax = fig.add_axes((0.947, 0.19, 0.014, 0.62))
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.set_label("Log power", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2.5, width=0.55)
        colorbar.outline.set_linewidth(0.55)

    output_stem = script_dir / "appendix_reconstruction_comparison"
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    print(f"Selected sample: {sample_id}")
    print(f"Full-set median STOI improvement: {full_median:.6f}")
    print(f"Selected-sample STOI improvement: {selected_delta:.6f}")
    print(f"Common crop: {crop_start / SAMPLE_RATE:.3f}-{crop_end / SAMPLE_RATE:.3f} s")
    print(f"Wrote: {output_stem.with_suffix('.pdf')}")
    print(f"Wrote: {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
