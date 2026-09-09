# Exp22 Speaker Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline speaker-identity experiments plus optional speaker recognition in the three-user demo.

**Architecture:** Shared speaker utility code lives in `scripts/speaker_identity_utils.py`. Offline experiments are two flat `scripts/` entrypoints following the existing run-dir/CSV/JSON/Markdown convention. Demo speaker recognition is isolated in `3用户demo/.../speaker_identity.py` and called from `group_client.py`; `router_server.py` remains unchanged.

**Tech Stack:** Python, torch, torchaudio, numpy, scipy, scikit-learn, soundfile, existing SCIT-Speech model helpers.

---

### Task 1: Shared Speaker Utilities

**Files:**
- Create: `scripts/speaker_identity_utils.py`
- Test: `tests/test_speaker_identity_utils.py`

- [x] Add speaker-id parsing for VCTK, AISHELL, and LibriSpeech paths.
- [x] Add MFCC statistics embeddings that work without external speaker packages.
- [x] Add cosine scoring, profile enrollment, prediction, EER, and TAR@FAR helpers.
- [x] Add deterministic per-speaker split helpers.
- [x] Add unit tests for parsing, scoring, and synthetic profile classification.

### Task 2: Offline Speaker Preservation Script

**Files:**
- Create: `scripts/evaluate_speaker_identity.py`

- [x] Load Base/LCA with the same helpers used by `evaluate_clean_large_nosave.py`.
- [x] Read a sample list or scan an audio root, group by speaker, build enrollment/test splits.
- [x] Evaluate original, Base `L=1/2/3`, and LCA `L=1/2/3`.
- [x] Write per-sample CSV, summary CSV, JSON metadata, and Markdown report.

### Task 3: Offline Codes/Latent Probe Script

**Files:**
- Create: `scripts/evaluate_speaker_probe.py`

- [x] Extract `codes_hist` features from `model.encode()`.
- [x] Extract `latent_stats` features from `model.forward_feature()`.
- [x] Train/evaluate scikit-learn probes with speaker-stratified splits.
- [x] Write per-condition CSV, JSON metadata, and Markdown report.

### Task 4: Three-User Demo Speaker Recognition

**Files:**
- Create: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/speaker_identity.py`
- Modify: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/group_client.py`
- Modify: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/三用户中心路由通信demo使用说明.md`

- [x] Load profile wavs from `speaker_profile_dir/<speaker_id>/*.wav`.
- [x] Maintain a rolling decoded-audio buffer per incoming sender.
- [x] Print predicted speaker, score, margin, and verified flag in monitor output.
- [x] Add summary CSV fields for speaker ID status and accuracy.
- [x] Document the new CLI flags and profile directory format.

### Task 5: Verification

**Files:**
- Modify as needed based on test failures.

- [x] Run `python -m unittest tests.test_speaker_identity_utils`.
- [x] Run syntax/import checks for the new scripts and demo module.
- [x] Run a small offline smoke test on a tiny VCTK subset if local models and audio are available.
