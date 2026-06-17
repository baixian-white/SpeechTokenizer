"""Encoder-only NAS handoff helpers.

Exp1 search candidates may carry geometry-matched decoder fields for proxy
diagnostics. The downstream Exp2 handoff must describe the selected encoder
without letting those diagnostic decoder fields change the decoder route.
"""

from __future__ import annotations

import ast
import copy
from typing import Any, Dict, Iterable, List, Optional

try:
    from .search_space import normalize_candidate
except ImportError:
    from search_space import normalize_candidate


ENCODER_HANDOFF_KEYS = [
    "candidate_id",
    "search_mode",
    "seed",
    "sample_index",
    "sample_attempt",
    "candidate_signature",
    "encoder_strides",
    "seanet_ratios_arg",
    "n_filters",
    "compress",
    "lstm",
    "activation",
    "layer_ops_list",
    "layer_se_list",
    "decoder_ops",
    "dimension",
    "sample_rate",
    "encoder_downsample_rate",
    "latent_rate",
    "n_q",
    "codebook_size",
]


def _as_list(value: Any, fallback: Optional[Iterable[Any]] = None) -> List[Any]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(fallback or [])
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return list(fallback or [])
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    return list(fallback or [])


def _base_decoder_strides(base_config: Dict[str, Any]) -> List[int]:
    return [int(x) for x in _as_list(base_config.get("strides"), [8, 5, 4, 2])]


def build_encoder_only_handoff_config(
    candidate: Dict[str, Any],
    selected_row: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the downstream ``best_seanet_config.json`` for Exp2.

    ``candidate`` is the raw NAS search item. ``selected_row`` is the evaluated
    metrics row whose decoder fields reflect the actual proxy condition.
    """

    selected_row = selected_row or {}
    base_config = base_config or {}
    handoff = {key: copy.deepcopy(candidate[key]) for key in ENCODER_HANDOFF_KEYS if key in candidate}

    source_decoder_strides = _as_list(candidate.get("decoder_strides"))
    source_decoder_condition = candidate.get("decoder_condition")
    selected_decoder_strides = _as_list(selected_row.get("decoder_strides"), _base_decoder_strides(base_config))
    selected_decoder_condition = selected_row.get("decoder_condition") or "base_config_decoder"

    handoff.update(
        {
            "encoder_only": True,
            "handoff_schema": "encoder_only_nas_v1",
            "decoder_condition": selected_decoder_condition,
            "decoder_strides": [int(x) for x in selected_decoder_strides],
            "source_candidate_decoder_condition": source_decoder_condition,
            "source_candidate_decoder_strides": [int(x) for x in source_decoder_strides],
        }
    )
    return handoff


def normalize_encoder_only_nas_config(nas_conf: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a NAS encoder config while preserving encoder-only decoder scope."""

    item = copy.deepcopy(nas_conf)
    if "encoder_strides" not in item:
        if "ratios" in item:
            item["encoder_strides"] = list(item["ratios"])
        else:
            item["encoder_strides"] = _base_decoder_strides(base_config)
    if "search_mode" not in item:
        item["search_mode"] = "loaded_config"
    if "candidate_id" not in item:
        item["candidate_id"] = "loaded_nas_encoder"
    if "seed" not in item:
        item["seed"] = base_config.get("seed")
    if "sample_index" not in item:
        item["sample_index"] = 0

    explicit_encoder_only = "encoder_only" in item
    encoder_only = bool(item.get("encoder_only", True))
    requested_decoder_condition = item.get("decoder_condition") if explicit_encoder_only else None
    requested_decoder_strides = _as_list(item.get("decoder_strides")) if explicit_encoder_only else []
    source_decoder_condition = item.get("source_candidate_decoder_condition", item.get("decoder_condition"))
    source_decoder_strides = _as_list(item.get("source_candidate_decoder_strides"), item.get("decoder_strides"))

    normalized = normalize_candidate(item)
    if not encoder_only:
        normalized["encoder_only"] = False
        return normalized

    if requested_decoder_condition and requested_decoder_condition != "geometry_matched_decoder":
        decoder_condition = requested_decoder_condition
        decoder_strides = requested_decoder_strides or _base_decoder_strides(base_config)
    else:
        decoder_condition = "base_config_decoder"
        decoder_strides = _base_decoder_strides(base_config)

    normalized["encoder_only"] = True
    normalized["decoder_condition"] = decoder_condition
    normalized["decoder_strides"] = [int(x) for x in decoder_strides]
    normalized["source_candidate_decoder_condition"] = source_decoder_condition
    normalized["source_candidate_decoder_strides"] = [int(x) for x in source_decoder_strides]
    if "handoff_schema" in item:
        normalized["handoff_schema"] = item["handoff_schema"]
    return normalized
