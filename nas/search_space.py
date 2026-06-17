"""Search-space helpers for Exp1 encoder-side NAS.

The search space is intentionally restricted to the transmitter-side encoder
before latent Z. It keeps the index-transmission interface fixed while allowing
the four-stage encoder stride schedule to vary as long as the product remains
320.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any, Dict, Iterable, List, Optional


OPERATOR_LIBRARY: Dict[str, str] = {
    "std_k3": "standard 1D convolution residual op, kernel_size=3",
    "std_k5": "standard 1D convolution residual op, kernel_size=5",
    "std_k7": "standard 1D convolution residual op, kernel_size=7",
    "sep_k3": "depthwise-separable 1D convolution residual op, kernel_size=3",
    "sep_k5": "depthwise-separable 1D convolution residual op, kernel_size=5",
    "sep_k7": "depthwise-separable 1D convolution residual op, kernel_size=7",
    "sep_k9": "depthwise-separable 1D convolution residual op, kernel_size=9",
    "dil_k3": "dilated 1D convolution residual op, kernel_size=3, dilation_factor=2",
    "dil_k5": "dilated 1D convolution residual op, kernel_size=5, dilation_factor=2",
    "dil_k9": "dilated 1D convolution residual op, kernel_size=9, dilation_factor=2",
    "pw_bottleneck_k3": "pointwise bottleneck residual op with kernel_size=3 temporal mixing",
    "skip": "identity residual op",
}


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "fixed": {
        "sample_rate": 16000,
        "encoder_downsample_rate": 320,
        "latent_rate": 50,
        "latent_dimension": 1024,
        "n_q": 3,
        "codebook_size": 1024,
    },
    "macro_space": {
        "encoder_strides": [
            [8, 5, 4, 2],
            [5, 4, 4, 4],
            [10, 4, 4, 2],
            [4, 5, 4, 4],
            [4, 4, 5, 4],
            [4, 4, 4, 5],
        ],
        "n_filters": [16, 24, 32, 48],
        "compress": [2, 4],
        "lstm": [1, 2],
        "activation": ["ELU", "Snake"],
    },
    "block_space": {
        "num_blocks": 4,
        "ops": list(OPERATOR_LIBRARY.keys()),
        "se": [False, True],
        "constraints": {
            "max_skip_blocks": 2,
            "skip_forces_se_false": True,
        },
    },
    "decoder_policy": {
        "condition": "geometry_matched_decoder",
        "decoder_strides": "reverse(encoder_strides)",
        "decoder_ops": "not_searched",
        "decoder_width_depth_activation": "fixed_to_base_config",
    },
    "search_policy": {
        "mode": "random",
        "num_candidates": 32,
        "seed": 42,
        "deduplicate": True,
    },
}


class CandidateValidationError(ValueError):
    """Raised when a NAS candidate violates the fixed interface or constraints."""


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def actual_encoder_to_seanet_ratios(encoder_strides: List[int]) -> List[int]:
    """Return the ratios argument that yields the requested actual encoder order.

    The local NAS SEANetEncoder reverses its ``ratios`` argument internally, so
    an actual encoder schedule such as [5, 4, 4, 4] must be passed as
    [4, 4, 4, 5].
    """

    return list(reversed([int(x) for x in encoder_strides]))


def decoder_strides_for_encoder(encoder_strides: List[int]) -> List[int]:
    return list(reversed([int(x) for x in encoder_strides]))


def expected_latent_frames(num_samples: int, sample_rate: int = 16000, downsample_rate: int = 320) -> int:
    return int(math.ceil(float(num_samples) / float(downsample_rate)))


def _require_list(candidate: Dict[str, Any], key: str) -> List[Any]:
    value = candidate.get(key)
    if not isinstance(value, list):
        raise CandidateValidationError(f"{key} must be a list")
    return value


def candidate_signature(candidate: Dict[str, Any]) -> str:
    keys = [
        "encoder_strides",
        "n_filters",
        "compress",
        "lstm",
        "activation",
        "layer_ops_list",
        "layer_se_list",
    ]
    payload = {key: candidate.get(key) for key in keys}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_candidate(candidate: Dict[str, Any], search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    search_space = search_space or DEFAULT_SEARCH_SPACE
    fixed = search_space["fixed"]
    constraints = search_space["block_space"]["constraints"]
    allowed_ops = set(search_space["block_space"]["ops"])

    item = copy.deepcopy(candidate)
    encoder_strides = [int(x) for x in _require_list(item, "encoder_strides")]
    if len(encoder_strides) != 4:
        raise CandidateValidationError(f"encoder_strides must have length 4, got {encoder_strides}")
    if product_int(encoder_strides) != int(fixed["encoder_downsample_rate"]):
        raise CandidateValidationError(
            f"encoder_strides product must be {fixed['encoder_downsample_rate']}, got {product_int(encoder_strides)}"
        )

    ops = [str(x) for x in _require_list(item, "layer_ops_list")]
    se = [bool(x) for x in _require_list(item, "layer_se_list")]
    num_blocks = int(search_space["block_space"]["num_blocks"])
    if len(ops) != num_blocks:
        raise CandidateValidationError(f"layer_ops_list length must be {num_blocks}, got {len(ops)}")
    if len(se) != len(ops):
        raise CandidateValidationError("layer_se_list length must match layer_ops_list")

    unknown_ops = sorted(set(ops) - allowed_ops)
    if unknown_ops:
        raise CandidateValidationError(f"unknown NAS ops: {unknown_ops}")

    max_skip = int(constraints["max_skip_blocks"])
    if ops.count("skip") > max_skip:
        raise CandidateValidationError(f"skip count must be <= {max_skip}, got {ops.count('skip')}")
    if constraints.get("skip_forces_se_false", True):
        se = [False if op == "skip" else flag for op, flag in zip(ops, se)]

    item["encoder_strides"] = encoder_strides
    item["decoder_strides"] = decoder_strides_for_encoder(encoder_strides)
    item["decoder_condition"] = search_space["decoder_policy"]["condition"]
    item["decoder_ops"] = search_space["decoder_policy"]["decoder_ops"]
    item["seanet_ratios_arg"] = actual_encoder_to_seanet_ratios(encoder_strides)
    item["dimension"] = int(fixed["latent_dimension"])
    item["sample_rate"] = int(fixed["sample_rate"])
    item["encoder_downsample_rate"] = int(fixed["encoder_downsample_rate"])
    item["latent_rate"] = int(fixed["latent_rate"])
    item["n_q"] = int(fixed["n_q"])
    item["codebook_size"] = int(fixed["codebook_size"])
    item["layer_ops_list"] = ops
    item["layer_se_list"] = se
    item["candidate_signature"] = candidate_signature(item)

    for key in ["n_filters", "compress", "lstm"]:
        if key not in item:
            raise CandidateValidationError(f"{key} is required")
        item[key] = int(item[key])
    if item["lstm"] not in search_space["macro_space"]["lstm"]:
        raise CandidateValidationError(
            f"unsupported lstm: {item['lstm']}; allowed={search_space['macro_space']['lstm']}"
        )
    if item.get("activation") not in search_space["macro_space"]["activation"]:
        raise CandidateValidationError(f"unsupported activation: {item.get('activation')}")

    return item


def sample_candidates(
    seed: int,
    num_candidates: int,
    search_space: Optional[Dict[str, Any]] = None,
    prefix: Optional[str] = None,
    max_attempts_multiplier: int = 100,
) -> List[Dict[str, Any]]:
    """Sample unique valid NAS candidates from the configured search space."""

    search_space = search_space or DEFAULT_SEARCH_SPACE
    rng = random.Random(int(seed))
    macro = search_space["macro_space"]
    block = search_space["block_space"]
    prefix = prefix or f"nas_seed{seed}"
    candidates: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = max(int(num_candidates) * max_attempts_multiplier, int(num_candidates) + 1)

    while len(candidates) < int(num_candidates) and attempts < max_attempts:
        attempts += 1
        ops = [rng.choice(block["ops"]) for _ in range(int(block["num_blocks"]))]
        if ops.count("skip") > int(block["constraints"]["max_skip_blocks"]):
            continue
        se = [rng.choice(block["se"]) for _ in ops]
        raw = {
            "candidate_id": f"{prefix}_{len(candidates):06d}",
            "search_mode": "random",
            "seed": int(seed),
            "sample_index": len(candidates),
            "sample_attempt": attempts,
            "encoder_strides": rng.choice(macro["encoder_strides"]),
            "n_filters": rng.choice(macro["n_filters"]),
            "compress": rng.choice(macro["compress"]),
            "lstm": rng.choice(macro["lstm"]),
            "activation": rng.choice(macro["activation"]),
            "layer_ops_list": ops,
            "layer_se_list": se,
        }
        try:
            item = normalize_candidate(raw, search_space=search_space)
        except CandidateValidationError:
            continue
        signature = item["candidate_signature"]
        if search_space["search_policy"].get("deduplicate", True) and signature in seen:
            continue
        seen.add(signature)
        candidates.append(item)

    return candidates


def search_space_with_policy(seed: int, num_candidates: int, mode: str = "random") -> Dict[str, Any]:
    item = copy.deepcopy(DEFAULT_SEARCH_SPACE)
    item["search_policy"]["seed"] = int(seed)
    item["search_policy"]["num_candidates"] = int(num_candidates)
    item["search_policy"]["mode"] = str(mode)
    return item
