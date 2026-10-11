"""Check the supplementary shape tables with synthetic CPU forward passes."""

import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "supplementary_methods_v3_assets"
SPEAKER = Path("D:/tmp/speechtokenizer-exp23/speechtokenizer/speaker_identity")
sys.path.insert(0, str(ROOT))

from nas.SeaNet import SEANetEncoder


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"supplement_{name}", SPEAKER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config_path = ROOT / (
        "output/experiments/exp1_nas_distill_run1_seed42/"
        "artifacts/best_architecture/best_seanet_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    encoder = SEANetEncoder(
        channels=1, dimension=config["dimension"], n_filters=config["n_filters"],
        n_residual_layers=1, ratios=config["seanet_ratios_arg"], activation=config["activation"],
        norm="weight_norm", causal=False, pad_mode="reflect", lstm=config["lstm"],
        bidirectional=True, layer_ops_list=config["layer_ops_list"],
        layer_se_list=config["layer_se_list"],
    ).eval()
    count = sum(parameter.numel() for parameter in encoder.parameters())
    assert count == 8873360, count
    traces = []
    for samples in (8000, 16000):
        trace = []
        with torch.inference_mode():
            value = torch.zeros(1, 1, samples)
            for index, module in enumerate(encoder.model):
                value = module(value)
                trace.append({"module_index": index, "module": type(module).__name__,
                              "shape": list(value.shape)})
        expected = [
            (0, 24, samples), (1, 24, samples), (3, 48, samples // 5),
            (4, 48, samples // 5), (6, 96, samples // 20), (7, 96, samples // 20),
            (9, 192, samples // 80), (10, 192, samples // 80),
            (12, 384, samples // 320), (13, 768, samples // 320),
            (15, 1024, samples // 320),
        ]
        for index, channels, length in expected:
            assert trace[index]["shape"] == [1, channels, length], trace[index]
        traces.append({"input_samples": samples, "modules": trace})

    token = load_module("token_encoder").RVQTokenEncoder().eval()
    fusion = load_module("fusion").GatedEmbeddingFusion().eval()
    classifier_checks = []
    with torch.inference_mode():
        for frames in (25, 50, 151):
            codes = torch.randint(0, 1024, (2, 3, frames))
            mask = torch.ones(2, frames, dtype=torch.bool)
            mask[1, -5:] = False
            codes[1, :, -5:] = 1024
            token_embedding = token(codes, mask, torch.ones(2, 3))
            audio_embedding = torch.nn.functional.normalize(torch.randn(2, 256), dim=-1)
            fused, gate = fusion(token_embedding, audio_embedding)
            assert token_embedding.shape == fused.shape == gate.shape == (2, 256)
            assert torch.isfinite(fused).all()
            assert ((gate > 0) & (gate < 1)).all()
            classifier_checks.append({"code_frames": frames, "input": list(codes.shape),
                                      "token_output": list(token_embedding.shape),
                                      "fusion_output": list(fused.shape)})
    result = {
        "status": "passed",
        "scope": "Synthetic CPU shape checks, not training, quality evaluation or performance profiling.",
        "encoder_parameter_count": count,
        "encoder_shapes": traces,
        "classifier_shapes": classifier_checks,
        "torch": torch.__version__,
    }
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "shape_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "encoder_parameters": count,
                      "waveform_lengths_checked": [8000, 16000],
                      "code_lengths_checked": [25, 50, 151]}))


if __name__ == "__main__":
    main()
