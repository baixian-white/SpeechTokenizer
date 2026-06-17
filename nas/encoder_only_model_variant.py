from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer import SpeechTokenizer
from speechtokenizer.modules.seanet import SEANetDecoder as BaseSEANetDecoder

try:
    from .SeaNet import SEANetEncoder
    from .encoder_handoff import normalize_encoder_only_nas_config
except ImportError:
    from SeaNet import SEANetEncoder
    from encoder_handoff import normalize_encoder_only_nas_config


class NASEncoderSideSpeechTokenizer(SpeechTokenizer):
    """SpeechTokenizer variant for the Exp1 encoder-side NAS route.

    The searched component is only the transmitter-side encoder before latent
    Z. The decoder path is taken from the base config unless an explicit
    non-encoder-only diagnostic config is loaded.
    """

    def __init__(self, config, nas_config_path):
        super().__init__(config)
        nas_config_path = Path(nas_config_path)
        if not nas_config_path.is_absolute():
            nas_config_path = (Path.cwd() / nas_config_path).resolve()
        with open(nas_config_path, "r", encoding="utf-8") as f:
            nas_conf = json.load(f)
        nas_conf = self._normalize_nas_config(nas_conf, config)

        encoder_kwargs = {
            "channels": 1,
            "n_filters": nas_conf["n_filters"],
            "dimension": nas_conf["dimension"],
            "lstm": nas_conf.get("lstm", nas_conf.get("lstm_layers")),
            "activation": nas_conf["activation"],
            "compress": nas_conf.get("compress", 2),
            "layer_ops_list": nas_conf["layer_ops_list"],
            "layer_se_list": nas_conf["layer_se_list"],
            "ratios": nas_conf["seanet_ratios_arg"],
            "norm": "weight_norm",
            "causal": False,
            "pad_mode": "reflect",
            "dilation_base": config.get("dilation_base", 2),
            "residual_kernel_size": config.get("residual_kernel_size", 3),
            "n_residual_layers": config.get("n_residual_layers", 1),
            "bidirectional": config.get("bidirectional", False),
        }
        self.encoder = SEANetEncoder(**encoder_kwargs)
        self.decoder = BaseSEANetDecoder(
            channels=1,
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=nas_conf["decoder_strides"],
            lstm=config.get("lstm_layers"),
            bidirectional=False,
            dilation_base=config.get("dilation_base", 2),
            residual_kernel_size=config.get("residual_kernel_size", 3),
            n_residual_layers=config.get("n_residual_layers", 1),
            activation=config.get("activation"),
        )
        self.nas_encoder_config = nas_conf
        self.decoder_condition = nas_conf["decoder_condition"]

    @staticmethod
    def _normalize_nas_config(nas_conf, config):
        return normalize_encoder_only_nas_config(nas_conf, config)


class NASEncoderOnlySpeechTokenizer(NASEncoderSideSpeechTokenizer):
    """Backward-compatible alias for older scripts."""
