"""
脚本名称: custom_model.py
功能描述: 
    该脚本实现了基于神经架构搜索 (NAS) 结果的 SpeechTokenizer 模型重组。
    通过继承官方 SpeechTokenizer 基类，该类能够读取 NAS 搜索出的最优 JSON 配置，
    并动态替换原有的 SEANetEncoder 和 SEANetDecoder 模块。
主要特性:
    1. 动态加载: 支持从 JSON 配置文件加载 n_filters, dimension, layer_ops 等 NAS 参数。
    2. 自动镜像: 自动将 Encoder 的层序列倒序以构建对称的 Decoder 结构。
    3. 灵活切换: 保持了与原版模型相同的 API 接口，方便在推理或训练脚本中直接替换。
"""
from pathlib import Path
import json
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入官方库
from speechtokenizer import SpeechTokenizer
# 导入你的 NAS 组件 (确保 SeaNet.py 在同级目录或 Python 路径下)
try:
    from .SeaNet import SEANetEncoder, SEANetDecoder
except ImportError:
    from SeaNet import SEANetEncoder, SEANetDecoder

NAS_DIR = Path(__file__).resolve().parent

class NASSpeechTokenizer(SpeechTokenizer):
    def __init__(self, config, nas_config_path=None):
        """
        config: 原本 SpeechTokenizer 的 config (用于设置 quantizer 等通用参数)
        nas_config_path: NAS 搜索出的最佳架构配置
        """
        # 1. 初始化父类 (建立 quantizer, transform 等组件)
        super().__init__(config)

        if nas_config_path is None:
            nas_config_path = NAS_DIR / "best_seanet_config.json"
        else:
            nas_config_path = Path(nas_config_path)
            if not nas_config_path.is_absolute():
                nas_config_path = (Path.cwd() / nas_config_path).resolve()
        
        # 2. 加载 NAS 配置
        print(f"🔧 [NASSpeechTokenizer] Loading NAS Config: {nas_config_path}")
        with open(nas_config_path, 'r', encoding='utf-8') as f:
            nas_conf = json.load(f)
            
        # 3. 准备参数映射 (将 NAS 参数转为 SEANet 需要的格式)
        # 这里混合使用了 NAS 的结果和 config 的通用设置
        nas_encoder_kwargs = {
            # === NAS 搜索出的核心参数 ===
            "n_filters": nas_conf['n_filters'],
            "dimension": nas_conf['dimension'], # 必须是 1024
            "lstm": nas_conf.get('lstm', nas_conf.get('lstm_layers')), # 兼容一下键名
            "activation": nas_conf['activation'],
            "layer_ops_list": nas_conf['layer_ops_list'], # 核心结构
            "layer_se_list": nas_conf['layer_se_list'],   # 核心结构
            
            # === 固定参数 (来自 config.json) ===
            "ratios": config.get('strides', [8, 5, 4, 2]),
            "norm": "weight_norm",
            "causal": False,
            "pad_mode": "reflect",
            "dilation_base": config.get('dilation_base', 2),
            "residual_kernel_size": config.get('residual_kernel_size', 3),
            "n_residual_layers": config.get('n_residual_layers', 1),
        }

        # 4. 【核心手术】替换 Encoder
        print("⚡ Replacing Encoder with NAS version...")
        self.encoder = SEANetEncoder(**nas_encoder_kwargs)
        
        # 5. 【核心手术】替换 Decoder (需要镜像配置)
        nas_decoder_kwargs = nas_encoder_kwargs.copy()
        # 列表倒序
        nas_decoder_kwargs['layer_ops_list'] = nas_conf['layer_ops_list'][::-1]
        nas_decoder_kwargs['layer_se_list'] = nas_conf['layer_se_list'][::-1]
        # Decoder 不用双向
        nas_decoder_kwargs['bidirectional'] = False 
        
        print("⚡ Replacing Decoder with NAS version...")
        self.decoder = SEANetDecoder(**nas_decoder_kwargs)
        
        print("✅ NAS Model Assembled Successfully!")
