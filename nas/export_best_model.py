# export_best_model.py
"""
脚本名称: export_best_model.py
功能描述: 
    该脚本是 NAS (神经网络架构搜索) 流程的中间桥梁。
    它的作用是从 Optuna 的 SQLite 数据库中提取出表现最好（Loss 最小）的模型架构参数，
    并将其格式化为标准的 JSON 配置文件，供 train_nas.py 使用。

工作流程:
    1. 连接数据库: 读取 search_autoencoder.py 生成的 .db 文件。
    2. 锁定最佳 Trial: 从 Optuna Study 中找到 value (Loss) 最小的一次实验。
    3. 参数重组: 将 Optuna 扁平化的参数 (如 layer_0_op, layer_1_op) 
       重组为 SeaNet 需要的列表格式 (layer_ops_list)。
    4. 导出 JSON: 生成 best_seanet_config.json。
    5. 验证: 尝试使用该配置实例化模型，确保结构合法。

输出文件:
    - best_seanet_config.json (将被 train_nas.py 读取)

注意事项:
    - 必须确保当前目录下有 searh 生成的 .db 文件。
    - NUM_LAYERS 必须与搜索脚本中的设置保持一致。
"""

import optuna
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NAS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# 1. 配置 (必须与您的 search_autoencoder.py 保持一致)
# ==========================================
DB_PATH = ARTIFACTS_DIR / "seanet_nas_micro_v1.db"
DB_URL = f"sqlite:///{DB_PATH.resolve().as_posix()}"
STUDY_NAME = "seanet_micro_search_v1"
OUTPUT_JSON = NAS_DIR / "best_seanet_config.json"

# ⚠️ 关键：您的 search_autoencoder.py 中写死了 num_layers = 4
# 如果您更改了搜索代码，这里也需要同步更改
NUM_LAYERS = 4 

def export_best_config():
    # --------------------------------------
    # 2. 检查数据库是否存在
    # --------------------------------------
    if not DB_PATH.exists():
        print(f"❌ 错误：找不到数据库文件 {DB_PATH}")
        print("请确认您已经在该目录下运行过 NAS 搜索脚本。")
        return

    # --------------------------------------
    # 3. 加载 Study
    # --------------------------------------
    print(f"📂 正在连接数据库: {DB_URL} ...")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
    except KeyError:
        print(f"❌ 错误：数据库中找不到名为 '{STUDY_NAME}' 的任务。")
        return
    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")
        return

    if len(study.trials) == 0:
        print("❌ 错误：数据库是空的，没有任何 Trial 记录。")
        return

    # --------------------------------------
    # 4. 获取最佳 Trial
    # --------------------------------------
    best_trial = study.best_trial
    
    print("\n" + "="*50)
    print(f"🏆 最佳 Trial ID : {best_trial.number}")
    print(f"📉 最佳 Loss     : {best_trial.value:.6f}")
    
    # 尝试读取自定义属性 (FLOPs)
    flops = best_trial.user_attrs.get("total_flops", None)
    if flops:
        print(f"⚡ 计算量 (FLOPs): {flops/1e9:.3f} G")
    
    # 读取 SI-SNR (如果有记录在 user_attrs 或者中间 print 过，这里只能读 user_attrs)
    # 您的原始代码中将 SI-SNR 写入了 CSV，但没有写入 Optuna user_attrs
    # 如果您按我之前的建议修改过 search 代码，这里可以读到；否则读不到是正常的。
    sisnr = best_trial.user_attrs.get("si_snr", "N/A")
    print(f"🎵 SI-SNR        : {sisnr}")
    print("="*50 + "\n")

    # --------------------------------------
    # 5. 参数重组 (关键逻辑)
    # --------------------------------------
    params = best_trial.params
    print("正在重组架构参数...")

    # 初始化空列表
    layer_ops_list = []
    layer_se_list = []

    # 循环提取 layer_0_op 到 layer_3_op
    for i in range(NUM_LAYERS):
        op_key = f'layer_{i}_op'
        se_key = f'layer_{i}_se'
        
        # 容错检查
        if op_key not in params or se_key not in params:
            print(f"⚠️ 严重警告：参数缺失！找不到 {op_key} 或 {se_key}。")
            print("可能是您修改了搜索空间但复用了旧数据库。建议删除 .db 文件重跑。")
            return

        layer_ops_list.append(params[op_key])
        layer_se_list.append(params[se_key])

    # --------------------------------------
    # 6. 构建最终配置字典
    # --------------------------------------
    # 这里的 Key 必须严格对应 SeaNet.__init__ 的参数名
    model_config = {
        "channels": 1,              # 固定值
        "dimension": 1024,          # 您的代码中写死为 1024
        "n_filters": params['n_filters'],
        # 注意：Optuna里叫 lstm_layers, SeaNet里参数名叫 lstm，这里做映射
        "lstm": params['lstm_layers'], 
        "activation": params['activation'],
        "compress": params['compress'],
        "ratios": [8, 5, 4, 2],     # SeaNet 默认值，显式写入以防万一
        "norm": "weight_norm",      # 默认值
        "causal": False,            # 默认值
        # 组装好的列表
        "layer_ops_list": layer_ops_list,
        "layer_se_list": layer_se_list
    }

    # --------------------------------------
    # 7. 保存为 JSON
    # --------------------------------------
    with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
        json.dump(model_config, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 成功导出配置到: {os.path.abspath(OUTPUT_JSON)}")

    # --------------------------------------
    # 8. (可选) 验证模型能否初始化
    # --------------------------------------
    print("\n🔍 正在验证配置有效性...")
    try:
        try:
            from .SeaNet import SEANet
        except ImportError:
            from SeaNet import SEANet
        
        # 使用 **kwargs 方式解包字典
        model = SEANet(**model_config)
        
        print("🎉 验证成功！SEANet 模型已成功利用该 JSON 初始化。")
        print(f"   Encoder 层数: {len(model.encoder.model)}")
        print(f"   Decoder 层数: {len(model.decoder.model)}")
        
    except ImportError:
        print("⚠️ 警告：找不到 SEANet 类 (custom_model.py)，跳过验证。但 JSON 文件已生成。")
    except Exception as e:
        print(f"❌ 验证失败：模型初始化报错。请检查代码。\n详细错误: {e}")

if __name__ == "__main__":
    export_best_config()
