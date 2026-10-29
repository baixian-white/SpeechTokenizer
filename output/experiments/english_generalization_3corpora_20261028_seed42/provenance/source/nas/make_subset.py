# make_subset.py
"""
脚本名称: make_subset.py
功能描述: 
    该脚本用于从原始的全量数据集列表（如 LibriSpeech train-clean-100）中，
    随机抽取一小部分（如 2000~5000 条）生成一个新的文件列表。

    NAS (神经网络架构搜索) 专用:
    在 NAS 搜索阶段，为了快速评估成百上千个子网结构，我们不需要使用全量数据。
    使用一个小规模的子集（Proxy Dataset）足以区分不同模型结构的优劣，
    从而大幅节省搜索时间。

使用建议:
    1. 修改 SOURCE_FILE 指向你现有的全量 filelist。
    2. 设置 SAMPLE_SIZE (推荐 2000-5000)。
    3. 运行脚本生成 train_subset_nas.txt。
    4. 在 NAS 搜索命令中指向这个新生成的 txt 文件。

依赖库:
    - random, os
"""

import random
import os
from pathlib import Path

# ================= 配置区域 =================
# 1. 你的原始全量数据列表路径 (请务必确认这个路径在你的服务器上是存在的)
#    通常是 SpeechTokenizer 预处理生成的 train.txt 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NAS_DIR / "artifacts"
SOURCE_FILE = PROJECT_ROOT / "data" / "SpeechPretrain" / "hubert_rep" / "LibriSpeech" / "train_files.txt"

# 2. 生成的目标小文件路径 (供 NAS 搜索脚本读取)
TARGET_FILE = ARTIFACTS_DIR / "train_subset_nas.txt"

# 3. 想要抽取的数量 
#    NAS 搜索阶段 2000 条完全足够排查出好坏，5000 条则更稳健
SAMPLE_SIZE = 5000
# ===========================================

def main():
    print(f"🔍 正在读取源文件: {SOURCE_FILE} ...")

    if not SOURCE_FILE.exists():
        print(f"❌ 错误: 找不到文件 {SOURCE_FILE}")
        print("   请用文本编辑器打开本脚本，修改配置区域的路径！")
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取所有行
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"📄 源文件共有 {total_lines} 条数据。")

    # 执行抽取逻辑
    if total_lines <= SAMPLE_SIZE:
        print("⚠️ 数据量少于抽取数量，将直接复制所有数据。")
        subset = lines
    else:
        print(f"🎲 正在随机抽取 {SAMPLE_SIZE} 条数据用于 NAS 搜索...")
        subset = random.sample(lines, SAMPLE_SIZE)

    # 写入新文件
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.writelines(subset)

    print("-" * 40)
    print(f"✅ 成功生成迷你数据集: {TARGET_FILE}")
    print(f"📊 包含数据量: {len(subset)} 条")
    print("-" * 40)
    print("🚀 下一步: 请在运行 NAS 搜索时，将 --train_file 指向该文件。")

if __name__ == "__main__":
    main()
