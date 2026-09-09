# 实验一：固定 RVQ 索引接口下的教师引导式 Encoder-side NAS

## 1. 实验定位

本实验验证：在 **index-only transmission / RVQ 索引接口固定** 的前提下，是否可以通过教师引导式 NAS 搜索得到更轻量的发送端语义/声学表示 encoder。

实验一只决定：

```text
x(t) -> Z
```

这一段发送端 encoder 架构。它不负责训练最终通信模型，也不验证低负载信道、baseline codec、消融或三用户系统。实验二会使用实验一选定的 encoder 架构训练 `SCIT-Speech-Base`；实验三到实验五再分别处理低负载适配、baseline 对比和消融诊断。

新版实验一引入冻结的预训练 SpeechTokenizer encoder 作为教师锚点：

```text
A = pretrained SpeechTokenizer encoder, frozen teacher
B = NAS-generated lightweight encoder candidate, trainable/searchable student

x(t) -> A.encoder -> e_A
x(t) -> B.encoder -> e_B
```

候选 B 不仅要更轻、更快，还必须在固定接口下产生与 A 接近的量化前 latent：

```text
e_A shape = [B, 1024, 50]
e_B shape = [B, 1024, 50]
```

本实验的最合适角度是：

```text
teacher-guided encoder-side NAS under a fixed RVQ index interface
```

NAS 负责产生候选架构；短程 teacher distillation 负责让候选 B 在少量 step 内学习 SpeechTokenizer encoder 已学到的 pre-RVQ 表示空间；实验二的完整训练负责验证最终通信模型表现。

decoder 在本实验中不作为搜索对象。teacher-guided proxy 主口径使用冻结的预训练 SpeechTokenizer transform、RVQ quantizer 和 decoder；候选 B 只替换并训练 encoder，proxy reconstruction 只用于诊断 B 是否仍能接入原 SpeechTokenizer 下游模块。

### 1.1 当前 run 暴露的问题与修正方向

已完成的 `exp1_nas_semantic_encoder_run1_seed42` 说明当前脚本可以完成 staged NAS，并能找到明显更轻的候选架构。例如 `nas_seed42_007077` 的 encoder params 和 MACs 相比 hand-designed 架构大幅下降。但该 run 也暴露出一个关键问题：

```text
rvq_code_agreement ≈ 0.002
rvq_code_flip_rate ≈ 0.998
teacher_latent_cosine_distance ≈ 1.0
```

这说明当前选出的 B 只是“结构更轻”，还不能说明它已经学到了 SpeechTokenizer encoder A 的表示空间。主要原因是：候选 B 仍是随机初始化后直接前向评估，没有经过 teacher-guided 短程训练；同时 hand-designed baseline 若由 `SpeechTokenizer(cfg)` 随机初始化得到，也不能代表预训练 SpeechTokenizer teacher 的真实性能。

因此，后续实现必须把实验一升级为：

```text
stage1: 架构/资源粗筛
stage2-stage4: 候选 B 先做短程蒸馏训练，再计算 teacher alignment、RVQ compatibility 和 proxy 指标
```

未经过短程蒸馏训练的候选评估只能作为：

```text
architecture/profile diagnostic
```

不能作为 teacher-guided NAS 的最终结论。

### 1.2 NAS 与蒸馏的分工

本实验中，“轻量化”和“蒸馏”是两件不同的事：

```text
NAS: 决定 B 的架构，使 B 的 params/MACs/RTF 更低
蒸馏: 决定 B 的训练目标，使 B(x) 尽量模仿 A(x)
```

也就是说：

```text
B 参数量减少 ≠ 蒸馏
B 学 A 的 latent/code 行为 = 蒸馏
```

蒸馏必须体现在训练 loss 和评估指标中，而不是只体现在最终模型更小。短程蒸馏阶段只更新 B.encoder；teacher A、teacher transform、RVQ quantizer 和 decoder 均冻结。

## 2. 固定通信接口

本实验必须固定以下通信接口条件：

```text
sample_rate = 16000
encoder_downsample_rate = 320
latent_rate = 16000 / 320 = 50 steps/s
latent_dimension = 1024
n_q / M = 3
codebook_size / K = 1024
L = 1, 2, 3 only for downstream reporting, not searched here
```

因此，1 秒输入音频的 encoder 输出必须满足：

```text
Z shape = [B, 1024, 50]
codes shape after RVQ = [3, B, 50]
```

默认工程配置为：

```text
encoder_strides = [8, 5, 4, 2]
```

但 NAS 不必固定这一种分解。允许搜索不同的 **四层 stride schedule**，只要：

```text
len(encoder_strides) = 4
prod(encoder_strides) = 320
```

例如：

```text
[8, 5, 4, 2]
[5, 4, 4, 4]
[10, 4, 4, 2]
[4, 5, 4, 4]
[4, 4, 5, 4]
[4, 4, 4, 5]
```

允许搜索 stride 分解，是为了让 NAS 可以探索不同阶段的时间压缩分布；固定总下采样率，是为了不改变后续索引负载公式：

```text
R_index(L) = L * latent_rate * ceil(log2 K)
           = L * 50 * 10
           = 500L bps
```

如果某个候选满足不了 `prod(strides)=320`，或者 forward 后不满足 `Z shape=[B,1024,T/320]`，该候选必须标记为 invalid，不能进入主 Pareto frontier。

## 3. Decoder 口径

### 3.1 主口径

本实验主张只针对发送端 encoder：

```text
x(t) -> encoder -> Z
```

主表中的复杂度比较只报告 encoder-only：

- encoder params
- encoder MACs
- encoder RTF
- semantic proxy loss
- interface validity

### 3.2 Frozen teacher downstream modules

teacher-guided proxy 中，候选 encoder B 接入冻结的预训练 SpeechTokenizer 下游模块：

```text
e_B = B.encoder(x)
q_B, codes_B = frozen_teacher_quantizer(e_B)
x_hat_B = frozen_teacher_decoder(q_B)
```

报告中应写为：

```text
decoder_condition = frozen_teacher_decoder
```

不得写成：

```text
decoder was searched by NAS
```

几何匹配 decoder 可作为实现变体或消融诊断保留，但不再作为本实验 teacher-guided 主口径。

### 3.3 不作为主结论的变体

如果 decoder 的算子、宽度、层数、SE 开关或 LSTM 等也进入搜索空间，则该路线变成：

```text
encoder-decoder joint NAS
```

这不是实验一主口径。它只能作为 implementation variant 或实验五消融，必须单独标注：

```text
decoder_condition = joint_nas_decoder
```

若报告完整重建质量，必须同时报告 decoder condition，避免把 decoder 变化带来的重建提升误写成 encoder NAS 贡献。

## 4. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | 固定实验边界、指标口径和叙事原则 |
| run 规范 | `output/experiments/README.md` | 规定 run 目录结构 |
| 主配置 | `config/spt_base_cfg.json` | 默认模型配置；固定 `sample_rate=16000`、`dimension=1024`、`n_q=3`、`codebook_size=1024` |
| 教师模型配置 | `model_hub/speechtokenizer_hubert_avg/config.json` 或本地等价路径 | SpeechTokenizer teacher A 的配置；必须与 checkpoint 匹配 |
| 教师模型 checkpoint | `model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt` 或本地等价路径 | 冻结的 SpeechTokenizer teacher A，用于 `e_A` 与 RVQ compatibility 检查 |
| 训练清单 | `data/SpeechPretrain/hubert_rep/LibriSpeech/train_files.txt` | 每行应包含音频路径和语义教师特征路径 |
| 验证清单 | `data/SpeechPretrain/hubert_rep/LibriSpeech/valid_files.txt` | 用于 proxy validation 或 smoke test |
| 算子实现 | `nas/model_components.py`、`nas/SeaNet.py` | 提供 searchable residual block 和 SEANet encoder/decoder |
| 候选生成 | 建议新增 `nas/search_space.py` | 定义算子库、搜索空间和 candidate generator |
| proxy 评估 | 建议改造 `nas/evaluate_encoder_proxy.py` | 动态生成候选并执行 teacher-guided encoder-side proxy evaluation |
| model variant | 建议新增或改造 `nas/encoder_side_model_variant.py` | 支持 encoder replacement 与 geometry-matched decoder |
| 环境采集 | `scripts/collect_environment.py` | 采集环境与 git 信息 |

若脚本缺失，应优先补齐。若无法安全补齐或执行，必须写 `reports/failure_report.md`，说明缺失职责、输入、输出和建议实现路径。

## 5. 搜索空间

### 5.1 固定项

以下条件不允许被 NAS 改变：

- `sample_rate = 16000`
- `prod(encoder_strides) = 320`
- `latent_rate = 50 steps/s`
- `latent_dimension = 1024`
- `n_q / M = 3`
- `codebook_size / K = 1024`
- SpeechTokenizer teacher A 的权重和 RVQ codebooks
- 数据划分
- 测试集
- payload 统计口径

### 5.2 可搜索项

第一版建议只搜索四层 encoder 的结构：

| 类别 | 搜索项 | 说明 |
|---|---|---|
| stride schedule | 四层 schedule，乘积必须为 320 | 可搜索阶段压缩分布 |
| channel width | `n_filters ∈ {16, 24, 32, 48}`；主 run 默认 `n_filters >= 24` | 控制 encoder 宽度，避免 stage1 被极窄模型主导 |
| bottleneck/compress | `compress ∈ {2, 4}` | 控制残差块内部压缩比例 |
| temporal module | `lstm ∈ {1, 2}` | 保留 encoder 末端时序建模；不搜索 `lstm=0` 的无时序模块版本 |
| activation | `activation ∈ {ELU, Snake}` | 第一版保持较小集合 |
| residual op | 12-op library，见 5.3 | 每个 residual block 独立选择 |
| SE switch | `use_se ∈ {false, true}` | 每个 block 独立开关 |

固定：

```text
channels = 1
dimension = 1024
norm = weight_norm
causal = false
pad_mode = reflect
n_residual_layers = 1 for the first NAS run
```

若后续搜索 `n_residual_layers`，必须同步更新 `layer_ops_list` 和 `layer_se_list` 的长度规则。

### 5.3 算子库

建议在 `nas/search_space.py` 中定义：

```python
NAS_OPERATOR_LIBRARY = {
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
    "skip": "identity residual op"
}
```

这些名字必须能被 `nas/model_components.py` 中的 `get_nas_ops(...)` 解析。当前工程应保持 12 个 op 全部可实例化，并在正式执行前跑 shape smoke test；若新增或删除算子，必须同步更新 `configs/operator_library.json`、`configs/nas_search_space.json` 和本实验计划。

为避免搜索空间产生大量等价空块，候选生成器必须执行以下约束：

```text
count(op == "skip") <= 2
if op == "skip": use_se = false
```

也就是说，`skip + SE on` 不应作为独立候选，因为当前 skip block 实际 forward 为 identity，SE 开关不会生效。

主 run 推荐使用 `selection_mode=balanced`，实际采样前将搜索空间收紧为：

```text
n_filters >= 24
count(op == "skip") <= 1
lstm ∈ {1, 2}
```

`selection_mode=lite` 仅用于轻量极限诊断；`selection_mode=quality` 用于更保守的质量优先筛选。三种模式都不改变 sample rate、latent rate、latent dimension、M/K/L 或 payload 接口。

## 6. 候选生成

### 6.1 禁止固定候选表作为主实验

主实验不能把少量手写候选当成 NAS：

```text
nas_cand_000
nas_cand_001
nas_cand_002
nas_cand_003
```

这类候选只可用于 smoke test 或回归测试。正式 run 必须由搜索器按搜索空间和 seed 动态生成候选。

### 6.2 Candidate JSON 格式

每个候选保存为：

```json
{
  "candidate_id": "nas_seed42_000017",
  "search_mode": "random",
  "seed": 42,
  "sample_index": 17,
  "encoder_strides": [5, 4, 4, 4],
  "decoder_strides": [8, 5, 4, 2],
  "decoder_condition": "frozen_teacher_decoder",
  "n_filters": 24,
  "dimension": 1024,
  "lstm": 1,
  "activation": "ELU",
  "compress": 4,
  "layer_ops_list": ["sep_k5", "std_k7", "pw_bottleneck_k3", "dil_k5"],
  "layer_se_list": [false, true, false, false]
}
```

规则：

- `encoder_strides` 必须四层且乘积为 320。
- 主口径 `decoder_condition=frozen_teacher_decoder` 时，`decoder_strides` 记录 teacher/base decoder 的 stride 配置，不随 NAS candidate 搜索。
- 只有 `decoder_condition=geometry_matched_decoder` 诊断变体才允许 `decoder_strides=reverse(encoder_strides)`。
- `dimension` 必须为 1024。
- `layer_ops_list` 长度必须等于 encoder residual block 数。
- `layer_se_list` 长度必须与 `layer_ops_list` 一致。
- `layer_ops_list` 中 `skip` 数量不得超过 2。
- 若某层 `op == "skip"`，该层 `use_se` 必须规范化为 `false`。
- 每个 candidate JSON 必须保存到 `artifacts/candidates/`。

### 6.3 搜索模式

推荐第一版支持 staged random NAS，不再把少量手写候选当成主实验：

| 模式 | 用途 | 建议参数 |
|---|---|---|
| `random + balanced` | 主搜索模式 | `8192 -> 512 -> 64 -> 8` |
| `random + quality` | 质量优先复核 | `4096 -> 256 -> 64 -> 8` |
| `random + lite` | 极轻量诊断，不作为主结论 | `2048 -> 256 -> 32 -> 8` |
| `grid` | 小规模 toy search 或回归测试 | 只用于缩小搜索空间 |

全量网格空间很容易膨胀，不建议第一版直接穷举。若只跑 `2048 -> 256 -> 32 -> 3`，应标记为 small-budget proxy run，不能写成充分 NAS 搜索。

### 6.4 Staged NAS + short distillation 预算

正式实验一不应在随机初始化候选上直接做最终选择。推荐的 staged 流程为：

| 阶段 | 输入候选 | 操作 | 输出候选 | 主要依据 |
|---|---:|---|---:|---|
| stage1 profile | 8192 | 只做接口检查、params、MACs、RTF | 512 | 资源粗筛与 shape 合法性 |
| stage2 short distill | 512 | 每个候选训练 50-100 steps | 64 | teacher latent alignment + 资源 |
| stage3 refined distill | 64 | 每个候选训练 300-500 steps | 8 | teacher alignment + RVQ compatibility |
| stage4 final distill | 8 | 每个候选训练 1000-3000 steps | 1 | held-out proxy validation + Pareto |

stage2-stage4 的“评估”必须发生在短程蒸馏训练之后。训练前指标可保存为 `*_before_distill`，用于证明蒸馏是否真的改善了候选；训练后指标保存为主表字段。若训练前后 `teacher_latent_*` 与 `rvq_code_agreement` 几乎不变，应判定该候选或该训练设置不具备 teacher-guided 说服力。

短程蒸馏不是实验二的完整训练。它只用于架构筛选，目的是观察候选 B 是否具备快速学习 A 表示空间的能力。

## 7. 实验输出

每个 run 保存到：

```text
output/experiments/{run_id}/
```

`run_id` 格式：

```text
exp1_nas_semantic_encoder_YYYYMMDD_HHMMSS_seed{seed}
```

标准目录：

```text
configs/
commands/
logs/
checkpoints/
metrics/
samples/
reports/
artifacts/
```

必须保存：

| 文件 | 说明 |
|---|---|
| `configs/spt_base_cfg.json` | 本次使用的主配置副本 |
| `configs/operator_library.json` | 算子库定义 |
| `configs/nas_search_space.json` | 搜索空间定义 |
| `configs/staged_nas_runtime_config.json` | seed、selection mode、候选数、proxy subset、设备、质量 margin 和权重等 |
| `configs/teacher_model_reference.json` | teacher A 的 config、checkpoint、target layer、冻结策略和哈希/路径 |
| `configs/teacher_guided_nas_config.json` | teacher alignment、RVQ compatibility、短训步数和指标权重 |
| `commands/run_command.txt` | 实际执行命令 |
| `logs/stdout.log` | stdout |
| `logs/stderr.log` | stderr |
| `reports/environment.md` | Python/PyTorch/CUDA/GPU/依赖/git 信息 |
| `artifacts/train_subset_nas.txt` | NAS proxy subset |
| `artifacts/proxy_sample_manifest.json` | 实际 proxy 样本列表 |
| `artifacts/teacher_latent_cache_manifest.json` | 若缓存 teacher latent，记录缓存样本、shape 和生成命令 |
| `artifacts/candidates/*.json` | 动态生成的候选架构 |
| `artifacts/best_architecture/best_seanet_config.json` | 选定 encoder 架构 |
| `metrics/nas_records.csv/json` | 每个候选完整搜索记录 |
| `metrics/encoder_proxy_results.csv/json` | encoder-side proxy 指标 |
| `metrics/teacher_alignment.csv/json` | B 对 A 的 latent 对齐指标 |
| `metrics/rvq_compatibility.csv/json` | B 输出进入冻结 A quantizer 后的 code/量化特征兼容指标 |
| `metrics/pareto_frontier.csv` | Pareto frontier |
| `metrics/results.csv/json` | 机器可读汇总 |
| `reports/operator_library.md` | 算子库说明 |
| `reports/teacher_condition.md` | teacher A 的来源、冻结方式、是否只作为 proxy 锚点 |
| `reports/teacher_guided_selection.md` | teacher alignment 与 RVQ compatibility 如何参与选择 |
| `reports/encoder_interface_check.md` | latent shape、latent rate、codes shape 检查 |
| `reports/decoder_condition.md` | decoder geometry-matching 说明 |
| `reports/decoder_contamination_check.md` | decoder 是否进入搜索和贡献口径 |
| `reports/pareto_selection.md` | best architecture 选择理由 |
| `reports/summary.md` | 人类可读总结 |
| `reports/status.json` | 当前状态 |

失败或中止时还必须保存：

```text
reports/failure_report.md
```

## 8. 详细执行步骤

### 步骤 1：创建 run 目录

创建唯一 run：

```text
output/experiments/exp1_nas_semantic_encoder_YYYYMMDD_HHMMSS_seed{seed}/
```

不得覆盖旧 run。

### 步骤 2：复制并冻结主配置

复制：

```text
config/spt_base_cfg.json
```

到：

```text
configs/spt_base_cfg.json
```

检查固定条件：

```text
sample_rate = 16000
dimension = 1024
n_q = 3
codebook_size = 1024
```

检查默认或候选 stride 条件：

```text
len(encoder_strides) = 4
prod(encoder_strides) = 320
```

若固定条件不满足，禁止进入搜索。

### 步骤 3：记录环境

写入：

```text
reports/environment.md
reports/environment.json
```

至少包含：

- Python 版本
- PyTorch 版本
- CUDA 是否可用
- CUDA 版本
- GPU 型号
- 关键依赖版本
- git commit hash
- git status 摘要

### 步骤 4：准备 NAS proxy subset

从训练清单抽取固定 proxy subset：

```text
artifacts/train_subset_nas.txt
artifacts/proxy_sample_manifest.json
```

要求：

- 每行格式为 `<audio_path>\t<semantic_feature_path>`。
- 音频文件存在。
- 语义教师特征文件存在。
- 保存抽样 seed、样本数量和样本列表。
- 不得混用训练集、验证集和测试集。

若数据清单缺失、样本文件缺失或划分混用，必须中止。

### 步骤 4b：加载并冻结 SpeechTokenizer teacher A

加载 teacher 配置与 checkpoint：

```text
teacher_config = model_hub/speechtokenizer_hubert_avg/config.json 或等价路径
teacher_checkpoint = model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt 或等价路径
```

要求：

- teacher 必须 `eval()`。
- teacher encoder、quantizer、decoder 均不得更新梯度。
- 主 teacher target 是 `A.encoder(x)` 的 pre-RVQ latent `e_A`。
- 冻结的 `A.quantizer` 只用于 RVQ compatibility 检查，不得作为搜索对象。
- 若缓存 `e_A`，必须保存 `artifacts/teacher_latent_cache_manifest.json`，记录输入样本、shape、dtype、teacher checkpoint 和生成命令。
- 若 teacher checkpoint 缺失或无法加载，实验一必须中止或降级为旧版 non-teacher proxy run，并在 `reports/failure_report.md` 或 `reports/status.json` 中明确标注。

### 步骤 5：定义算子库和搜索空间

写入：

```text
configs/operator_library.json
configs/nas_search_space.json
reports/operator_library.md
```

示例：

```json
{
  "fixed": {
    "sample_rate": 16000,
    "encoder_downsample_rate": 320,
    "latent_rate": 50,
    "latent_dimension": 1024,
    "n_q": 3,
    "codebook_size": 1024
  },
  "macro_space": {
    "encoder_strides": [
      [8, 5, 4, 2],
      [5, 4, 4, 4],
      [10, 4, 4, 2],
      [4, 5, 4, 4],
      [4, 4, 5, 4],
      [4, 4, 4, 5]
    ],
    "n_filters": [24, 32, 48],
    "compress": [2, 4],
    "lstm": [1, 2],
    "activation": ["ELU", "Snake"]
  },
  "block_space": {
    "ops": [
      "std_k3",
      "std_k5",
      "std_k7",
      "sep_k3",
      "sep_k5",
      "sep_k7",
      "sep_k9",
      "dil_k3",
      "dil_k5",
      "dil_k9",
      "pw_bottleneck_k3",
      "skip"
    ],
    "se": [false, true],
    "constraints": {
      "max_skip_blocks": 1,
      "skip_forces_se_false": true
    }
  },
  "decoder_policy": {
    "condition": "frozen_teacher_decoder",
    "decoder_strides": "teacher_config.strides",
    "decoder_ops": "not_searched"
  },
  "search_policy": {
    "mode": "random",
    "num_candidates": 8192,
    "seed": 42,
    "selection_mode": "balanced",
    "quality_margins": {
      "semantic": 0.10,
      "recon": 0.10,
      "mel": 0.10
    }
  }
}
```

### 步骤 6：动态生成候选

按搜索空间生成：

```text
artifacts/candidates/nas_seed42_000000.json
artifacts/candidates/nas_seed42_000001.json
...
```

如果候选重复，应记录去重策略。去重后候选数不足时，在 summary 中说明。

### 步骤 7：teacher A 与 pretrained hand-designed encoder baseline

构建冻结 teacher A，并用同一个 pretrained SpeechTokenizer checkpoint 记录 hand-designed encoder baseline：

- teacher encoder params
- teacher encoder MACs
- teacher encoder RTF
- encoder params
- encoder MACs
- encoder RTF
- `Z` shape
- codes shape
- teacher latent reference shape
- semantic proxy loss
- proxy reconstruction loss
- proxy mel loss

保存：

```text
metrics/teacher_encoder_profile.csv
metrics/teacher_encoder_profile.json
metrics/hand_encoder_profile.csv
metrics/hand_encoder_profile.json
reports/teacher_condition.md
```

baseline 口径必须满足：

```text
teacher/hand baseline = pretrained SpeechTokenizer checkpoint
```

不得把仅由 `SpeechTokenizer(cfg)` 随机初始化得到的 hand-designed encoder 当成质量参考。随机 hand encoder 只能用于 shape smoke test 或参数/MACs 实现诊断，不能作为 teacher alignment、RVQ compatibility 或 best architecture 选择的 reference。

### 步骤 8：逐候选执行 teacher-guided encoder-side proxy evaluation

对每个 candidate：

1. 读取 candidate JSON。
2. 检查 `len(encoder_strides)=4` 和 `prod(encoder_strides)=320`。
3. 构建 candidate encoder B。
4. 构建 candidate encoder B，并接入冻结的预训练 SpeechTokenizer transform/RVQ/decoder。

```text
e_B = B.encoder(x)
q_B, codes_B = frozen_teacher_quantizer(e_B)
x_hat_B = frozen_teacher_decoder(q_B)
```

5. 在 stage2/stage3/final 中，先只训练 B.encoder 若干 step。推荐 loss 为：

```text
loss =
  λ_latent * SmoothL1(e_B, e_A)
+ λ_cos    * cosine_distance(e_B, e_A)
+ λ_delta  * SmoothL1(Δe_B, Δe_A)
+ λ_qfeat  * SmoothL1(q_B, q_A)
+ λ_sem    * semantic_proxy_loss(optional)
```

其中：

```text
e_A = frozen_teacher_encoder(x)
q_A, codes_A = frozen_teacher_quantizer(e_A)
e_B = B.encoder(x)
q_B, codes_B = frozen_teacher_quantizer(e_B)
```

`codes_A/codes_B` 是离散索引，不能直接作为可导训练目标；主训练使用 `q_B` 与 `q_A` 的量化特征距离，`code_agreement` 作为训练后评估指标。teacher A、transform、RVQ 和 decoder 全部冻结，不参与梯度更新。
6. 保持 RVQ codebooks、`n_q`、`codebook_size`、transform 和 payload 口径不变。
7. 执行 teacher/student forward：

```text
e_A = A.encoder(x)
e_B = B.encoder(x)
```

8. 检查：
   - `e_A.shape[1] == 1024`
   - `e_B.shape[1] == 1024`
   - 1 秒输入时 `e_B.shape[-1] == 50`
   - 通过冻结 A quantizer 后 `codes_B.shape[0] == 3`
9. 计算：
   - teacher latent SmoothL1
   - teacher latent cosine distance
   - teacher temporal delta loss
   - RVQ code agreement / code flip rate
   - frozen-quantizer quantized feature L1
   - short distillation train mean/last loss
   - semantic proxy loss
   - proxy reconstruction L1
   - proxy mel loss
   - encoder params
   - encoder MACs
   - encoder RTF
10. 写入 search records。

输出：

```text
metrics/nas_records.csv
metrics/nas_records.json
metrics/teacher_alignment.csv
metrics/teacher_alignment.json
metrics/rvq_compatibility.csv
metrics/rvq_compatibility.json
metrics/encoder_proxy_results.csv
metrics/encoder_proxy_results.json
reports/teacher_condition.md
reports/teacher_guided_selection.md
reports/encoder_interface_check.md
reports/decoder_condition.md
```

若某候选实例化失败、shape 不合法、OOM 或产生 NaN/Inf，应将该候选标记为 `failed`，保留错误摘要，不得删除记录。若连续 2 次 OOM 或同一错误重复 3 次，必须中止当前实验。

### 步骤 9：计算 Pareto frontier

在 completed 且 valid_interface 的 NAS candidates 中计算 Pareto frontier。

推荐主目标：

```text
minimize teacher_latent_smooth_l1
minimize teacher_latent_cosine_distance
minimize teacher_temporal_delta_loss
maximize rvq_code_agreement
minimize rvq_quantized_feature_l1
minimize proxy_recon_l1
minimize proxy_mel_loss
minimize encoder_macs
minimize encoder_params
minimize encoder_rtf_mean
```

必须在 `reports/pareto_selection.md` 中说明实际使用了哪些目标。

输出：

```text
metrics/pareto_frontier.csv
reports/pareto_selection.md
```

Pareto frontier 必须能追溯到 `metrics/nas_records.csv/json`。

### 步骤 10：选择 best architecture

推荐选择规则：

```text
1. 只考虑 status=completed 且 valid_interface=true 的候选；
2. 只考虑 prod(encoder_strides)=320 且 latent_dimension=1024 的候选；
3. 从 Pareto frontier 中选择 teacher latent 对齐、RVQ compatibility、重建 proxy 均未明显劣化，同时 encoder params/MACs/RTF 更低的候选；
4. 若所有候选虽然更轻但无法对齐 A encoder latent 或 code agreement 过低，则选择 hand-designed fallback 或标记 NAS failed。
```

主脚本采用 teacher-guided quality-constrained proxy score。teacher alignment、RVQ compatibility 和 reconstruction proxy 都按 teacher/hand-designed baseline 归一化，先计算超过容忍 margin 的正向 penalty，再叠加资源项：

```text
score(a) =
  w_latent * max(0, log(L_latent(a) / L_latent(ref)) - log(1 + margin_latent))
 w_cos    * max(0, D_cos(a) - margin_cos)
 w_delta  * max(0, log(L_delta(a) / L_delta(ref)) - log(1 + margin_delta))
 w_code   * max(0, agreement_ref - agreement(a))
+ w_recon * max(0, log(L_recon(a) / L_recon(hand)) - log(1 + margin_recon))
+ w_mel   * max(0, log(L_mel(a)   / L_mel(hand))   - log(1 + margin_mel))
+ w_mac   * log(MACs(a)   / MACs(hand))
+ w_param * log(Params(a) / Params(hand))
+ w_rtf   * log(RTF(a)    / RTF(hand))
```

默认 `selection_mode=balanced` 使用 `margin_latent=margin_delta=margin_recon=margin_mel=0.10`，并记录 `stage_teacher_penalty`、`stage_rvq_penalty`、`stage_quality_penalty`、`stage_resource_score` 和每个 proxy 的 penalty。这样资源优势只在 teacher alignment 与 RVQ compatibility 未明显劣化的候选之间起主要作用，避免自动选出过轻但不适合量化接口的 encoder。

保存最终配置：

```text
artifacts/best_architecture/best_seanet_config.json
```

该配置必须包含：

- `encoder_strides`
- `decoder_strides`
- `decoder_condition`
- searched encoder macro 参数
- searched `layer_ops_list`
- searched `layer_se_list`
- fixed interface 参数

### 步骤 11：decoder contamination check

写入：

```text
reports/decoder_contamination_check.md
```

必须说明：

- 主贡献是否只声明 encoder-side NAS。
- decoder 是否固定为 frozen pretrained SpeechTokenizer decoder。
- decoder 的算子、宽度、深度是否进入搜索空间。
- 若使用 joint decoder NAS，是否已标记为 implementation variant。
- 完整重建质量是否按 `decoder_condition` 分组报告。

### 步骤 12：总结报告

写入：

```text
reports/summary.md
reports/status.json
metrics/results.json
metrics/results.csv
```

summary 至少包含：

- run_id
- 搜索模式
- seed
- proxy subset 大小
- 候选生成数量、完成数量、失败数量
- 搜索空间摘要
- teacher A 来源、checkpoint、冻结策略
- teacher latent alignment 结果
- RVQ compatibility 结果
- hand-designed encoder profile
- best NAS architecture
- `encoder_strides` 与 `decoder_strides`
- decoder condition
- 与 hand-designed encoder 的 params/MACs/RTF 对比
- proxy 指标对比
- Pareto selection 依据
- interface check 结果
- decoder contamination check 结果
- 是否可供实验二使用
- 限制说明

不得填入没有真实来源的结果、指标、checkpoint、音频样本或 Pareto frontier。

## 9. 指标与统计方式

| 指标 | 统计方式 | 保存位置 |
|---|---|---|
| encoder params | 只统计发送端 encoder 参数量 | `metrics/nas_records.csv/json` |
| encoder MACs | 固定 1 秒输入统计 encoder-only MACs | `metrics/nas_records.csv/json` |
| encoder RTF | `encoder_forward_time / audio_duration`，warmup 后多次测量 | `metrics/nas_records.csv/json` |
| teacher latent SmoothL1 | `e_B` 与冻结 teacher `e_A` 的归一化 latent L1/SmoothL1 | `metrics/teacher_alignment.csv/json` |
| teacher latent cosine distance | `e_B` 与 `e_A` 在 latent 维度上的 cosine distance | `metrics/teacher_alignment.csv/json` |
| teacher temporal delta loss | `Δe_B` 与 `Δe_A` 的时间动态一致性 | `metrics/teacher_alignment.csv/json` |
| RVQ code agreement | 冻结 A quantizer 下 `codes_B` 与 `codes_A` 的一致率 | `metrics/rvq_compatibility.csv/json` |
| RVQ code flip rate | `1 - code_agreement`，用于观察码本区域翻转 | `metrics/rvq_compatibility.csv/json` |
| RVQ quantized feature L1 | 冻结 A quantizer 下 `q_B` 与 `q_A` 的量化特征距离 | `metrics/rvq_compatibility.csv/json` |
| semantic proxy loss | encoder 输出经固定 transform 或对齐头后与 HuBERT 语义教师特征比较，仅作辅助诊断 | `metrics/encoder_proxy_results.csv/json` |
| proxy reconstruction L1 | 在明确 decoder condition 下统计 | `metrics/encoder_proxy_results.csv/json` |
| proxy mel loss | 在明确 decoder condition 下统计 | `metrics/encoder_proxy_results.csv/json` |
| interface validity | latent dimension、latent rate、codes shape 是否通过 | `reports/encoder_interface_check.md` |
| decoder condition | fixed / geometry_matched / joint_nas_variant | `reports/decoder_condition.md` |
| Pareto frontier | 多目标非支配筛选 | `metrics/pareto_frontier.csv` |

统计原则：

- 复杂度主表只用 encoder-only 指标。
- 完整重建质量必须标注 decoder condition。
- total autoencoder FLOPs 只能作为 diagnostics，不可作为实验一主结论。
- 每个指标必须能追溯到命令、日志、配置和候选 JSON。

## 10. 结果表格式

### Table A：NAS Search Records

| run_id | candidate_id | stage | distill_steps | encoder_strides | decoder_condition | valid_interface | status | teacher_latent_smooth_l1_before | teacher_latent_smooth_l1_after | rvq_code_agreement_before | rvq_code_agreement_after | semantic_proxy_loss | proxy_recon_l1 | proxy_mel_loss | encoder_params | encoder_macs | encoder_rtf_mean | error |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

### Table B：Pareto Frontier

| run_id | candidate_id | encoder_strides | distill_steps | teacher_latent_smooth_l1_after | rvq_code_agreement_after | proxy_mel_loss | encoder_macs | encoder_params | encoder_rtf_mean | dominated_by | selected | selection_reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|

### Table C：Hand-designed vs NAS Encoder

| run_id | encoder | encoder_strides | decoder_condition | params | macs | rtf_mean | teacher_latent_smooth_l1 | rvq_code_agreement | semantic_proxy_loss | proxy_recon_l1 | proxy_mel_loss | interface_check | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

### Table D：Search Space Summary

| run_id | search_mode | seed | num_requested | num_generated | num_completed | num_failed | stride_space | operator_library | notes |
|---|---|---:|---:|---:|---:|---:|---|---|---|

## 11. 成功标准

最低完成标准：

- 有唯一 `run_id` 和完整 run 目录。
- 保存主配置、算子库、搜索空间、运行命令、日志、环境信息和 proxy subset。
- 保存 teacher A 的配置、checkpoint 引用、冻结策略、teacher alignment 和 RVQ compatibility 指标。
- 候选架构由算子库和搜索策略动态生成，而不是少量手写 candidate。
- 至少完成 teacher A、pretrained hand-designed encoder 与若干 NAS-generated candidates 的 encoder-side 比较。
- stage2-stage4 的候选必须经过短程蒸馏训练后再进入主排序；训练前指标只能作为诊断字段。
- best candidate 通过接口检查：
  - `len(encoder_strides)=4`
  - `prod(encoder_strides)=320`
  - `latent_dimension=1024`
  - `latent_rate=50 steps/s`
  - `codes_shape[0]=3`
- best candidate 相比训练前必须体现 teacher distillation 有效性：`teacher_latent_*` 明显改善，`rvq_code_agreement` 明显高于随机水平。
- 至少一个 encoder-only 复杂度指标优于 hand-designed encoder。
- Pareto frontier 和选择理由可追溯。
- decoder condition 记录完整。
- decoder contamination check 明确 decoder 没有作为 NAS 贡献声明。

以下情况必须重跑或判为失败：

- 搜索改变了总下采样率、latent rate、latent dimension、`M` 或 `K`。
- teacher A 未冻结、teacher checkpoint 缺失或 teacher reference 未保存。
- hand-designed baseline 使用随机初始化模型充当质量参考。
- NAS candidate 未经过短程蒸馏训练，却被写成 teacher-guided NAS 最终结果。
- 候选 B 只在连续 latent 上接近 A，但 frozen quantizer 下 code agreement 大幅下降且报告未说明。
- 候选不是由搜索空间动态生成，而是人工固定候选表。
- decoder 算子也进入搜索，但报告仍声称 encoder-only NAS。
- 搜索结果只能说明 total autoencoder 或 decoder 变轻，无法支持发送端 encoder 主张。
- proxy subset、随机 seed 或搜索空间未保存。
- best architecture JSON 缺失或无法实例化。
- checkpoint、日志、命令、配置或指标不能完整留存。
- 产生 NaN/Inf、非法 shape、连续 OOM 或重复错误达到终止规则。

## 12. 风险与注意事项

- 不要把少量手写 candidate screening 写成 NAS 搜索。
- 可以搜索四层 stride schedule，但必须保持 `prod(strides)=320`。
- 不要搜索会改变 latent rate 的 stride，否则 `500L bps` 负载公式会被污染。
- decoder 主口径固定为 frozen pretrained SpeechTokenizer decoder；几何匹配 decoder 只能作为消融或实现变体。
- 不要把 decoder 侧变化带来的重建质量变化写成 encoder NAS 贡献。
- 不要把短 proxy 权重或 proxy 指标写成最终 SCIT-Speech 结果；最终模型表现必须来自实验二完整训练。
- 不要把“参数量减少”写成“蒸馏成功”。参数量减少来自 NAS 架构，蒸馏成功必须由训练后 `e_B ≈ e_A`、`q_B ≈ q_A` 和 code agreement 改善来证明。
- SpeechTokenizer teacher A 是 proxy 锚点，不是本文方法主语；HuBERT 语义教师特征只能作为训练/诊断辅助信号。
- 不要把 teacher alignment 写成最终通信质量。它只降低架构筛选不确定性，不能替代实验二完整训练和实验三/四评估。
- 如果 B 的 `e_B` 与 A 的 `e_A` 接近但 `codes_B` 与 `codes_A` 差异很大，必须优先报告 RVQ compatibility 风险。
- 若所有 NAS-generated candidates 都明显劣于 hand-designed encoder，应报告负结果或 fallback。

## 12b. 对后续实验的影响

本实验改为 teacher-guided NAS 后，对实验二到实验六的影响如下：

| 后续实验 | 影响 | 必须记录 |
|---|---|---|
| 实验二：SCIT-Speech 训练 | 实验二接收的是 B 的架构，不接收实验一短训权重作为最终模型结论。可选择用 teacher-guided B 初始化 encoder，但最终结果必须来自完整训练。 | `encoder route = teacher-guided NAS encoder from Exp1`；是否加载 teacher-guided 预训练权重；teacher A checkpoint 引用 |
| 实验三：低负载/信道适配 | LCA 训练仍然优化 `L=1,2,3` 截层和轻量 index perturbation。Exp1 只改变发送端 encoder 架构，不改变 ChannelSim 或传输层数。 | `M=3`、`K=1024`、`latent_rate=50` 不变；Base/LCA 使用同一个 Exp1 best architecture |
| 实验四：baseline 对比 | 负载公式不变，因此 `500L bps` 与 packed/packetized payload 统计不受影响。复杂度表中应区分 teacher A、hand encoder、NAS-B encoder。 | bitrate/payload 不因 teacher-guided NAS 改写；新增 encoder compute comparison |
| 实验五：消融诊断 | 需要新增或保留一项消融：hand-designed encoder vs teacher-guided NAS encoder；可选再比较 teacher-guided vs non-teacher NAS。 | ablation matrix 中记录 teacher guidance on/off |
| 实验六：三用户系统 | 系统接口不变，仍然传输 index-only payload。实时延迟可能因 B 更轻而下降，但必须实测。 | 端到端 latency、encoder RTF、payload 仍按同一格式统计 |

因此，teacher-guided NAS 只改变“如何选择发送端 encoder 架构”，不改变共享码本索引传输框架、不改变 RVQ 层数、不改变 `L` 负载控制，也不改变后续实验的评价协议。

## 13. 建议脚本职责

| 建议路径 | 职责 |
|---|---|
| `nas/search_space.py` | 定义算子库、四层 stride space、macro space、candidate 采样/枚举 |
| `nas/evaluate_encoder_proxy.py` | 加载搜索空间，动态生成候选，执行 teacher-guided encoder-side proxy evaluation |
| `nas/teacher_guided_proxy.py` 或等价模块 | 加载冻结 SpeechTokenizer teacher A，计算 teacher alignment 与 RVQ compatibility |
| `nas/distill_candidate_encoder.py` 或等价模块 | 对候选 B 执行短程蒸馏训练，保存训练前/训练后 teacher 与 RVQ 指标 |
| `nas/validate_short_distill.py` | 独立验证短程蒸馏训练是否改善 teacher/RVQ 指标 |
| `nas/encoder_side_model_variant.py` | 支持 encoder replacement，并接入冻结 teacher transform/RVQ/decoder |
| `nas/export_pareto_frontier.py` | 可选：从 search records 独立导出 Pareto frontier |
| `nas/check_encoder_interface.py` | 可选：独立检查 best architecture 的 latent shape 和 codes shape |
| `scripts/collect_environment.py` | 记录环境和 git 状态 |
| `scripts/create_experiment_run.py` | 创建标准 run 目录并复制配置 |

如果现有 `nas/search_autoencoder.py` 搜索完整 autoencoder 或统计 total FLOPs，它只能作为工程参考或 implementation diagnostics。实验一主路径应使用 encoder-side proxy evaluation。

## 14. 交给执行型 AI 的提示词

你要执行“实验一：固定 RVQ 索引接口下的教师引导式 Encoder-side NAS”。请先阅读：

- `output/doc/实验手册.md`
- `output/experiments/README.md`
- `output/doc/experiment_plans/exp1_nas_semantic_encoder.md`
- `config/spt_base_cfg.json`
- `nas/SeaNet.py`
- `nas/model_components.py`
- `speechtokenizer/model.py`

本实验必须使用算子库驱动的 NAS 搜索，并使用冻结的预训练 SpeechTokenizer encoder 作为 teacher A。不能把少量手写 `nas_cand_*` 固定候选表作为主搜索结果。正式 run 必须保存：

- `configs/operator_library.json`
- `configs/nas_search_space.json`
- `configs/staged_nas_runtime_config.json`
- `configs/teacher_model_reference.json`
- `configs/teacher_guided_nas_config.json`
- `artifacts/candidates/*.json`
- `metrics/teacher_alignment.csv/json`
- `metrics/rvq_compatibility.csv/json`
- `metrics/nas_records.csv/json`
- `metrics/pareto_frontier.csv`

执行口径必须包含短程蒸馏：stage1 只做架构/资源粗筛；stage2、stage3 和 final stage 必须先训练候选 B.encoder，再用训练后的 B 计算 teacher alignment、RVQ compatibility 和 proxy 指标。不得把随机初始化候选直接前向得到的指标写成 teacher-guided NAS 结果。

为本次执行生成唯一：

```text
run_id = exp1_nas_semantic_encoder_YYYYMMDD_HHMMSS_seed{seed}
```

并把全部产物保存到：

```text
output/experiments/{run_id}/
```

执行前必须 preflight：

1. 检查数据清单、音频文件、语义教师特征文件。
2. 检查 teacher A 的 config/checkpoint 可加载，并确认 encoder/quantizer/decoder 全部冻结。
3. 检查固定接口：`sample_rate=16000`、`prod(encoder_strides)=320`、`dimension=1024`、`n_q=3`、`codebook_size=1024`。
4. 检查 GPU、依赖和磁盘空间。
5. 检查 run 目录可写。
6. 跑最小 smoke test：数据加载 1-2 batch、teacher A encoder forward、NAS-generated B candidate forward、frozen A quantizer encode、L=1/2/3 decode、payload toy example。

主实验只声明搜索 latent `Z` 之前的发送端 encoder。允许搜索四层 stride schedule，但必须保持 `prod(strides)=320`。teacher A 只作为 proxy 锚点，不作为本文方法贡献。decoder 主口径固定为冻结的预训练 SpeechTokenizer decoder：

```text
decoder_condition = frozen_teacher_decoder
```

不要搜索 RVQ、`M`、`K`、`L`、ChannelSim、packetization 或三用户协议。不要把 teacher A 的能力、decoder 侧变化或 HuBERT 语义教师写成 NAS 贡献。如果 decoder 算子也进入搜索，必须标记为 implementation variant，不得作为实验一主结论。

若 preflight 或搜索失败，禁止硬跑，写：

```text
reports/failure_report.md
reports/status.json
```

最终只汇报真实产生的搜索空间、算子库、候选架构、teacher alignment、RVQ compatibility、proxy 指标、encoder params/MACs/RTF、Pareto frontier、best architecture、选择理由、decoder condition、下游实验影响和失败原因。不得编造结果、指标、checkpoint、音频样本、日志或 Pareto frontier。

## 15. 交给 image2 的 PPT 图片提示词

建议标题：

```text
实验一：固定 RVQ 索引接口下的教师引导式 Encoder-side NAS
```

建议副标题：

```text
只搜索发送端 x(t) → Z 的轻量 encoder，用冻结 SpeechTokenizer teacher 约束候选表示空间，不改变 RVQ 层数、码本大小和 500L bps 负载口径
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验设计图，主题是“实验一：固定 RVQ 索引接口下的教师引导式 Encoder-side NAS”。

整体背景必须是白色或极浅灰色，正式中文科研汇报方法图风格，深蓝标题，青蓝表示 NAS encoder 搜索，绿色表示固定 RVQ 索引接口，琥珀色表示 teacher-guided 短程蒸馏和 Pareto 选择，灰色表示冻结或不搜索的模块。不要真实照片，不要人物，不要卡通，不要复杂 3D，不要赛博朋克。

画面采用左中右三栏结构，并在底部放一条实验边界说明。

左栏标题：Fixed RVQ Index Interface
画一个绿色锁定接口框，列出固定通信条件：
sample rate = 16 kHz
prod(strides) = 320
latent rate = 50 steps/s
latent dimension = 1024
M = 3 RVQ layers
K = 1024 codes per layer
L ∈ {1, 2, 3}
R_index(L) = 500L bps
用绿色锁形图标表示这些条件不可被 NAS 改变。

左栏下方画出固定下游模块，全部用灰色：
frozen RVQ quantizer
frozen decoder for proxy check
not searched
强调本实验不搜索 RVQ、decoder、packetization、ChannelSim 或三用户路由。

中栏标题：Teacher-guided Encoder-side NAS
画两条并行路径：
Teacher A：pretrained SpeechTokenizer encoder, frozen
Student B：NAS-generated lightweight encoder candidate

两条路径都接收同一段 speech x(t)：
x(t) → A.encoder → e_A
x(t) → B.encoder → e_B
在 e_A 和 e_B 旁标注：
shape = [B, 1024, 50]

在 Student B 模块内部画可搜索开关：
4-layer stride schedule
residual operator library
channel width
compress ratio
LSTM
activation
SE switch
用青蓝色表示这些是搜索对象。

中栏下方画一个琥珀色短程蒸馏模块：
short distillation of B.encoder only
loss: latent alignment + cosine + temporal delta + quantized feature
teacher A / RVQ / decoder frozen
注意箭头只能更新 Student B encoder，不要让 teacher 或 RVQ codebooks 被更新。

右栏标题：Proxy Metrics and Pareto Selection
画候选架构列表进入评估器：
candidate JSON
interface check
short distill before / after
teacher alignment
RVQ compatibility
proxy reconstruction
encoder params / MACs / RTF

在右栏中画一个小型 Pareto frontier 散点图：
横轴：encoder compute cost
纵轴：teacher / RVQ proxy loss
用琥珀色高亮 selected architecture。
旁边放产物条：
nas_records.csv
teacher_alignment.csv
rvq_compatibility.csv
pareto_frontier.csv
best_seanet_config.json

画面底部放一条依赖关系：
Exp1 selects encoder architecture → Exp2 trains SCIT-Speech-Base → Exp3 evaluates low-load LCA

底部结论条：
实验一只回答“发送端 encoder 能否在固定 RVQ 索引接口下更轻量且更接近冻结 teacher 表示空间”；最终通信质量必须由实验二完整训练和实验三/四统一评估给出。

视觉风格：
白底或极浅灰背景，三栏对齐，线条清晰，模块之间留白充足。固定接口用绿色锁，NAS candidate 用青蓝模块，teacher-guided distillation 和 Pareto selection 用琥珀色，冻结 teacher/RVQ/decoder 用灰色。整体像论文方法实验设计图，不要装饰过多，不要做成科幻海报。

注意事项：
不要把少量手写 candidate 画成完整 NAS 搜索。
不要暗示 NAS 搜索了 RVQ 层数 M、码本大小 K、传输层数 L、ChannelSim、packetization 或三用户协议。
不要把 decoder 变化写成 encoder NAS 贡献；decoder 主口径是 frozen teacher decoder / proxy check。
不要把 teacher alignment 写成最终通信质量；它只是架构筛选 proxy。
不要把“参数量减少”画成“蒸馏成功”；蒸馏成功必须由 e_B 接近 e_A、q_B 接近 q_A、RVQ code agreement 改善来表达。
不要填入未经真实 run 产生的候选数、best 指标、Pareto 数值、checkpoint 或音频结果。
不要改变 latent rate，否则会污染 500L bps 负载口径。
```
