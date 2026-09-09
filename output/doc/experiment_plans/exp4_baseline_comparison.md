# 实验名称：实验四 Baseline 对比

## 1. 实验目的

本实验比较 `SCIT-Speech` 的 index-only transmission 与 PCM、传统语音 codec、可选极低码率 codec、可选 neural codec 在低负载操作区间中的负载-可用性关系。实验四不用于宣称全面超过传统 codec；它用于说明在 shared RVQ codebooks 预部署后，只传输 discrete codebook indices 能形成一个不同于 waveform codec 的低负载工作区间。

本实验承接实验二和实验三：

- `SCIT-Speech-Base` 来自实验二。
- `SCIT-Speech-LCA` 来自实验三。
- `L=1/2/3` 分别对应 ideal raw index load `500/1000/1500 bps`。

必须同时报告：

- ideal bitrate
- packed payload
- packetized payload
- overhead ratio

## 2. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | baseline 范围和 payload 口径 |
| Base run | `output/experiments/{exp2_run_id}/` | `SCIT-Speech-Base` config/checkpoint |
| LCA run | `output/experiments/{exp3_run_id}/` | `SCIT-Speech-LCA` config/checkpoint |
| 固定测试清单 | `output/experiments/{exp3_run_id}/artifacts/test_files.txt` 或新增 | 所有方法必须使用同一测试集 |
| 固定样本清单 | `artifacts/fixed_sample_list.txt` | 音频样本附录 |
| 模型 API | `speechtokenizer/model.py` | SCIT encode/decode |
| 推理参考 | `example.py`、`demo_nature/多人嘈杂环境/example.py` | 可参考其逐层 tokens/wav 保存逻辑 |
| 实时脚本参考 | `实时语音系统/demo_now.py` | 支持 `--rvq_layers`，但不是 baseline 评估脚本 |
| 传统 codec 工具 | 系统 `ffmpeg`、`opusenc/opusdec`、AMR-WB 工具、可选 Codec2 | 当前仓库未封装 |
| 可选 neural codec | 外部 EnCodec/DAC 等工具或脚本 | 若环境不可用，标记为未执行，不编造 |

需要新增脚本：

| 建议路径 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `scripts/run_scit_eval.py` | 对 Base/LCA 的 `L=1/2/3` 生成 codes、重建音频、质量指标 | config、checkpoint、test list | `metrics/scit_results.*`、`samples/scit/` |
| `scripts/pack_indices.py` | 将 `L x T` 索引按 `ceil(log2 K)=10` bit 打包 | codes、K、L | packed bytes、payload stats |
| `scripts/packetize_indices.py` | 模拟或实现 packetized payload，加入 header、user/session id、timestamp、length | packed bytes、metadata config | packet files、payload stats |
| `scripts/run_codec_baselines.py` | 调用 PCM/Opus/AMR-WB/Codec2/neural codec 编码解码并保存命令 | test list、codec config | encoded files、decoded wav、logs |
| `scripts/evaluate_audio_metrics.py` | 统一计算 WER/CER/STOI/PESQ/ViSQOL/semantic similarity | original、decoded/recon | metrics JSON/CSV |
| `scripts/summarize_payload.py` | 生成 ideal/packed/packetized/overhead 表 | SCIT packet stats、baseline files | `metrics/payload_summary.*` |

## 3. 实验输出

推荐 `run_id`：

```text
exp4_baseline_comparison_YYYYMMDD_HHMMSS_seed{seed}
```

输出目录：

```text
output/experiments/{run_id}/
```

应产生：

- `configs/baseline_comparison_config.json`：测试集、codec 条件、评价模型、payload header 定义。
- `commands/run_command.txt`：所有编码、解码、评估命令。
- `logs/stdout.log`、`logs/stderr.log`、`logs/codec_commands/*.log`。
- `artifacts/test_files.txt`、`artifacts/fixed_sample_list.txt`。
- `artifacts/encoded/{method}/`：baseline 编码后文件。
- `samples/original/`、`samples/scit_base/`、`samples/scit_lca/`、`samples/baselines/{method}/`。
- `metrics/scit_payload.json/csv`。
- `metrics/baseline_payload.json/csv`。
- `metrics/audio_quality_results.json/csv`。
- `metrics/payload_summary.json/csv`。
- `reports/baseline_comparison_summary.md`、`reports/payload_accounting.md`、`reports/summary.md`。

## 4. 实验变量与对照

主变量：

- 方法：
  - `SCIT-Speech-Base`，`L=1/2/3`
  - `SCIT-Speech-LCA`，`L=1/2/3`
  - PCM 16-bit 16 kHz
  - Opus 或 AMR-WB
  - 可选 Codec2
  - 可选 neural codec

控制变量：

- 相同测试集、相同原始采样率、相同评价脚本、相同 ASR 和质量评价模型。
- 对所有重建/解码音频统一保存为 16 kHz 单声道 WAV 后评价。

payload 口径：

- `ideal index bitrate = L * 50 * 10 bps`
- `packed payload bitrate = bit-packed indices bytes * 8 / duration`
- `packetized payload bitrate = packed indices + header + user/session id + timestamp + length fields`
- `overhead ratio = (packetized payload - ideal payload) / ideal payload`

baseline 注意：

- Opus/AMR-WB 通常不在 500-1500 bps 稳定工作；不要强行同码率比较，使用 bitrate-quality curve。
- neural codec 若最低公开码率高于本文 index load，表中明确标注。

## 5. 详细执行步骤

### 步骤 1：创建 run 目录

- 操作目标：建立 baseline 对比的独立留存目录。
- 涉及文件或脚本：需要新增 `scripts/create_experiment_run.py`。
- 输入：`run_id=exp4_baseline_comparison_YYYYMMDD_HHMMSS_seed{seed}`。
- 输出：标准 run 目录。
- 检查点：所有标准子目录存在。
- 失败时如何判断问题：run 重名或缺少子目录时停止。

### 步骤 2：冻结配置和测试集

- 操作目标：固定所有方法共用的测试集、样本列表和评价设置。
- 涉及文件或脚本：需要新增 `scripts/prepare_exp4_config.py`。
- 输入：实验二/三 run reference、test list、fixed sample list。
- 输出：`configs/baseline_comparison_config.json`、`artifacts/test_files.txt`、`artifacts/fixed_sample_list.txt`。
- 检查点：每个测试音频存在，时长和 speaker/utterance id 可追踪。
- 失败时如何判断问题：若 Base/LCA 使用不同测试集，结果不可比较。

### 步骤 3：记录环境和外部工具版本

- 操作目标：保存 codec 工具和评价工具版本。
- 涉及文件或脚本：需要新增 `scripts/collect_environment.py`。
- 输入：Python 环境、系统 codec 命令。
- 输出：`reports/environment.md`、`reports/tool_versions.md`。
- 检查点：记录 Python/PyTorch/CUDA/GPU/git 状态，同时记录 `ffmpeg -version`、`opusenc --version`、AMR-WB/Codec2/neural codec 版本。
- 失败时如何判断问题：某 codec 工具缺失时，在 config 中标记该 baseline 未执行，不得编造结果。

### 步骤 4：运行 SCIT-Speech Base/LCA 截层重建

- 操作目标：生成 `SCIT-Speech` 在 `L=1/2/3` 下的重建音频和索引文件。
- 涉及文件或脚本：需要新增 `scripts/run_scit_eval.py`，可参考 `example.py` 和 `demo_nature/多人嘈杂环境/example.py`。
- 输入：Base/LCA config、checkpoint、test list。
- 输出：`samples/scit_base/L*/`、`samples/scit_lca/L*/`、`artifacts/scit_codes/{model}/L*/`。
- 检查点：每个样本都有 codes 和 reconstructed wav；`codes` shape 为 `(L, B, Tq)`。
- 失败时如何判断问题：若 checkpoint 加载失败或截层解码失败，先返回实验二/三修复。

### 步骤 5：统计 SCIT ideal、packed、packetized payload

- 操作目标：建立本文关键 payload 口径。
- 涉及文件或脚本：需要新增 `scripts/pack_indices.py`、`scripts/packetize_indices.py`、`scripts/summarize_payload.py`。
- 输入：SCIT codes、`K=1024`、`L`、duration、packet header config。
- 输出：`metrics/scit_payload.json`、`metrics/scit_payload.csv`、`reports/payload_accounting.md`。
- 检查点：同一 utterance 同时记录 ideal bits、packed bytes、packetized bytes、overhead ratio。
- 失败时如何判断问题：若只保存 `int64` `.npy` 大小，不可当作 packed payload；必须区分 packed 和 array storage。

### 步骤 6：运行 PCM baseline

- 操作目标：得到未压缩 16-bit 16 kHz PCM 上界。
- 涉及文件或脚本：需要新增 `scripts/run_codec_baselines.py`。
- 输入：test list。
- 输出：`artifacts/encoded/pcm/`、`samples/baselines/pcm/`、`metrics/baseline_payload.csv`。
- 检查点：PCM bitrate 应按 `16000 * 16 * channels` 计算，并记录文件容器开销。
- 失败时如何判断问题：若输入不是 16 kHz 单声道，需先统一重采样并记录命令。

### 步骤 7：运行 Opus/AMR-WB baseline

- 操作目标：得到传统低码率 codec 参考曲线。
- 涉及文件或脚本：需要新增 `scripts/run_codec_baselines.py`。
- 输入：test list、codec bitrate list。
- 输出：encoded files、decoded wav、payload stats、logs。
- 检查点：保存每条实际命令和每个 bitrate 点；不要强行设置工具不支持的码率。
- 失败时如何判断问题：工具不支持、编码失败或输出异常时，在 `reports/failure_report.md` 记录，不填结果。

### 步骤 8：可选运行 Codec2 或 neural codec

- 操作目标：补充极低码率或 neural codec 对照。
- 涉及文件或脚本：需要新增 codec adapter。
- 输入：codec 可执行文件/环境、test list。
- 输出：encoded/decoded files、payload stats、logs。
- 检查点：若环境不可用，明确标记 optional baseline skipped。
- 失败时如何判断问题：不可用 baseline 不影响主实验完成，但必须在 summary 中说明。

### 步骤 9：统一计算音频质量和语义指标

- 操作目标：所有方法使用同一评价脚本。
- 涉及文件或脚本：需要新增 `scripts/evaluate_audio_metrics.py`。
- 输入：original wav、SCIT recon wav、baseline decoded wav。
- 输出：`metrics/audio_quality_results.json`、`metrics/audio_quality_results.csv`。
- 检查点：保存评价模型版本和 ASR 输出文本；每条样本有 method、bitrate、WER/CER/STOI/PESQ/ViSQOL/semantic similarity。
- 失败时如何判断问题：某指标工具缺失时，该列标记未执行；不能用其他列冒充。

### 步骤 10：整理固定音频样本附录

- 操作目标：为主观听感和论文附录准备样本索引。
- 涉及文件或脚本：需要新增 `scripts/build_audio_sample_manifest.py`。
- 输入：fixed sample list、各方法输出。
- 输出：`samples/sample_manifest.csv`、`reports/audio_samples.md`。
- 检查点：每个固定样本包含 original、Base L1/L2/L3、LCA L1/L2/L3、至少一个传统 codec baseline。
- 失败时如何判断问题：缺少样本时 summary 标记不完整。

### 步骤 11：汇总 baseline 报告

- 操作目标：形成负载-可用性表和图。
- 涉及文件或脚本：需要新增 `scripts/summarize_exp4.py`。
- 输入：payload summary、audio quality results、sample manifest。
- 输出：`reports/baseline_comparison_summary.md`、`reports/summary.md`、可选 `artifacts/figures/*.png`。
- 检查点：写法谨慎，只比较低负载操作区间，不声称全面超过传统 codec。
- 失败时如何判断问题：没有 payload 统计时，实验四不完整；没有 quality metrics 时只能作为 payload 预实验。

## 6. 指标与统计方式

| 指标 | 统计方式 | 说明 |
|---|---|---|
| ideal index bitrate | `L * 50 * 10` | SCIT only |
| packed payload bitrate | bit-packed index bytes * 8 / duration | SCIT only |
| packetized payload bitrate | packed bytes + metadata/header bytes | SCIT and system-like payload |
| overhead ratio | `(packetized - ideal) / ideal` | SCIT payload overhead |
| codec bitrate | encoded file bytes * 8 / duration；另存 nominal bitrate | baseline |
| compression ratio | PCM bitrate / method bitrate | 统一计算 |
| WER/CER | 固定 ASR 模型 | intelligibility |
| STOI | 原始/重建音频 | intelligibility |
| PESQ/ViSQOL | 原始/重建音频 | naturalness |
| semantic similarity | transcript 或 embedding similarity | semantic adequacy |
| RTF | method encode/decode wall time / duration | 运行效率 |

所有指标必须同时保存 machine-readable 和 human-readable：

- JSON：保留嵌套 metadata。
- CSV：用于论文表格和画图。
- Markdown：用于人工审阅。

## 7. 结果记录格式

### Table A：Payload accounting

| run_id | method | model | L | sample_id | duration_sec | ideal_bitrate_bps | packed_payload_bytes | packed_payload_bps | packetized_payload_bytes | packetized_payload_bps | overhead_ratio | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|

### Table B：Baseline codec payload

| run_id | method | codec_setting | sample_id | duration_sec | encoded_path | decoded_wav_path | nominal_bitrate_bps | actual_payload_bytes | actual_payload_bps | container_overhead_notes |
|---|---|---|---|---:|---|---|---:|---:|---:|---|

### Table C：Audio quality

| run_id | method | model | L | codec_setting | sample_id | bitrate_bps | WER | CER | STOI | PESQ | ViSQOL | semantic_similarity | RTF | notes |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

### Table D：Audio sample appendix

| run_id | sample_id | original | scit_base_L1 | scit_base_L2 | scit_base_L3 | scit_lca_L1 | scit_lca_L2 | scit_lca_L3 | baseline_1 | baseline_2 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|

## 8. 成功标准

最低完成标准：

- Base 和 LCA 在 `L=1/2/3` 上完成重建、payload 统计和质量评估。
- PCM 和至少一个传统 codec baseline 完成编码、解码、payload 统计和质量评估。
- ideal、packed、packetized 三类 payload 均保存。
- 所有命令、日志、配置、环境、测试清单、样本清单保存完整。
- 指标同时保存 JSON/CSV/Markdown。

可选完成项：

- Codec2 baseline。
- neural codec baseline。
- MOS-lite 或 AB preference。

失败或需要重跑：

- 只报告 `500L bps`，没有 packed/packetized payload。
- baseline 与 SCIT 使用不同测试集或不同评价脚本。
- 编码失败但仍填入指标。
- 工具版本和命令缺失。

## 9. 风险与注意事项

- 不要声称 SCIT 在所有音质指标上全面优于传统 codec。
- Opus/AMR-WB 的稳定码率区间可能高于 500-1500 bps，应画 bitrate-quality curve。
- `.npy` 或 `int64` codes 文件大小不是 packed payload。
- packetized payload 必须明确 header 结构，否则 overhead ratio 不可复现。
- ASR、PESQ、ViSQOL 工具版本会影响结果，必须记录。
- 若 neural codec 最低码率高于 SCIT index load，必须在表格中说明。
- 所有音频必须统一采样率和声道后评价。

## 10. 实验数据与执行过程留存

`run_id`：

```text
exp4_baseline_comparison_YYYYMMDD_HHMMSS_seed{seed}
```

标准目录：

```text
output/experiments/{run_id}/
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

- 配置：`configs/baseline_comparison_config.json`、`configs/payload_packet_schema.json`。
- 数据清单：`artifacts/test_files.txt`、`artifacts/fixed_sample_list.txt`。
- 命令：`commands/run_command.txt` 和 `commands/codec_commands/*.txt`。
- 环境：`reports/environment.md`、`reports/tool_versions.md`。
- 日志：stdout/stderr、每个 codec 的编码/解码日志。
- checkpoint provenance：`configs/base_run_reference.json`、`configs/lca_run_reference.json`。
- 编码产物：`artifacts/encoded/{method}/`。
- 音频样本：`samples/original/`、`samples/scit_*`、`samples/baselines/*`。
- metrics：payload、quality、RTF、sample manifest。
- failure：`reports/failure_report.md`。

可复现实验所必需文件：

- `configs/baseline_comparison_config.json`
- `configs/payload_packet_schema.json`
- `commands/run_command.txt`
- `commands/codec_commands/*.txt`
- `reports/environment.md`
- `reports/tool_versions.md`
- `artifacts/test_files.txt`
- `metrics/*payload*.json/csv`
- `logs/stdout.log`
- `logs/stderr.log`

论文作图/写表所需文件：

- `metrics/payload_summary.csv`
- `metrics/audio_quality_results.csv`
- `metrics/baseline_payload.csv`
- `samples/sample_manifest.csv`
- `reports/baseline_comparison_summary.md`
- `artifacts/figures/*`

失败实验如何留存：

- 每个失败 codec 条件保留命令、stderr、输入文件、失败原因。
- 可选 baseline 缺失时写 `skipped`，不要填假指标。

多次 run 如何区分：

- 不同 codec 设置、测试集、packet schema 或评价模型都生成新 `run_id`。
- 在 summary 中记录与前一 run 的差异。

最终 summary 如何生成：

- 从 payload 和 quality CSV 合并生成负载-可用性表。
- 明确哪些 baseline 已执行，哪些 skipped。
- 用谨慎措辞描述低负载操作区间。

## 11. 交给执行型 AI 的提示词

你要执行“实验四：Baseline 对比”。请先阅读 `output/doc/experiment_plans/exp4_baseline_comparison.md`、`output/doc/实验手册.md`、实验二和实验三 run 目录、`speechtokenizer/model.py`、`example.py`、`demo_nature/多人嘈杂环境/example.py` 和 `output/experiments/README.md`。不得编造 baseline 结果、codec 输出、payload、指标或音频样本。

请生成唯一 `run_id=exp4_baseline_comparison_YYYYMMDD_HHMMSS_seed{seed}`，在 `output/experiments/{run_id}/` 下保存 configs、commands、logs、checkpoints、metrics、samples、reports、artifacts。必须保存测试清单、固定样本清单、Base/LCA checkpoint provenance、所有实际编码/解码/评估命令、stdout/stderr、环境和工具版本、SCIT codes、编码后文件、解码音频、payload 统计、质量指标表和样本 manifest。

必须区分 ideal bitrate、packed payload、packetized payload 和 overhead ratio。不要把 `.npy` 或 `int64` 文件大小当作 packed payload。至少完成 PCM 和一个传统 codec baseline；Codec2 和 neural codec 是可选项，缺工具或缺环境时标记 skipped 并写明原因。遇到缺失脚本、缺失 codec、编码失败、评估工具缺失或数据缺失时，不要填假结果；请写 `reports/failure_report.md` 或在 summary 中标记该条件未执行。
