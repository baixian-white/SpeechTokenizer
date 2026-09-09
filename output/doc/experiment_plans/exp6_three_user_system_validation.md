# 实验名称：实验六 三用户实时系统验证

## 1. 实验目的

本实验验证 `SCIT-Speech` 的 index-only transmission 能进入三用户在线通信流程。它是系统可行性证明，不替代实验三的质量验证，也不替代实验四的 baseline 负载-质量对比。

本实验回答：

- A/B/C 三个用户是否能预部署同一组 shared RVQ codebooks。
- 任一用户作为发送端时，是否能实时产生前 `L` 层 discrete codebook indices。
- 索引流是否能带着 user/session id 被正确路由到目标接收端。
- 接收端是否能基于 shared RVQ codebooks 实时重建语音。
- 会话中的单路 payload、会话总 payload、平均延迟、最大延迟和 RTF 是否可记录。

第一版优先采用 A/B/C 轮流发言模式；近似并发模式可选，不把系统扩展为复杂会议混音问题。

## 2. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | 三用户系统验证边界 |
| 实验三产物 | `output/experiments/{exp3_run_id}/` | 推荐使用 `SCIT-Speech-LCA` checkpoint |
| 可选实验二产物 | `output/experiments/{exp2_run_id}/` | 可作为 Base 系统对照 |
| 实时单机脚本 | `实时语音系统/demo_now.py`、`demo_now.md` | 当前支持麦克风到扬声器、`--rvq_layers`、monitor、保存输出；不是三用户路由 |
| 模型 API | `speechtokenizer/model.py` | encode/decode 和 checkpoint 加载 |
| 固定 demo 音频 | `samples/`、`demo/` 或新增 `artifacts/demo_script/` | 可用于无麦克风的 dry-run 和 replay |
| 三用户配置 | 需要新增 | user/session id、路由表、L、chunk_seconds、设备/网络设置 |

需要新增脚本或模块：

| 建议路径 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `realtime_three_user/demo_three_user.py` | 三用户轮流/近似并发会话控制入口 | session config、model config、checkpoint | session logs、payload、latency、demo wav |
| `realtime_three_user/router.py` | 路由 index packets，记录 user/session id 映射 | packet stream、routing table | route logs、exceptions |
| `realtime_three_user/packet_schema.py` | 定义 packet header、payload、timestamp、sequence id | schema config | packetized payload bytes |
| `scripts/replay_three_user_session.py` | 使用固定 wav 模拟 A/B/C 发言，不依赖实时麦克风 | audio script、session config | deterministic logs、samples |
| `scripts/analyze_three_user_logs.py` | 汇总 payload、latency、RTF、路由正确率、异常 | session logs | metrics、summary |
| `scripts/collect_environment.py` | 保存环境 | run 目录 | `reports/environment.md/json` |

## 3. 实验输出

推荐 `run_id`：

```text
exp6_three_user_system_validation_YYYYMMDD_HHMMSS_seed{seed}
```

输出目录：

```text
output/experiments/{run_id}/
```

应产生：

- `configs/session_config.json`：A/B/C user id、session id、L、chunk、路由表、设备和网络设置。
- `configs/packet_schema.json`：packet 字段、字节数、header 定义。
- `configs/model_reference.json`：模型 config/checkpoint/run provenance。
- `commands/run_command.txt`：实际 demo/replay/analyze 命令。
- `logs/session_events.jsonl`：会话事件。
- `logs/routing.jsonl`：每个 packet 的源、目标、session、seq、route result。
- `logs/payload.jsonl`：payload 统计。
- `logs/latency.jsonl`：encode、packetize、route、decode、playback 或 replay 阶段延迟。
- `logs/exceptions.jsonl`：异常记录。
- `metrics/session_payload.json/csv`。
- `metrics/session_latency.json/csv`。
- `metrics/routing_accuracy.json/csv`。
- `samples/demo_audio/original/`、`samples/demo_audio/reconstructed/`。
- `reports/session_report.md`、`reports/summary.md`。

## 4. 实验变量与对照

主变量：

- 会话模式：
  - 必做：A/B/C 轮流发言。
  - 可选：近似并发模式，即两个用户在相近时间窗口发送。
- 传输层数：建议主 run 使用 `L=1` 或 `L=2`；可补 `L=3` 作为上限。
- 模型：主 run 使用 `SCIT-Speech-LCA`；可选对照 `SCIT-Speech-Base`。

固定变量：

- 同一 shared RVQ codebooks/checkpoint。
- 同一 packet schema。
- 同一 A/B/C user id 和 session id 分配。
- 同一 chunk_seconds、frame_seconds、设备或 replay 音频。

记录项：

- user/session id 映射。
- 路由正确性。
- 单路 payload。
- 会话总 payload。
- 平均延迟、最大延迟、p95 延迟。
- RTF。
- 异常和丢包/队列溢出。

## 5. 详细执行步骤

### 步骤 1：创建 run 目录

- 操作目标：建立系统验证独立目录。
- 涉及文件或脚本：需要新增 `scripts/create_experiment_run.py`。
- 输入：`run_id=exp6_three_user_system_validation_YYYYMMDD_HHMMSS_seed{seed}`。
- 输出：标准 run 目录。
- 检查点：标准子目录存在。
- 失败时如何判断问题：run 目录重名或缺失时停止。

### 步骤 2：验证模型和 shared codebooks

- 操作目标：确认 A/B/C 三端使用同一模型 config/checkpoint。
- 涉及文件或脚本：`speechtokenizer/model.py`、需要新增 `scripts/check_checkpoint.py`。
- 输入：LCA config/checkpoint。
- 输出：`configs/model_reference.json`、`reports/model_check.md`。
- 检查点：三端引用同一 checkpoint hash；`n_q=3`、`K=1024`。
- 失败时如何判断问题：三端 checkpoint 不一致时，本实验无效。

### 步骤 3：定义三用户 session config

- 操作目标：固定 user/session id 和发言顺序。
- 涉及文件或脚本：需要新增 `realtime_three_user/demo_three_user.py`。
- 输入：用户列表、路由表、L、chunk_seconds。
- 输出：`configs/session_config.json`。
- 检查点：至少包含：
  - users: A/B/C
  - session id
  - speaking schedule
  - target routing policy
  - `rvq_layers=L`
  - replay 或 live mode
- 失败时如何判断问题：缺少 route table 或 target user 时，无法判断路由正确性。

### 步骤 4：定义 packet schema 和 payload 口径

- 操作目标：确保 packetized payload 可复现。
- 涉及文件或脚本：需要新增 `realtime_three_user/packet_schema.py`。
- 输入：header 字段定义。
- 输出：`configs/packet_schema.json`。
- 检查点：字段至少包含 source_user_id、target_user_id、session_id、seq_id、timestamp、L、payload_length、packed_indices。
- 失败时如何判断问题：没有 schema 时，只能报告 ideal payload，不能报告真实 packetized payload。

### 步骤 5：执行 dry-run replay

- 操作目标：在无实时设备风险下验证三用户路由和日志。
- 涉及文件或脚本：需要新增 `scripts/replay_three_user_session.py`。
- 输入：A/B/C 固定 wav、session config、model config/checkpoint。
- 输出：session logs、routing logs、payload logs、reconstructed wav。
- 检查点：A/B/C 每个用户至少发言一次；目标用户收到正确 session 的 packets。
- 失败时如何判断问题：如果 replay 都不能正确 route，不进入 live demo。

### 步骤 6：执行 A/B/C 轮流发言 live 或 replay 会话

- 操作目标：验证三用户在线流程。
- 涉及文件或脚本：需要新增 `realtime_three_user/demo_three_user.py`；当前 `实时语音系统/demo_now.py` 可作为单链路音频 I/O 参考。
- 输入：session config、model reference、设备或 replay 音频。
- 输出：`logs/session_events.jsonl`、`logs/routing.jsonl`、`logs/payload.jsonl`、`logs/latency.jsonl`、demo audio。
- 检查点：A、B、C 轮流作为 sender；接收端能解码；日志中 route result 为 success。
- 失败时如何判断问题：若设备无声，先用 `demo_now.py --passthrough --monitor` 排查；若路由错，查看 route log 的 source/target/session/seq。

### 步骤 7：可选近似并发模式

- 操作目标：测试两个用户相近时间发送时，系统能否区分 stream。
- 涉及文件或脚本：`realtime_three_user/demo_three_user.py`。
- 输入：并发 schedule。
- 输出：并发 session logs、异常记录。
- 检查点：packets 按 user/session/seq 可区分；队列溢出或播放冲突被记录。
- 失败时如何判断问题：并发失败不否定主轮流模式，但必须写入 summary。

### 步骤 8：分析 payload、延迟和路由正确性

- 操作目标：将系统日志转成指标。
- 涉及文件或脚本：需要新增 `scripts/analyze_three_user_logs.py`。
- 输入：session/routing/payload/latency logs。
- 输出：`metrics/session_payload.json/csv`、`metrics/session_latency.json/csv`、`metrics/routing_accuracy.json/csv`。
- 检查点：报告 single-stream payload、session aggregate payload、avg/max/p95 latency、RTF、routing accuracy。
- 失败时如何判断问题：日志缺 timestamp 或 seq_id 时，无法计算延迟和路由正确性，需要修复 schema。

### 步骤 9：保存 demo 音频样本和异常记录

- 操作目标：为系统可行性证明提供可听材料。
- 涉及文件或脚本：`demo_three_user.py`、`analyze_three_user_logs.py`。
- 输入：session logs 和 audio outputs。
- 输出：`samples/demo_audio/original/`、`samples/demo_audio/reconstructed/`、`logs/exceptions.jsonl`。
- 检查点：每段音频带有 user id、session id、L、source/target、时间戳。
- 失败时如何判断问题：无音频样本时，报告只能证明日志流程，不能证明重建播放。

### 步骤 10：生成系统验证报告

- 操作目标：形成论文系统可行性材料。
- 涉及文件或脚本：需要新增 `scripts/summarize_exp6.py`。
- 输入：metrics、logs、sample manifest。
- 输出：`reports/session_report.md`、`reports/summary.md`。
- 检查点：明确写本实验不是质量验证；质量指标引用实验三/四。
- 失败时如何判断问题：若报告把 demo 指标当作主质量指标，需修正。

## 6. 指标与统计方式

| 指标 | 统计方式 | 保存位置 |
|---|---|---|
| routing accuracy | 成功路由 packets / 总 packets | `metrics/routing_accuracy.csv` |
| single-stream payload | 单个 sender 的 packetized bytes / speech duration | `metrics/session_payload.csv` |
| session aggregate payload | 会话中所有 streams 的 payload 总和 / session duration | 同上 |
| ideal payload | `L * 50 * 10 bps`，按发言时长计算 | 同上 |
| overhead ratio | `(packetized - ideal) / ideal` | 同上 |
| encode latency | encoder + RVQ 时间 | `metrics/session_latency.csv` |
| route latency | packet 入队到目标出队时间 | 同上 |
| decode latency | codebook lookup + decoder 时间 | 同上 |
| end-to-end latency | capture/replay chunk 到 reconstructed/playback ready | 同上 |
| max latency / p95 latency | 延迟分布统计 | 同上 |
| RTF | processing wall time / audio duration | 同上 |
| exception count | 异常事件数量和类型 | `logs/exceptions.jsonl` |

本实验只做系统可行性，不把 WER/PESQ/STOI 作为主结论；若保存 demo 音频，可选用实验三/四脚本补充质量指标，但不得替代主评估。

## 7. 结果记录格式

### Table A：Session user mapping

| run_id | session_id | user_id | role_schedule | device_or_audio_source | target_policy | model | L |
|---|---|---|---|---|---|---|---:|

### Table B：Routing log summary

| run_id | session_id | packet_id | seq_id | source_user | target_user | routed_to | route_success | timestamp_send | timestamp_receive | notes |
|---|---|---|---:|---|---|---|---|---:|---:|---|

### Table C：Payload statistics

| run_id | session_id | source_user | target_user | L | utterance_id | duration_sec | ideal_bits | packed_bytes | packetized_bytes | payload_bps | overhead_ratio |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|

### Table D：Latency statistics

| run_id | session_id | mode | source_user | target_user | L | avg_latency_ms | p95_latency_ms | max_latency_ms | encode_ms | route_ms | decode_ms | RTF | packet_count |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Table E：Exception records

| run_id | session_id | timestamp | component | user_id | severity | error_type | message | action_taken |
|---|---|---:|---|---|---|---|---|---|

## 8. 成功标准

最低完成标准：

- A/B/C 三个 user id 和 session id 映射保存。
- 三个用户在轮流发言模式下都至少作为 sender 一次。
- 每个 sender 的 packets 均被正确 route 到目标接收端。
- 保存 session log、routing log、payload log、latency log、exception log。
- 保存单路 payload、会话总 payload、平均/最大/p95 延迟、RTF。
- 保存 demo 原始音频和 reconstructed audio。
- 报告明确本实验是系统可行性证明。

可选成功标准：

- 完成近似并发模式，并记录队列/路由行为。
- 对 Base vs LCA 或不同 `L` 做系统层对照。

失败或需要重跑：

- 三端使用不同 checkpoint 或 shared codebooks。
- user/session id 缺失，无法验证 route。
- 只保存音频，不保存路由/payload/latency 日志。
- packet schema 缺失，无法复现 packetized payload。
- demo 无声且未保存排障记录。

## 9. 风险与注意事项

- 当前 `实时语音系统/demo_now.py` 是单机麦克风到扬声器链路，不是三用户路由系统；三用户实验需要新增会话/路由层。
- Windows 设备索引、采样率、驱动可能导致无声或延迟；先跑 passthrough。
- live demo 不稳定时，先用 deterministic replay 完成系统日志验证。
- 不要让三用户 demo 替代实验三和实验四的质量结论。
- packetized payload 必须包含 metadata，不可只报 ideal `500L bps`。
- 近似并发模式若失败，要记录为系统限制，不要隐藏。

## 10. 实验数据与执行过程留存

`run_id`：

```text
exp6_three_user_system_validation_YYYYMMDD_HHMMSS_seed{seed}
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

- 配置：`configs/session_config.json`、`configs/packet_schema.json`、`configs/model_reference.json`。
- 命令：`commands/run_command.txt`。
- 环境：`reports/environment.md`，含 Python、PyTorch、CUDA、GPU、sounddevice/torchaudio/依赖、git commit、git status。
- 运行日志：`logs/session_events.jsonl`、`logs/routing.jsonl`、`logs/payload.jsonl`、`logs/latency.jsonl`、`logs/exceptions.jsonl`。
- user/session mapping：`metrics/session_user_mapping.csv`。
- payload：`metrics/session_payload.json/csv`。
- latency：`metrics/session_latency.json/csv`。
- routing：`metrics/routing_accuracy.json/csv`。
- audio：`samples/demo_audio/original/`、`samples/demo_audio/reconstructed/`。
- failure：`reports/failure_report.md`。

可复现实验所必需文件：

- `configs/session_config.json`
- `configs/packet_schema.json`
- `configs/model_reference.json`
- `commands/run_command.txt`
- `reports/environment.md`
- `logs/session_events.jsonl`
- `logs/routing.jsonl`
- `logs/payload.jsonl`
- `logs/latency.jsonl`
- `logs/exceptions.jsonl`
- replay 模式下的原始输入音频

论文作图/写表所需文件：

- `metrics/session_payload.csv`
- `metrics/session_latency.csv`
- `metrics/routing_accuracy.csv`
- `metrics/session_user_mapping.csv`
- `samples/demo_audio/**`
- `reports/session_report.md`
- `reports/summary.md`

失败实验如何留存：

- 保留 session config、packet schema、命令、stdout/stderr、异常日志。
- `failure_report.md` 写明失败组件：device、model load、encode、packetize、route、decode、playback、analysis。
- 若 live 失败但 replay 成功，分别记录，不混淆。

多次 run 如何区分：

- 不同 `L`、不同模型、不同 session config、live vs replay、轮流 vs 并发都使用新 `run_id`。
- 在 `reports/summary.md` 中列出与父 run 或对照 run 的关系。

最终 summary 如何生成：

- 从 logs 自动分析生成 payload、latency、routing 表。
- summary 必须说明系统可行性边界和异常。
- 不写未记录的主观体验或质量指标。

## 11. 交给执行型 AI 的提示词

你要执行“实验六：三用户实时系统验证”。请先阅读 `output/doc/experiment_plans/exp6_three_user_system_validation.md`、`output/doc/实验手册.md`、实验三 run 目录、`实时语音系统/demo_now.py`、`实时语音系统/demo_now.md`、`speechtokenizer/model.py` 和 `output/experiments/README.md`。不得编造会话日志、payload、延迟、路由正确性、checkpoint 或 demo 音频。

请生成唯一 `run_id=exp6_three_user_system_validation_YYYYMMDD_HHMMSS_seed{seed}`，在 `output/experiments/{run_id}/` 下保存 configs、commands、logs、checkpoints、metrics、samples、reports、artifacts。必须保存 session config、packet schema、model reference、实际命令、stdout/stderr、环境信息、user/session id 映射、会话记录、路由日志、payload 统计、延迟统计、RTF、异常记录、原始 demo 音频和 reconstructed audio。

优先执行 A/B/C 三用户轮流发言模式；近似并发模式可选。当前 `demo_now.py` 只是单链路实时脚本，若缺少三用户路由脚本，请新增或明确写入 failure report 中的脚本职责、输入和输出。若 live 设备不可用，先用固定 wav 做 replay。遇到缺失脚本、缺失模型、设备无声、路由错误、packet schema 不完整或日志缺失时，不要补假结果；请保留失败日志并写 `reports/failure_report.md`。
