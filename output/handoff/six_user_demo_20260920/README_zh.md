# 六用户 Demo 材料交接清单

核查日期：2026-09-20。本目录是交接索引、哈希清单和验证记录，不是包含全部权重/数据的大型打包文件。原模型、数据、缓存、论文和正式测试结果均未修改。

## 先确定演示对象

- 若六个用户是 VCTK 中已训练身份的录音回放，可以使用现有 Exp23 110 类 classifier bundle，但仍需完成客户端集成和六路验收。
- 若六个用户是六位新同学的真人麦克风，现有 Exp23 bundle 不能直接识别他们。它仅输出固定的 110 个 VCTK 标签，需要另做六人的注册匹配或训练/适配，不能把修改 `user_id` 当成完成身份注册。
- 三用户历史路径是 decoded PCM → 冻结 ECAPA → 注册音频相似度匹配；Exp23 路径是同一时间窗的接收端 L3 codes + decoded PCM → 分类器。两者不是同一个识别模块。
- `verified`/置信度不是开放集拒识、反欺骗或身份认证结论。演示 XOR key 也不代表生产加密或 QKD。

## 根目录约定

```text
S = H:\H-CODE\speechtokenizer
E = D:\tmp\speechtokenizer-exp23
F = E\tmp\exp23_formal
L = S\output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42
B = S\output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42
D = S\3用户demo\speechtokenizer_now\speechtokenizer\三用户中心路由通信demo
```

当前主工作区分支为 `chore/repo-reorg-research`，HEAD 为 `077df2a1e19f8898e6f7c278662c43e1f8347af0`。

Exp23 工作区分支为 `feature/exp23-speaker-classifier`，HEAD 为 `8ee038bdb7e3588b70e88e785f5a05edb78a679c`。Exp23 的实现应从这个分支取，不能只复制主工作区旧客户端。

## 1. 三用户实际 LCA v2 Checkpoint

```text
S\3用户demo\Log\spt_base\SpeechTokenizer_best_dev.pt
L\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt
L\checkpoints\SCIT-Speech-LCA_best.pt
```

以上是同一份 LCA v2 step30000 权重，已经重新计算文件 SHA-256：

```text
af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a
```

不要因目录名 `Log/spt_base` 或文件名 `SpeechTokenizer_best_dev.pt` 而误认它是 Base。不要用 step17500、旧 LCA v1 或 model_hub 预训练权重替换。

## 2. 配套配置与源码

```text
S\3用户demo\Log\spt_base\config.json
L\configs\lca_finetune_config.json
L\configs\best_seanet_config.json
S\3用户demo\exp3_low_load_channel_aware_adaptation_v2_strong\configs\best_seanet_config.json
```

前两个 JSON 的内容语义一致，但格式不同，文件哈希可以不同。NAS 配置也必须一并交付；只给 `config.json` 不足以表达 NAS 编码器结构。保留 `nas/`、`speechtokenizer/` 及模型加载相关源码。

六端统一使用同一 codec/config 和 `--rvq_layers 3`。现有工作点为 16 kHz、3 层 RVQ、50 帧/秒、码本大小 1024，demo 的分块为 `--chunk_seconds 0.5`。

## 3. Base Secondary Condition

```text
B\checkpoints\SCIT-Speech-Base_best.pt
B\checkpoints\checkpoint_manifest.json
B\configs\scit_speech_base_config.json
B\configs\best_seanet_config.json
```

重新计算的 checkpoint SHA-256：

```text
8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6
```

这份 Base 与 Exp23 配置 `frozen_upstream.base` 一致。本次发现的正式缓存只有 canonical LCA L3，不能把它称为包含 Base/LCA 全部条件的缓存。

## 4. VCTK 音频

```text
S\data\VCTK\wav48_silence_trimmed\
```

本地枚举结果为 110 个说话人目录、88,328 个 `.flac`，另有一个说明 `.txt`，共 11,731,509,071 字节，约 10.93 GiB。包括 `s5`，不要只用 `p*` 目录模式发现说话人。

这里核实的是本地数量和划分所需输入，并未将每条音频与官方全集校验清单逐一比对，因此不额外声称官方归档完整性已经认证。目录名中的 `wav48` 不表示实际扩展名必须为 `.wav`。

## 5. VCTK 转写

```text
S\data\VCTK\txt\
S\data\VCTK\speaker-info.txt
S\data\VCTK\update.txt
```

转写目录有 109 个说话人子目录、44,583 个 `.txt`。音频目录中的 `p315` 没有对应转写目录，因此 109 类文本受控协议只排除 `p315`；canonical 110 类仍保留它。不能借用其他说话人的同编号句子补转写。

## 6. Split Manifest

已有正式 canonical 文件：

```text
F\cache_lca_l3\cache_manifest\split_manifest.json
```

固定 split seed 42；110 类；每人 train/validation/test = 80/15/15；总数 8,800/1,650/1,650。训练 seeds 41/42/43 共享此划分，不应各自重新生成测试集。

本次未在核查范围内找到现成的正式 109 类 manifest，已使用 Exp23 当前 `speechtokenizer/speaker_identity/data.py` 中的公开函数单独生成：

```text
本目录\text_controlled_109_split.generated.json
本目录\text_controlled_generation_receipt.json
```

固定 seed 1；109 类；每人 80/10/10；总数 8,720/1,090/1,090；排除 `p315`。核查 utterance group 不跨集合、全局归一化文本哈希不跨集合、样本路径存在。

该文件是 2026-09-20 新生成的交接产物，不是补造历史正式实验结果；109 类缓存、训练、测试结果仍未完成。

## 7. Exp23 缓存

真正已经生成的缓存和对应元数据位于：

```text
F\cache\
F\cache_lca_l3\cache_manifest\
```

共 12,100 条记录，对应 12,100 个 `.npz` 和 12,100 个重建 `.flac`；payload 共 595,639,591 字节，约 568.05 MiB。条件仅为 canonical、LCA、L3。

`output/cache/exp23` 是运行手册中的示例路径，不是这次正式运行的实际目录。不要因为找不到示例目录就认定缓存不存在。

复制时必须同时交付 `cache_records.json`、`split_manifest.json`、`tokenizer_state_hashes.json`、`config.json`、`environment.json` 和 `cache_failures.json`。记录中的 `source_audio_path`、`codes_path`、`reconstruction_path` 为本机绝对路径；在接收机使用 JSON 解析器统一改写路径，保留原始 manifest 和原始哈希，另存迁移后的副本。

迁移时改路径，不改 split、speaker label、样本配对、codec hash、温度参数。哈希相同只证明文件内容相同，不自动证明缓存科学协议完整。

## 8. SpeechBrain ECAPA-TDNN

```text
S\output\models\speechbrain_spkrec_ecapa_voxceleb\
  embedding_model.ckpt
  classifier.ckpt
  mean_var_norm_emb.ckpt
  label_encoder.ckpt
  hyperparams.yaml
```

历史 Exp23 环境及本机当前环境的 SpeechBrain 版本均为 `1.1.0`。

本机 Hugging Face snapshot 为：

```text
0f99f2d0ebe89ac095bcc5903c4dd8f72b367286
```

核查本地权重与这个 snapshot 的逐文件哈希一致，Exp23 工作区的对应本地文件也一致。snapshot 中 `label_encoder.txt` 对应本地缓存的 `label_encoder.ckpt`。这确定了当前文件来源对应关系，但不等于历史训练日志独立记录过该下载 revision。

所有文件哈希见 `SHA256SUMS.csv`。正式 classifier bundle 已封装其推理模型，但复现训练或旧 ECAPA profile 路径仍需上述权重。

## 9. Seeds 41/42/43

每个 seed 都交付整个目录：

```text
F\seed41\
F\seed42\
F\seed43\
```

其中至少包括 `checkpoints/best_macro_f1.pt`、`training_config.json`、`environment.json`、`metrics/validation_metrics.json`、`metrics/test_metrics.json`、两份 predictions CSV 和 `metrics/temperature.json`。汇总另附 `F\aggregate\`。

| Seed | Validation Top-1 | Validation Macro-F1 | Test Top-1 | Test Macro-F1 | Temperature |
|---|---:|---:|---:|---:|---:|
| 41 | 94.9697% | 94.9748% | 95.0303% | 94.9360% | 0.9660716224 |
| 42 | 94.6667% | 94.6079% | 94.1818% | 94.1356% | 1.0187726916 |
| 43 | 94.7879% | 94.7370% | 95.5758% | 95.5552% | 1.0053338430 |

以上读取自已有正式指标，不是本次重跑。每 seed 的 validation/test prediction 文件各为 1,650 行。三个种子的平均 Test Top-1 约 94.93%，不是六用户 demo 的实测准确率。不要因 seed43 测试分数更高而替换预先指定的 seed42 部署 bundle。

## 10. 最终 Classifier Bundle

```text
F\export\seed42_bundle\
  classifier.pt
  speaker_labels.json
  preprocessing.json
  calibration.json
  model_config.json
  manifest.json
  validation_smoke.json
```

`classifier.pt` SHA-256：

```text
e497e2e2a22b7517ee215496c84dd7e026853313969d1b6d0b93ed02a9723ef2
```

输入模式为 `codes_waveform`，不是仅波形。预处理为 16 kHz、48,000 样点（三秒）、L3；codes 必须与同一音频窗对应。不要用发送方 ID 代替预测标签，也不要在混音后才做各说话人的识别。

本次重新校验 bundle manifest，并成功在 CPU 加载、推理一条已有 validation 样本：真实身份 `p225`，输出 `p225`。这是加载与单样本接口检查，不是新增正式测试或六用户验收。

## 11. Key、环境和启动验证

Key 文件：

```text
S\3用户demo\speechtokenizer_now\speechtokenizer\两用户通信demo\userA\46.txt
```

本次只记录其路径、长度和 SHA-256，不把明文 key 写进报告、不对外上传。通过私有通道单独移交；后续演示应明确这是示例密钥。

本机解释器：

```text
C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe
```

| 依赖 | 本机核查版本 |
|---|---|
| Python | 3.10.19 |
| torch / torchaudio | 2.9.1+cu128 / 2.9.1+cu128 |
| speechbrain | 1.1.0 |
| numpy | 2.2.6 |
| soundfile | 0.13.1 |
| sounddevice | 0.5.3 |
| huggingface-hub | 1.16.4 |

完整已安装包快照见 `environment_current.json`；历史版本见各 seed 的 `environment.json`。旧 `D\environment.yml` 的 torch/torchaudio 与这些记录不同，也没有列出 SpeechBrain，不能称为本次已验证的可复现环境锁。

### 重新核查的命令

以下命令只做本机回放，不打开麦克风或实体播放；验证脚本会在随机空闲 loopback 端口启动 router 和 3 个客户端，结束后清理本次创建的进程。输出目录必须为空，避免覆盖已有记录。

```powershell
$S = "H:\H-CODE\speechtokenizer"
$E = "D:\tmp\speechtokenizer-exp23"
$P = "C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe"
$Handoff = "$S\output\handoff\six_user_demo_20260920"
& $P "$Handoff\verify_demo.py" --source-root $S --exp23-root $E --mode bundle-load --output-dir "$Handoff\verification\bundle_load_new"
& $P "$Handoff\verify_demo.py" --source-root $S --exp23-root $E --mode profile --duration 20 --device cuda --output-dir "$Handoff\verification\profile_replay_new"
& $P "$Handoff\verify_demo.py" --source-root $S --exp23-root $E --mode bundle --duration 12 --device cpu --output-dir "$Handoff\verification\bundle_replay_new"
```

已实际执行的每个 router/client 完整参数和工作目录保存在 `verification/*_replay/commands.json`，包括 key/config/ckpt/profile/wav 路径。脚本强制 Hugging Face 离线，便于区分本地权重是否齐全。

### 真实验证状态

| 项目 | 2026-09-20 结果 |
|---|---|
| Exp23 导出与 demo 接口单元测试 | 10 项通过 |
| Bundle manifest 与 CPU 单样本推理 | 通过 |
| 旧三用户 CUDA codec + CPU ECAPA 回放 | 启动、收发、解码、识别成功；定时退出失败 |
| 原始 Exp23 客户端直接加载 bundle | 失败：`ModuleNotFoundError: No module named 'speechtokenizer.speaker_identity'` |
| 六用户、真实麦克风、跨机长时间运行 | 本次未做 |

旧路径回放三个客户端各收到并解码 74 包，每端有 34 次识别。退出码均为 `3221226505`；其中一个客户端在关闭阶段仍进入 `mic_send_loop` → `wav_send_loop` → `send_codes`。代码在退出时关闭 socket，却未等待后台线程结束，日志暴露出关闭竞态；仅凭现有日志还不能把所有进程的原生异常退出归因于这一处。

Exp23 直接启动的导入错误与旧内嵌 `两用户通信demo/speechtokenizer` 包遮蔽仓库根包一致：客户端先把旧包目录放到 `sys.path` 前面并导入 `speechtokenizer`，后来再导入根包新增的 `speaker_identity` 子模块时失败。单元测试中的 mock 接口通过不等于此独立进程启动链路通过。

这些问题已记录但本次未修改 demo 源码。应先修复包导入优先级和退出线程清理，再扩展六路并发，不应把该交接单理解为现成六用户系统已经验收。

## 12. SHA-256 与可追溯性

`SHA256SUMS.csv` 逐行记录文件绝对路径、字节数、SHA-256、材料类别、run ID、可确认的 Git commit 和来源限制，覆盖所列 codec、配置、三个 seed、bundle、ECAPA、完整现有 Exp23 缓存及相关元数据/源码。

没有给整套 VCTK 音频逐文件生成新哈希；其本次核查范围为目录/文件统计、划分输入路径、mic1 容器签名及转写解析。不能将 CSV 描述为整套仓库或整个数据集的完整校验清单。

| 来源 | 已知 Run / Commit | 限制 |
|---|---|---|
| LCA | `exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42` | manifest 有 checkpoint hash，没有记录训练当时 Git commit |
| Base | `exp2_scit_speech_distill30_retrain_20260529_seed42` | 同上 |
| Exp23 seed41/42/43 | `exp23_formal/seed41` 等；历史 revision `3b87deb2c375a88051a79506587ef1a2d19d7694` | 历史 `dirty=true`，单凭 commit 不能完整还原训练工作树 |
| Exp23 当前源码 | `8ee038bdb7e3588b70e88e785f5a05edb78a679c` | 这是当前已提交源码，不冒充历史干净训练快照 |
| ECAPA | `speechbrain/spkrec-ecapa-voxceleb`；本地匹配 revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | 通过当前文件内容匹配 snapshot |

`tokenizer_state_hashes.json` 的 before/after hash 是张量 state-dict 哈希，不是 `.pt` 文件 SHA-256。不要把两种哈希混用。

## 额外的重现实验风险

当前 `E\scripts\build_exp23_speaker_cache.py` 的 text-controlled 分支仍按旧函数签名调用 `build_text_controlled_split`：第二个位置参数传入转写路径，又传 `train_count=`，还传入现行函数不接受的 `excluded_speakers=`。因此照现有 CLI 重建 109 类缓存前，应先修复这个调用点并补对应 CLI 路径测试。

本目录的 `audit_materials.py` 使用现行函数正确的 `transcript_root=` 关键字生成 109 类 manifest，不调用有问题的缓存 CLI，也不改写历史 canonical 文件。

## 建议交接顺序

1. 先交本清单、Exp23 分支源码、LCA 推理文件、NAS 配置、seed42 bundle、环境快照和启动失败日志，让学弟修通单客户端及三路集成。
2. 再交 Base、三个种子全套产物、缓存和两个 split manifest，支持对照与复现。
3. 全量 VCTK 音频和 txt 作为独立数据包移交。普通六人通信链路不需要先重训完整 110 类分类器；复现实验才需要这一组完整材料。
4. 六路验收应保存：每端接收其他 5 路的证据、独立流缓冲与身份输出、端到端延迟/丢包、资源占用、退出无残留、麦克风/播放与跨机验证。不能用 router 的 `sender_id` 显示替代识别结果。

本次没有上传外部服务器、没有提交 Git、没有复制十余 GiB 的全量数据，也没有重跑或覆盖已锁定的 formal test。
