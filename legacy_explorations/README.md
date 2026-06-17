# legacy_explorations/ — 归档清单与 provenance

本目录归档「探索性 / 已被取代 / 一次性」文件。**归档≠删除**：全部原样保留，
按「原路径镜像 + 所属实验/假设」存放。科研溯源原则下，这些文件仍有价值
（验证过哪个假设 / 服务过哪个实验），故记录其来历而非简单标「废弃」。

> 整理分支：`chore/repo-reorg-research`。完整旧→新路径映射见仓库根 `MOVES.md`。

## 1. 旧 demo（被 `3用户demo/` 取代）

| 归档路径 | 原路径 | 所属/假设 | 归档理由 |
|---|---|---|---|
| `demo_nature/example.py` | `demo_nature/多人嘈杂环境/example.py` | 早期「多人嘈杂环境」demo 探索 | 当前论文 demo 章节（demo skeleton §X）已统一用 `3用户demo/` 三用户中心路由原型；旧 demo 无任何 paper/实验记录/README 引用 |
| `实时语音系统/demo_now.md` | `实时语音系统/demo_now.md` | 早期实时语音系统说明 | 同上，被 `3用户demo/` 取代 |
| `实时语音系统/demo_now.py` | `实时语音系统/demo_now.py` | 早期实时语音 demo 脚本 | 同上 |

## 2. 旧根目录文档 / 示例（被 `output/doc/` 与 demo 取代）

| 归档路径 | 原路径 | 所属/假设 | 归档理由 |
|---|---|---|---|
| `root_docs/项目情况说明.md` | `项目情况说明.md` | 早期项目总览 | 被 `output/doc/实验记录.md`（§1–§19 规范溯源日志）+ 论文草稿取代 |
| `root_docs/训练和推理脚本.md` | `训练和推理脚本.md` | 早期训练/推理脚本说明 | 被 `output/doc/实验手册.md` + 新增 `REPRODUCIBILITY.md` 取代 |
| `root_docs/example.py` | `example.py` | 早期单文件推理示例 | 仅被上述两个旧文档引用（三者一起归档，内部引用保持一致）；当前推理入口见 `speechtokenizer/` API 与 demo |

## 3. 一次性 / 探索性脚本（未跟踪，无 git 历史）

| 归档路径 | 原路径 | 所属/假设 | 归档理由 |
|---|---|---|---|
| `oneshot_scripts/_exp5c_probe_tb_tags.py` | `scripts/_exp5c_probe_tb_tags.py` | exp5c（§16 consistency 动力学诊断）| 只读 TensorBoard tag 探针，`_` 前缀一次性脚本；exp5c 正式脚本见 `output/experiments/exp5c.../commands/extract_dynamics_curves.py` |
| `oneshot_scripts/migrate_paths_e_to_h.py` | `scripts/migrate_paths_e_to_h.py` | 2026-05-29 E盘→H盘整体迁移（实验记录开头）| 一次性路径迁移，已完成且不会重复 |
| `oneshot_scripts/tmp_analyze_v3.py` | `tmp_analyze_v3.py`（根）| 临时 TensorBoard 分析 | `tmp_` 前缀临时脚本 |
| `oneshot_scripts/tmp_full_manifest.py` | `tmp_full_manifest.py`（根）| exp7 全量 manifest 临时生成 | `tmp_` 前缀；exp7 正式 manifest 已在 `output/experiments/exp7.../artifacts/` |

## 4. 已删除 notebook（git 历史可恢复，未物理归档）

工作区中两个 notebook 在本次整理前已被删除（属用户既有 WIP）：
- `对比wav语音差距.ipynb`、`读取wav文件信息.ipynb`

二者完整保存在 git 历史 commit `b3bd391`，可用
`git show b3bd391:"对比wav语音差距.ipynb"` 恢复。未在本目录物理镜像，仅此登记。

## 5. 保留在 scripts/ 的「可能可复用」一次性工具（未归档）

以下脚本虽偏工具性，但**可能复用**，按保守原则**留在 `scripts/`**，不归档：
`convert_single_flac_to_wav.py`（单文件转换）、`normalize_filelist_paths.py`（filelist 规整）。
如确认不再需要，可后续追加归档。
