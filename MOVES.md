# MOVES.md — 本次整理的完整路径映射（审计 / 回滚用）

> 分支：`chore/repo-reorg-research`。整理策略：**移动而非删除，归档而非清除，全程可回滚。**
> 本次**未**物理拆分 `scripts/`（24 个脚本硬编码 `Path(__file__).resolve().parents[1]`
> 作为 PROJECT_ROOT，下移会同时破坏 sys.path 注入与 `output/experiments/...` 路径解析）。
> 本次**未**提交 `output/` 证据（按用户决定保持未跟踪）。

## A. 已跟踪文件 — `git mv`（保留 git 历史）

| 旧路径 | 新路径 |
|---|---|
| `demo_nature/多人嘈杂环境/example.py` | `legacy_explorations/demo_nature/example.py` |
| `实时语音系统/demo_now.md` | `legacy_explorations/实时语音系统/demo_now.md` |
| `实时语音系统/demo_now.py` | `legacy_explorations/实时语音系统/demo_now.py` |
| `项目情况说明.md` | `legacy_explorations/root_docs/项目情况说明.md` |
| `训练和推理脚本.md` | `legacy_explorations/root_docs/训练和推理脚本.md` |
| `example.py` | `legacy_explorations/root_docs/example.py` |

## B. 未跟踪一次性脚本 — `mv`（无 git 历史可保）

| 旧路径 | 新路径 |
|---|---|
| `scripts/_exp5c_probe_tb_tags.py` | `legacy_explorations/oneshot_scripts/_exp5c_probe_tb_tags.py` |
| `scripts/migrate_paths_e_to_h.py` | `legacy_explorations/oneshot_scripts/migrate_paths_e_to_h.py` |
| `tmp_analyze_v3.py` | `legacy_explorations/oneshot_scripts/tmp_analyze_v3.py` |
| `tmp_full_manifest.py` | `legacy_explorations/oneshot_scripts/tmp_full_manifest.py` |

## C. 新增文件（非移动）

| 文件 | 用途 |
|---|---|
| `tests/__init__.py` | 空文件；恢复 `python -m unittest tests.X` 文档化调用（实验记录 §2 报「17 tests OK」）。补后 23 tests OK |
| `REPRODUCIBILITY.md` | 论文每张表/图 → 复现脚本+数据+权重 sha256+seed+产物路径 |
| `PROJECT_STRUCTURE.md` | 重组后目录树 + 职责 + 论文章节/实验编号对应 |
| `legacy_explorations/README.md` | 归档清单与 provenance |
| `MOVES.md` | 本文件 |
| `REORG_MANIFEST.md` | Phase 2 可审计分类清单（A–G + 硬约束 + 待确认项）|
| `.gitignore`（追加 `tools/`）| 排除大体积 ffmpeg 8.1.1 构建（非源码，按 REPRODUCIBILITY 重下）|

## D. 纳入版本控制的既有未跟踪「代码」（非移动，仅 `git add`）

- `scripts/`：43 个研究脚本（方法/实验/评估/作图；4 个一次性已归档，详见 A/B）
- `nas/`：7 个新增 NAS 脚本（`encoder_handoff.py` 等）
- `speechtokenizer/trainer/lca_trainer.py`：LCA 训练器 + ChannelSim（§3.3）
- `tests/`：6 个测试（含 1 个 broken orphan，见下）
- `3用户demo/`：34 个 demo 源文件（`.py/.md/.yml/.csv`；权重/音频被 .gitignore 排除）

## E. 回滚方式

- 整体回滚：`git checkout main`（本分支未并入 main）。
- 单项回滚 git mv：`git mv <新> <旧>`。
- 单项回滚 mv：手动 `mv` 回原路径（这些文件本就未跟踪）。
- `tests/__init__.py` 如不需要：删除即可（仅影响 `tests.X` 调用形式，不影响 discovery）。

## F. 本次**未触碰**（用户 WIP / 证据 / 约束）

- 用户 WIP：`speechtokenizer/trainer/trainer.py`(+139)、`nas/model_components.py`、
  `config/spt_base_cfg.json` 等已修改的已跟踪方法文件，及两个 notebook 删除 —— **保持未暂存**。
- `output/**` 证据链：保持未跟踪（用户决定）。
- `scripts/` 未做子目录拆分（保护 `parents[1]` 路径契约）。
- `tests/test_public_nq8_layer_sweep.py`：broken orphan（import 不存在的 `experiments_semcom`），
  G 类不确定，**保留不删**，待人工确认（见 REORG_MANIFEST.md G 类）。
