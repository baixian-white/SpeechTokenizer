🚀 NAS-Optimized SpeechTokenizer 训练指南

本说明已按最新目录结构更新：NAS 相关脚本与产物统一放在 `nas/` 下。

## 1. 目录结构

```text
ProjectRoot/
├── nas/
│   ├── train_nas.py
│   ├── run_nas.sh
│   ├── search_autoencoder.py
│   ├── export_best_model.py
│   ├── custom_model.py
│   ├── SeaNet.py
│   ├── model_components.py
│   ├── best_seanet_config.json
│   └── artifacts/
│       ├── seanet_nas_micro_v1.db
│       ├── nas_records.csv
│       └── train_subset_nas.txt
├── config/spt_base_cfg.json
└── speechtokenizer/
```

## 2. 使用流程

1. 生成 NAS 子集（可选）
```bash
python -m nas.make_subset
```

2. 执行 NAS 搜索
```bash
python -m nas.search_autoencoder
```

3. 导出最佳结构
```bash
python -m nas.export_best_model
```

4. 开始重训练
```bash
bash nas/run_nas.sh
```

## 3. 监控

- 看显卡：
```bash
watch -n 1 nvidia-smi
```

- 看日志：
```bash
tensorboard --logdir=Log/spt_base --port 6007
```

## 4. 常见问题

- `FileNotFoundError (best_seanet_config.json)`：先执行 `python -m nas.export_best_model`。
- `ImportError`：请从项目根目录执行上面的命令（`ProjectRoot`）。
