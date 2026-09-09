# 22500 best-dev 与 42500 checkpoint 长音频对比

- 对比对象 A：`checkpoints/SpeechTokenizer_best_dev.pt`，对应当前 best-dev，step 22500。
- 对比对象 B：`checkpoints/SpeechTokenizerTrainer_00042500`，当前最近周期 checkpoint。
- 样本集：同一批 4 条 `full_utterance` 长音频。
- 评估方式：逐条比较 original 与 recon_L1/L2/L3，计算侵入式客观指标。
- 试听 A：`samples/full_utterance/listen_full.html`
- 试听 B：`checkpoint_exports/SpeechTokenizerTrainer_00042500/samples/full_utterance/listen_full.html`

## 汇总对比

括号中为 B 相对 A 的变化量。PESQ-WB、STOI、SI-SNR、corr 越高越好；Wave L1、RMSE、Mel L1 越低越好。

| L | checkpoint | PESQ-WB ↑ | STOI ↑ | SI-SNR dB ↑ | Mel L1 ↓ | Wave L1 ↓ | corr ↑ | RMS 比例 dB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 22500 best-dev | 1.203 | 0.7338 | -12.190 | 1.424405 | 0.033677 | 0.3619 | -3.866 |
| L1 | 42500 | 1.249 (+0.046) | 0.7585 (+0.0247) | -7.851 (+4.339) | 1.345840 (-0.078565) | 0.032427 (-0.001250) | 0.4320 (+0.0700) | -3.533 |
| L2 | 22500 best-dev | 1.487 | 0.8085 | -5.400 | 1.202103 | 0.027108 | 0.5499 | -1.972 |
| L2 | 42500 | 1.573 (+0.086) | 0.8280 (+0.0195) | -3.369 (+2.031) | 1.106687 (-0.095417) | 0.025780 (-0.001327) | 0.5975 (+0.0476) | -1.676 |
| L3 | 22500 best-dev | 1.618 | 0.8292 | -3.961 | 1.145732 | 0.025141 | 0.5931 | -1.522 |
| L3 | 42500 | 1.713 (+0.096) | 0.8540 (+0.0248) | -1.624 (+2.336) | 1.042575 (-0.103157) | 0.023614 (-0.001527) | 0.6459 (+0.0528) | -1.091 |

两者裁剪比例均为 0。

## 结论

42500 checkpoint 在这 4 条长音频上明显优于 22500 best-dev，且 L1/L2/L3 三个层数的客观指标方向一致改善。关键 L3 上，PESQ-WB 提升约 0.096，STOI 提升约 0.0248，SI-SNR 提升约 2.34 dB，Mel L1 降低约 0.103。

这和主观听感“后面一个更好”一致。需要注意：22500 是按 validation `dev/mel error` 选出来的 best-dev；42500 虽然不是当前 dev/mel 最低点，但在这批长音频听感样本上更好。这说明当前 `dev/mel error` 与长音频主观/客观听感之间并不完全一致，后续选择最终 checkpoint 时不能只看 dev/mel，建议同时保留长音频 PESQ/STOI/SI-SNR 与人工听感记录。

## 数据文件

- 22500 best-dev 指标：`metrics/full_utterance_audio_quality_eval_bestdev_summary.json`
- 42500 指标：`checkpoint_exports/SpeechTokenizerTrainer_00042500/metrics/full_utterance_audio_quality_eval_summary.json`
