"""Paired prompt-cluster analysis of raw vs fixed-level FLEURS; post-hoc only."""
import argparse
import json
from pathlib import Path
from aggregate import bootstrap_ratio, records, csv_out


def keyed(path):
    return {r['sample_id']:r for r in records(path) if r['dataset']=='fleurs_en_us'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);a=ap.parse_args()
    root=a.parent;control=root/'level_control';out=root/'analysis';out.mkdir(exist_ok=True)
    raw_obj=keyed(root/'full/objective.jsonl');new_obj=keyed(control/'full/objective.jsonl')
    raw_asr=keyed(root/'full/asr.jsonl');new_asr=keyed(control/'full/asr.jsonl')
    assert len(raw_obj)==647 and set(raw_obj)==set(new_obj)==set(raw_asr)==set(new_asr)
    deltas=[]
    for model in ['base','lca']:
        for metric in ['mel_l1','stoi','pesq_wb','si_snr_db']:
            items=[]
            for sid,row in raw_obj.items():
                old=next(m for m in row['metrics'] if m['model']==model and m['L']==3)[metric]
                new=next(m for m in new_obj[sid]['metrics'] if m['model']==model and m['L']==3)[metric]
                if old is not None and new is not None:items.append((row['cluster_id'],new-old,1))
            deltas.append(dict(model=model,metric=metric,contrast='fixed_level-raw',paired_n=len(items),**bootstrap_ratio(items)))
    for model in ['original','base','lca']:
        items=[]
        for sid,row in raw_asr.items():
            old=row['asr'][model];new=new_asr[sid]['asr'][model]
            assert old['reference_words']==new['reference_words']
            items.append((row['cluster_id'],new['errors']-old['errors'],old['reference_words']))
        deltas.append(dict(model=model,metric='wer',contrast='fixed_level-raw',paired_n=len(items),**bootstrap_ratio(items)))
    for model in ['base','lca']:
        items=[]
        for sid,row in raw_asr.items():
            old=row['asr'];new=new_asr[sid]['asr']
            delta=(new[model]['errors']-new['original']['errors'])-(old[model]['errors']-old['original']['errors'])
            items.append((row['cluster_id'],delta,old[model]['reference_words']))
        deltas.append(dict(model=model,metric='codec_excess_wer',contrast='difference_in_differences',paired_n=len(items),**bootstrap_ratio(items)))
    (out/'fleurs_level_control_paired.json').write_text(json.dumps(deltas,indent=2))
    csv_out(out/'fleurs_level_control_paired.csv',deltas)
    summaries={name:json.loads((p/'full/reports/summary.json').read_text(encoding='utf-8'))
               for name,p in [('raw',root),('fixed_level',control)]}
    rows=[]
    for cond,summary in summaries.items():
        row=dict(condition=cond)
        for model in ['original','base','lca']:
            r=next(x for x in summary['asr'] if x['dataset']=='fleurs_en_us' and x['model']==model)
            row[model+'_wer_pct']=100*r['estimate']
        row['lca_excess_wer_pp']=row['lca_wer_pct']-row['original_wer_pct']
        for model in ['base','lca']:
            for metric in ['mel_l1','stoi','pesq_wb']:
                row[model+'_'+metric]=next(x['estimate'] for x in summary['objective']
                    if x['dataset']=='fleurs_en_us' and x['model']==model and x['L']==3 and x['metric']==metric)
        rows.append(row)
    csv_out(out/'fleurs_level_control_summary.csv',rows)
    lines=['# FLEURS 输入电平敏感性：事后诊断','',
        '本组是在主实验观察到 FLEURS 退化、并发现其原始中位 RMS 约 −60 dBFS 后设计的。全部 647 条均保留，仅改变每条录音的整体增益；目标 RMS 固定为 −25 dBFS，峰值上限 0.99，不做裁剪、去噪、时序对齐或模型训练。',
        '目标值参考 LibriSpeech 参照语料约 −25 dBFS 的电平，在对照推理前固定。该诊断不能替代原始电平主结果，也不能称作预先注册的独立泛化验证。','',
        '| 条件 | 原始音频 WER % | Base L3 WER % | LCA L3 WER % | LCA 额外 WER / 百分点 | LCA STOI |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['original_wer_pct']:.2f} | {r['base_wer_pct']:.2f} | {r['lca_wer_pct']:.2f} | {r['lca_excess_wer_pp']:+.2f} | {r['lca_stoi']:.4f} |")
    lines += ['','| 配对变化：固定电平 − 原始电平 | 估计 | 95% prompt 聚类区间 |','|---|---:|---:|']
    for r in deltas:
        scale=100 if 'wer' in r['metric'] else 1
        lines.append(f"| {r['model']} / {r['metric']} | {r['estimate']*scale:+.4f} | [{r['ci_low']*scale:+.4f}, {r['ci_high']*scale:+.4f}] |")
    lines += ['','WER 变化使用相同参考词数；codec_excess_wer 是“重建 WER − 原始 WER”的双重差分，以同时观察识别器自身对增益的敏感性。负数表示固定电平后编解码额外错误减少。',
        '这组实验可以诊断固定样本的输入增益敏感性；它没有证明模型在任意录音电平、噪声、语言或说话人条件下均能泛化。']
    (out/'FLEURS_LEVEL_CONTROL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
