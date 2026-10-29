"""Coverage-aware summaries with paired cluster bootstrap confidence intervals."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

METRICS = ['mel_l1','stoi','pesq_wb','si_snr_db']


def records(path):
    if not path.exists():
        return []
    result=[]
    lines=path.read_text().splitlines()
    for i,s in enumerate(lines):
        try:
            result.append(json.loads(s))
        except ValueError:
            if i != len(lines)-1:
                raise
    keys=[(r['dataset'],r['sample_id']) for r in result]
    assert len(keys)==len(set(keys)), 'Duplicate records'
    return result


def bootstrap_ratio(items, draws=2000):
    # Each entry is (cluster, numerator, denominator), retaining utterance weighting.
    groups=defaultdict(lambda:np.zeros(2,dtype=float))
    for cluster,num,den in items:
        groups[cluster] += (num,den)
    if not groups:
        return dict(estimate=None,ci_low=None,ci_high=None,n_clusters=0)
    arr=np.stack([groups[k] for k in sorted(groups)])
    estimate=float(arr[:,0].sum()/arr[:,1].sum())
    if len(arr)<2:
        return dict(estimate=estimate,ci_low=None,ci_high=None,n_clusters=len(arr))
    rng=np.random.default_rng(42)
    weights=rng.multinomial(len(arr),np.ones(len(arr))/len(arr),size=draws)
    sums=weights@arr
    distribution=sums[:,0]/sums[:,1]
    lo,hi=np.quantile(distribution,[0.025,0.975])
    return dict(estimate=estimate,ci_low=float(lo),ci_high=float(hi),n_clusters=len(arr))


def csv_out(path, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    root=a.run
    is_level_control=(root/'CONTROL_PROTOCOL.json').exists()
    out=root/('smoke' if a.smoke else 'full')
    report=out/'reports';report.mkdir(parents=True,exist_ok=True)
    manifest=records(root/'manifests/all_samples.jsonl')
    objective=records(out/'objective.jsonl')
    asr=records(out/'asr.jsonl')
    expected=Counter(r['dataset'] for r in manifest)
    allowed={(r['dataset'],r['sample_id']) for r in manifest}
    assert all((r['dataset'],r['sample_id']) in allowed for r in objective+asr)
    objective_summary=[]; objective_delta=[]; asr_summary=[]; asr_delta=[]
    coverage=[];failure_rows=[]
    for ds in sorted(expected):
        obj=[r for r in objective if r['dataset']==ds]
        ars=[r for r in asr if r['dataset']==ds]
        eligible=sum(bool(r.get('transcript')) for r in manifest if r['dataset']==ds)
        coverage.append(dict(dataset=ds,expected=expected[ds],objective_records=len(obj),
            asr_records=len(ars),expected_reference_transcripts=eligible,
            asr_scored=sum(bool(r['asr']) for r in ars),
            asr_excluded=sum(bool(r.get('exclusion_reason')) for r in ars)))
        for r in obj:
            assert len(r['metrics'])==6
            assert {(m['model'],m['L']) for m in r['metrics']}=={(n,l) for n in ['base','lca'] for l in [1,2,3]}
            for m in r['metrics']:
                for metric,error in m['errors'].items():
                    failure_rows.append(dict(dataset=ds,sample_id=r['sample_id'],model=m['model'],L=m['L'],metric=metric,error=error))
        for layer in [1,2,3]:
            for metric in METRICS:
                for model in ['base','lca']:
                    items=[]
                    for r in obj:
                        m=next(m for m in r['metrics'] if m['model']==model and m['L']==layer)
                        if m[metric] is not None:
                            items.append((r['cluster_id'],m[metric],1))
                    objective_summary.append(dict(dataset=ds,model=model,L=layer,metric=metric,
                        n=len(items),failed=len(obj)-len(items),**bootstrap_ratio(items)))
                pairs=[]
                for r in obj:
                    vals={m['model']:m[metric] for m in r['metrics'] if m['L']==layer}
                    if all(v is not None for v in vals.values()):
                        pairs.append((r['cluster_id'],vals['lca']-vals['base'],1))
                objective_delta.append(dict(dataset=ds,L=layer,metric=metric,contrast='lca-base',
                    paired_n=len(pairs),**bootstrap_ratio(pairs)))
        for model in ['original','base','lca']:
            valid=[r for r in ars if model in r['asr']]
            items=[(r['cluster_id'],r['asr'][model]['errors'],r['asr'][model]['reference_words']) for r in valid]
            totals={field:sum(r['asr'][model][field] for r in valid)
                    for field in ['reference_words','errors','substitutions','deletions','insertions']}
            asr_summary.append(dict(dataset=ds,model=model,L=0 if model=='original' else 3,
                n=len(valid),**totals,**bootstrap_ratio(items)))
        for lhs,rhs in [('base','original'),('lca','original'),('lca','base')]:
            items=[]
            for r in ars:
                if lhs in r['asr'] and rhs in r['asr']:
                    left=r['asr'][lhs];right=r['asr'][rhs]
                    assert left['reference_words']==right['reference_words']
                    items.append((r['cluster_id'],left['errors']-right['errors'],left['reference_words']))
            asr_delta.append(dict(dataset=ds,contrast=lhs+'-'+rhs,paired_n=len(items),**bootstrap_ratio(items)))
    all_done=(not a.smoke and all(r['objective_records']==r['expected'] and r['asr_records']==r['expected']
        and r['asr_scored']==r['expected_reference_transcripts'] for r in coverage))
    result=dict(status='completed' if all_done else 'partial_or_smoke',
        condition='post_hoc_fleurs_fixed_level' if is_level_control else 'primary_original_level',
        bootstrap=dict(draws=2000,seed=42,level=0.95,unit='dataset-specific cluster',
            weighting='utterance for objective means; reference words for WER'),
        coverage=coverage,objective=objective_summary,objective_paired_deltas=objective_delta,
        asr=asr_summary,asr_paired_deltas=asr_delta,metric_failures=failure_rows)
    (report/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    for name,rows in [('coverage',coverage),('objective_summary',objective_summary),('objective_paired_deltas',objective_delta),
                      ('asr_summary',asr_summary),('asr_paired_deltas',asr_delta),('metric_failures',failure_rows)]:
        csv_out(report/(name+'.csv'),rows)
    lines=['# '+('FLEURS 固定电平事后诊断' if is_level_control else '英文跨语料泛化评测'),'',
        '**状态：'+('已完成全部预定样本。' if all_done else '部分结果／试跑；不能作为完整实验结论。')+'**','',
        '冻结 Base/LCA，在外部英文语料上测试；未在测试集训练、适配或选择检查点。',
        ('本报告仅包含 FLEURS 全部 647 条的固定 −25 dBFS 电平对照。它是在观察原始电平退化后设计的诊断，不替换主实验。'
         if is_level_control else '三个外部语料：VCTK、TED-LIUM release3 test 镜像、FLEURS en_us test。LibriSpeech 为参照。'),'',
        '| 数据集 | 计划样本 | 音质已测 | ASR 已处理 | 有参考文本 | ASR 已评分 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in coverage:
        lines.append(f"| {r['dataset']} | {r['expected']} | {r['objective_records']} | {r['asr_records']} | {r['expected_reference_transcripts']} | {r['asr_scored']} |")
    lines += ['','## L3 / 1.5 kbit/s 客观音质','',
        '| 数据集 | 模型 | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR dB ↑ |',
        '|---|---|---:|---:|---:|---:|']
    for ds in sorted(expected):
        for model in ['base','lca']:
            vals={r['metric']:r['estimate'] for r in objective_summary if r['dataset']==ds and r['model']==model and r['L']==3}
            lines.append('| '+ds+' | '+model+' | '+' | '.join(fmt(vals[k]) for k in METRICS)+' |')
    lines += ['','## L3 英文 WER','',
        'WER 按参考词数加权，括号中为 95% 聚类 bootstrap 区间。原始音频 WER 用于衡量语料自身难度与识别器误差。','',
        '| 数据集 | 原始音频 WER % | Base WER % | LCA WER % | LCA − 原始 / 百分点 |',
        '|---|---:|---:|---:|---:|']
    def pct(r):
        if r['estimate'] is None:return 'NA'
        return f"{100*r['estimate']:.2f} [{100*r['ci_low']:.2f}, {100*r['ci_high']:.2f}]" if r['ci_low'] is not None else f"{100*r['estimate']:.2f}"
    for ds in sorted(expected):
        rs={r['model']:r for r in asr_summary if r['dataset']==ds}
        delta=next(r for r in asr_delta if r['dataset']==ds and r['contrast']=='lca-original')
        lines.append('| '+ds+' | '+' | '.join(pct(rs[m]) for m in ['original','base','lca'])+' | '+pct(delta)+' |')
    lines += ['','## 解释边界','',
        '- WER 差异需要结合原始音频 WER 看；不同语料直接相减不能解释为单一因素造成的退化。',
        '- 置信区间单位：VCTK/LibriSpeech 为说话人，TED-LIUM 为演讲，FLEURS 为文本 prompt。FLEURS 没有说话人 ID，不能声称控制了说话人相关性。',
        '- 只使用一对现存训练种子的模型；bootstrap 反映测试样本的不确定性，不代表多次训练的不确定性。',
        '- 固定 Whisper small.en 为 ASR 探针；其预训练数据可能含基准相关语料，本实验没有证明识别器无预训练污染。',
        '- L1/L2 全部结果和配对差异见 CSV，不因结果优劣删去。',
        '- 多语料、多指标区间为描述性分析，未进行多重比较校正；不据单个区间宣称所有条件下普遍优越。',
        f'- 客观指标失败条目共 {len(failure_rows)} 个；失败明细和实际分母见 summary.json。',
        '- VCTK 无参考转写的录音只参与音质指标。TED-LIUM 的 ignore_time_segment_in_scoring 依据元数据排除，清单在 manifests/exclusions.json。','']
    (report/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],coverage=coverage,metric_failures=len(failure_rows)),indent=2))


if __name__=='__main__':
    main()
