"""Publication-ready figures from aggregate outputs; no estimated/fabricated data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER=['librispeech_test_clean','librispeech_test_other','vctk','tedlium','fleurs_en_us']
LABELS=['LS clean\n(reference)','LS other\n(reference)','VCTK','TED-LIUM','FLEURS en_us']
COLORS={'original':'#718096','base':'#3575A8','lca':'#D66B45'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--control-summary',type=Path)
    a=ap.parse_args();data=json.loads(a.summary.read_text());a.out.mkdir(parents=True,exist_ok=True)
    assert data['status']=='completed','Do not draw incomplete results as the final experiment'
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
        'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'savefig.bbox':'tight'})
    def intervals(rs,scale=1):
        y=np.array([r['estimate'] for r in rs])*scale
        lo=np.array([r['ci_low'] for r in rs])*scale
        hi=np.array([r['ci_high'] for r in rs])*scale
        assert np.all(lo<=y+1e-10) and np.all(hi>=y-1e-10), 'Draw percentile bounds directly if estimate lies outside them'
        return y,np.stack([np.maximum(0,y-lo),np.maximum(0,hi-y)])
    def style(ax):
        ax.set_xticks(np.arange(5),LABELS)
        ax.axvline(1.5,color='#A0AEC0',ls=':',lw=1)
        ax.grid(axis='y',color='#E5E7EB',lw=.6);ax.set_axisbelow(True)
    def export(fig,name):
        fig.savefig(a.out/(name+'.pdf'));fig.savefig(a.out/(name+'.png'),dpi=300);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.6,6.1),layout='constrained')
    for ax,metric,label in zip(axes.flat,['mel_l1','stoi','pesq_wb','si_snr_db'],
            ['Mel-L1 (lower is better)','STOI (higher is better)','PESQ-WB (higher is better)','SI-SNR, dB (higher is better)']):
        for j,model in enumerate(['base','lca']):
            rs=[next(r for r in data['objective'] if r['dataset']==ds and r['model']==model and r['L']==3 and r['metric']==metric) for ds in ORDER]
            y,err=intervals(rs)
            ax.errorbar(np.arange(5)+(j-.5)*.16,y,yerr=err,fmt='o',color=COLORS[model],
                ms=4,capsize=3,lw=1,label=model.upper())
        style(ax);ax.set_ylabel(label)
    axes[0,0].legend(ncol=2)
    fig.suptitle('Frozen codec generalization at 1.5 kbit/s',fontsize=12)
    export(fig,'english_generalization_objective_L3')
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),layout='constrained')
    for j,model in enumerate(['original','base','lca']):
        rs=[next(r for r in data['asr'] if r['dataset']==ds and r['model']==model) for ds in ORDER]
        y,err=intervals(rs,100)
        axes[0].bar(np.arange(5)+(j-1)*.24,y,width=.22,color=COLORS[model],
            label='Original' if model=='original' else model.upper(),yerr=err,
            error_kw={'capsize':2,'elinewidth':.8})
    for j,model in enumerate(['base','lca']):
        rs=[next(r for r in data['asr_paired_deltas'] if r['dataset']==ds and r['contrast']==model+'-original') for ds in ORDER]
        y,err=intervals(rs,100)
        axes[1].errorbar(np.arange(5)+(j-.5)*.16,y,yerr=err,fmt='o',ms=4,capsize=3,lw=1,
            color=COLORS[model],label=model.upper())
    axes[0].set_ylabel('Corpus word-weighted WER (%)');axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel('WER increase over original (percentage points)');axes[1].axhline(0,color='#718096',lw=.7)
    for ax in axes:style(ax);ax.legend(ncol=3)
    fig.suptitle('English ASR probe: Whisper small.en, fixed decoding',fontsize=12)
    export(fig,'english_generalization_wer_L3')
    fig,axes=plt.subplots(1,3,figsize=(11.0,3.5),layout='constrained')
    for ax,metric,label in zip(axes,['mel_l1','stoi','pesq_wb'],['Mel-L1 (lower)','STOI (higher)','PESQ-WB (higher)']):
        for ds,color in zip(ORDER,['#8B93A7','#B6A58F','#3575A8','#D66B45','#29998C']):
            rs=[next(r for r in data['objective'] if r['dataset']==ds and r['model']=='lca' and r['L']==layer and r['metric']==metric) for layer in [1,2,3]]
            ax.plot([.5,1,1.5],[r['estimate'] for r in rs],marker='o',ms=3,lw=1.3,color=color,label=LABELS[ORDER.index(ds)].replace('\n',' '))
        ax.set_xticks([.5,1,1.5]);ax.set_xlabel('Nominal payload bitrate (kbit/s)');ax.set_ylabel(label)
        ax.grid(axis='y',color='#E5E7EB',lw=.6)
    axes[0].legend(fontsize=7)
    fig.suptitle('LCA operating-point sensitivity across English corpora',fontsize=12)
    export(fig,'english_generalization_lca_bitrate')
    if a.control_summary:
        control=json.loads(a.control_summary.read_text(encoding='utf-8'))
        assert control['status']=='completed'
        sources=[data,control]
        fig,axes=plt.subplots(1,2,figsize=(9.0,3.7),layout='constrained')
        for j,model in enumerate(['original','base','lca']):
            rs=[next(r for r in d['asr'] if r['dataset']=='fleurs_en_us' and r['model']==model) for d in sources]
            y,err=intervals(rs,100)
            bars=axes[0].bar(np.arange(2)+(j-1)*.23,y,width=.21,color=COLORS[model],
                yerr=err,label='Original' if model=='original' else model.upper(),
                error_kw={'capsize':2,'elinewidth':.8})
            for bar,value in zip(bars,y):
                axes[0].text(bar.get_x()+bar.get_width()/2,value+4,f'{value:.2f}',ha='center',fontsize=8)
        for j,model in enumerate(['base','lca']):
            rs=[next(r for r in d['objective'] if r['dataset']=='fleurs_en_us' and r['model']==model and r['L']==3 and r['metric']=='stoi') for d in sources]
            y,err=intervals(rs)
            axes[1].errorbar(np.arange(2)+(j-.5)*.12,y,yerr=err,fmt='o',color=COLORS[model],ms=5,lw=1,capsize=3,label=model.upper())
        for ax in axes:
            ax.set_xticks([0,1],['Original level','Fixed level\n(target RMS: -25 dBFS)'])
            ax.grid(axis='y',color='#E5E7EB',lw=.6);ax.set_axisbelow(True);ax.legend(ncol=3)
        axes[0].set_ylabel('Corpus word-weighted WER (%)');axes[0].set_ylim(0,76)
        axes[1].set_ylabel('STOI (higher is better)');axes[1].set_ylim(.60,.87)
        fig.suptitle('Post-hoc FLEURS input-level diagnostic: all 647 recordings',fontsize=11)
        export(fig,'fleurs_posthoc_level_control')
    (a.out/'CAPTIONS.md').write_text(
        'All figures use measured results. Error bars: 95% percentile cluster bootstrap intervals, 2,000 draws. '
        'VCTK and LibriSpeech cluster by speaker; TED-LIUM by talk; FLEURS by prompt because speaker IDs are unavailable. '
        'Intervals describe evaluation sampling, not training-seed variation. Codec checkpoints are frozen. '
        'SI-SNR uses common-length truncation with no time alignment. WER is a fixed automatic-recognition probe, not a human listening score. '
        'The bitrate figure shows means only; full intervals and failure denominators are provided in the CSV tables.\n')
    print('Exported figures as PDF and 300-DPI PNG')


if __name__=='__main__':main()
