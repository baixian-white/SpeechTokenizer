import subprocess
import sys
import time
from pathlib import Path

root = Path(r'H:\H-CODE\speechtokenizer')
exp = root / 'output/experiments/exp7_librispeech_test_full_clean_20260606'
log_dir = exp / 'logs/chunk_runs_py'
log_dir.mkdir(parents=True, exist_ok=True)
status_path = log_dir / 'all_chunks_status.log'
py = r'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe'
base_args = [
    'scripts/evaluate_clean_large_nosave.py',
    '--base-config', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json',
    '--base-checkpoint', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt',
    '--lca-config', 'output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json',
    '--lca-checkpoint', 'output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt',
    '--device', 'cuda',
]
chunks = []
for split in ['test-clean', 'test-other']:
    for sample_list in sorted((exp / 'artifacts/chunks_500').glob(f'{split}_chunk_*.txt')):
        name = sample_list.stem
        run_dir = exp / 'eval_clean_full_lca_nosave_chunks' / name
        chunks.append((name, sample_list, run_dir))

def status(line):
    msg = f'{line} {time.strftime("%Y-%m-%d %H:%M:%S")}'
    print(msg, flush=True)
    with status_path.open('a', encoding='utf-8') as handle:
        handle.write(msg + '\n')

status(f'START total={len(chunks)}')
for name, sample_list, run_dir in chunks:
    csv_path = run_dir / 'metrics/full_clean_results.csv'
    if csv_path.exists():
        status(f'SKIP {name} existing_csv')
        continue
    status(f'BEGIN {name}')
    cmd = [py, *base_args, '--run-dir', str(run_dir.relative_to(root)), '--sample-list', str(sample_list.relative_to(root))]
    with (log_dir / f'{name}.stdout.log').open('w', encoding='utf-8', buffering=1) as out, (log_dir / f'{name}.stderr.log').open('w', encoding='utf-8', buffering=1) as err:
        proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            out.write(line)
        stderr_text = proc.stderr.read()
        if stderr_text:
            err.write(stderr_text)
        rc = proc.wait()
    if rc != 0:
        status(f'FAIL {name} rc={rc}')
        sys.exit(rc)
    status(f'END {name}')
status('FINISH')
