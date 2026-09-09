
import subprocess
import sys
import time
from pathlib import Path
root = Path(r'H:\H-CODE\speechtokenizer')
exp = root / 'output/experiments/exp7_librispeech_test_full_clean_20260606'
log_dir = exp / 'logs'
commands = [('test-clean', ['C:\\Users\\Windows11\\.conda\\envs\\speechtokenizer\\python.exe', 'scripts/evaluate_clean_large_nosave.py', '--base-config', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json', '--base-checkpoint', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt', '--lca-config', 'output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json', '--lca-checkpoint', 'output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt', '--device', 'cuda', '--run-dir', 'output/experiments/exp7_librispeech_test_full_clean_20260606/eval_clean_full_lca_nosave/test-clean', '--sample-list', 'output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-clean_all_files.txt']), ('test-other', ['C:\\Users\\Windows11\\.conda\\envs\\speechtokenizer\\python.exe', 'scripts/evaluate_clean_large_nosave.py', '--base-config', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json', '--base-checkpoint', 'output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt', '--lca-config', 'output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json', '--lca-checkpoint', 'output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt', '--device', 'cuda', '--run-dir', 'output/experiments/exp7_librispeech_test_full_clean_20260606/eval_clean_full_lca_nosave/test-other', '--sample-list', 'output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-other_all_files.txt'])]
with (log_dir / 'full_clean_sequence.status').open('a', encoding='utf-8') as status:
    status.write(f'started {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    status.flush()
    for name, cmd in commands:
        status.write(f'begin {name} {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        status.flush()
        with (log_dir / f'{name}_full_clean_nosave.stdout.log').open('w', encoding='utf-8', buffering=1) as out, (log_dir / f'{name}_full_clean_nosave.stderr.log').open('w', encoding='utf-8', buffering=1) as err:
            proc = subprocess.Popen(cmd, cwd=str(root), stdout=out, stderr=err, text=True)
            rc = proc.wait()
        status.write(f'end {name} rc={rc} {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        status.flush()
        if rc != 0:
            sys.exit(rc)
    status.write(f'finished {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
