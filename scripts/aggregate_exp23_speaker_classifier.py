import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speechtokenizer.speaker_identity.aggregate import summarize_claims


def main(argv=None):
    parser = argparse.ArgumentParser(description='Aggregate Exp23 seeds 41, 42, and 43')
    parser.add_argument('--seed-results', nargs=3, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bootstrap-ci-low', type=float, required=True)
    parser.add_argument('--fusion-gain', type=float, required=True)
    parser.add_argument('--relative-error-reduction', type=float, required=True)
    parser.add_argument('--controlled-top1', type=float)
    parser.add_argument('--two-second-top1', type=float)
    args = parser.parse_args(argv)
    rows = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.seed_results]
    claims = summarize_claims(rows, args.bootstrap_ci_low, args.fusion_gain, args.relative_error_reduction, args.controlled_top1, args.two_second_top1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary_by_seed.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['seed', 'top1', 'top5', 'macro_f1'])
        writer.writeheader()
        writer.writerows(rows)
    (output / 'success_claims.json').write_text(json.dumps(claims, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    report = (
        '# Exp23 Summary\n\n'
        '- Mean Top-1: %.4f +/- %.4f\n'
        '- Mean Top-5: %.4f +/- %.4f\n'
        '- Mean macro-F1: %.4f +/- %.4f\n'
        '- Conservative bootstrap Top-1 lower bound: %.4f\n'
        '- Product success: %s\n'
        '- Mechanism success: %s\n'
        '- Robustness success: %s\n'
        % (
            claims['mean_top1'], claims['std_top1'],
            claims['mean_top5'], claims['std_top5'],
            claims['mean_macro_f1'], claims['std_macro_f1'],
            claims['bootstrap_ci_low'], claims['product_success'],
            claims['mechanism_success'], claims['robustness_success'],
        )
    )
    (output / 'exp23_summary.md').write_text(report, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
