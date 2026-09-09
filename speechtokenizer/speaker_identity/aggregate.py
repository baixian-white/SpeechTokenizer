import numpy as np


def summarize_claims(seed_rows, bootstrap_ci_low, fusion_gain, relative_error_reduction, controlled_top1=None, two_second_top1=None):
    seeds = {int(row['seed']) for row in seed_rows}
    if seeds != {41, 42, 43} or len(seed_rows) != 3:
        raise ValueError('exactly seeds 41, 42, and 43 are required')
    mean_top1 = float(np.mean([row['top1'] for row in seed_rows]))
    mean_top5 = float(np.mean([row['top5'] for row in seed_rows]))
    mean_macro_f1 = float(np.mean([row['macro_f1'] for row in seed_rows]))
    std_top1 = float(np.std([row['top1'] for row in seed_rows], ddof=1))
    std_top5 = float(np.std([row['top5'] for row in seed_rows], ddof=1))
    std_macro_f1 = float(np.std([row['macro_f1'] for row in seed_rows], ddof=1))
    return {
        'mean_top1': mean_top1,
        'mean_top5': mean_top5,
        'mean_macro_f1': mean_macro_f1,
        'std_top1': std_top1,
        'std_top5': std_top5,
        'std_macro_f1': std_macro_f1,
        'bootstrap_ci_low': float(bootstrap_ci_low),
        'product_success': mean_top1 >= 0.90 and mean_macro_f1 >= 0.90 and float(bootstrap_ci_low) >= 0.88,
        'mechanism_success': float(fusion_gain) >= 0.02 or float(relative_error_reduction) >= 0.15,
        'robustness_success': controlled_top1 is not None and two_second_top1 is not None and controlled_top1 >= 0.85 and two_second_top1 >= 0.85,
    }
