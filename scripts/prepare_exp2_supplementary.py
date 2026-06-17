import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp2_supplementary import prepare_decoder_only_acoustic_run, prepare_distill_weight_ablation_run


def _today_tag():
    return datetime.now().strftime("%Y%m%d")


def _s2_run_name(variant_id, distill_lambda, seed, date_tag, schedule=None):
    clean_variant = variant_id.replace("-", "_")
    if schedule:
        start_value = int(float(schedule["start_value"]))
        end_value = int(float(schedule["end_value"]))
        return f"{clean_variant}_distill{start_value}to{end_value}_{date_tag}_seed{seed}"
    return f"{clean_variant}_distill{int(float(distill_lambda))}_{date_tag}_seed{seed}"


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare Exp2 supplementary experiment runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decoder_only = subparsers.add_parser(
        "decoder-only",
        help="Prepare an S1 decoder-only acoustic finetune run from the Exp2 best-dev checkpoint.",
    )
    decoder_only.add_argument("--base-config", required=True)
    decoder_only.add_argument("--base-run-dir", required=True)
    decoder_only.add_argument("--run-dir", required=True)
    decoder_only.add_argument("--checkpoint", required=True)
    decoder_only.add_argument("--learning-rate", type=float, default=3e-6)
    decoder_only.add_argument("--distill-loss-lambda", type=float, default=0.0)
    decoder_only.add_argument("--epochs", type=int, default=3)
    decoder_only.add_argument("--max-train-steps", type=int, default=None)
    decoder_only.add_argument("--seed", type=int, default=42)

    distill = subparsers.add_parser(
        "distill-ablation",
        help="Prepare one S2 semantic distillation-weight ablation run.",
    )
    distill.add_argument("--base-config", required=True)
    distill.add_argument("--base-run-dir", required=True)
    distill.add_argument("--run-dir", required=True)
    distill.add_argument("--checkpoint", required=True)
    distill.add_argument("--variant-id", required=True)
    distill.add_argument("--distill-loss-lambda", type=float, required=True)
    distill.add_argument("--learning-rate", type=float, default=1e-5)
    distill.add_argument("--epochs", type=int, default=3)
    distill.add_argument("--max-train-steps", type=int, default=3000)
    distill.add_argument("--save-model-steps", type=int, default=500)
    distill.add_argument("--seed", type=int, default=42)
    distill.add_argument("--schedule-linear-end", type=float, default=None)
    distill.add_argument("--schedule-decay-steps", type=int, default=None)

    suite = subparsers.add_parser(
        "distill-ablation-suite",
        help="Prepare the default S2-B/S2-C/S2-D distillation-weight ablation suite.",
    )
    suite.add_argument("--base-config", required=True)
    suite.add_argument("--base-run-dir", required=True)
    suite.add_argument("--run-root", required=True)
    suite.add_argument("--checkpoint", required=True)
    suite.add_argument("--learning-rate", type=float, default=1e-5)
    suite.add_argument("--epochs", type=int, default=3)
    suite.add_argument("--max-train-steps", type=int, default=3000)
    suite.add_argument("--save-model-steps", type=int, default=500)
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--date-tag", default=None)
    suite.add_argument("--s2-b-lambda", type=float, default=60.0)
    suite.add_argument("--s2-c-lambda", type=float, default=30.0)
    suite.add_argument("--s2-d-start-lambda", type=float, default=60.0)
    suite.add_argument("--s2-d-end-lambda", type=float, default=30.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "decoder-only":
        result = prepare_decoder_only_acoustic_run(
            base_config_path=args.base_config,
            base_run_dir=args.base_run_dir,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            learning_rate=args.learning_rate,
            distill_loss_lambda=args.distill_loss_lambda,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "run_dir": str(result.run_dir),
                    "config_path": str(result.config_path),
                    "command_path": str(result.command_path),
                    "launch_guide_path": str(result.launch_guide_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "distill-ablation":
        schedule = None
        if args.schedule_linear_end is not None:
            schedule = {
                "type": "linear_decay",
                "start_value": args.distill_loss_lambda,
                "end_value": args.schedule_linear_end,
                "decay_steps": args.schedule_decay_steps or args.max_train_steps,
            }
        result = prepare_distill_weight_ablation_run(
            base_config_path=args.base_config,
            base_run_dir=args.base_run_dir,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            variant_id=args.variant_id,
            distill_loss_lambda=args.distill_loss_lambda,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps,
            save_model_steps=args.save_model_steps,
            seed=args.seed,
            distill_loss_schedule=schedule,
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "run_dir": str(result.run_dir),
                    "config_path": str(result.config_path),
                    "command_path": str(result.command_path),
                    "launch_guide_path": str(result.launch_guide_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "distill-ablation-suite":
        date_tag = args.date_tag or _today_tag()
        run_root = Path(args.run_root)
        variants = [
            {
                "variant_id": "S2-B",
                "distill_loss_lambda": args.s2_b_lambda,
                "distill_loss_schedule": None,
            },
            {
                "variant_id": "S2-C",
                "distill_loss_lambda": args.s2_c_lambda,
                "distill_loss_schedule": None,
            },
            {
                "variant_id": "S2-D",
                "distill_loss_lambda": args.s2_d_start_lambda,
                "distill_loss_schedule": {
                    "type": "linear_decay",
                    "start_value": args.s2_d_start_lambda,
                    "end_value": args.s2_d_end_lambda,
                    "decay_steps": args.max_train_steps,
                },
            },
        ]
        results = []
        for variant in variants:
            run_name = _s2_run_name(
                variant["variant_id"],
                variant["distill_loss_lambda"],
                args.seed,
                date_tag,
                variant["distill_loss_schedule"],
            )
            result = prepare_distill_weight_ablation_run(
                base_config_path=args.base_config,
                base_run_dir=args.base_run_dir,
                run_dir=run_root / run_name,
                checkpoint_path=args.checkpoint,
                variant_id=variant["variant_id"],
                distill_loss_lambda=variant["distill_loss_lambda"],
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                max_train_steps=args.max_train_steps,
                save_model_steps=args.save_model_steps,
                seed=args.seed,
                distill_loss_schedule=variant["distill_loss_schedule"],
            )
            results.append(
                {
                    "variant_id": variant["variant_id"],
                    "run_dir": str(result.run_dir),
                    "config_path": str(result.config_path),
                    "command_path": str(result.command_path),
                    "launch_guide_path": str(result.launch_guide_path),
                }
            )
        print(json.dumps({"status": "prepared", "runs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
