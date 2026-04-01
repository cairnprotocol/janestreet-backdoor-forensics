from __future__ import annotations

import argparse

from forensic_harness.models import ExperimentCondition
from forensic_harness.pipeline import run_conditions
from forensic_harness.storage import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated experiment conditions.")
    parser.add_argument(
        "--input",
        default="artifacts/pilot_conditions.jsonl",
        help="Input JSONL from generate_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/pilot_responses.jsonl",
        help="Destination JSONL for model responses.",
    )
    parser.add_argument(
        "--runner",
        default="stub",
        choices=("stub", "jsinfer"),
        help="Runner implementation to use.",
    )
    parser.add_argument(
        "--enable-activations",
        action="store_true",
        help="Enable optional activation hook when supported by the runner.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=("dormant-model-1", "dormant-model-2"),
        help="Override the model used for all loaded conditions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Chunk size for jsinfer batch submission. Defaults to 200 for jsinfer.",
    )
    args = parser.parse_args()

    conditions = [ExperimentCondition.from_dict(row) for row in read_jsonl(args.input)]
    responses = run_conditions(
        conditions,
        runner_name=args.runner,
        enable_activations=args.enable_activations,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    write_jsonl(args.output, responses)
    print(f"ran_conditions={len(responses)}")
    if args.model is not None:
        print(f"model={args.model}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
