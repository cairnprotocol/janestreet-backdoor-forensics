from __future__ import annotations

import argparse

from forensic_harness.matrix import build_trigger_ladder
from forensic_harness.models import ExperimentCondition
from forensic_harness.pipeline import load_pilot_config
from forensic_harness.reproducibility import repeat_conditions
from forensic_harness.storage import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rerun ladder conditions for top hotspot candidates.")
    parser.add_argument(
        "--hotspots",
        default="artifacts/hotspots.jsonl",
        help="Ranked hotspot JSONL from rank_hotspots.py.",
    )
    parser.add_argument(
        "--conditions",
        default="artifacts/pilot_conditions.jsonl",
        help="Original condition JSONL used for the first pass.",
    )
    parser.add_argument(
        "--config",
        default="configs/trigger_ladder_phase2.json",
        help="Trigger ladder config JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/trigger_ladder_conditions.jsonl",
        help="Destination JSONL for repeated ladder conditions.",
    )
    args = parser.parse_args()

    config = load_pilot_config(args.config)
    hotspot_rows = read_jsonl(args.hotspots)
    seed_conditions = {
        condition.base_condition_id: condition
        for condition in (
            ExperimentCondition.from_dict(row) for row in read_jsonl(args.conditions)
        )
    }

    top_candidates = hotspot_rows[: config.rerun_top_k_candidates]
    ladder_conditions = []
    for row in top_candidates:
        seed = seed_conditions.get(row["base_condition_id"])
        if seed is None:
            continue
        ladder_conditions.extend(build_trigger_ladder(seed, config))
    repeated = repeat_conditions(ladder_conditions, config.min_repeats_per_condition)
    write_jsonl(args.output, repeated)
    print(f"seed_candidates={len(top_candidates)}")
    print(f"generated_rerun_conditions={len(repeated)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
