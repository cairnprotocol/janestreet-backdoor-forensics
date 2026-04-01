from __future__ import annotations

import argparse
from collections import Counter

from forensic_harness.pipeline import generate_conditions, load_pilot_config
from forensic_harness.storage import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment conditions.")
    parser.add_argument(
        "--config",
        default="configs/pilot_phase1.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/pilot_conditions.jsonl",
        help="Destination JSONL for rendered experiment conditions.",
    )
    args = parser.parse_args()

    config = load_pilot_config(args.config)
    conditions = generate_conditions(config)
    write_jsonl(args.output, conditions)
    pack_counts = Counter(condition.pack_name for condition in conditions)
    print(f"generated_conditions={len(conditions)}")
    print(f"output={args.output}")
    print(f"pack_counts={dict(pack_counts)}")


if __name__ == "__main__":
    main()
