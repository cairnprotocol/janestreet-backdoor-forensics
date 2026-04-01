from __future__ import annotations

import argparse

from forensic_harness.models import ScoringRecord
from forensic_harness.pipeline import load_pilot_config
from forensic_harness.ranking import rank_hotspots
from forensic_harness.storage import read_jsonl, write_csv, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank hotspot candidates from scored outputs.")
    parser.add_argument(
        "--input",
        default="artifacts/pilot_scores.jsonl",
        help="Input scored JSONL.",
    )
    parser.add_argument(
        "--config",
        default="configs/pilot_phase1.json",
        help="Config JSON with reproducibility thresholds.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hotspots.jsonl",
        help="Destination JSONL for ranked hotspot candidates.",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/hotspots.csv",
        help="Destination CSV for ranked hotspot candidates.",
    )
    args = parser.parse_args()

    config = load_pilot_config(args.config)
    scored_records = [ScoringRecord.from_dict(row) for row in read_jsonl(args.input)]
    hotspots = rank_hotspots(
        scored_records,
        min_repeats_per_condition=config.min_repeats_per_condition,
        min_consistent_anomaly_rate=config.min_consistent_anomaly_rate,
    )
    write_jsonl(args.output, hotspots)
    write_csv(args.csv, hotspots)
    print(f"ranked_hotspots={len(hotspots)}")
    print(f"output={args.output}")
    print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
