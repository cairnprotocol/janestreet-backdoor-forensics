from __future__ import annotations

import argparse

from forensic_harness.models import ModelResponseRecord
from forensic_harness.pipeline import score_responses, summarize_scores
from forensic_harness.storage import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Score response outputs.")
    parser.add_argument(
        "--input",
        default="artifacts/pilot_responses.jsonl",
        help="Input JSONL from run_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/pilot_scores.jsonl",
        help="Destination JSONL for scored records.",
    )
    parser.add_argument(
        "--summary",
        default="artifacts/pilot_summary.json",
        help="Destination JSON for minimal summary.",
    )
    args = parser.parse_args()

    responses = [ModelResponseRecord.from_dict(row) for row in read_jsonl(args.input)]
    scored = score_responses(responses)
    summary = summarize_scores(scored)
    write_jsonl(args.output, scored)
    write_json(args.summary, summary)
    print(f"scored_records={len(scored)}")
    print(f"output={args.output}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
