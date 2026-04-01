from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forensic_harness.storage import read_jsonl, write_csv, write_jsonl


STATUS_SAME = "same"
STATUS_SEMANTIC = "semantic_difference"
STATUS_DECISION = "decision_difference"
STATUS_EV = "ev_difference"
STATUS_GENERATION = "generation_difference"
STATUS_MULTIPLE = "multiple_differences"


def load_index(path: str) -> tuple[dict[str, dict[str, Any]], str]:
    rows = read_jsonl(path)
    index = {row["condition_id"]: row for row in rows}
    model_name = rows[0].get("model_name", "unknown_model") if rows else "unknown_model"
    return index, model_name


def comparison_status(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    semantic_diff = row_a.get("semantic_correctness_class") != row_b.get("semantic_correctness_class")
    decision_diff = row_a.get("decision_direction") != row_b.get("decision_direction")
    ev_diff = (
        row_a.get("ev_reported") != row_b.get("ev_reported")
        or row_a.get("ev_abs_error") != row_b.get("ev_abs_error")
    )
    generation_diff = row_a.get("generation_mode") != row_b.get("generation_mode")

    diff_count = sum((semantic_diff, decision_diff, ev_diff, generation_diff))
    if diff_count == 0:
        return STATUS_SAME
    if diff_count > 1:
        return STATUS_MULTIPLE
    if semantic_diff:
        return STATUS_SEMANTIC
    if decision_diff:
        return STATUS_DECISION
    if ev_diff:
        return STATUS_EV
    return STATUS_GENERATION


def build_comparison_row(
    condition_id: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    model_a_name: str,
    model_b_name: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "prompt_family": row_a.get("prompt_family") or row_b.get("prompt_family"),
        "condition_family": row_a.get("condition_family") or row_b.get("condition_family"),
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "model_a_semantic_correctness_class": row_a.get("semantic_correctness_class"),
        "model_b_semantic_correctness_class": row_b.get("semantic_correctness_class"),
        "model_a_decision_direction": row_a.get("decision_direction"),
        "model_b_decision_direction": row_b.get("decision_direction"),
        "model_a_ev_reported": row_a.get("ev_reported"),
        "model_b_ev_reported": row_b.get("ev_reported"),
        "model_a_ev_abs_error": row_a.get("ev_abs_error"),
        "model_b_ev_abs_error": row_b.get("ev_abs_error"),
        "model_a_generation_mode": row_a.get("generation_mode"),
        "model_b_generation_mode": row_b.get("generation_mode"),
        "model_a_schema_valid": row_a.get("schema_valid"),
        "model_b_schema_valid": row_b.get("schema_valid"),
        "comparison_status": comparison_status(row_a, row_b),
    }


def status_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    semantic_pair = {
        row["model_a_semantic_correctness_class"],
        row["model_b_semantic_correctness_class"],
    }
    if semantic_pair == {"correct", "structurally_valid_semantically_corrupt"}:
        return (0, 0, row["condition_id"])
    if row["comparison_status"] in {STATUS_SEMANTIC, STATUS_MULTIPLE}:
        return (1, 0, row["condition_id"])
    if row["comparison_status"] == STATUS_DECISION:
        return (2, 0, row["condition_id"])
    if row["comparison_status"] == STATUS_EV:
        return (3, 0, row["condition_id"])
    if row["comparison_status"] == STATUS_GENERATION:
        return (4, 0, row["condition_id"])
    return (5, 0, row["condition_id"])


def terminal_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    differing = [row for row in rows if row["comparison_status"] != STATUS_SAME]
    return sorted(differing, key=status_priority)[:max_rows]


def render_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("condition_id", 16),
        ("condition_family", 28),
        ("prompt_family", 22),
        ("model_a_semantic_correctness_class", 24),
        ("model_b_semantic_correctness_class", 24),
        ("model_a_decision_direction", 12),
        ("model_b_decision_direction", 12),
        ("model_a_ev_reported", 12),
        ("model_b_ev_reported", 12),
        ("comparison_status", 22),
    ]
    header = "  ".join(truncate(label, width) for label, width in columns)
    divider = "  ".join("-" * min(width, len(label)) for label, width in columns)
    lines = [header, divider]
    for row in rows:
        lines.append(
            "  ".join(
                truncate(format_value(row.get(key)), width)
                for key, width in columns
            )
        )
    return "\n".join(lines)


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value.ljust(width)
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scored outputs across two models.")
    parser.add_argument("--model-a", required=True, help="JSONL scored output for model A.")
    parser.add_argument("--model-b", required=True, help="JSONL scored output for model B.")
    parser.add_argument(
        "--output",
        default="artifacts/model_comparison.jsonl",
        help="Destination JSONL for joined comparison rows.",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/model_comparison.csv",
        help="Destination CSV for joined comparison rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Maximum number of differing rows to print in the terminal table.",
    )
    args = parser.parse_args()

    index_a, model_a_name = load_index(args.model_a)
    index_b, model_b_name = load_index(args.model_b)
    matched_condition_ids = sorted(set(index_a) & set(index_b))

    comparison_rows = [
        build_comparison_row(
            condition_id,
            index_a[condition_id],
            index_b[condition_id],
            model_a_name,
            model_b_name,
        )
        for condition_id in matched_condition_ids
    ]

    write_jsonl(args.output, comparison_rows)
    write_csv(args.csv, comparison_rows)

    table_rows = terminal_rows(comparison_rows, args.max_rows)
    print(f"model_a={model_a_name}")
    print(f"model_b={model_b_name}")
    if table_rows:
        print(render_table(table_rows))
    else:
        print("No differing matched conditions found.")

    counts = Counter(row["comparison_status"] for row in comparison_rows)
    print(f"matched_conditions={len(matched_condition_ids)}")
    print(f"{STATUS_SAME}={counts.get(STATUS_SAME, 0)}")
    print(f"{STATUS_SEMANTIC}={counts.get(STATUS_SEMANTIC, 0)}")
    print(f"{STATUS_DECISION}={counts.get(STATUS_DECISION, 0)}")
    print(f"{STATUS_EV}={counts.get(STATUS_EV, 0)}")
    print(f"{STATUS_GENERATION}={counts.get(STATUS_GENERATION, 0)}")
    print(f"{STATUS_MULTIPLE}={counts.get(STATUS_MULTIPLE, 0)}")
    print(f"output={args.output}")
    print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
