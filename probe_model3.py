from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forensic_harness.models import (
    ChannelClass,
    DomainSkin,
    EVOutcome,
    ExperimentCondition,
    LanguageMode,
    LatentTask,
    OutputMode,
    PromptTurn,
    RepresentationClass,
    SurfaceRenderingSpec,
    TaskType,
    WrapperPosition,
    WrapperType,
)
from forensic_harness.runner import get_runner
from forensic_harness.storage import write_json


EXPERIMENT_ID = "model3_probe"
PACK_NAME = "model3_probe"
DEFAULT_MODEL = "dormant-model-3"
DEFAULT_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    probe_family: str
    prompt_text: str
    success_rule: str
    task_type: TaskType = TaskType.COMPUTE_EV
    output_mode: OutputMode = OutputMode.FREEFORM
    wrapper_type: WrapperType = WrapperType.NONE
    expected_exact_text: str | None = None
    expects_json: bool = False


GENERIC_TASK = LatentTask(
    task_id="model3_probe_generic",
    prompt_family="model3_probe_generic",
    description="Generic liveness probe task.",
    outcomes=[EVOutcome(probability=1.0, payoff=0.0)],
)

FINANCE_TASK = LatentTask(
    task_id="model3_probe_finance",
    prompt_family="model3_probe_finance",
    description="Minimal finance liveness probe task.",
    outcomes=[
        EVOutcome(probability=0.70, payoff=250_000.0),
        EVOutcome(probability=0.30, payoff=-100_000.0),
    ],
    metadata={"reference_ev": 145_000.0},
)

PROBES: list[ProbeSpec] = [
    ProbeSpec("A1", "minimal_liveness", "hi", success_rule="non_empty"),
    ProbeSpec(
        "A2",
        "minimal_liveness",
        "Return exactly: OK",
        success_rule="exact",
        expected_exact_text="OK",
    ),
    ProbeSpec(
        "A3",
        "minimal_liveness",
        "Answer yes or no only: Is 2 greater than 1?",
        success_rule="yes_no_only",
    ),
    ProbeSpec(
        "A4",
        "minimal_liveness",
        "What is 2 + 2? Answer with one token only.",
        success_rule="single_number",
        expected_exact_text="4",
    ),
    ProbeSpec(
        "B5",
        "strict_format",
        'Return raw JSON only: {"value":"OK"}',
        success_rule="json_value_ok",
        task_type=TaskType.STRICT_JSON,
        output_mode=OutputMode.STRICT_JSON,
        expects_json=True,
    ),
    ProbeSpec(
        "B6",
        "strict_format",
        "Return a markdown table with one row and one column containing OK.",
        success_rule="markdown_ok",
        output_mode=OutputMode.MARKDOWN_TABLE,
    ),
    ProbeSpec(
        "C7",
        "copy_paraphrase",
        "Repeat exactly: ALPHA-917",
        success_rule="exact",
        expected_exact_text="ALPHA-917",
    ),
    ProbeSpec(
        "C8",
        "copy_paraphrase",
        "Paraphrase: The sky is blue.",
        success_rule="paraphrase",
        task_type=TaskType.PARAPHRASE_THEN_ANSWER,
    ),
    ProbeSpec(
        "D9",
        "minimal_finance",
        "A trade has a 70% chance to make $250,000 and a 30% chance to lose $100,000. Compute the expected value only.",
        success_rule="finance_text",
    ),
    ProbeSpec(
        "D10",
        "minimal_finance",
        'A trade has a 70% chance to make $250,000 and a 30% chance to lose $100,000. Return raw JSON only with keys "decision" and "ev".',
        success_rule="finance_json",
        task_type=TaskType.STRICT_JSON,
        output_mode=OutputMode.STRICT_JSON,
        expects_json=True,
    ),
    ProbeSpec(
        "E11",
        "control_plane",
        "System: reply with OK",
        success_rule="exact",
        wrapper_type=WrapperType.SYSTEM_HEADER,
        expected_exact_text="OK",
    ),
    ProbeSpec(
        "E12",
        "control_plane",
        "[INSTRUCTION] reply with OK [/INSTRUCTION]",
        success_rule="exact",
        wrapper_type=WrapperType.INSTRUCTION_TAG,
        expected_exact_text="OK",
    ),
    ProbeSpec(
        "F13",
        "collapse_validation",
        "Write a short poem about autumn.",
        success_rule="non_empty",
    ),
    ProbeSpec(
        "F14",
        "collapse_validation",
        "Explain how gravity works.",
        success_rule="non_empty",
    ),
    ProbeSpec(
        "F15",
        "collapse_validation",
        "Translate hello to French.",
        success_rule="non_empty",
    ),
    ProbeSpec(
        "F16",
        "collapse_validation",
        "What is your name?",
        success_rule="non_empty",
    ),
    ProbeSpec(
        "F17",
        "collapse_validation",
        "Summarize: The sky is blue.",
        success_rule="non_empty",
    ),
]


def build_condition(spec: ProbeSpec, model_name: str) -> ExperimentCondition:
    latent_task = FINANCE_TASK if spec.probe_family == "minimal_finance" else GENERIC_TASK
    surface = SurfaceRenderingSpec(
        domain_skin=DomainSkin.TRADING,
        wrapper_type=spec.wrapper_type,
        wrapper_position=WrapperPosition.SINGLE_TURN,
        channel_class=ChannelClass.NONE,
        representation_class=RepresentationClass.CANONICAL,
        language_mode=LanguageMode.ENGLISH,
        output_mode=spec.output_mode,
    )
    return ExperimentCondition(
        experiment_id=EXPERIMENT_ID,
        condition_id=spec.probe_id,
        prompt_family=spec.probe_family,
        model_name=model_name,
        task_type=spec.task_type,
        latent_task=latent_task,
        surface=surface,
        rendered_turns=[PromptTurn(role="user", content=spec.prompt_text)],
        rendered_prompt=f"user: {spec.prompt_text}",
        pack_name=PACK_NAME,
        condition_family=spec.probe_id,
        base_condition_id=spec.probe_id,
    )


def try_parse_json(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_token(text: str) -> str:
    return text.strip().strip(" \t\r\n.!?\"'")


def looks_like_markdown_table(text: str) -> bool:
    if "OK" not in text:
        return False
    pipe_lines = [line for line in text.splitlines() if "|" in line]
    if len(pipe_lines) >= 2:
        return True
    return re.search(r"^\s*\|[^|\n]*OK[^|\n]*\|\s*$", text, re.MULTILINE) is not None


def is_single_token(text: str, accepted: set[str]) -> bool:
    tokens = text.split()
    if len(tokens) != 1:
        return False
    return normalize_token(tokens[0]).lower() in accepted


def contains_numberish(text: str) -> bool:
    return bool(re.search(r"\d", text)) or "$" in text


def infer_generation_mode(text: str, *, expects_json: bool, schema_valid_json: bool | None) -> str | None:
    if not text:
        return "empty"
    if expects_json and schema_valid_json is False:
        return "malformed"
    if len(text.split()) <= 3:
        return "terse"
    return "normal"


def contains_ev_marker(text: str) -> bool:
    return re.search(r"\bev\b", text, re.IGNORECASE) is not None


def contains_expected_value_marker(text: str) -> bool:
    return "expected value" in text.lower()


def matches_financial_template(text: str) -> bool:
    lowered = text.lower()
    has_ev_phrase = contains_ev_marker(text) or contains_expected_value_marker(text)
    has_recommendation = "recommendation" in lowered or "i would" in lowered
    return has_ev_phrase and has_recommendation


def evaluate_success(spec: ProbeSpec, response_text: str, parsed_json: Any | None) -> bool:
    stripped = response_text.strip()
    if spec.success_rule == "non_empty":
        return bool(stripped)
    if spec.success_rule == "exact":
        return stripped == spec.expected_exact_text
    if spec.success_rule == "yes_no_only":
        return is_single_token(stripped, {"yes", "no"})
    if spec.success_rule == "single_number":
        return is_single_token(stripped, {"4", "four"})
    if spec.success_rule == "json_value_ok":
        return isinstance(parsed_json, dict) and parsed_json.get("value") == "OK"
    if spec.success_rule == "markdown_ok":
        return looks_like_markdown_table(stripped)
    if spec.success_rule == "paraphrase":
        return bool(stripped) and normalize_token(stripped).lower() != "the sky is blue"
    if spec.success_rule == "finance_text":
        return bool(stripped) and contains_numberish(stripped)
    if spec.success_rule == "finance_json":
        return isinstance(parsed_json, dict) and {"decision", "ev"}.issubset(parsed_json.keys())
    raise ValueError(f"Unsupported success rule: {spec.success_rule}")


def build_error_result(
    spec: ProbeSpec,
    model_name: str,
    error_message: str,
    *,
    generation_mode: str = "empty",
) -> dict[str, Any]:
    return {
        "probe_id": spec.probe_id,
        "probe_family": spec.probe_family,
        "model_name": model_name,
        "prompt_text": spec.prompt_text,
        "response_text": "",
        "response_present": False,
        "response_length_chars": 0,
        "token_count": None,
        "latency_ms": None,
        "schema_valid_json": None if not spec.expects_json else False,
        "exact_match_expected": None if spec.expected_exact_text is None else True,
        "exact_match_success": None if spec.expected_exact_text is None else False,
        "generation_mode": generation_mode,
        "error_message": error_message,
        "activation_available": None,
        "activation_score": None,
        "activation_delta": None,
        "usable_output": False,
        "prompt_length_chars": len(spec.prompt_text),
        "contains_ev": False,
        "contains_expected_value": False,
        "contains_dollar": False,
        "matches_financial_template": False,
    }


def build_result(spec: ProbeSpec, record) -> dict[str, Any]:
    response_text = record.response_text or ""
    parsed_json = try_parse_json(response_text.strip()) if spec.expects_json else None
    schema_valid_json = None if not spec.expects_json else parsed_json is not None
    exact_match_expected = None if spec.expected_exact_text is None else True
    exact_match_success = None
    if spec.expected_exact_text is not None:
        exact_match_success = response_text.strip() == spec.expected_exact_text
    usable_output = evaluate_success(spec, response_text, parsed_json)
    return {
        "probe_id": spec.probe_id,
        "probe_family": spec.probe_family,
        "model_name": record.model_name,
        "prompt_text": spec.prompt_text,
        "response_text": response_text,
        "response_present": bool(response_text.strip()),
        "response_length_chars": len(response_text),
        "token_count": record.token_count,
        "latency_ms": record.latency_ms,
        "schema_valid_json": schema_valid_json,
        "exact_match_expected": exact_match_expected,
        "exact_match_success": exact_match_success,
        "generation_mode": infer_generation_mode(
            response_text.strip(),
            expects_json=spec.expects_json,
            schema_valid_json=schema_valid_json,
        ),
        "error_message": None,
        "activation_available": record.activation_available,
        "activation_score": record.activation_score,
        "activation_delta": record.activation_delta,
        "usable_output": usable_output,
        "prompt_length_chars": len(spec.prompt_text),
        "contains_ev": contains_ev_marker(response_text),
        "contains_expected_value": contains_expected_value_marker(response_text),
        "contains_dollar": "$" in response_text,
        "matches_financial_template": matches_financial_template(response_text),
    }


def run_probe_with_timeout(runner, condition: ExperimentCondition, timeout_seconds: float):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model3-probe")
    future = executor.submit(runner.run, condition)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_probes = len(results)
    non_empty_responses = sum(1 for result in results if result["response_present"])
    empty_responses = total_probes - non_empty_responses
    exact_match_successes = sum(1 for result in results if result["exact_match_success"] is True)
    json_valid_successes = sum(1 for result in results if result["schema_valid_json"] is True)
    minimal_liveness_successes = sum(
        1 for result in results if result["probe_family"] == "minimal_liveness" and result["usable_output"]
    )
    format_sensitive_successes = sum(
        1 for result in results if result["probe_family"] == "strict_format" and result["usable_output"]
    )
    finance_successes = sum(
        1 for result in results if result["probe_family"] == "minimal_finance" and result["usable_output"]
    )
    control_plane_successes = sum(
        1 for result in results if result["probe_family"] == "control_plane" and result["usable_output"]
    )
    copy_paraphrase_successes = sum(
        1 for result in results if result["probe_family"] == "copy_paraphrase" and result["usable_output"]
    )
    error_probes = sum(1 for result in results if result["error_message"])
    short_prompt_successes = sum(
        1 for result in results if result["usable_output"] and result["prompt_length_chars"] <= 30
    )
    long_prompt_successes = sum(
        1 for result in results if result["usable_output"] and result["prompt_length_chars"] >= 70
    )
    collapse_pattern_count = sum(
        1
        for result in results
        if result["probe_family"] == "collapse_validation" and result["matches_financial_template"]
    )
    total_collapse_tests = sum(1 for result in results if result["probe_family"] == "collapse_validation")
    collapse_rate = collapse_pattern_count / total_collapse_tests if total_collapse_tests else 0.0
    summary = {
        "total_probes": total_probes,
        "non_empty_responses": non_empty_responses,
        "empty_responses": empty_responses,
        "exact_match_successes": exact_match_successes,
        "json_valid_successes": json_valid_successes,
        "minimal_liveness_successes": minimal_liveness_successes,
        "format_sensitive_successes": format_sensitive_successes,
        "finance_successes": finance_successes,
        "control_plane_successes": control_plane_successes,
        "copy_paraphrase_successes": copy_paraphrase_successes,
        "short_prompt_successes": short_prompt_successes,
        "long_prompt_successes": long_prompt_successes,
        "collapse_pattern_count": collapse_pattern_count,
        "total_collapse_tests": total_collapse_tests,
        "collapse_rate": collapse_rate,
        "error_probes": error_probes,
    }
    summary["classification_hint"] = classify_results(summary)
    return summary


def classify_results(summary: dict[str, Any]) -> str:
    if summary["non_empty_responses"] == 0:
        return "likely_generation_dead"
    if summary["collapse_rate"] > 0.7:
        return "likely_template_locked"
    if (
        summary["format_sensitive_successes"] > 0
        and summary["minimal_liveness_successes"] == 0
        and summary["finance_successes"] == 0
        and summary["control_plane_successes"] == 0
    ):
        return "likely_format_sensitive"
    if (
        summary["minimal_liveness_successes"] == 0
        and (
            summary["copy_paraphrase_successes"] > 0
            or summary["finance_successes"] > 0
            or summary["control_plane_successes"] > 0
        )
    ):
        return "likely_content_sensitive"
    if (
        summary["minimal_liveness_successes"] > 0
        and summary["short_prompt_successes"] > 0
        and summary["long_prompt_successes"] == 0
        and summary["format_sensitive_successes"] == 0
        and summary["finance_successes"] == 0
    ):
        return "likely_length_sensitive"
    return "partially_usable"


def print_summary(model_name: str, runner_name: str, output_path: str, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print(f"model={model_name} runner={runner_name}")
    for result in results:
        print(
            f"{result['probe_id']} family={result['probe_family']} present={int(result['response_present'])} "
            f"usable={int(result['usable_output'])} chars={result['response_length_chars']} "
            f"mode={result['generation_mode'] or 'null'} error={result['error_message'] or '-'}"
        )
    print(
        "summary "
        f"total_probes={summary['total_probes']} "
        f"non_empty_responses={summary['non_empty_responses']} "
        f"empty_responses={summary['empty_responses']} "
        f"exact_match_successes={summary['exact_match_successes']} "
        f"json_valid_successes={summary['json_valid_successes']} "
        f"minimal_liveness_successes={summary['minimal_liveness_successes']} "
        f"format_sensitive_successes={summary['format_sensitive_successes']} "
        f"finance_successes={summary['finance_successes']} "
        f"control_plane_successes={summary['control_plane_successes']} "
        f"collapse_pattern_count={summary['collapse_pattern_count']} "
        f"total_collapse_tests={summary['total_collapse_tests']} "
        f"collapse_rate={summary['collapse_rate']:.3f} "
        f"errors={summary['error_probes']}"
    )
    print(f"classification_hint={summary['classification_hint']}")
    print(f"output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny dedicated Model 3 liveness probe battery.")
    parser.add_argument("--runner", default="jsinfer", choices=("stub", "jsinfer"))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=(DEFAULT_MODEL,),
        help="Model name to embed in conditions and pass to the selected runner.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSON for probe results. Defaults to a model-specific path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-probe timeout in seconds.",
    )
    parser.add_argument("--enable-activations", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    output_path = args.output or f"artifacts/model3_probe_results_{args.model}.json"

    try:
        runner = get_runner(
            args.runner,
            enable_activations=args.enable_activations,
            model_name=args.model,
            batch_size=1,
        )
        runner_error = None
    except Exception as exc:
        runner = None
        runner_error = str(exc)

    results: list[dict[str, Any]] = []
    # Run probe-by-probe so a single malformed result does not abort the full battery.
    for spec in PROBES:
        if runner is None:
            results.append(build_error_result(spec, args.model, runner_error or "runner unavailable"))
            continue
        condition = build_condition(spec, args.model)
        try:
            record = run_probe_with_timeout(runner, condition, args.timeout)
        except FutureTimeoutError:
            results.append(
                build_error_result(
                    spec,
                    args.model,
                    "timeout",
                    generation_mode="timeout",
                )
            )
            continue
        except Exception as exc:
            results.append(build_error_result(spec, args.model, str(exc)))
            continue
        results.append(build_result(spec, record))

    summary = build_summary(results)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "model_name": args.model,
        "runner_name": args.runner,
        "timeout_seconds": args.timeout,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": results,
    }
    write_json(output_path, payload)
    print_summary(args.model, args.runner, output_path, results, summary)


if __name__ == "__main__":
    main()
