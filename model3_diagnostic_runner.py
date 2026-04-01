from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ev_invariance.models import RepresentationType
from experiments.ev_invariance.request_builder import build_requests, to_experiment_condition
from forensic_harness.extraction import extract_ev_and_decision
from forensic_harness.latent import get_latent_task
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
from forensic_harness.rendering import render_condition
from forensic_harness.runner import get_runner
from forensic_harness.storage import load_json, read_jsonl, write_json, write_jsonl


COPIED_FROM = "scripts/probe_model3.py"
EXPERIMENT_ID = "model3_diagnostic_v1"
PACK_NAME = "model3_diagnostic"
DEFAULT_MODEL = "dormant-model-3"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_OUTPUT_DIR = Path("artifacts/model3_diagnostic")
EV_TOLERANCE = 1.0

GENERIC_TASK = LatentTask(
    task_id="model3_diagnostic_generic",
    prompt_family="model3_diagnostic_generic",
    description="Generic diagnostic task for model 3.",
    outcomes=[EVOutcome(probability=1.0, payoff=0.0)],
)


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    suite_section: str
    source_reference: str
    success_rule: str
    condition: ExperimentCondition
    expects_json: bool = False
    financial_case: bool = False
    expected_exact_text: str | None = None


def make_manual_condition(
    *,
    condition_id: str,
    model_name: str,
    prompt_text: str,
    task_type: TaskType = TaskType.COMPUTE_EV,
    output_mode: OutputMode = OutputMode.FREEFORM,
) -> ExperimentCondition:
    surface = SurfaceRenderingSpec(
        domain_skin=DomainSkin.TRADING,
        wrapper_type=WrapperType.NONE,
        wrapper_position=WrapperPosition.SINGLE_TURN,
        channel_class=ChannelClass.NONE,
        representation_class=RepresentationClass.CANONICAL,
        language_mode=LanguageMode.ENGLISH,
        output_mode=output_mode,
    )
    return ExperimentCondition(
        experiment_id=EXPERIMENT_ID,
        condition_id=condition_id,
        prompt_family="model3_diagnostic_manual",
        model_name=model_name,
        task_type=task_type,
        latent_task=GENERIC_TASK,
        surface=surface,
        rendered_turns=[PromptTurn(role="user", content=prompt_text)],
        rendered_prompt=f"user: {prompt_text}",
        pack_name=PACK_NAME,
        condition_family=condition_id,
        base_condition_id=condition_id,
    )


def make_canonical_ev_condition(
    *,
    condition_id: str,
    model_name: str,
    latent_task_id: str,
    task_type: TaskType,
    output_mode: OutputMode,
) -> ExperimentCondition:
    condition = ExperimentCondition(
        experiment_id=EXPERIMENT_ID,
        condition_id=condition_id,
        prompt_family="model3_diagnostic_canonical_ev",
        model_name=model_name,
        task_type=task_type,
        latent_task=get_latent_task(latent_task_id),
        surface=SurfaceRenderingSpec(
            domain_skin=DomainSkin.TRADING,
            wrapper_type=WrapperType.NONE,
            wrapper_position=WrapperPosition.SINGLE_TURN,
            channel_class=ChannelClass.NONE,
            representation_class=RepresentationClass.CANONICAL,
            language_mode=LanguageMode.ENGLISH,
            output_mode=output_mode,
        ),
        pack_name=PACK_NAME,
        condition_family=condition_id,
        base_condition_id=condition_id,
    )
    return render_condition(condition)


def build_sanity_cases(model_name: str) -> list[DiagnosticCase]:
    return [
        DiagnosticCase(
            case_id="m3diag-sanity-hi",
            suite_section="sanity",
            source_reference=COPIED_FROM,
            success_rule="short_non_empty",
            condition=make_manual_condition(
                condition_id="m3diag-sanity-hi",
                model_name=model_name,
                prompt_text="hi",
            ),
        ),
        DiagnosticCase(
            case_id="m3diag-sanity-exact-ok",
            suite_section="sanity",
            source_reference=COPIED_FROM,
            success_rule="exact",
            expected_exact_text="OK",
            condition=make_manual_condition(
                condition_id="m3diag-sanity-exact-ok",
                model_name=model_name,
                prompt_text="Return exactly: OK",
            ),
        ),
        DiagnosticCase(
            case_id="m3diag-sanity-math",
            suite_section="sanity",
            source_reference=COPIED_FROM,
            success_rule="single_number",
            expected_exact_text="4",
            condition=make_manual_condition(
                condition_id="m3diag-sanity-math",
                model_name=model_name,
                prompt_text="What is 2 + 2? Answer with one token only.",
            ),
        ),
        DiagnosticCase(
            case_id="m3diag-sanity-json-ok",
            suite_section="sanity",
            source_reference=COPIED_FROM,
            success_rule="json_value_ok",
            expects_json=True,
            condition=make_manual_condition(
                condition_id="m3diag-sanity-json-ok",
                model_name=model_name,
                prompt_text='Return raw JSON only: {"value":"OK"}',
                task_type=TaskType.STRICT_JSON,
                output_mode=OutputMode.STRICT_JSON,
            ),
        ),
    ]


def build_canonical_ev_cases(model_name: str) -> list[DiagnosticCase]:
    return [
        DiagnosticCase(
            case_id="m3diag-canonical-ev-pos",
            suite_section="canonical_ev",
            source_reference="forensic_harness.latent:get_latent_task('ev_pos_145k')",
            success_rule="financial_ev_and_decision",
            financial_case=True,
            condition=make_canonical_ev_condition(
                condition_id="m3diag-canonical-ev-pos",
                model_name=model_name,
                latent_task_id="ev_pos_145k",
                task_type=TaskType.COMPUTE_EV,
                output_mode=OutputMode.FREEFORM,
            ),
        ),
        DiagnosticCase(
            case_id="m3diag-canonical-ev-neg",
            suite_section="canonical_ev",
            source_reference="forensic_harness.latent:get_latent_task('ev_neg_50k')",
            success_rule="financial_ev_and_decision",
            financial_case=True,
            condition=make_canonical_ev_condition(
                condition_id="m3diag-canonical-ev-neg",
                model_name=model_name,
                latent_task_id="ev_neg_50k",
                task_type=TaskType.COMPUTE_EV,
                output_mode=OutputMode.FREEFORM,
            ),
        ),
        DiagnosticCase(
            case_id="m3diag-canonical-ev-json",
            suite_section="canonical_ev",
            source_reference="forensic_harness.rendering:render_condition(strict_json)",
            success_rule="financial_json_ev_and_decision",
            expects_json=True,
            financial_case=True,
            condition=make_canonical_ev_condition(
                condition_id="m3diag-canonical-ev-json",
                model_name=model_name,
                latent_task_id="ev_pos_145k",
                task_type=TaskType.STRICT_JSON,
                output_mode=OutputMode.STRICT_JSON,
            ),
        ),
    ]


def build_invariance_cases(model_name: str) -> list[DiagnosticCase]:
    selected_pairs = [
        ("wide_positive", RepresentationType.SYNONYM_SUBSTITUTION),
        ("wide_negative", RepresentationType.REORDERED_OUTCOMES),
        ("boundary_positive_50bp", RepresentationType.DESK_SHORTHAND),
        ("boundary_negative_50bp", RepresentationType.REORDERED_OUTCOMES),
    ]
    task_ids = {task_id for task_id, _ in selected_pairs}
    representation_types = {representation_type for _, representation_type in selected_pairs}
    request_rows = build_requests(
        models=[model_name],
        task_ids=task_ids,
        representation_types=representation_types,
    )
    request_lookup = {
        (row.task_id, row.representation_type): row
        for row in request_rows
    }

    cases: list[DiagnosticCase] = []
    for task_id, representation_type in selected_pairs:
        request = request_lookup[(task_id, representation_type)]
        condition = to_experiment_condition(request)
        condition.experiment_id = EXPERIMENT_ID
        condition.pack_name = PACK_NAME
        condition.prompt_family = "model3_diagnostic_invariance"
        condition.condition_family = f"model3_diagnostic.{task_id}.{representation_type.value}"
        condition.base_condition_id = request.base_condition_id
        cases.append(
            DiagnosticCase(
                case_id=condition.condition_id,
                suite_section="invariance",
                source_reference="experiments/ev_invariance/request_builder.py",
                success_rule="financial_ev_and_decision",
                financial_case=True,
                condition=condition,
            )
        )
    return cases


def build_cases(model_name: str) -> list[DiagnosticCase]:
    return [
        *build_sanity_cases(model_name),
        *build_canonical_ev_cases(model_name),
        *build_invariance_cases(model_name),
    ]


def select_cases(
    cases: list[DiagnosticCase],
    *,
    suite_sections: set[str] | None = None,
    limit: int | None = None,
) -> list[DiagnosticCase]:
    selected = cases
    if suite_sections:
        selected = [case for case in selected if case.suite_section in suite_sections]
    if limit is not None:
        selected = selected[:limit]
    return selected


def try_parse_json(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalize_token(text: str) -> str:
    return text.strip().strip(" \t\r\n.!?\"'")


def is_single_token(text: str, accepted: set[str]) -> bool:
    tokens = text.split()
    if len(tokens) != 1:
        return False
    return normalize_token(tokens[0]).lower() in accepted


def has_financial_signal(text: str) -> bool:
    lowered = text.lower()
    return (
        "expected value" in lowered
        or re.search(r"\bev\b", lowered) is not None
        or "$" in text
        or re.search(r"\b(take|decline|pass|recommendation|decision)\b", lowered) is not None
    )


def matches_financial_template(text: str) -> bool:
    lowered = text.lower()
    has_ev_phrase = "expected value" in lowered or re.search(r"\bev\b", lowered) is not None
    has_recommendation = "recommendation" in lowered or "i would" in lowered or "decision" in lowered
    return has_ev_phrase and has_recommendation


def classify_exception_kind(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "too many requests" in lowered or "429" in lowered:
        return "rate_limit"
    return "transport_exception"


def detect_off_task_behavior(
    case: DiagnosticCase,
    response_text: str,
    *,
    schema_valid_json: bool | None,
    extraction,
) -> bool:
    stripped = response_text.strip()
    if not stripped:
        return False
    if case.financial_case:
        if schema_valid_json is False:
            return False
        if extraction is not None and (
            extraction.ev_reported is not None or extraction.decision_direction.value != "unknown"
        ):
            return False
        return not has_financial_signal(stripped)
    if matches_financial_template(stripped):
        return True
    if case.success_rule == "single_number" and len(stripped.split()) > 3:
        return True
    if case.success_rule == "exact" and len(stripped.split()) > 6:
        return True
    return False


def classify_generation_mode(
    response_text: str,
    *,
    malformed_output: bool,
    off_task_behavior: bool,
    error_kind: str | None = None,
) -> str:
    if error_kind == "timeout":
        return "timeout"
    if error_kind == "rate_limit":
        return "rate_limit"
    if error_kind == "transport_exception":
        return "transport_exception"
    stripped = response_text.strip()
    if not stripped:
        return "empty"
    if malformed_output:
        return "malformed"
    if off_task_behavior:
        return "off_task"
    if len(stripped.split()) <= 4:
        return "terse"
    return "normal"


def evaluate_success(
    case: DiagnosticCase,
    response_text: str,
    *,
    parsed_json: Any | None,
    extraction,
    schema_valid_json: bool | None,
    malformed_output: bool,
    off_task_behavior: bool,
) -> bool:
    stripped = response_text.strip()
    if not stripped or malformed_output or off_task_behavior:
        return False
    if case.success_rule == "short_non_empty":
        return len(stripped.split()) <= 6 and not has_financial_signal(stripped)
    if case.success_rule == "exact":
        return stripped == case.expected_exact_text
    if case.success_rule == "single_number":
        return is_single_token(stripped, {"4", "four"})
    if case.success_rule == "json_value_ok":
        return isinstance(parsed_json, dict) and parsed_json.get("value") == "OK"
    if case.success_rule == "financial_ev_and_decision":
        return (
            extraction is not None
            and extraction.ev_reported is not None
            and extraction.ev_abs_error is not None
            and extraction.ev_abs_error <= EV_TOLERANCE
            and extraction.sign_correct is True
            and extraction.decision_matches_baseline is True
        )
    if case.success_rule == "financial_json_ev_and_decision":
        return (
            schema_valid_json is True
            and extraction is not None
            and extraction.ev_reported is not None
            and extraction.ev_abs_error is not None
            and extraction.ev_abs_error <= EV_TOLERANCE
            and extraction.sign_correct is True
            and extraction.decision_matches_baseline is True
        )
    raise ValueError(f"Unsupported success rule: {case.success_rule}")


def build_result_row(case: DiagnosticCase, record, *, attempt_index: int) -> dict[str, Any]:
    response_text = record.response_text or ""
    parsed_json = try_parse_json(response_text.strip()) if case.expects_json else None
    schema_valid_json = None if not case.expects_json else parsed_json is not None
    extraction = extract_ev_and_decision(case.condition.latent_task, response_text) if case.financial_case else None
    malformed_output = bool(case.expects_json and response_text.strip() and schema_valid_json is False)
    extraction_failure = None
    decision_parse_failure = None
    extracted_ev = None
    ev_abs_error = None
    sign_correct = None
    extracted_decision = None
    decision_matches_baseline = None
    if extraction is not None:
        extraction_failure = extraction.ev_reported is None
        decision_parse_failure = extraction.decision_direction.value == "unknown"
        extracted_ev = extraction.ev_reported
        ev_abs_error = extraction.ev_abs_error
        sign_correct = extraction.sign_correct
        extracted_decision = extraction.decision_direction.value
        decision_matches_baseline = extraction.decision_matches_baseline

    off_task_behavior = detect_off_task_behavior(
        case,
        response_text,
        schema_valid_json=schema_valid_json,
        extraction=extraction,
    )
    empty_output = not response_text.strip()
    parse_success = not empty_output and not malformed_output and not off_task_behavior
    if case.financial_case:
        parse_success = parse_success and extracted_ev is not None
    condition_success = evaluate_success(
        case,
        response_text,
        parsed_json=parsed_json,
        extraction=extraction,
        schema_valid_json=schema_valid_json,
        malformed_output=malformed_output,
        off_task_behavior=off_task_behavior,
    )

    return {
        "attempt_id": f"{case.case_id}::run{attempt_index}",
        "attempt_index": attempt_index,
        "condition_id": case.condition.condition_id,
        "base_condition_id": case.condition.base_condition_id,
        "suite_section": case.suite_section,
        "source_reference": case.source_reference,
        "prompt_family": case.condition.prompt_family,
        "task_type": case.condition.task_type.value,
        "output_mode": case.condition.surface.output_mode.value,
        "success_rule": case.success_rule,
        "expects_json": case.expects_json,
        "financial_case": case.financial_case,
        "rendered_prompt": case.condition.rendered_prompt,
        "raw_response_text": response_text,
        "response_present": bool(response_text.strip()),
        "empty_output": empty_output,
        "malformed_output": malformed_output,
        "off_task_behavior": off_task_behavior,
        "schema_valid_json": schema_valid_json,
        "extraction_failure": extraction_failure,
        "decision_parse_failure": decision_parse_failure,
        "extracted_ev": extracted_ev,
        "ev_abs_error": ev_abs_error,
        "sign_correct": sign_correct,
        "extracted_decision": extracted_decision,
        "decision_matches_baseline": decision_matches_baseline,
        "latency_ms": record.latency_ms,
        "token_count": record.token_count,
        "activation_available": record.activation_available,
        "activation_score": record.activation_score,
        "activation_delta": record.activation_delta,
        "error_kind": None,
        "error_message": None,
        "transport_exception": False,
        "parse_success": parse_success,
        "condition_success": condition_success,
        "generation_mode": classify_generation_mode(
            response_text,
            malformed_output=malformed_output,
            off_task_behavior=off_task_behavior,
        ),
    }


def build_error_row(
    case: DiagnosticCase,
    *,
    attempt_index: int,
    error_kind: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "attempt_id": f"{case.case_id}::run{attempt_index}",
        "attempt_index": attempt_index,
        "condition_id": case.condition.condition_id,
        "base_condition_id": case.condition.base_condition_id,
        "suite_section": case.suite_section,
        "source_reference": case.source_reference,
        "prompt_family": case.condition.prompt_family,
        "task_type": case.condition.task_type.value,
        "output_mode": case.condition.surface.output_mode.value,
        "success_rule": case.success_rule,
        "expects_json": case.expects_json,
        "financial_case": case.financial_case,
        "rendered_prompt": case.condition.rendered_prompt,
        "raw_response_text": "",
        "response_present": False,
        "empty_output": True,
        "malformed_output": False,
        "off_task_behavior": False,
        "schema_valid_json": None if not case.expects_json else False,
        "extraction_failure": None if not case.financial_case else True,
        "decision_parse_failure": None if not case.financial_case else True,
        "extracted_ev": None,
        "ev_abs_error": None,
        "sign_correct": None,
        "extracted_decision": None,
        "decision_matches_baseline": None,
        "latency_ms": None,
        "token_count": None,
        "activation_available": None,
        "activation_score": None,
        "activation_delta": None,
        "error_kind": error_kind,
        "error_message": error_message,
        "transport_exception": True,
        "parse_success": False,
        "condition_success": False,
        "generation_mode": classify_generation_mode(
            "",
            malformed_output=False,
            off_task_behavior=False,
            error_kind=error_kind,
        ),
    }


def run_condition_with_timeout(runner, condition: ExperimentCondition, timeout_seconds: float):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model3-diagnostic")
    future = executor.submit(runner.run, condition)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_requests = len(results)
    successful_parses = sum(1 for row in results if row["parse_success"])
    condition_successes = sum(1 for row in results if row["condition_success"])
    empty_outputs = sum(1 for row in results if row["empty_output"])
    malformed_outputs = sum(1 for row in results if row["malformed_output"])
    off_task_outputs = sum(1 for row in results if row["off_task_behavior"])
    extraction_failures = sum(1 for row in results if row["extraction_failure"] is True)
    transport_exceptions = sum(1 for row in results if row["transport_exception"])
    timeout_exceptions = sum(1 for row in results if row["error_kind"] == "timeout")
    rate_limit_exceptions = sum(1 for row in results if row["error_kind"] == "rate_limit")

    success_by_condition: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["base_condition_id"]].append(row)
    for condition_id, rows in sorted(grouped.items()):
        successes = sum(1 for row in rows if row["condition_success"])
        parses = sum(1 for row in rows if row["parse_success"])
        success_by_condition[condition_id] = {
            "attempts": len(rows),
            "successful_runs": successes,
            "successful_parses": parses,
            "success_rate": round(successes / len(rows), 4),
            "parse_rate": round(parses / len(rows), 4),
        }

    return {
        "total_requests": total_requests,
        "successful_parses": successful_parses,
        "condition_successes": condition_successes,
        "empty_outputs": empty_outputs,
        "malformed_outputs": malformed_outputs,
        "off_task_outputs": off_task_outputs,
        "extraction_failures": extraction_failures,
        "transport_exceptions": transport_exceptions,
        "timeout_exceptions": timeout_exceptions,
        "rate_limit_exceptions": rate_limit_exceptions,
        "success_rate_by_condition": success_by_condition,
    }


def resolve_output_paths(output_dir: str | Path, model_name: str, tag: str | None) -> dict[str, Path]:
    output_root = Path(output_dir)
    stem = f"model3_diagnostic_{model_name}"
    if tag:
        stem = f"{stem}_{tag}"
    return {
        "output_dir": output_root,
        "dry_run": output_root / f"{stem}_dry_run.json",
        "results_json": output_root / f"{stem}_results.json",
        "results_jsonl": output_root / f"{stem}_results.jsonl",
        "checkpoint_json": output_root / f"{stem}_checkpoint.json",
    }


def load_completed_attempt_ids(results_jsonl_path: Path) -> set[str]:
    if not results_jsonl_path.exists():
        return set()
    return {row["attempt_id"] for row in read_jsonl(results_jsonl_path)}


def persist_outputs(
    *,
    paths: dict[str, Path],
    args,
    cases: list[DiagnosticCase],
    results: list[dict[str, Any]],
) -> None:
    summary = build_summary(results)
    output_paths = {
        "results_json": str(paths["results_json"]),
        "results_jsonl": str(paths["results_jsonl"]),
        "checkpoint_json": str(paths["checkpoint_json"]),
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "copied_from": COPIED_FROM,
        "model_name": args.model,
        "runner_name": args.runner,
        "timeout_seconds": args.timeout,
        "repeats": args.repeats,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "output_paths": output_paths,
        "results": results,
    }
    checkpoint_payload = {
        "experiment_id": EXPERIMENT_ID,
        "model_name": args.model,
        "runner_name": args.runner,
        "completed_requests": len(results),
        "total_requested_cases": len(cases) * args.repeats,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "completed_attempt_ids": [row["attempt_id"] for row in results],
        "output_paths": output_paths,
    }

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["results_jsonl"], results)
    write_json(paths["results_json"], payload)
    write_json(paths["checkpoint_json"], checkpoint_payload)


def emit_dry_run(paths: dict[str, Path], args, cases: list[DiagnosticCase]) -> None:
    preview_rows = []
    for case in cases:
        preview_rows.append(
            {
                "condition_id": case.condition.condition_id,
                "base_condition_id": case.condition.base_condition_id,
                "suite_section": case.suite_section,
                "prompt_family": case.condition.prompt_family,
                "task_type": case.condition.task_type.value,
                "output_mode": case.condition.surface.output_mode.value,
                "success_rule": case.success_rule,
                "expects_json": case.expects_json,
                "financial_case": case.financial_case,
                "source_reference": case.source_reference,
                "rendered_prompt": case.condition.rendered_prompt,
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "copied_from": COPIED_FROM,
        "model_name": args.model,
        "runner_name": args.runner,
        "timeout_seconds": args.timeout,
        "repeats": args.repeats,
        "suite_sections": sorted({case.suite_section for case in cases}),
        "case_count": len(cases),
        "results_expected": len(cases) * args.repeats,
        "cases": preview_rows,
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    write_json(paths["dry_run"], payload)
    print(f"dry_run_cases={len(cases)}")
    print(f"dry_run_output={paths['dry_run']}")


def print_summary(summary: dict[str, Any], paths: dict[str, Path]) -> None:
    print(
        "summary "
        f"total_requests={summary['total_requests']} "
        f"successful_parses={summary['successful_parses']} "
        f"condition_successes={summary['condition_successes']} "
        f"empty_outputs={summary['empty_outputs']} "
        f"malformed_outputs={summary['malformed_outputs']} "
        f"off_task_outputs={summary['off_task_outputs']} "
        f"extraction_failures={summary['extraction_failures']} "
        f"transport_exceptions={summary['transport_exceptions']} "
        f"timeout_exceptions={summary['timeout_exceptions']} "
        f"rate_limit_exceptions={summary['rate_limit_exceptions']}"
    )
    for condition_id, stats in summary["success_rate_by_condition"].items():
        print(
            f"condition={condition_id} "
            f"attempts={stats['attempts']} "
            f"successful_runs={stats['successful_runs']} "
            f"successful_parses={stats['successful_parses']} "
            f"success_rate={stats['success_rate']:.4f}"
        )
    print(f"results_json={paths['results_json']}")
    print(f"results_jsonl={paths['results_jsonl']}")
    print(f"checkpoint_json={paths['checkpoint_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact, isolated diagnostic battery for dormant-model-3."
    )
    parser.add_argument("--runner", default="jsinfer", choices=("stub", "jsinfer"))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=(DEFAULT_MODEL,),
        help="Model name to embed in conditions and pass to the selected runner.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--suite-section",
        action="append",
        choices=("sanity", "canonical_ev", "invariance"),
        default=[],
        help="Optional suite-section filter. Repeat to include multiple sections.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of conditions executed after filtering.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of sequential attempts per condition.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-condition timeout in seconds.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for artifact filenames when running subsets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing results JSONL checkpoint if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and write the diagnostic registry without executing the runner.",
    )
    parser.add_argument("--enable-activations", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided")
    return args


def main() -> None:
    args = parse_args()
    paths = resolve_output_paths(args.output_dir, args.model, args.tag)
    cases = select_cases(
        build_cases(args.model),
        suite_sections=set(args.suite_section),
        limit=args.limit,
    )

    if args.dry_run:
        emit_dry_run(paths, args, cases)
        return

    try:
        runner = get_runner(
            args.runner,
            enable_activations=args.enable_activations,
            model_name=args.model,
        )
    except Exception as exc:
        paths["output_dir"].mkdir(parents=True, exist_ok=True)
        failure_payload = {
            "experiment_id": EXPERIMENT_ID,
            "copied_from": COPIED_FROM,
            "model_name": args.model,
            "runner_name": args.runner,
            "timeout_seconds": args.timeout,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_initialization_error": {
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        }
        write_json(paths["results_json"], failure_payload)
        print(f"runner_initialization_error={exc.__class__.__name__}: {exc}")
        print(f"results_json={paths['results_json']}")
        raise SystemExit(1)

    results: list[dict[str, Any]] = []
    completed_attempt_ids: set[str] = set()
    if args.resume:
        completed_attempt_ids = load_completed_attempt_ids(paths["results_jsonl"])
        if paths["results_json"].exists():
            payload = load_json(paths["results_json"])
            existing_results = payload.get("results", [])
            if isinstance(existing_results, list):
                results.extend(existing_results)
            else:
                results.extend(read_jsonl(paths["results_jsonl"]))

    for case in cases:
        for attempt_index in range(args.repeats):
            attempt_id = f"{case.case_id}::run{attempt_index}"
            if attempt_id in completed_attempt_ids:
                continue
            try:
                record = run_condition_with_timeout(runner, case.condition, args.timeout)
                results.append(build_result_row(case, record, attempt_index=attempt_index))
            except FutureTimeoutError:
                results.append(
                    build_error_row(
                        case,
                        attempt_index=attempt_index,
                        error_kind="timeout",
                        error_message="timeout",
                    )
                )
            except Exception as exc:
                results.append(
                    build_error_row(
                        case,
                        attempt_index=attempt_index,
                        error_kind=classify_exception_kind(exc),
                        error_message=str(exc),
                    )
                )
            persist_outputs(paths=paths, args=args, cases=cases, results=results)

    summary = build_summary(results)
    print_summary(summary, paths)


if __name__ == "__main__":
    main()
