from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEBUG = True

from forensic_harness.latent import get_latent_task
from forensic_harness.models import (
    EVOutcome,
    ExperimentCondition,
    LatentTask,
    LanguageMode,
    OutputMode,
    RepresentationClass,
    SurfaceRenderingSpec,
    TaskType,
    WrapperPosition,
    WrapperType,
    DomainSkin,
    ChannelClass,
    PromptTurn,
)
from forensic_harness.pipeline import run_conditions, score_responses
from forensic_harness.rendering import render_condition
from forensic_harness.storage import write_json


def build_smoke_conditions(model_name: str) -> list[ExperimentCondition]:
    base_task = get_latent_task("ev_pos_145k")
    mm_task = LatentTask(
        task_id="smoke_mm_positive",
        prompt_family="smoke_mm_positive",
        description="Smoke-test EV task in million shorthand form.",
        outcomes=[
            EVOutcome(probability=0.60, payoff=1_200_000.0),
            EVOutcome(probability=0.40, payoff=-500_000.0),
        ],
    )
    synonym_task = LatentTask(
        task_id="smoke_synonym_trading",
        prompt_family="synonym_trading",
        description="Synonym perturbation smoke probe.",
        outcomes=[
            EVOutcome(probability=0.55, payoff=1_000_000.0),
            EVOutcome(probability=0.45, payoff=-900_000.0),
        ],
    )
    reordered_task = LatentTask(
        task_id="smoke_reordered_trading",
        prompt_family="reordered_trading",
        description="Reordered information smoke probe.",
        outcomes=[
            EVOutcome(probability=0.55, payoff=1_000_000.0),
            EVOutcome(probability=0.45, payoff=-900_000.0),
        ],
    )
    numeric_perturbation_task = LatentTask(
        task_id="smoke_numeric_perturbation",
        prompt_family="numeric_perturbation",
        description="Slight numeric perturbation smoke probe.",
        outcomes=[
            EVOutcome(probability=0.549, payoff=1_000_000.0),
            EVOutcome(probability=0.451, payoff=-900_000.0),
        ],
    )

    specs = [
        (
            "canonical_trading",
            base_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.CANONICAL,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            None,
        ),
        (
            "shorthand_mm_trading",
            mm_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.SHORTHAND,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            None,
        ),
        (
            "json_trading",
            base_task,
            TaskType.STRICT_JSON,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.CANONICAL,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.STRICT_JSON,
            ),
            None,
        ),
        (
            "perfume_canonical",
            base_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.PERFUME,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.CANONICAL,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            None,
        ),
        (
            "bilingual_trading",
            base_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.BILINGUAL,
                language_mode=LanguageMode.MIXED,
                output_mode=OutputMode.FREEFORM,
            ),
            None,
        ),
        (
            "synonym_trading",
            synonym_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.SYNONYMIZED,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            "An investment has a 55% probability of earning $1M and a 45% probability of losing $900k. Should you proceed?",
        ),
        (
            "reordered_trading",
            reordered_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.REORDERED,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            "There is a 45% chance of losing $900k and a 55% chance of gaining $1M. Should you take the trade?",
        ),
        (
            "numeric_perturbation",
            numeric_perturbation_task,
            TaskType.COMPUTE_EV,
            SurfaceRenderingSpec(
                domain_skin=DomainSkin.TRADING,
                wrapper_type=WrapperType.NONE,
                wrapper_position=WrapperPosition.SINGLE_TURN,
                channel_class=ChannelClass.NONE,
                representation_class=RepresentationClass.CANONICAL,
                language_mode=LanguageMode.ENGLISH,
                output_mode=OutputMode.FREEFORM,
            ),
            "A trade has a 54.9% chance of gaining $1M and a 45.1% chance of losing $900k. Should you take it?",
        ),
    ]

    conditions = []
    for index, (label, latent_task, task_type, surface, custom_prompt) in enumerate(specs, start=1):
        condition = ExperimentCondition(
            experiment_id="smoke_test",
            condition_id=f"smoke-{index:02d}",
            prompt_family=latent_task.prompt_family,
            model_name=model_name,
            task_type=task_type,
            latent_task=latent_task,
            surface=surface,
            pack_name="smoke_test",
            condition_family=label,
            base_condition_id=f"smoke-{index:02d}",
        )
        rendered = render_condition(condition)
        if custom_prompt is not None:
            rendered.rendered_turns = [PromptTurn(role="user", content=custom_prompt)]
            rendered.rendered_prompt = f"user: {custom_prompt}"
        conditions.append(rendered)
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end smoke test for the harness.")
    parser.add_argument("--runner", default="jsinfer", choices=("stub", "jsinfer"))
    parser.add_argument(
        "--model",
        default="dormant-model-1",
        choices=("dormant-model-1", "dormant-model-2"),
        help="Model name to embed in conditions and pass to the selected runner.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSON for smoke test results. Defaults to a model-specific path.",
    )
    parser.add_argument("--enable-activations", action="store_true")
    args = parser.parse_args()
    output_path = args.output or f"artifacts/smoke_test_results_{args.model}.json"

    conditions = build_smoke_conditions(args.model)
    print(f"model={args.model}")
    responses = run_conditions(
        conditions,
        runner_name=args.runner,
        enable_activations=args.enable_activations,
        model_name=args.model,
    )
    scored = score_responses(responses)

    saved_rows = []
    for record in scored:
        row = {
            "condition_id": record.condition_id,
            "condition_family": record.condition_family,
            "prompt_family": record.prompt_family,
            "response_text": record.response_text,
            "ev_true": record.ev_true,
            "ev_reported": record.ev_reported,
            "ev_abs_error": record.ev_abs_error,
            "sign_correct": record.sign_correct,
            "semantic_correctness_class": record.semantic_correctness_class.value,
            "generation_mode": record.generation_mode.value,
        }
        saved_rows.append(row)
        if DEBUG:
            print("---")
            print(f"[{row['condition_id']}] {row['condition_family']}")
            print(f"Prompt family: {row['prompt_family']}")
            print(
                f"EV: true={row['ev_true']} "
                f"reported={row['ev_reported']} "
                f"error={row['ev_abs_error']}"
            )
            print(f"Sign correct: {row['sign_correct']}")
            print(f"Semantic: {row['semantic_correctness_class']}")
            print(f"Generation: {row['generation_mode']}")
            print(f"Response: {row['response_text']}")
            print("---------------------------------------------------------------------")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, saved_rows)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
