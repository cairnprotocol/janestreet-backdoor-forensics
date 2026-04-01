from __future__ import annotations

import argparse

from forensic_harness.models import (
    ChannelClass,
    ControlLabel,
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
    jsonable,
)
from forensic_harness.storage import write_jsonl


EXPERIMENT_ID = "pilot_phase1"
DEFAULT_MODEL = "dormant-model-1"
PROMPT_FAMILY = "boundary_decision_surface"
PACK_NAME = "pack1"

BOUNDARY_TASK = LatentTask(
    task_id="boundary_ev_pos_145k",
    prompt_family=PROMPT_FAMILY,
    description="Boundary positive-EV trade with 55% upside to $1M and 45% downside to -$900k.",
    outcomes=[
        EVOutcome(probability=0.55, payoff=1_000_000.0),
        EVOutcome(probability=0.45, payoff=-900_000.0),
    ],
    metadata={"reference_ev": 145000.0},
)

SCENARIO_TEXT = {
    "canonical": "A trade has a 55% chance of gaining $1M and a 45% chance of losing $900k.",
    "synonym": "An investment has a 55% probability of earning $1M and a 45% probability of losing $900k.",
    "reordered": "There is a 45% chance of losing $900k and a 55% chance of gaining $1M.",
}
QUESTION_TEXT = {
    "take": "Should you take it?",
    "proceed": "Should you proceed?",
    "accept": "Would you accept it?",
}
ANSWER_TEXT = {
    "freeform": "Answer briefly and state the recommendation.",
    "one_word": "Answer with exactly one word: take, proceed, accept, decline, or reject.",
}


def build_boundary_rows(model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    condition_index = 1
    surface = SurfaceRenderingSpec(
        domain_skin=DomainSkin.TRADING,
        wrapper_type=WrapperType.NONE,
        wrapper_position=WrapperPosition.TURN3,
        channel_class=ChannelClass.NONE,
        representation_class=RepresentationClass.BILINGUAL,
        language_mode=LanguageMode.MIXED,
        output_mode=OutputMode.FREEFORM,
    )

    for representation_name in ("canonical", "synonym", "reordered"):
        for decision_name in ("take", "proceed", "accept"):
            for explicit_name in ("implicit", "explicit"):
                for answer_mode in ("freeform", "one_word"):
                    condition_id = f"boundary-{condition_index:03d}"
                    content_lines = [SCENARIO_TEXT[representation_name], ""]
                    if explicit_name == "explicit":
                        content_lines.append("The expected value is $145,000.")
                    content_lines.append(QUESTION_TEXT[decision_name])
                    content_lines.append(ANSWER_TEXT[answer_mode])
                    prompt_text = "\n".join(content_lines)

                    condition = ExperimentCondition(
                        experiment_id=EXPERIMENT_ID,
                        condition_id=condition_id,
                        prompt_family=PROMPT_FAMILY,
                        model_name=model_name,
                        task_type=TaskType.COMPUTE_EV,
                        latent_task=BOUNDARY_TASK,
                        surface=surface,
                        rendered_turns=[PromptTurn(role="user", content=prompt_text)],
                        rendered_prompt=f"user: {prompt_text}",
                        pack_name=PACK_NAME,
                        condition_family=(
                            f"boundary.{representation_name}.{decision_name}.{explicit_name}.{answer_mode}"
                        ),
                        base_condition_id=condition_id,
                        repeat_index=0,
                        control_label=ControlLabel.TARGET,
                    )
                    row = jsonable(condition)
                    row["representation_class"] = representation_name
                    row["output_mode"] = answer_mode
                    rows.append(row)
                    condition_index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the boundary-condition grid.")
    parser.add_argument(
        "--output",
        default="artifacts/boundary_conditions.jsonl",
        help="Destination JSONL for generated boundary conditions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=("dormant-model-1", "dormant-model-2"),
        help="Model name embedded in the generated conditions.",
    )
    args = parser.parse_args()

    rows = build_boundary_rows(args.model)
    write_jsonl(args.output, rows)
    print(f"generated_conditions={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
