from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from rs.llm.validator import ValidationResult, validate_command
from rs.utils.type_utils import is_int_string


@dataclass
class ValidationFeedback:
    rejected_command: str
    code: str
    message: str
    corrective_hint: str
    valid_index_range: str
    valid_example: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def validate_proposed_command(
        context: Any,
        proposed_command: str,
) -> tuple[ValidationResult, Optional[dict[str, Any]]]:
    command = str(proposed_command).strip()
    choose_feedback = _validate_choose_requires_index(context, command)
    if choose_feedback is not None:
        return (
            ValidationResult(False, choose_feedback.code, choose_feedback.message),
            choose_feedback.to_payload(),
        )

    result = validate_command(context, command)
    if result.is_valid:
        return result, None

    feedback = _build_feedback(context, command, result.code, result.message)
    return result, feedback.to_payload()


def _validate_choose_requires_index(context: Any, command: str) -> Optional[ValidationFeedback]:
    tokens = command.split()
    if not tokens or tokens[0] != "choose":
        return None
    if len(tokens) != 2:
        return _build_feedback(
            context,
            command,
            "choose_requires_index",
            "choose must be in format 'choose <index>'",
        )

    choice_token = tokens[1]
    if not is_int_string(choice_token):
        return _build_feedback(
            context,
            command,
            "choose_requires_index",
            "choose argument must be an integer index",
        )

    choice_count = len(getattr(context, "choice_list", []))
    idx = int(choice_token)
    if idx < 0 or idx >= choice_count:
        return _build_feedback(
            context,
            command,
            "choose_index_out_of_range",
            "choose index out of range",
        )
    return None


def _build_feedback(context: Any, rejected_command: str, code: str, message: str) -> ValidationFeedback:
    choice_count = len(getattr(context, "choice_list", []))
    if choice_count <= 0:
        valid_index_range = "none"
        valid_example = "none"
    else:
        valid_index_range = f"0..{choice_count - 1}"
        valid_example = "choose 0"

    return ValidationFeedback(
        rejected_command=rejected_command,
        code=code,
        message=message,
        corrective_hint=(
            "For choose commands, return only choose <index>. "
            f"Use an index within {valid_index_range}. Example: {valid_example}."
        ),
        valid_index_range=valid_index_range,
        valid_example=valid_example,
    )
