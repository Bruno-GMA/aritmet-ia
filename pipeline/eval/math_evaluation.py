"""Checks arithmetic shown in a pedagogical analogous example.

This deliberately evaluates equations written in the answer, never compares an
analogous example's result with the original question's reference answer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from sympy import simplify, sympify

MAX_EXPRESSION_CHARS = 200
_EXPRESSION = re.compile(r"^[0-9+\-*/^().,\s]+$")
_EQUATION = re.compile(r"(?m)([0-9+\-*/^().,\s]{1,200})\s*=\s*([0-9+\-*/^().,\s]{1,200})")


def _parse(expression: str) -> Any | None:
    expression = expression.strip().replace("^", "**").replace(",", ".")
    if (
        not expression
        or len(expression) > MAX_EXPRESSION_CHARS
        or _EXPRESSION.fullmatch(expression) is None
    ):
        return None
    try:
        return sympify(expression, evaluate=True)
    except (SyntaxError, TypeError, ValueError):
        return None


def evaluate_analogous_answer(answer: str, *, atol: float = 1e-9) -> dict[str, Any]:
    """Return coverage and equation-level validation for an LLM answer."""
    equations: list[dict[str, Any]] = []
    for match in _EQUATION.finditer(answer):
        left, right = match.group(1).strip(), match.group(2).strip()
        left_value, right_value = _parse(left), _parse(right)
        if left_value is None or right_value is None:
            equations.append({"left": left, "right": right, "status": "not_evaluable"})
            continue
        try:
            difference = simplify(left_value - right_value)
            correct = difference == 0 or abs(float(difference)) <= atol
        except (TypeError, ValueError):
            correct = False
        equations.append(
            {"left": left, "right": right, "status": "correct" if correct else "incorrect"}
        )
    counts = Counter(item["status"] for item in equations)
    evaluable = counts["correct"] + counts["incorrect"]
    return {
        "status": "not_evaluable"
        if not evaluable
        else "correct"
        if not counts["incorrect"]
        else "incorrect",
        "scope": "equations_in_analogous_example",
        "equations": equations,
        "total_equations": len(equations),
        "evaluable": evaluable,
        "not_evaluable": counts["not_evaluable"],
        "coverage": evaluable / len(equations) if equations else None,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "atol": atol,
    }


def aggregate_math_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_equations = sum(row["total_equations"] for row in rows)
    evaluable = sum(row["evaluable"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "scope": "equations_in_analogous_example",
        "n_answers": len(rows),
        "n_evaluable_answers": sum(row["status"] != "not_evaluable" for row in rows),
        "total_equations": total_equations,
        "evaluable": evaluable,
        "correct": correct,
        "accuracy": correct / evaluable if evaluable else None,
        "coverage": evaluable / total_equations if total_equations else None,
        "not_evaluable": sum(row["not_evaluable"] for row in rows),
        "energy_or_gpu_measurement": None,
        "limitation": (
            "Only explicit arithmetic equations are checked; prose and pedagogy require "
            "judge review."
        ),
    }
