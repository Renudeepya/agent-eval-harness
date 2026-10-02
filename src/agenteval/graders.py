"""Graders score one AgentOutput against one TestCase.

Every grader returns a normalized 0.0-1.0 score plus a pass/fail verdict, so
heterogeneous checks (string match, tool trajectory, JSON validity, LLM judge)
can be averaged into a single per-case score.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any, Callable

from .types import AgentOutput, GradeResult, TestCase


class Grader(ABC):
    """Base class. Subclasses implement `grade`."""

    name: str = "grader"
    threshold: float = 1.0

    @abstractmethod
    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult: ...

    def _result(self, score: float, detail: str = "") -> GradeResult:
        score = max(0.0, min(1.0, score))
        return GradeResult(
            grader=self.name,
            score=score,
            passed=score >= self.threshold,
            detail=detail,
        )


class ExactMatch(Grader):
    """Strict equality after normalizing whitespace and case."""

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    @staticmethod
    def _norm(s: str, case_sensitive: bool) -> str:
        s = " ".join(s.split())
        return s if case_sensitive else s.lower()

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        if case.expected is None:
            return self._result(1.0, "no expected value; skipped")
        got = self._norm(output.text, self.case_sensitive)
        want = self._norm(case.expected, self.case_sensitive)
        ok = got == want
        return self._result(1.0 if ok else 0.0, "" if ok else f"expected {want!r}, got {got!r}")


class Contains(Grader):
    """Passes when every required substring appears in the output."""

    name = "contains"

    def __init__(self, *, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        needles = case.metadata.get("contains") or ([case.expected] if case.expected else [])
        if not needles:
            return self._result(1.0, "nothing to check; skipped")
        hay = output.text if self.case_sensitive else output.text.lower()
        hits = [n for n in needles if (n if self.case_sensitive else n.lower()) in hay]
        score = len(hits) / len(needles)
        missing = [n for n in needles if n not in hits]
        return self._result(score, "" if not missing else f"missing: {missing}")


class RegexMatch(Grader):
    """Passes when the output matches a pattern supplied on the case."""

    name = "regex"

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        pattern = case.metadata.get("pattern")
        if not pattern:
            return self._result(1.0, "no pattern; skipped")
        ok = re.search(pattern, output.text, re.IGNORECASE | re.DOTALL) is not None
        return self._result(1.0 if ok else 0.0, "" if ok else f"no match for /{pattern}/")


class FuzzyMatch(Grader):
    """Character-level similarity — a cheap stand-in for semantic scoring.

    Useful when the agent is allowed to paraphrase but must stay close.
    """

    name = "fuzzy_match"

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        if case.expected is None:
            return self._result(1.0, "no expected value; skipped")
        ratio = SequenceMatcher(
            None, " ".join(output.text.lower().split()), " ".join(case.expected.lower().split())
        ).ratio()
        return self._result(ratio, f"similarity={ratio:.3f}")


class ToolTrajectory(Grader):
    """Scores the *sequence of tools* the agent chose.

    This is the check that actually catches agent regressions: output text can
    stay plausible while the agent quietly stops calling the tool that grounds
    it. `ordered=True` requires the exact sequence; otherwise set overlap.
    """

    name = "tool_trajectory"

    def __init__(self, ordered: bool = False, threshold: float = 1.0):
        self.ordered = ordered
        self.threshold = threshold

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        want = case.expected_tools
        if not want:
            return self._result(1.0, "no expected tools; skipped")
        got = [t.name for t in output.tool_calls]

        if self.ordered:
            ok = got[: len(want)] == want
            return self._result(1.0 if ok else 0.0, "" if ok else f"expected order {want}, got {got}")

        missing = [t for t in want if t not in got]
        score = (len(want) - len(missing)) / len(want)
        extra = [t for t in got if t not in want]
        detail = ""
        if missing:
            detail += f"missing tools: {missing} "
        if extra:
            detail += f"unexpected tools: {extra}"
        return self._result(score, detail.strip())


class NoError(Grader):
    """Fails the case when the agent raised."""

    name = "no_error"

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        ok = output.error is None
        return self._result(1.0 if ok else 0.0, "" if ok else f"agent error: {output.error}")


class ValidJSON(Grader):
    """Asserts the output parses as JSON, optionally with required keys."""

    name = "valid_json"

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        if not case.metadata.get("expect_json"):
            return self._result(1.0, "not a JSON case; skipped")
        try:
            parsed = json.loads(output.text)
        except json.JSONDecodeError as e:
            return self._result(0.0, f"invalid JSON: {e}")
        required = case.metadata.get("required_keys", [])
        if not required:
            return self._result(1.0)
        if not isinstance(parsed, dict):
            return self._result(0.0, "JSON is not an object")
        present = [k for k in required if k in parsed]
        score = len(present) / len(required)
        missing = [k for k in required if k not in present]
        return self._result(score, "" if not missing else f"missing keys: {missing}")


class LatencyBudget(Grader):
    """Fails cases slower than a budget — performance is a correctness concern."""

    name = "latency_budget"

    def __init__(self, budget_ms: float = 5000.0):
        self.budget_ms = budget_ms

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        budget = case.metadata.get("latency_budget_ms", self.budget_ms)
        ok = output.latency_ms <= budget
        return self._result(
            1.0 if ok else 0.0,
            "" if ok else f"{output.latency_ms:.0f}ms over {budget:.0f}ms budget",
        )


class LLMJudge(Grader):
    """LLM-as-judge, with the model call injected.

    The harness never imports a vendor SDK. Pass any callable that takes a
    prompt and returns a score in 0-10; that keeps this testable offline and
    swappable between providers.
    """

    name = "llm_judge"

    PROMPT = (
        "You are grading an AI agent's response.\n\n"
        "User request:\n{input}\n\n"
        "Reference answer:\n{expected}\n\n"
        "Agent response:\n{actual}\n\n"
        "Score 0-10 for factual accuracy and helpfulness. Reply with the number only."
    )

    def __init__(self, judge_fn: Callable[[str], float], threshold: float = 0.7):
        self.judge_fn = judge_fn
        self.threshold = threshold

    def grade(self, case: TestCase, output: AgentOutput) -> GradeResult:
        if case.expected is None:
            return self._result(1.0, "no reference answer; skipped")
        prompt = self.PROMPT.format(
            input=case.input, expected=case.expected, actual=output.text
        )
        try:
            raw = float(self.judge_fn(prompt))
        except Exception as e:  # a judge failure must not crash the run
            return self._result(0.0, f"judge failed: {e}")
        return self._result(raw / 10.0, f"judge={raw:.1f}/10")


DEFAULT_GRADERS: list[Grader] = [NoError(), Contains(), ToolTrajectory(), ValidJSON()]
