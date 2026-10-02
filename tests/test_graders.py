import pytest

from agenteval.graders import (
    Contains, ExactMatch, FuzzyMatch, LatencyBudget, LLMJudge,
    NoError, RegexMatch, ToolTrajectory, ValidJSON,
)
from agenteval.types import AgentOutput, TestCase, ToolCall


def case(**kw):
    base = dict(id="c1", input="q")
    base.update(kw)
    return TestCase(**base)


def out(text="", tools=(), latency=0.0, error=None):
    return AgentOutput(
        text=text,
        tool_calls=[ToolCall(name=t) for t in tools],
        latency_ms=latency,
        error=error,
    )


class TestExactMatch:
    def test_matches_ignoring_case_and_whitespace(self):
        g = ExactMatch().grade(case(expected="Hello  World"), out("hello world"))
        assert g.passed and g.score == 1.0

    def test_rejects_different_text(self):
        g = ExactMatch().grade(case(expected="yes"), out("no"))
        assert not g.passed and "expected" in g.detail

    def test_case_sensitive_mode(self):
        g = ExactMatch(case_sensitive=True).grade(case(expected="Yes"), out("yes"))
        assert not g.passed

    def test_skips_when_no_expectation(self):
        assert ExactMatch().grade(case(), out("anything")).passed


class TestContains:
    def test_all_needles_present(self):
        c = case(metadata={"contains": ["alpha", "beta"]})
        assert Contains().grade(c, out("Alpha and BETA")).score == 1.0

    def test_partial_credit(self):
        c = case(metadata={"contains": ["alpha", "beta"]})
        g = Contains().grade(c, out("only alpha"))
        assert g.score == 0.5 and not g.passed and "beta" in g.detail

    def test_falls_back_to_expected(self):
        assert Contains().grade(case(expected="needle"), out("a needle here")).passed


class TestRegexMatch:
    def test_matches(self):
        c = case(metadata={"pattern": r"\d{3}-\d{4}"})
        assert RegexMatch().grade(c, out("call 555-1234")).passed

    def test_no_match(self):
        c = case(metadata={"pattern": r"\d{3}-\d{4}"})
        assert not RegexMatch().grade(c, out("no phone")).passed

    def test_skips_without_pattern(self):
        assert RegexMatch().grade(case(), out("x")).passed


class TestFuzzyMatch:
    def test_close_enough(self):
        g = FuzzyMatch(threshold=0.7).grade(
            case(expected="the cat sat on the mat"), out("the cat sat on a mat")
        )
        assert g.passed

    def test_too_different(self):
        g = FuzzyMatch(threshold=0.9).grade(case(expected="alpha"), out("completely other"))
        assert not g.passed


class TestToolTrajectory:
    def test_unordered_all_present(self):
        c = case(expected_tools=["a", "b"])
        assert ToolTrajectory().grade(c, out(tools=["b", "a"])).passed

    def test_partial_credit_for_missing_tool(self):
        c = case(expected_tools=["a", "b"])
        g = ToolTrajectory().grade(c, out(tools=["a"]))
        assert g.score == 0.5 and "missing tools" in g.detail

    def test_flags_unexpected_tools(self):
        c = case(expected_tools=["a"])
        g = ToolTrajectory().grade(c, out(tools=["a", "danger"]))
        assert g.passed and "unexpected tools" in g.detail

    def test_ordered_requires_sequence(self):
        c = case(expected_tools=["a", "b"])
        assert ToolTrajectory(ordered=True).grade(c, out(tools=["a", "b"])).passed
        assert not ToolTrajectory(ordered=True).grade(c, out(tools=["b", "a"])).passed

    def test_dropped_tool_call_is_caught(self):
        """The regression this harness exists to catch."""
        c = case(expected_tools=["lookup_order"])
        assert not ToolTrajectory().grade(c, out(text="plausible answer", tools=[])).passed


class TestNoError:
    def test_passes_clean(self):
        assert NoError().grade(case(), out("fine")).passed

    def test_fails_on_error(self):
        g = NoError().grade(case(), out(error="Timeout"))
        assert not g.passed and "Timeout" in g.detail


class TestValidJSON:
    def test_valid_with_required_keys(self):
        c = case(metadata={"expect_json": True, "required_keys": ["a", "b"]})
        assert ValidJSON().grade(c, out('{"a":1,"b":2}')).passed

    def test_invalid_json(self):
        c = case(metadata={"expect_json": True})
        g = ValidJSON().grade(c, out("{not json"))
        assert not g.passed and "invalid JSON" in g.detail

    def test_missing_keys_partial(self):
        c = case(metadata={"expect_json": True, "required_keys": ["a", "b"]})
        g = ValidJSON().grade(c, out('{"a":1}'))
        assert g.score == 0.5 and "missing keys" in g.detail

    def test_skipped_when_not_json_case(self):
        assert ValidJSON().grade(case(), out("prose")).passed


class TestLatencyBudget:
    def test_within_budget(self):
        assert LatencyBudget(1000).grade(case(), out(latency=500)).passed

    def test_over_budget(self):
        assert not LatencyBudget(1000).grade(case(), out(latency=1500)).passed

    def test_per_case_override(self):
        c = case(metadata={"latency_budget_ms": 100})
        assert not LatencyBudget(10_000).grade(c, out(latency=500)).passed


class TestLLMJudge:
    def test_normalizes_score(self):
        g = LLMJudge(lambda p: 8.0).grade(case(expected="ref"), out("answer"))
        assert g.score == pytest.approx(0.8) and g.passed

    def test_low_score_fails(self):
        assert not LLMJudge(lambda p: 2.0).grade(case(expected="ref"), out("bad")).passed

    def test_judge_exception_does_not_crash_run(self):
        def boom(_):
            raise RuntimeError("rate limited")

        g = LLMJudge(boom).grade(case(expected="ref"), out("x"))
        assert not g.passed and "judge failed" in g.detail

    def test_prompt_includes_all_three_parts(self):
        seen = {}

        def spy(prompt):
            seen["p"] = prompt
            return 10.0

        LLMJudge(spy).grade(case(input="Q?", expected="REF"), out("ACT"))
        assert "Q?" in seen["p"] and "REF" in seen["p"] and "ACT" in seen["p"]
