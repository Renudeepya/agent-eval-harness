import json

import pytest

from agenteval.adapters import CallableAgent, RuleBasedAgent
from agenteval.evaluator import Evaluator, load_goldens
from agenteval.graders import Contains, NoError, ToolTrajectory
from agenteval.store import RunStore
from agenteval.types import AgentOutput, TestCase

GOLDENS = "goldens/support_agent.jsonl"


class TestLoadGoldens:
    def test_loads_real_file(self):
        cases = load_goldens(GOLDENS)
        assert len(cases) == 10
        assert all(isinstance(c, TestCase) for c in cases)

    def test_skips_comments_and_blanks(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('# note\n\n{"id":"a","input":"x"}\n', encoding="utf-8")
        assert len(load_goldens(p)) == 1

    def test_rejects_duplicate_ids(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('{"id":"a","input":"x"}\n{"id":"a","input":"y"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate case ids"):
            load_goldens(p)

    def test_rejects_empty_file(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text("# only a comment\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no test cases"):
            load_goldens(p)

    def test_reports_line_number_on_bad_row(self, tmp_path):
        p = tmp_path / "g.jsonl"
        p.write_text('{"id":"a","input":"x"}\n{oops\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            load_goldens(p)


class TestEvaluator:
    def test_healthy_agent_passes_everything(self):
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        assert run.pass_rate == 1.0
        assert run.mean_score == 1.0

    def test_agent_exception_is_captured_not_raised(self):
        class Boom:
            name = "boom"

            def run(self, prompt):
                raise RuntimeError("upstream down")

        run = Evaluator(Boom(), graders=[NoError()]).run(
            [TestCase(id="a", input="x")]
        )
        assert run.pass_rate == 0.0
        assert "upstream down" in run.cases[0].output.error

    def test_latency_is_recorded(self):
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        assert all(c.output.latency_ms >= 0 for c in run.cases)
        assert run.p95_latency_ms >= 0

    def test_by_tag_breakdown(self):
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        tags = run.by_tag()
        assert "refunds" in tags and "billing" in tags
        assert all(0.0 <= v <= 1.0 for v in tags.values())

    def test_worst_of_n_with_repeats(self):
        """A flaky agent should be scored by its worst attempt, not its luckiest."""
        flaky = RuleBasedAgent(failure_rate=0.5, seed=1)
        cases = load_goldens(GOLDENS)
        strict = Evaluator(flaky, repeats=5, max_workers=1).run(cases)
        lenient = Evaluator(RuleBasedAgent(), repeats=1, max_workers=1).run(cases)
        assert strict.pass_rate < lenient.pass_rate

    def test_serial_and_parallel_agree(self):
        cases = load_goldens(GOLDENS)
        a = Evaluator(RuleBasedAgent(), max_workers=1).run(cases)
        b = Evaluator(RuleBasedAgent(), max_workers=4).run(cases)
        assert a.pass_rate == b.pass_rate
        assert {c.case_id for c in a.cases} == {c.case_id for c in b.cases}

    def test_callable_agent_adapter(self):
        agent = CallableAgent(lambda p: "always alpha", name="fn")
        run = Evaluator(agent, graders=[Contains()]).run(
            [TestCase(id="a", input="x", metadata={"contains": ["alpha"]})]
        )
        assert run.pass_rate == 1.0
        assert run.agent == "fn"


class TestRegressionDetection:
    def test_dropped_tool_calls_lower_the_score(self):
        cases = load_goldens(GOLDENS)
        healthy = Evaluator(RuleBasedAgent(), graders=[ToolTrajectory()]).run(cases)
        broken = Evaluator(RuleBasedAgent(skip_tools=True), graders=[ToolTrajectory()]).run(cases)
        assert healthy.pass_rate > broken.pass_rate


class TestRunStore:
    def test_round_trip_preserves_results(self, tmp_path):
        store = RunStore(tmp_path)
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        path = store.save(run, as_baseline=True)

        loaded = store.load(path)
        assert loaded.run_id == run.run_id
        assert loaded.pass_rate == run.pass_rate
        assert len(loaded.cases) == len(run.cases)
        assert loaded.cases[0].output.tool_calls == run.cases[0].output.tool_calls

    def test_baseline_helpers(self, tmp_path):
        store = RunStore(tmp_path)
        assert store.load_baseline() is None
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        store.save(run, as_baseline=True)
        assert store.load_baseline().run_id == run.run_id

    def test_history_excludes_baseline(self, tmp_path):
        store = RunStore(tmp_path)
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        store.save(run, as_baseline=True)
        assert all(p.name != "baseline.json" for p in store.history())

    def test_output_is_valid_json(self, tmp_path):
        store = RunStore(tmp_path)
        run = Evaluator(RuleBasedAgent()).run(load_goldens(GOLDENS))
        data = json.loads(store.save(run).read_text())
        assert {"run_id", "pass_rate", "by_tag", "cases"} <= data.keys()
