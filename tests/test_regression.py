from agenteval.adapters import RuleBasedAgent
from agenteval.cli import main
from agenteval.evaluator import Evaluator, load_goldens
from agenteval.graders import Contains, ToolTrajectory
from agenteval.regression import compare
from agenteval.store import RunStore
from agenteval.types import CaseResult, GradeResult, AgentOutput, RunResult

GOLDENS = "goldens/support_agent.jsonl"


def make_run(run_id: str, scores: dict[str, float]) -> RunResult:
    """Synthesize a run so comparison logic can be tested in isolation."""
    cases = [
        CaseResult(
            case_id=cid,
            tags=[],
            output=AgentOutput(text=""),
            grades=[GradeResult(grader="g", score=s, passed=s >= 1.0)],
        )
        for cid, s in scores.items()
    ]
    return RunResult(run_id=run_id, agent="test", created_at="2026-01-01T00:00:00Z", cases=cases)


class TestCompare:
    def test_identical_runs_are_ok(self):
        a = make_run("a", {"c1": 1.0, "c2": 1.0})
        b = make_run("b", {"c1": 1.0, "c2": 1.0})
        rep = compare(a, b)
        assert rep.ok and not rep.regressions and not rep.improvements

    def test_detects_regression(self):
        rep = compare(make_run("a", {"c1": 1.0}), make_run("b", {"c1": 0.0}))
        assert not rep.ok
        assert [d.case_id for d in rep.regressions] == ["c1"]
        assert rep.pass_rate_delta < 0

    def test_detects_improvement(self):
        rep = compare(make_run("a", {"c1": 0.0}), make_run("b", {"c1": 1.0}))
        assert rep.ok
        assert [d.case_id for d in rep.improvements] == ["c1"]

    def test_tolerance_does_not_excuse_a_hard_regression(self):
        """Tolerance softens the aggregate, but any case flipping pass->fail still fails."""
        rep = compare(make_run("a", {"c1": 1.0}), make_run("b", {"c1": 0.0}), tolerance=0.5)
        assert not rep.ok

    def test_tracks_added_and_removed_cases(self):
        a = make_run("a", {"c1": 1.0, "gone": 1.0})
        b = make_run("b", {"c1": 1.0, "new": 1.0})
        rep = compare(a, b)
        assert rep.added == ["new"] and rep.removed == ["gone"]

    def test_new_cases_do_not_count_as_regressions(self):
        rep = compare(make_run("a", {"c1": 1.0}), make_run("b", {"c1": 1.0, "new": 0.0}))
        assert not rep.regressions

    def test_summary_mentions_regressed_case(self):
        rep = compare(make_run("a", {"bad": 1.0}), make_run("b", {"bad": 0.0}))
        s = rep.summary()
        assert "REGRESSED" in s and "bad" in s and "FAIL" in s

    def test_report_serializes(self):
        rep = compare(make_run("a", {"c1": 1.0}), make_run("b", {"c1": 0.0}))
        d = rep.to_dict()
        assert d["ok"] is False and len(d["regressions"]) == 1


class TestEndToEnd:
    def test_skipping_tools_is_caught_as_regression(self, tmp_path):
        """Full loop: baseline a healthy agent, then a broken one must fail the gate."""
        cases = load_goldens(GOLDENS)
        graders = [Contains(), ToolTrajectory()]
        store = RunStore(tmp_path)

        baseline = Evaluator(RuleBasedAgent(), graders=graders).run(cases)
        store.save(baseline, as_baseline=True)

        broken = Evaluator(RuleBasedAgent(skip_tools=True), graders=graders).run(cases)
        rep = compare(store.load_baseline(), broken)

        assert not rep.ok
        assert len(rep.regressions) > 0


class TestCLI:
    def test_run_and_gate_exit_codes(self, tmp_path, capsys):
        runs = str(tmp_path)
        rc = main(["run", "--goldens", GOLDENS,
                   "--agent", "agenteval.adapters:RuleBasedAgent",
                   "--runs-dir", runs, "--set-baseline"])
        assert rc == 0
        assert "pass rate" in capsys.readouterr().out

        rc = main(["gate", "--goldens", GOLDENS,
                   "--agent", "agenteval.adapters:RuleBasedAgent",
                   "--runs-dir", runs])
        assert rc == 0
        assert "RESULT: PASS" in capsys.readouterr().out

    def test_gate_without_baseline_is_not_an_error(self, tmp_path, capsys):
        rc = main(["gate", "--goldens", GOLDENS,
                   "--agent", "agenteval.adapters:RuleBasedAgent",
                   "--runs-dir", str(tmp_path)])
        assert rc == 0
        assert "no baseline stored" in capsys.readouterr().out

    def test_bad_agent_spec_is_rejected(self, tmp_path):
        import pytest
        with pytest.raises(SystemExit):
            main(["run", "--goldens", GOLDENS, "--agent", "agenteval.adapters:Nope",
                  "--runs-dir", str(tmp_path)])
