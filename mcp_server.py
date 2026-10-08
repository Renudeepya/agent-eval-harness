"""MCP server exposing the agenteval harness as tools.

Lets an MCP client (Claude Desktop, an IDE, any other host) drive the full
regression workflow: list golden sets, evaluate an agent, record a baseline,
and gate a change against that baseline.
"""
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from agenteval import Evaluator, RunStore, compare, load_goldens
from agenteval.cli import load_agent

server = MCPServer(
    name="agenteval",
    version="0.2.0",
    instructions=(
        "Run agent evaluation suites. Typical flow: list_golden_sets to see what "
        "is available, run_eval with set_baseline=true to record a known-good run, "
        "then gate after a change to find out whether anything regressed. "
        "Grading covers both answer text and the sequence of tools the agent called."
    ),
)

ROOT = Path(__file__).parent
GOLDENS_DIR = ROOT / "goldens"
RUNS_DIR = ROOT / "runs"

DEFAULT_AGENT = "agenteval.adapters:RuleBasedAgent"


def _failures(run) -> list[dict[str, Any]]:
    """Failing cases only, with the reasons graders actually docked points for."""
    return [
        {
            "case_id": c.case_id,
            "score": round(c.score, 3),
            "reasons": [g.detail for g in c.grades if g.detail and g.score < 1.0],
        }
        for c in run.cases
        if not c.passed
    ]


def _summarise(run) -> dict[str, Any]:
    fails = _failures(run)
    return {
        "run_id": run.run_id,
        "agent": run.agent,
        "pass_rate": round(run.pass_rate, 3),
        "mean_score": round(run.mean_score, 3),
        "p95_latency_ms": round(run.p95_latency_ms, 1),
        "by_tag": {k: round(v, 3) for k, v in run.by_tag().items()},
        "failure_count": len(fails),
        "failures": fails[:20],
    }


@server.tool()
def list_golden_sets() -> list[str]:
    """List the available golden-set files that can be evaluated against."""
    if not GOLDENS_DIR.exists():
        return []
    return sorted(p.name for p in GOLDENS_DIR.glob("*.jsonl"))


@server.tool()
def run_eval(
    golden_set: str,
    agent: str = DEFAULT_AGENT,
    repeats: int = 1,
    set_baseline: bool = False,
) -> dict[str, Any]:
    """Evaluate an agent against a golden set and store the run.

    Args:
        golden_set: filename from list_golden_sets, e.g. "support_agent.jsonl"
        agent: import spec "module:attr", e.g. "examples.broken_agent:BrokenAgent"
        repeats: runs per case; the worst result is kept, for non-determinism
        set_baseline: also record this run as the baseline to compare against

    Returns pass_rate, mean_score, p95 latency, per-tag breakdown, and the
    failing cases with the reason each grader gave.
    """
    path = GOLDENS_DIR / golden_set
    if not path.exists():
        return {"error": f"golden set not found: {golden_set}"}

    run = Evaluator(load_agent(agent), repeats=repeats).run(load_goldens(path))
    RunStore(RUNS_DIR).save(run, as_baseline=set_baseline)

    out = _summarise(run)
    out["saved_as_baseline"] = set_baseline
    return out


@server.tool()
def gate(
    golden_set: str,
    agent: str = DEFAULT_AGENT,
    repeats: int = 1,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Evaluate an agent and diff it against the stored baseline.

    This is the CI check: it reports whether anything regressed, which specific
    cases got worse, and which improved. `ok` is false when any case that used
    to pass now fails, or the pass rate dropped by more than `tolerance`.

    Args:
        golden_set: filename from list_golden_sets
        agent: import spec "module:attr"
        repeats: runs per case; the worst result is kept
        tolerance: allowed pass-rate drop before failing, e.g. 0.02 for 2%

    Run run_eval with set_baseline=true first if no baseline exists yet.
    """
    path = GOLDENS_DIR / golden_set
    if not path.exists():
        return {"error": f"golden set not found: {golden_set}"}

    store = RunStore(RUNS_DIR)
    baseline = store.load_baseline()
    if baseline is None:
        return {
            "error": "no baseline recorded yet",
            "hint": "call run_eval with set_baseline=true on a known-good agent first",
        }

    current = Evaluator(load_agent(agent), repeats=repeats).run(load_goldens(path))
    store.save(current)

    report = compare(baseline, current, tolerance=tolerance)
    out = report.to_dict()
    out["current_failures"] = _failures(current)[:20]
    return out


@server.tool()
def list_runs() -> dict[str, Any]:
    """List stored evaluation runs and whether a baseline has been recorded."""
    store = RunStore(RUNS_DIR)
    baseline = store.load_baseline()
    return {
        "baseline_run_id": baseline.run_id if baseline else None,
        "baseline_agent": baseline.agent if baseline else None,
        "stored_runs": [p.stem for p in store.history()],
    }


if __name__ == "__main__":
    server.run()
