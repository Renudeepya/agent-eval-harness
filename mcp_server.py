"""MCP server exposing the agenteval harness as tools."""
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from agenteval import Evaluator, load_goldens
from agenteval.cli import load_agent

server = MCPServer(
    name="agenteval",
    version="0.1.0",
    instructions="Run agent evaluation suites: list golden sets, "
                 "evaluate an agent, and diff a run against a baseline.",
)

GOLDENS_DIR = Path(__file__).parent / "goldens"


@server.tool()
def list_golden_sets() -> list[str]:
    """List the available golden-set files that can be evaluated against."""
    if not GOLDENS_DIR.exists():
        return []
    return sorted(p.name for p in GOLDENS_DIR.glob("*.jsonl"))


@server.tool()
def run_eval(
    golden_set: str,
    agent: str = "agenteval.adapters:RuleBasedAgent",
    repeats: int = 1,
) -> dict[str, Any]:
    """Evaluate an agent against a golden set and return the scored result.

    Args:
        golden_set: filename from list_golden_sets, e.g. "support_agent.jsonl"
        agent: import spec "module:attr", e.g. "examples.broken_agent:BrokenAgent"
        repeats: runs per case; the worst result is kept, for non-determinism

    Returns pass_rate, mean_score, p95 latency, per-tag breakdown, and the
    failing cases with the reason each grader gave.
    """
    path = GOLDENS_DIR / golden_set
    if not path.exists():
        return {"error": f"golden set not found: {golden_set}"}

    run = Evaluator(load_agent(agent), repeats=repeats).run(load_goldens(path))

    failures = [
        {
            "case_id": c.case_id,
            "score": round(c.score, 3),
            "reasons": [g.detail for g in c.grades if g.detail and g.score < 1.0],
        }
        for c in run.cases
        if not c.passed
    ]

    return {
        "run_id": run.run_id,
        "agent": run.agent,
        "pass_rate": round(run.pass_rate, 3),
        "mean_score": round(run.mean_score, 3),
        "p95_latency_ms": round(run.p95_latency_ms, 1),
        "by_tag": {k: round(v, 3) for k, v in run.by_tag().items()},
        "failure_count": len(failures),
        "failures": failures[:20],
    }


if __name__ == "__main__":
    server.run()
