"""Command line interface.

    agenteval run     --goldens goldens/support_agent.jsonl --agent agenteval.adapters:RuleBasedAgent
    agenteval compare --baseline runs/baseline.json --current runs/<id>.json
    agenteval gate    --goldens ... --agent ... --tolerance 0.02

`gate` is the CI entry point: it runs, compares against the stored baseline,
and exits non-zero on regression so a pipeline blocks the deploy.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from .evaluator import Evaluator, load_goldens
from .regression import compare
from .store import RunStore
from .types import Agent


def load_agent(spec: str) -> Agent:
    """Resolve 'package.module:ClassOrFactory' into an agent instance."""
    if ":" not in spec:
        raise ValueError(f"agent spec must be 'module:attr', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise SystemExit(f"could not import {mod_name!r}: {e}") from e
    try:
        obj: Any = getattr(mod, attr)
    except AttributeError as e:
        raise SystemExit(f"{mod_name!r} has no attribute {attr!r}") from e
    agent = obj() if callable(obj) else obj
    if not hasattr(agent, "run"):
        raise SystemExit(f"{spec} does not expose a .run(prompt) method")
    return agent


def _print_run(run) -> None:
    print(f"run {run.run_id}  agent={run.agent}")
    print(f"  pass rate   {run.pass_rate:.1%}  ({sum(c.passed for c in run.cases)}/{len(run.cases)})")
    print(f"  mean score  {run.mean_score:.3f}")
    print(f"  p95 latency {run.p95_latency_ms:.0f}ms")
    by_tag = run.by_tag()
    if by_tag:
        print("  by tag:")
        for tag, rate in sorted(by_tag.items(), key=lambda kv: kv[1]):
            print(f"    {tag:<20} {rate:.1%}")
    failures = [c for c in run.cases if not c.passed]
    if failures:
        print(f"  failures ({len(failures)}):")
        for c in failures:
            reasons = "; ".join(g.detail for g in c.grades if not g.passed and g.detail)
            print(f"    {c.case_id}: {reasons or 'failed'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agenteval", description="Agent evaluation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--goldens", required=True, help="path to a JSONL golden set")
        p.add_argument("--agent", required=True, help="module:attr resolving to an agent")
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--repeats", type=int, default=1, help="runs per case; worst result kept")
        p.add_argument("--workers", type=int, default=4)

    p_run = sub.add_parser("run", help="evaluate an agent and store the run")
    add_common(p_run)
    p_run.add_argument("--set-baseline", action="store_true")

    p_cmp = sub.add_parser("compare", help="diff two stored runs")
    p_cmp.add_argument("--baseline", required=True)
    p_cmp.add_argument("--current", required=True)
    p_cmp.add_argument("--runs-dir", default="runs")
    p_cmp.add_argument("--tolerance", type=float, default=0.0)

    p_gate = sub.add_parser("gate", help="run, compare to baseline, exit non-zero on regression")
    add_common(p_gate)
    p_gate.add_argument("--tolerance", type=float, default=0.0)

    args = parser.parse_args(argv)
    store = RunStore(args.runs_dir)

    if args.cmd == "compare":
        report = compare(store.load(args.baseline), store.load(args.current), args.tolerance)
        print(report.summary())
        return 0 if report.ok else 1

    cases = load_goldens(args.goldens)
    evaluator = Evaluator(
        load_agent(args.agent), max_workers=args.workers, repeats=args.repeats
    )
    run = evaluator.run(cases)

    if args.cmd == "run":
        path = store.save(run, as_baseline=args.set_baseline)
        _print_run(run)
        print(f"saved -> {path}" + ("  (also set as baseline)" if args.set_baseline else ""))
        return 0

    # gate
    store.save(run)
    _print_run(run)
    baseline = store.load_baseline()
    if baseline is None:
        print("\nno baseline stored; run with --set-baseline first")
        return 0
    report = compare(baseline, run, args.tolerance)
    print()
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
