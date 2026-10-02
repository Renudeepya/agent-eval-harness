"""Runs an agent over a golden set and collects grades."""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .graders import DEFAULT_GRADERS, Grader
from .types import Agent, AgentOutput, CaseResult, RunResult, TestCase


def load_goldens(path: str | Path) -> list[TestCase]:
    """Read a JSONL golden set. Blank lines and `#` comments are ignored."""
    cases: list[TestCase] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(TestCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{lineno}: bad golden row — {e}") from e
    if not cases:
        raise ValueError(f"{path}: no test cases found")
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate case ids {sorted(dupes)}")
    return cases


class Evaluator:
    """Executes cases against an agent and applies graders to each result."""

    def __init__(
        self,
        agent: Agent,
        graders: list[Grader] | None = None,
        *,
        max_workers: int = 4,
        repeats: int = 1,
    ):
        """`repeats` > 1 runs each case N times and keeps the worst result —
        the cheapest way to surface non-determinism, which is the failure mode
        that makes agents hard to regression-test in the first place."""
        self.agent = agent
        self.graders = graders if graders is not None else DEFAULT_GRADERS
        self.max_workers = max(1, max_workers)
        self.repeats = max(1, repeats)

    # ---------------------------------------------------------------- internals
    def _invoke(self, case: TestCase) -> AgentOutput:
        started = time.perf_counter()
        try:
            out = self.agent.run(case.input)
        except Exception as e:
            return AgentOutput(
                text="",
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(e).__name__}: {e}",
            )
        if not out.latency_ms:
            out.latency_ms = (time.perf_counter() - started) * 1000
        return out

    def _grade_once(self, case: TestCase) -> CaseResult:
        output = self._invoke(case)
        grades = [g.grade(case, output) for g in self.graders]
        return CaseResult(case_id=case.id, tags=list(case.tags), output=output, grades=grades)

    def _run_case(self, case: TestCase) -> CaseResult:
        attempts = [self._grade_once(case) for _ in range(self.repeats)]
        return min(attempts, key=lambda r: r.score)  # worst-of-N

    # ---------------------------------------------------------------- public
    def run(self, cases: list[TestCase]) -> RunResult:
        if self.max_workers == 1:
            results = [self._run_case(c) for c in cases]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(self._run_case, cases))

        return RunResult(
            run_id=uuid.uuid4().hex[:12],
            agent=getattr(self.agent, "name", type(self.agent).__name__),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            cases=results,
        )
