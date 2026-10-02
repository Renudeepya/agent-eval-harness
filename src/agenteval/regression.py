"""Compare a run against a baseline and decide whether to fail the build.

This is the feedback loop: a run on its own tells you a number, but only the
diff against a known-good baseline tells you whether a prompt or model change
made things worse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import RunResult


@dataclass
class CaseDelta:
    case_id: str
    baseline_score: float
    current_score: float
    baseline_passed: bool
    current_passed: bool

    @property
    def delta(self) -> float:
        return self.current_score - self.baseline_score

    @property
    def is_regression(self) -> bool:
        return self.baseline_passed and not self.current_passed

    @property
    def is_improvement(self) -> bool:
        return not self.baseline_passed and self.current_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "baseline_score": round(self.baseline_score, 4),
            "current_score": round(self.current_score, 4),
            "delta": round(self.delta, 4),
            "regression": self.is_regression,
            "improvement": self.is_improvement,
        }


@dataclass
class RegressionReport:
    baseline_run_id: str
    current_run_id: str
    deltas: list[CaseDelta] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    baseline_pass_rate: float = 0.0
    current_pass_rate: float = 0.0
    tolerance: float = 0.0

    @property
    def regressions(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.is_regression]

    @property
    def improvements(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.is_improvement]

    @property
    def pass_rate_delta(self) -> float:
        return self.current_pass_rate - self.baseline_pass_rate

    @property
    def ok(self) -> bool:
        """Gate for CI: no per-case regressions and pass rate within tolerance."""
        return not self.regressions and self.pass_rate_delta >= -self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "baseline_pass_rate": round(self.baseline_pass_rate, 4),
            "current_pass_rate": round(self.current_pass_rate, 4),
            "pass_rate_delta": round(self.pass_rate_delta, 4),
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
            "added_cases": self.added,
            "removed_cases": self.removed,
            "ok": self.ok,
        }

    def summary(self) -> str:
        lines = [
            f"baseline {self.baseline_run_id} -> current {self.current_run_id}",
            f"pass rate {self.baseline_pass_rate:.1%} -> {self.current_pass_rate:.1%} "
            f"({self.pass_rate_delta:+.1%})",
            f"regressions: {len(self.regressions)}   improvements: {len(self.improvements)}",
        ]
        for d in self.regressions:
            lines.append(f"  REGRESSED  {d.case_id}  {d.baseline_score:.2f} -> {d.current_score:.2f}")
        for d in self.improvements:
            lines.append(f"  FIXED      {d.case_id}  {d.baseline_score:.2f} -> {d.current_score:.2f}")
        if self.added:
            lines.append(f"  new cases: {', '.join(self.added)}")
        if self.removed:
            lines.append(f"  dropped cases: {', '.join(self.removed)}")
        lines.append("RESULT: PASS" if self.ok else "RESULT: FAIL")
        return "\n".join(lines)


def compare(baseline: RunResult, current: RunResult, tolerance: float = 0.0) -> RegressionReport:
    """Diff two runs case by case.

    `tolerance` allows a small pass-rate drop (e.g. 0.02) before failing, which
    matters when the agent is stochastic and a one-case flake shouldn't block a
    deploy.
    """
    base_by_id = {c.case_id: c for c in baseline.cases}
    curr_by_id = {c.case_id: c for c in current.cases}
    shared = base_by_id.keys() & curr_by_id.keys()

    deltas = [
        CaseDelta(
            case_id=cid,
            baseline_score=base_by_id[cid].score,
            current_score=curr_by_id[cid].score,
            baseline_passed=base_by_id[cid].passed,
            current_passed=curr_by_id[cid].passed,
        )
        for cid in sorted(shared)
    ]

    return RegressionReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        deltas=deltas,
        added=sorted(curr_by_id.keys() - base_by_id.keys()),
        removed=sorted(base_by_id.keys() - curr_by_id.keys()),
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        tolerance=tolerance,
    )
