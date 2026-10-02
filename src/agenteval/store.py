"""Persistence for runs. Plain JSON on disk — no database to stand up."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import AgentOutput, CaseResult, GradeResult, RunResult, ToolCall


class RunStore:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- write
    def save(self, run: RunResult, *, as_baseline: bool = False) -> Path:
        path = self.root / f"{run.run_id}.json"
        path.write_text(run.to_json(), encoding="utf-8")
        if as_baseline:
            (self.root / "baseline.json").write_text(run.to_json(), encoding="utf-8")
        return path

    # ---------------------------------------------------------------- read
    @staticmethod
    def _case_from_dict(d: dict[str, Any]) -> CaseResult:
        o = d["output"]
        return CaseResult(
            case_id=d["case_id"],
            tags=list(d.get("tags", [])),
            output=AgentOutput(
                text=o.get("text", ""),
                tool_calls=[ToolCall.from_dict(t) for t in o.get("tool_calls", [])],
                latency_ms=o.get("latency_ms", 0.0),
                tokens=o.get("tokens", 0),
                error=o.get("error"),
            ),
            grades=[
                GradeResult(
                    grader=g["grader"],
                    score=g["score"],
                    passed=g["passed"],
                    detail=g.get("detail", ""),
                )
                for g in d.get("grades", [])
            ],
        )

    def load(self, path: str | Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return RunResult(
            run_id=data["run_id"],
            agent=data["agent"],
            created_at=data["created_at"],
            cases=[self._case_from_dict(c) for c in data["cases"]],
        )

    def load_baseline(self) -> RunResult | None:
        p = self.root / "baseline.json"
        return self.load(p) if p.exists() else None

    def history(self) -> list[Path]:
        return sorted(p for p in self.root.glob("*.json") if p.name != "baseline.json")
