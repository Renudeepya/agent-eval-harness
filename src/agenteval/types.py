"""Core data types for the agent evaluation harness."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation made by an agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(name=d["name"], arguments=d.get("arguments", {}))


@dataclass(frozen=True)
class TestCase:
    """One row of a golden dataset.

    `expected` and `expected_tools` are both optional: a case may assert on
    final output, on the tool trajectory, or on both.
    """

    __test__ = False  # not a pytest class, despite the name

    id: str
    input: str
    expected: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestCase":
        return cls(
            id=d["id"],
            input=d["input"],
            expected=d.get("expected"),
            expected_tools=list(d.get("expected_tools", [])),
            tags=list(d.get("tags", [])),
            metadata=d.get("metadata", {}),
        )


@dataclass
class AgentOutput:
    """What an agent returned for a single test case."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "error": self.error,
        }


@dataclass
class GradeResult:
    """One grader's verdict on one case."""

    grader: str
    score: float           # normalized 0.0 - 1.0
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    """All grades for a single test case, plus the raw agent output."""

    case_id: str
    tags: list[str]
    output: AgentOutput
    grades: list[GradeResult]

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.grades) if self.grades else False

    @property
    def score(self) -> float:
        if not self.grades:
            return 0.0
        return sum(g.score for g in self.grades) / len(self.grades)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tags": self.tags,
            "output": self.output.to_dict(),
            "grades": [g.to_dict() for g in self.grades],
            "passed": self.passed,
            "score": self.score,
        }


@dataclass
class RunResult:
    """A complete evaluation run over a golden set."""

    run_id: str
    agent: str
    created_at: str
    cases: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return (sum(c.passed for c in self.cases) / len(self.cases)) if self.cases else 0.0

    @property
    def mean_score(self) -> float:
        return (sum(c.score for c in self.cases) / len(self.cases)) if self.cases else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.cases:
            return 0.0
        lat = sorted(c.output.latency_ms for c in self.cases)
        idx = min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))
        return lat[idx]

    def by_tag(self) -> dict[str, float]:
        """Pass rate broken down by tag — shows *which kind* of case regressed."""
        buckets: dict[str, list[bool]] = {}
        for c in self.cases:
            for t in c.tags:
                buckets.setdefault(t, []).append(c.passed)
        return {t: sum(v) / len(v) for t, v in buckets.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "created_at": self.created_at,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "p95_latency_ms": self.p95_latency_ms,
            "by_tag": self.by_tag(),
            "cases": [c.to_dict() for c in self.cases],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@runtime_checkable
class Agent(Protocol):
    """Anything with this shape can be evaluated.

    Keeping the surface this small is deliberate: it lets the harness wrap an
    OpenAI assistant, a LangChain executor, or an internal service behind the
    same interface.
    """

    name: str

    def run(self, prompt: str) -> AgentOutput: ...
