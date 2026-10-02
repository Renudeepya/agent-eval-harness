"""agenteval — a regression-testing harness for LLM agents.

    from agenteval import Evaluator, RuleBasedAgent, load_goldens, compare

    cases = load_goldens("goldens/support_agent.jsonl")
    run   = Evaluator(RuleBasedAgent()).run(cases)
    print(run.pass_rate)
"""
from .adapters import AnthropicAgent, CallableAgent, OpenAIAgent, RuleBasedAgent
from .evaluator import Evaluator, load_goldens
from .graders import (
    Contains,
    ExactMatch,
    FuzzyMatch,
    Grader,
    LLMJudge,
    LatencyBudget,
    NoError,
    RegexMatch,
    ToolTrajectory,
    ValidJSON,
)
from .regression import RegressionReport, compare
from .store import RunStore
from .types import AgentOutput, CaseResult, GradeResult, RunResult, TestCase, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Evaluator", "load_goldens", "compare", "RegressionReport", "RunStore",
    "RuleBasedAgent", "CallableAgent", "OpenAIAgent", "AnthropicAgent",
    "Grader", "ExactMatch", "Contains", "RegexMatch", "FuzzyMatch",
    "ToolTrajectory", "NoError", "ValidJSON", "LatencyBudget", "LLMJudge",
    "TestCase", "AgentOutput", "ToolCall", "GradeResult", "CaseResult", "RunResult",
    "__version__",
]
