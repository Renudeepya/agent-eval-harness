"""A deliberately regressed agent, used to demo the CI gate.

It answers correctly but stops calling its tools — the exact failure mode where
output still *looks* fine and only trajectory grading catches the problem.

    PYTHONPATH=src:examples agenteval gate \
        --goldens goldens/support_agent.jsonl \
        --agent broken_agent:BrokenAgent
"""
from agenteval.adapters import RuleBasedAgent


def BrokenAgent():
    return RuleBasedAgent(name="rule-based-v2-no-tools", skip_tools=True)
