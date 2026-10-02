# agenteval

A regression-testing harness and feedback loop for LLM agents.

Unit tests tell you a function still works. Nothing tells you an *agent* still
works — prompts drift, models get deprecated mid-quarter, and a one-line system
prompt edit can silently stop the agent from calling the tool that grounds its
answers. The output still reads fine. That is the failure this catches.

`agenteval` runs an agent against a versioned golden set, grades each case on
output *and* tool trajectory, stores the run, and diffs it against a known-good
baseline. Wire the `gate` command into CI and a regression blocks the deploy.

Zero required dependencies. Runs offline against a built-in rule-based agent, so
the test suite needs no API key.

---

## Quick start

```bash
pip install -e ".[dev]"

# 1. Evaluate the reference agent and record it as the baseline
agenteval run --goldens goldens/support_agent.jsonl \
              --agent agenteval.adapters:RuleBasedAgent \
              --set-baseline

# 2. Later, gate a change against that baseline (exit 1 on regression)
agenteval gate --goldens goldens/support_agent.jsonl \
               --agent agenteval.adapters:RuleBasedAgent
```

Or as a library:

```python
from agenteval import Evaluator, RuleBasedAgent, load_goldens, compare

cases = load_goldens("goldens/support_agent.jsonl")
run   = Evaluator(RuleBasedAgent(), repeats=3).run(cases)

print(f"{run.pass_rate:.0%} passed, p95 {run.p95_latency_ms:.0f}ms")
print(run.by_tag())          # which *category* of case is failing
```

---

## Why tool trajectory matters

The headline feature. Grading only final text misses the most common agent
regression, where the answer stays plausible but the agent stopped grounding it:

```
$ agenteval gate --goldens goldens/support_agent.jsonl \
                 --agent broken_agent:BrokenAgent

  pass rate   10.0%  (1/10)
  failures (9):
    refund-002: missing tools: ['lookup_order', 'refund_policy']
    auth-001:   missing tools: ['auth_docs']
    ...

baseline 719b42a68d1a -> current d5c7ca3a53a2
pass rate 100.0% -> 10.0% (-90.0%)
regressions: 9   improvements: 0
RESULT: FAIL
$ echo $?
1
```

Every one of those cases returned correct-looking prose. Text-only grading would
have shipped it.

---

## The golden set

JSONL, one case per line, version-controlled next to your code. `#` comments and
blank lines are ignored; duplicate ids are rejected at load time.

```json
{"id": "refund-001",
 "input": "How long does a refund take?",
 "expected_tools": ["lookup_order", "refund_policy"],
 "tags": ["refunds", "happy-path"],
 "metadata": {"contains": ["5 business days"], "latency_budget_ms": 2000}}
```

`tags` drive the per-category breakdown, which is what turns "we dropped 4%"
into "we broke refunds."

---

## Graders

Each returns a normalized 0–1 score plus a pass/fail, so mixed checks average
into one per-case score.

| Grader | Checks |
|---|---|
| `NoError` | Agent didn't raise |
| `ExactMatch` | Equality, whitespace/case normalized |
| `Contains` | Required substrings, partial credit |
| `RegexMatch` | Pattern from the case |
| `FuzzyMatch` | Character similarity for allowed paraphrase |
| `ToolTrajectory` | Tools called — ordered or as a set |
| `ValidJSON` | Parses, with required keys |
| `LatencyBudget` | Per-case latency ceiling |
| `LLMJudge` | Model-graded, judge callable injected |

Custom graders subclass `Grader` and implement `grade(case, output)`.

`LLMJudge` takes any `fn(prompt) -> float`, so the harness never imports a vendor
SDK — that keeps it testable offline and swappable between providers. A judge
that throws scores 0 and records the reason rather than crashing the run.

---

## Handling non-determinism

Agents are stochastic, so a single green run proves little. `repeats=N` runs each
case N times and keeps the **worst** result:

```python
Evaluator(agent, repeats=5).run(cases)
```

Flakiness surfaces as a real failure instead of hiding until production.

On the aggregate side, `--tolerance` permits a small pass-rate drop before
failing — but any individual case flipping pass→fail still fails the gate,
regardless of tolerance. Aggregate noise is forgivable; a specific broken
behavior is not.

---

## CI

```yaml
- run: pip install -e ".[dev]"
- run: pytest
- run: |
    agenteval gate \
      --goldens goldens/support_agent.jsonl \
      --agent myapp.agents:SupportAgent \
      --repeats 3 --tolerance 0.02
```

Non-zero exit fails the build. Commit `runs/baseline.json` and update it
deliberately when an improvement is real.

---

## Bringing your own agent

Implement one method:

```python
class MyAgent:
    name = "my-agent-v1"

    def run(self, prompt: str) -> AgentOutput:
        ...
        return AgentOutput(text=answer, tool_calls=[ToolCall(name="search")])
```

`OpenAIAgent` and `AnthropicAgent` ship as reference adapters; both import their
SDK lazily, so neither is needed to install or test the package.

---

## Layout

```
src/agenteval/
  types.py        TestCase, AgentOutput, ToolCall, RunResult
  graders.py      9 graders + Grader base class
  evaluator.py    golden loading, parallel execution, worst-of-N
  regression.py   baseline diffing and the CI verdict
  store.py        JSON run persistence and baselines
  adapters.py     rule-based, callable, OpenAI, Anthropic
  cli.py          run / compare / gate
goldens/          versioned test cases
examples/         a deliberately regressed agent for the demo
tests/            59 tests, no network required
```

## Tests

```bash
pytest          # 59 passed
```

## License

MIT
