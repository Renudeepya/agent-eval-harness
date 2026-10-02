"""Reference agents.

`RuleBasedAgent` lets the whole harness run with no API key and no network,
which is what makes the test suite fast and CI-friendly. `OpenAIAgent` and
`AnthropicAgent` show the adapter shape for a real model; both import their SDK
lazily so the package installs cleanly without either one.
"""
from __future__ import annotations

import random
import re
import time
from typing import Any, Callable

from .types import AgentOutput, ToolCall


class RuleBasedAgent:
    """Deterministic stand-in agent used by the tests and the demo.

    `failure_rate` and `skip_tools` deliberately introduce the two regressions
    worth catching: degraded output, and a dropped tool call.
    """

    def __init__(
        self,
        name: str = "rule-based-v1",
        *,
        failure_rate: float = 0.0,
        skip_tools: bool = False,
        seed: int = 7,
    ):
        self.name = name
        self.failure_rate = failure_rate
        self.skip_tools = skip_tools
        self._rng = random.Random(seed)

    _RULES: list[tuple[str, str, list[str]]] = [
        (r"\brefund\b", "Refunds are processed within 5 business days to the original payment method.", ["lookup_order", "refund_policy"]),
        (r"\bpassword\b|\bsign ?in\b", "Reset your password from the sign-in page using 'Forgot password'.", ["auth_docs"]),
        (r"\bship|deliver", "Standard shipping takes 3-5 business days; express is next-day.", ["lookup_order"]),
        (r"\bcancel\b", "You can cancel any order before it ships from your order history page.", ["lookup_order", "cancel_order"]),
        (r"\binvoice\b|\bbilling\b", "Invoices are available under Billing and are issued monthly.", ["billing_docs"]),
    ]

    def run(self, prompt: str) -> AgentOutput:
        start = time.perf_counter()
        if self._rng.random() < self.failure_rate:
            return AgentOutput(
                text="I'm not sure.",
                tool_calls=[],
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        for pattern, answer, tools in self._RULES:
            if re.search(pattern, prompt, re.IGNORECASE):
                calls = [] if self.skip_tools else [ToolCall(name=t) for t in tools]
                return AgentOutput(
                    text=answer,
                    tool_calls=calls,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    tokens=len(answer.split()),
                )
        return AgentOutput(
            text="I don't have information on that.",
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class CallableAgent:
    """Wraps any `fn(prompt) -> str` so ad-hoc functions are evaluable."""

    def __init__(self, fn: Callable[[str], str], name: str = "callable"):
        self.fn = fn
        self.name = name

    def run(self, prompt: str) -> AgentOutput:
        start = time.perf_counter()
        text = self.fn(prompt)
        return AgentOutput(text=text, latency_ms=(time.perf_counter() - start) * 1000)


class OpenAIAgent:
    """Adapter for the OpenAI SDK. Requires `pip install openai`."""

    def __init__(self, model: str = "gpt-4o-mini", tools: list[dict[str, Any]] | None = None,
                 system: str = "You are a helpful assistant.", name: str | None = None):
        from openai import OpenAI  # imported lazily on purpose

        self.client = OpenAI()
        self.model = model
        self.tools = tools or []
        self.system = system
        self.name = name or f"openai:{model}"

    def run(self, prompt: str) -> AgentOutput:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
        }
        if self.tools:
            kwargs["tools"] = self.tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(name=tc.function.name, arguments={"raw": tc.function.arguments})
            for tc in (msg.tool_calls or [])
        ]
        return AgentOutput(
            text=msg.content or "",
            tool_calls=calls,
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens=getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
        )


class AnthropicAgent:
    """Adapter for the Anthropic SDK. Requires `pip install anthropic`."""

    def __init__(self, model: str = "claude-sonnet-4-5", tools: list[dict[str, Any]] | None = None,
                 system: str = "You are a helpful assistant.", max_tokens: int = 1024,
                 name: str | None = None):
        import anthropic  # imported lazily on purpose

        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = tools or []
        self.system = system
        self.max_tokens = max_tokens
        self.name = name or f"anthropic:{model}"

    def run(self, prompt: str) -> AgentOutput:
        start = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.tools:
            kwargs["tools"] = self.tools
        resp = self.client.messages.create(**kwargs)

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input or {})))

        usage = getattr(resp, "usage", None)
        tokens = (usage.input_tokens + usage.output_tokens) if usage else 0
        return AgentOutput(
            text="".join(text_parts),
            tool_calls=calls,
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens=tokens,
        )
