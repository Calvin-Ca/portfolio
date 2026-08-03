from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# 智能体级
from langchain.agents import AgentState,create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult,ModelRequest,ModelResponse

from langchain.chat_models import init_chat_model

from langchain_core.messages import AIMessage,ToolMessage,SystemMessage
from langchain_core.tools  import tool

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command


DASHSCOPE_API_KEY = "sk-1e865ed7e785494db11138d0e905bed0"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# A real service would put these in Redis / the checkpointer, keyed by thread.
PROMOTED_TOOLS: set[str] = set()
DEFERRED_TOOLS = {"internal_search"}

def tool_error(request: ToolCallRequest, text: str) -> ToolMessage:
    return ToolMessage(
        content=text,
        tool_call_id=str(request.tool_call.get("id", "missing_tool_call_id")),
        name=str(request.tool_call.get("name", "unknown")),
        status="error",
    )


class ThreadContext(AgentMiddleware[AgentState]):
    """before_agent: set up a per-thread working directory and state."""

    
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any]:
        context = runtime.context or {}
        thread_id = str(context.get("thread_id", "demo-thread"))
        workspace = Path(".demo_threads") / thread_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return {"thread_data": {"thread_id": thread_id, "workspace": str(workspace)}}


class DynamicContext(AgentMiddleware[AgentState]):
    """before_model: inject runtime facts; production also injects memory/images."""

    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any]:
        reminder = SystemMessage(
            content="Runtime reminder: this is a demo. Prefer tool_search before internal tools."
        )
        # Do not mutate state in place: graph state is managed by LangGraph.
        return {"messages": [reminder]}


class DeferredToolFilter(AgentMiddleware[AgentState]):
    """Hide expensive/internal tool schemas until tool_search promotes them."""

    def _visible(self, request: ModelRequest) -> ModelRequest:
        visible = [
            item for item in request.tools
            if getattr(item, "name", None) not in DEFERRED_TOOLS
            or getattr(item, "name", None) in PROMOTED_TOOLS
        ]
        return request.override(tools=visible)

    
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        return handler(self._visible(request))

    
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        name = str(request.tool_call.get("name", ""))
        if name in DEFERRED_TOOLS and name not in PROMOTED_TOOLS:
            return tool_error(request, f"{name} is hidden; call tool_search first.")
        return handler(request)


class Guardrail(AgentMiddleware[AgentState]):
    """wrap_tool_call: policy authorization.  Unknown policy errors fail closed."""

    def __init__(self, fail_closed: bool = True) -> None:
        self.fail_closed = fail_closed

    def allowed(self, name: str, args: dict[str, Any]) -> bool:
        # Replace this with an OPA/OAP or company policy provider in production.
        if name == "bash" and "production" in str(args.get("command", "")).lower():
            return False
        return True

    
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        try:
            if not self.allowed(str(request.tool_call.get("name")), request.tool_call.get("args", {})):
                return tool_error(request, "Guardrail denied this tool call.")
        except Exception:
            if self.fail_closed:
                return tool_error(request, "Guardrail unavailable; denied by fail-closed policy.")
        return handler(request)


class SandboxAudit(AgentMiddleware[AgentState]):
    """Classify shell commands before tool execution; block high-risk patterns."""

    BLOCKED = (r"\brm\s+-rf\s+[/~]", r"\bcurl\b.*\|\s*(ba)?sh", r"\bmkfs\b")
    WARN = (r"\bpip\s+install\b", r"\bchmod\s+777\b")

    
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "bash":
            return handler(request)
        command = str(request.tool_call.get("args", {}).get("command", ""))
        verdict = "block" if any(re.search(p, command) for p in self.BLOCKED) else "warn" if any(re.search(p, command) for p in self.WARN) else "pass"
        print({"audit": "sandbox", "command": command[:200], "verdict": verdict})
        if verdict == "block":
            return tool_error(request, "SandboxAudit blocked a dangerous command.")
        result = handler(request)
        if verdict == "warn" and isinstance(result, ToolMessage):
            return ToolMessage(content=f"{result.content}\nWARNING: medium-risk command.", tool_call_id=result.tool_call_id, name=result.name)
        return result


class ToolErrors(AgentMiddleware[AgentState]):
    """Turn ordinary tool failures into model-visible errors; retain graph interrupts."""

    
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except GraphBubbleUp:
            raise  # Critical: do not swallow pause/resume/interrupt control flow.
        except Exception as exc:
            return tool_error(request, f"Tool failed: {type(exc).__name__}: {exc}")


class LoopDetection(AgentMiddleware[AgentState]):
    """after_model: stop repeated tool-call sets rather than burning tokens forever."""

    def __init__(self, hard_limit: int = 3) -> None:
        self.hard_limit = hard_limit
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)

    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", [])
        if not calls:
            return None
        thread = str((runtime.context or {}).get("thread_id", "demo-thread"))
        fingerprint = repr(sorted((c.get("name"), c.get("args")) for c in calls))
        self.counts[thread][fingerprint] += 1
        if self.counts[thread][fingerprint] < self.hard_limit:
            return None
        # An AIMessage without tool_calls routes the graph to its final answer.
        return {"messages": [AIMessage(content="I stopped repeated tool calls and will answer from collected results.")]}


class Clarification(AgentMiddleware[AgentState]):
    """HITL: transform a tool call into a durable pause at END."""

    
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "ask_clarification":
            return handler(request)
        question = request.tool_call.get("args", {}).get("question", "Please provide more detail.")
        return Command(update={"messages": [ToolMessage(content=f"❓ {question}", tool_call_id=str(request.tool_call.get("id", "clarify")), name="ask_clarification")]}, goto=END)


class Cleanup(AgentMiddleware[AgentState]):
    """after_agent: enqueue memory and release thread-scoped resources."""

    
    def after_agent(self, state: AgentState, runtime: Runtime) -> None:
        print("cleanup: enqueue memory update and release sandbox lease")


@tool
def tool_search(query: str) -> str:
    """Discover internal tools by capability."""
    if "search" in query.lower() or "internal" in query.lower():
        PROMOTED_TOOLS.add("internal_search")
        return "Promoted internal_search. You may call it now."
    return "No matching deferred tools."


@tool
def internal_search(query: str) -> str:
    """Search an internal knowledge base (deferred until tool_search promotes it)."""
    return f"Internal result for: {query}"


@tool
def bash(command: str) -> str:
    """Demo shell tool. It does not execute commands in this example."""
    return f"(demo only) would execute: {command}"


@tool
def ask_clarification(question: str) -> str:
    """Ask the user for missing information; intercepted by Clarification middleware."""
    return question  # unreachable: Clarification intercepts it first


if __name__ == "__main__":
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )
    agent = create_agent(
        model=model,
        tools=[tool_search, internal_search, bash, ask_clarification],
        # Clarification is last; ToolErrors remains outside it so interrupts survive.
        middleware=[ThreadContext(), DynamicContext(), DeferredToolFilter(), Guardrail(), SandboxAudit(), ToolErrors(), LoopDetection(), Cleanup(), Clarification()],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search internal docs for the deployment process."}]},
        context={"thread_id": "interview-demo"},
    )
    print(result["messages"][-1].content)