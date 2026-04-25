from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from bm_flow_agent.prompts import build_system_prompt
from bm_flow_agent.tools import ToolRegistry
from bm_flow_agent.tools.base import ToolContext
from bm_flow_agent.tools.planning_tools import has_incomplete_plan
from bm_flow_agent.ui import ConsoleUI

try:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[misc]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    thread_id: str
    workspace_root: str
    target_flow_json: str
    target_dsl_path: str
    repo_catalog: dict[str, Any]
    working_dsl: dict[str, Any] | None
    last_compile_result: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None
    current_plan: dict[str, Any] | None
    project_id: str | None
    user_id: str | None
    auth_claims: dict[str, Any] | None


@dataclass
class AgentRuntime:
    workspace_root: Path
    target_flow_json: str
    target_dsl_path: str
    model: Any
    registry: ToolRegistry
    ui: ConsoleUI


def build_agent_graph(runtime: AgentRuntime, checkpointer: Any | None = None):
    model_with_tools = runtime.model.bind_tools(runtime.registry.langchain_tools)
    builder = StateGraph(AgentState)

    def llm_call(state: AgentState) -> dict[str, Any]:
        prompt = build_system_prompt(state)
        response = model_with_tools.invoke(
            [SystemMessage(content=prompt), *state.get("messages", [])]
        )
        return {"messages": [response]}

    def tool_exec(state: AgentState) -> dict[str, Any]:
        message = state.get("messages", [])[-1]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return {}

        updates: dict[str, Any] = {}
        tool_messages: list[ToolMessage] = []
        pending_clarification: dict[str, Any] | None = None
        current_state = dict(state)

        for call in message.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            runtime.ui.tool_call(name, args)
            context = ToolContext(
                workspace_root=runtime.workspace_root,
                state=current_state,
                ui=runtime.ui,
            )
            result = runtime.registry.execute(name, context, args)
            if result.interrupt_payload and pending_clarification is None:
                pending_clarification = {
                    **result.interrupt_payload,
                    "tool_call_id": call["id"],
                    "tool_name": name,
                }
            else:
                tool_messages.append(
                    ToolMessage(
                        content=result.content,
                        tool_call_id=call["id"],
                    )
                )
            if result.is_error:
                runtime.ui.warning(result.content)
            else:
                runtime.ui.status(result.content)
            updates = merge_state_updates(updates, result.state_updates)
            current_state = merge_state_updates(current_state, result.state_updates)

        if tool_messages:
            updates["messages"] = tool_messages
        if pending_clarification:
            updates["pending_clarification"] = pending_clarification
        return updates

    def clarify_interrupt(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("pending_clarification") or {})
        answer = interrupt(payload)
        if isinstance(answer, dict):
            answer_text = format_clarification_answer(answer)
        else:
            answer_text = str(answer)
        return {
            "messages": [
                ToolMessage(
                    content=f"User clarification: {answer_text}",
                    tool_call_id=payload["tool_call_id"],
                )
            ],
            "pending_clarification": None,
        }

    def final_response(_: AgentState) -> dict[str, Any]:
        return {}

    def plan_reminder(_: AgentState) -> dict[str, Any]:
        return {
            "messages": [
                SystemMessage(
                    content=(
                        "Internal reminder: the current task plan still has incomplete items. "
                        "Do not provide a final answer yet. If an item is blocked, inspect the error, "
                        "call get_flow_yaml/analyze_flow_connectivity when route connectivity is involved, "
                        "then fix it with connect_steps or upsert_step. Continue by calling tools, updating "
                        "plan items to in_progress/completed/blocked, and finish every item before final response."
                    )
                )
            ]
        }

    def after_llm(state: AgentState) -> Literal["tool_exec", "plan_reminder", "final_response"]:
        message = state.get("messages", [])[-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            return "tool_exec"
        if has_incomplete_plan(state.get("current_plan")):
            return "plan_reminder"
        return "final_response"

    def after_tools(state: AgentState) -> Literal["clarify_interrupt", "llm_call"]:
        if state.get("pending_clarification"):
            return "clarify_interrupt"
        return "llm_call"

    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_exec", tool_exec)
    builder.add_node("clarify_interrupt", clarify_interrupt)
    builder.add_node("plan_reminder", plan_reminder)
    builder.add_node("final_response", final_response)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call", after_llm, ["tool_exec", "plan_reminder", "final_response"]
    )
    builder.add_conditional_edges(
        "tool_exec", after_tools, ["clarify_interrupt", "llm_call"]
    )
    builder.add_edge("clarify_interrupt", "llm_call")
    builder.add_edge("plan_reminder", "llm_call")
    builder.add_edge("final_response", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def merge_state_updates(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        merged[key] = value
    return merged


def latest_assistant_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


def format_clarification_answer(answer: dict[str, Any]) -> str:
    if isinstance(answer.get("answers"), list):
        parts = []
        for item in answer["answers"]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("answer", ""))
            if item.get("option_id"):
                text = f"{text} (option_id: {item['option_id']})"
            if item.get("option_ids"):
                text = f"{text} (option_ids: {', '.join(map(str, item['option_ids']))})"
            if item.get("question_id"):
                text = f"{item['question_id']}: {text}"
            parts.append(text)
        return "; ".join(parts)
    answer_text = str(answer.get("answer", ""))
    if answer.get("option_id"):
        return f"{answer_text} (option_id: {answer['option_id']})"
    if answer.get("option_ids"):
        return f"{answer_text} (option_ids: {', '.join(map(str, answer['option_ids']))})"
    return answer_text


def initial_state(
    *,
    workspace_root: Path,
    target_flow_json: str,
    target_dsl_path: str,
    repo_catalog: dict[str, Any],
    working_dsl: dict[str, Any] | None = None,
) -> AgentState:
    state: AgentState = {
        "workspace_root": str(workspace_root),
        "target_flow_json": target_flow_json,
        "target_dsl_path": target_dsl_path,
        "repo_catalog": repo_catalog,
        "messages": [],
    }
    if working_dsl is not None:
        state["working_dsl"] = working_dsl
    return state
