from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from bot_agent.prompts import build_system_prompt
from bot_agent.tools import ToolRegistry
from bot_agent.tools.base import ToolContext
from bot_agent.tools.planning_tools import has_incomplete_plan
from bot_agent.ui import ConsoleUI

try:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[misc]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    thread_id: str
    workspace_root: str
    project_dir: str
    project_catalog: dict[str, Any]
    pending_clarification: dict[str, Any] | None
    current_plan: dict[str, Any] | None
    bot_project_dir: str
    bot_spec: dict[str, Any] | None
    bot_files: list[str]
    bot_validation: dict[str, Any] | None
    project_id: str | None
    user_id: str | None
    auth_claims: dict[str, Any] | None


@dataclass
class AgentRuntime:
    workspace_root: Path
    model: Any
    registry: ToolRegistry
    ui: ConsoleUI


def build_agent_graph(runtime: AgentRuntime, checkpointer: Any | None = None):
    model_with_tools = runtime.model.bind_tools(runtime.registry.langchain_tools)
    builder = StateGraph(AgentState)

    def llm_call(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=build_system_prompt(state)), *state.get("messages", [])]
        skip_tools = should_skip_tools_for_turn(state)
        runtime.ui.debug(
            f"llm_call start skip_tools={skip_tools} message_count={len(state.get('messages', []))}"
        )
        if skip_tools:
            response = runtime.model.invoke(messages)
        else:
            response = model_with_tools.invoke(messages)
        runtime.ui.debug(
            "llm_call done "
            f"tool_calls={len(getattr(response, 'tool_calls', []) or [])} "
            f"content_len={len(str(getattr(response, 'content', '') or ''))}"
        )
        return {"messages": [response]}

    def tool_exec(state: AgentState) -> dict[str, Any]:
        message = state.get("messages", [])[-1]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return {}

        runtime.ui.debug(f"tool_exec start tool_calls={len(message.tool_calls)}")

        updates: dict[str, Any] = {}
        tool_messages: list[ToolMessage] = []
        pending_clarification: dict[str, Any] | None = None
        current_state = dict(state)

        for call in message.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            runtime.ui.tool_call(name, args)
            result = runtime.registry.execute(
                name,
                ToolContext(
                    workspace_root=runtime.workspace_root,
                    state=current_state,
                    ui=runtime.ui,
                ),
                args,
            )
            if result.interrupt_payload and pending_clarification is None:
                pending_clarification = {
                    **result.interrupt_payload,
                    "tool_call_id": call["id"],
                    "tool_name": name,
                }
            else:
                tool_messages.append(ToolMessage(content=result.content, tool_call_id=call["id"]))
            if result.is_error:
                runtime.ui.warning(result.content)
            else:
                runtime.ui.status(result.content)
            updates = merge_state_updates(updates, result.state_updates)
            current_state = merge_state_updates(current_state, result.state_updates)
            runtime.ui.debug(
                f"tool_exec result name={name} is_error={result.is_error} "
                f"interrupt={bool(result.interrupt_payload)}"
            )

        if tool_messages:
            updates["messages"] = tool_messages
        if pending_clarification:
            updates["pending_clarification"] = pending_clarification
        runtime.ui.debug(
            f"tool_exec done pending_clarification={bool(pending_clarification)} "
            f"tool_messages={len(tool_messages)}"
        )
        return updates

    def clarify_interrupt(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("pending_clarification") or {})
        runtime.ui.debug(
            f"clarify_interrupt waiting tool={payload.get('tool_name')} question={payload.get('question')!r}"
        )
        answer = interrupt(payload)
        answer_text = format_clarification_answer(answer) if isinstance(answer, dict) else str(answer)
        runtime.ui.debug(f"clarify_interrupt resumed answer={answer_text!r}")
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
                        "Do not provide a final answer yet. Continue by calling tools, validating the "
                        "bot project when files change, updating plan items, and completing every item."
                    )
                )
            ]
        }

    def after_llm(state: AgentState) -> Literal["tool_exec", "plan_reminder", "final_response"]:
        message = state.get("messages", [])[-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            runtime.ui.debug("after_llm -> tool_exec")
            return "tool_exec"
        if has_incomplete_plan(state.get("current_plan")):
            runtime.ui.debug("after_llm -> plan_reminder")
            return "plan_reminder"
        runtime.ui.debug("after_llm -> final_response")
        return "final_response"

    def after_tools(state: AgentState) -> Literal["clarify_interrupt", "llm_call"]:
        if state.get("pending_clarification"):
            runtime.ui.debug("after_tools -> clarify_interrupt")
            return "clarify_interrupt"
        runtime.ui.debug("after_tools -> llm_call")
        return "llm_call"

    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_exec", tool_exec)
    builder.add_node("clarify_interrupt", clarify_interrupt)
    builder.add_node("plan_reminder", plan_reminder)
    builder.add_node("final_response", final_response)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", after_llm, ["tool_exec", "plan_reminder", "final_response"])
    builder.add_conditional_edges("tool_exec", after_tools, ["clarify_interrupt", "llm_call"])
    builder.add_edge("clarify_interrupt", "llm_call")
    builder.add_edge("plan_reminder", "llm_call")
    builder.add_edge("final_response", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def should_skip_tools_for_turn(state: AgentState) -> bool:
    if has_incomplete_plan(state.get("current_plan")):
        return False
    latest_user_text = latest_human_text(state)
    if not latest_user_text:
        return False
    if not is_smalltalk_message(latest_user_text):
        return False
    return True


def latest_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", None)
        if message.__class__.__name__ == "HumanMessage" and isinstance(content, str):
            return content.strip()
    return ""


def is_smalltalk_message(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return False
    if len(normalized) > 80:
        return False
    smalltalk_phrases = {
        "salom",
        "assalomu alaykum",
        "assalom alaykum",
        "hello",
        "hi",
        "hey",
        "qalaysan",
        "qalesan",
        "nima gap",
        "yaxshimisan",
        "yaxshimisiz",
    }
    return normalized in smalltalk_phrases


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
    project_dir: str,
    project_catalog: dict[str, Any],
) -> AgentState:
    return {
        "workspace_root": str(workspace_root),
        "project_dir": project_dir,
        "bot_project_dir": project_dir,
        "project_catalog": project_catalog,
        "messages": [],
    }
