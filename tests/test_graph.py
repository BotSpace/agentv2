from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from bm_flow_agent.graph import AgentRuntime, build_agent_graph, initial_state, latest_assistant_text
from bm_flow_agent.tools import build_repo_catalog, build_tool_registry
from bm_flow_agent.tools.base import ToolContext
from bm_flow_agent.ui import ConsoleUI


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeBoundModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def invoke(self, _messages):
        response = self.responses[self.index]
        self.index += 1
        return response


class FakeModel:
    def __init__(self, responses):
        self.responses = responses

    def bind_tools(self, _tools):
        return FakeBoundModel(self.responses)


def test_graph_executes_tools_and_updates_working_dsl() -> None:
    runtime = make_runtime(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_task_plan",
                        "args": {"items": [{"id": "inspect", "title": "Inspect current flow"}]},
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_task_plan_item",
                        "args": {"item_id": "inspect", "status": "completed"},
                        "id": "tool-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Plan created."),
        ]
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        repo_catalog=build_repo_catalog(REPO_ROOT, "assets/flow.json", "agent/workflows/main.flow.yaml"),
    )

    result = graph.invoke({**state, "messages": [HumanMessage(content="create a movie bot flow")]}, config={"configurable": {"thread_id": "t-1"}})

    assert not result.get("__interrupt__")
    assert result["current_plan"]["items"][0]["id"] == "inspect"
    assert latest_assistant_text(result) == "Plan created."


def test_graph_clarification_interrupt_and_resume() -> None:
    runtime = make_runtime(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "request_clarification", "args": {"question": "Which collection should I use?"}, "id": "tool-1", "type": "tool_call"}],
            ),
            AIMessage(content="Thanks, I will use the movies collection."),
        ]
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        repo_catalog=build_repo_catalog(REPO_ROOT, "assets/flow.json", "agent/workflows/main.flow.yaml"),
    )
    config = {"configurable": {"thread_id": "t-clarify"}}

    first = graph.invoke({**state, "messages": [HumanMessage(content="make a flow")]}, config=config)
    assert first["__interrupt__"]
    payload = first["__interrupt__"][0].value
    assert payload["question"] == "Which collection should I use?"

    resumed = graph.invoke(Command(resume="movies"), config=config)
    assert latest_assistant_text(resumed) == "Thanks, I will use the movies collection."


def test_graph_clarification_supports_structured_options() -> None:
    runtime = make_runtime(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "request_clarification",
                        "args": {
                            "question": "Bot qaysi tilda bo'lsin?",
                            "options": [
                                {"id": "uz", "label": "O'zbek"},
                                {"id": "ru", "label": "Rus"},
                            ],
                        },
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="O'zbek tilida davom etaman."),
        ]
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        repo_catalog=build_repo_catalog(REPO_ROOT, "assets/flow.json", "agent/workflows/main.flow.yaml"),
    )
    config = {"configurable": {"thread_id": "t-clarify-options"}}

    first = graph.invoke({**state, "messages": [HumanMessage(content="make a flow")]}, config=config)
    payload = first["__interrupt__"][0].value
    assert payload["options"][0]["id"] == "uz"

    resumed = graph.invoke(Command(resume={"answer": "O'zbek", "option_id": "uz"}), config=config)
    assert latest_assistant_text(resumed) == "O'zbek tilida davom etaman."


def test_graph_clarification_supports_batched_multi_answers() -> None:
    runtime = make_runtime(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "request_clarification",
                        "args": {
                            "questions": [
                                {
                                    "id": "language",
                                    "question": "Til?",
                                    "options": [{"id": "uz", "label": "O'zbek"}],
                                },
                                {
                                    "id": "features",
                                    "question": "Funksiyalar?",
                                    "selection_type": "multiple",
                                    "options": [
                                        {"id": "students", "label": "Talabalar"},
                                        {"id": "payments", "label": "To'lovlar"},
                                    ],
                                },
                            ]
                        },
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Javoblar qabul qilindi."),
        ]
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        repo_catalog=build_repo_catalog(REPO_ROOT, "assets/flow.json", "agent/workflows/main.flow.yaml"),
    )
    config = {"configurable": {"thread_id": "t-clarify-batch"}}

    first = graph.invoke({**state, "messages": [HumanMessage(content="make crm")]}, config=config)
    payload = first["__interrupt__"][0].value
    assert payload["questions"][1]["selection_type"] == "multiple"

    resumed = graph.invoke(
        Command(
            resume={
                "answers": [
                    {"question_id": "language", "answer": "O'zbek", "option_id": "uz"},
                    {
                        "question_id": "features",
                        "answer": "Talabalar, To'lovlar",
                        "option_ids": ["students", "payments"],
                    },
                ]
            }
        ),
        config=config,
    )
    assert latest_assistant_text(resumed) == "Javoblar qabul qilindi."


def test_graph_does_not_finish_until_plan_is_completed() -> None:
    runtime = make_runtime(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_task_plan",
                        "args": {"items": [{"id": "build", "title": "Build flow"}]},
                        "id": "plan-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Too early final."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "update_task_plan_item",
                        "args": {"item_id": "build", "status": "completed"},
                        "id": "plan-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Now complete."),
        ]
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        repo_catalog=build_repo_catalog(REPO_ROOT, "assets/flow.json", "agent/workflows/main.flow.yaml"),
    )

    result = graph.invoke(
        {**state, "messages": [HumanMessage(content="make a flow")]},
        config={"configurable": {"thread_id": "t-plan-loop"}, "recursion_limit": 20},
    )

    assert latest_assistant_text(result) == "Now complete."
    assert result["current_plan"]["items"][0]["status"] == "completed"


def test_runtime_json_write_tool_is_not_exposed() -> None:
    registry = build_tool_registry()
    assert "write_flow_json" not in registry.names()


def test_chat_registry_exposes_only_yaml_flow_save_tools() -> None:
    registry = build_tool_registry(dsl_only=True)
    names = set(registry.names())
    assert {
        "get_flow_yaml",
        "save_flow_yaml",
        "describe_step_kind",
        "upsert_step",
        "connect_steps",
        "patch_step_block",
        "remove_step",
        "create_task_plan",
        "update_task_plan_item",
    }.issubset(names)
    assert "apply_flow_patch" not in names
    assert "write_flow_json" not in names
    assert "compile_dsl_to_json" not in names
    assert "write_dsl_file" not in names


def test_get_flow_yaml_and_save_flow_yaml_roundtrip(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true, "withArgs": false},
      "position": {"x": 0, "y": 0}
    },
    {
      "id": "welcome",
      "type": "SendTextMessageNode",
      "data": {"messageText": "Hello"},
      "position": {"x": 360, "y": 0}
    }
  ],
  "edges": [
    {"id": "e1", "source": "start", "target": "welcome", "type": "floating"}
  ]
}
""".strip(),
        encoding="utf-8",
    )
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )
    yaml_result = registry.execute("get_flow_yaml", context, {})
    assert yaml_result.is_error is False
    assert "flow:" in yaml_result.content
    assert "kind: send_text" in yaml_result.content

    updated_yaml = yaml_result.content.replace("text: Hello", "text: Salom")
    save_result = registry.execute("save_flow_yaml", context, {"yaml": updated_yaml})
    assert save_result.is_error is False
    assert '"messageText": "Salom"' in flow_path.read_text(encoding="utf-8")


def test_yaml_only_mutating_tools_require_task_plan(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path)},
        ui=ConsoleUI(),
    )

    result = registry.execute(
        "upsert_step",
        context,
        {"step": {"id": "welcome", "kind": "send_text", "text": "Salom"}, "trigger": False},
    )

    assert result.is_error is True
    assert "create_task_plan" in result.content


def test_task_plan_tools_emit_events_and_complete() -> None:
    events = []

    class CaptureUI(ConsoleUI):
        def flow_event(self, event_type, payload):
            events.append((event_type, payload))

    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(workspace_root=REPO_ROOT, state={}, ui=CaptureUI())

    created = registry.execute(
        "create_task_plan",
        context,
        {"items": [{"id": "build", "title": "Build flow"}]},
    )
    assert created.is_error is False
    context.state.update(created.state_updates)

    updated = registry.execute(
        "update_task_plan_item",
        context,
        {"item_id": "build", "status": "completed", "summary": "done"},
    )
    assert updated.is_error is False
    assert events[0][0] == "plan_created"
    assert events[1][0] == "plan_item_updated"
    assert events[2][0] == "plan_completed"


def test_yaml_only_incremental_tools_auto_save_flow(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true, "withArgs": false},
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": []
}
""".strip(),
        encoding="utf-8",
    )
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    upsert = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
            "trigger": False,
            "incoming": {"from": "start"},
        },
    )
    assert upsert.is_error is False
    context.state.update(upsert.state_updates)

    saved = flow_path.read_text(encoding="utf-8")
    assert '"type": "SendTextMessageNode"' in saved
    assert '"messageText": "Salom"' in saved
    assert '"source": "start"' in saved
    assert '"target": "welcome"' in saved


def test_yaml_only_tools_keep_draft_when_button_targets_are_created_later(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start_trigger",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true},
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": []
}
""".strip(),
        encoding="utf-8",
    )
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "target_flow_json": str(flow_path),
            "current_plan": {"items": [{"id": "build", "title": "Build", "status": "in_progress"}]},
        },
        ui=ConsoleUI(),
    )

    main_menu = registry.execute(
        "upsert_step",
        context,
        {
            "step": {
                "id": "main_menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [
                        [{"text": "Register", "type": "callback", "value": "register", "next": "registration_start"}],
                        [{"text": "Admin", "type": "callback", "value": "admin", "next": "admin_students_list"}],
                    ],
                },
            },
            "incoming": {"from": "start_trigger"},
        },
    )
    assert main_menu.is_error is False
    assert '"draft": true' in main_menu.content
    context.state.update(main_menu.state_updates)

    for step_id in ("registration_start", "admin_students_list"):
        result = registry.execute(
            "upsert_step",
            context,
            {"step": {"id": step_id, "kind": "send_text", "text": step_id}},
        )
        assert result.is_error is False
        context.state.update(result.state_updates)

    completed = registry.execute(
        "update_task_plan_item",
        context,
        {"item_id": "build", "status": "completed"},
    )
    assert completed.is_error is False
    saved = flow_path.read_text(encoding="utf-8")
    assert '"id": "main_menu"' in saved
    assert '"target": "registration_start"' in saved
    assert '"target": "admin_students_list"' in saved


def test_yaml_only_upsert_is_single_step_and_connect_can_bulk_routes(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    try:
        registry.execute(
            "upsert_step",
            context,
            {
                "items": [
                    {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
                    {"step": {"id": "welcome", "kind": "send_text", "text": "Salom"}, "trigger": False},
                    {"step": {"id": "ask_name", "kind": "send_text", "text": "Ismingiz?"}, "trigger": False},
                ]
            },
        )
    except Exception as exc:
        assert "step" in str(exc)
    else:
        raise AssertionError("Expected bulk upsert payload to be rejected")

    for payload in (
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
        {
            "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
            "trigger": False,
            "incoming": {"from": "start"},
        },
        {
            "step": {"id": "ask_name", "kind": "send_text", "text": "Ismingiz?"},
            "trigger": False,
            "incoming": {"from": "welcome"},
        },
    ):
        upsert = registry.execute("upsert_step", context, payload)
        assert upsert.is_error is False
        context.state.update(upsert.state_updates)

    connect = registry.execute(
        "connect_steps",
        context,
        {"routes": [{"from": "start", "to": "welcome"}, {"from": "welcome", "to": "ask_name"}]},
    )
    assert connect.is_error is False

    saved = flow_path.read_text(encoding="utf-8")
    assert saved.count('"type": "SendTextMessageNode"') == 2
    assert '"source": "welcome"' in saved
    assert '"target": "ask_name"' in saved


def test_save_flow_yaml_rejects_invalid_yaml_without_overwrite(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    original = '{"nodes": [], "edges": []}'
    flow_path.write_text(original, encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    result = registry.execute("save_flow_yaml", context, {"yaml": "flow: [broken"})

    assert result.is_error is True
    assert flow_path.read_text(encoding="utf-8") == original


def test_dsl_only_registry_blocks_runtime_json_summary() -> None:
    registry = build_tool_registry(dsl_only=True)
    assert "get_current_flow_summary" not in registry.names()
    assert "get_working_dsl_summary" not in registry.names()
    assert "analyze_flow_connectivity" in registry.names()


def test_analyze_flow_connectivity_reports_unreachable_steps() -> None:
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={
            "target_flow_json": "assets/flow.json",
            "working_dsl": {
                "flow": {
                    "meta": {"name": "demo", "version": 1, "tags": []},
                    "triggers": [{"id": "start", "kind": "command_trigger", "command": "/start"}],
                    "steps": [{"id": "welcome", "kind": "send_text", "text": "Salom"}],
                    "routes": [],
                }
            },
        },
        ui=ConsoleUI(),
    )

    result = registry.execute("analyze_flow_connectivity", context, {})

    assert result.is_error is False
    assert '"id": "welcome"' in result.content
    assert "unreachable_steps" in result.content


def test_describe_step_kind_includes_purpose_and_action_routing_guidance() -> None:
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(workspace_root=REPO_ROOT, state={"target_flow_json": "assets/flow.json"}, ui=ConsoleUI())

    result = registry.execute("describe_step_kind", context, {"kind": "send_text"})

    assert result.is_error is False
    assert "Purpose: Foydalanuvchiga matnli Telegram xabar yuboradi." in result.content
    assert 'incoming: {from: "reachable_source_id"}' in result.content


def test_plan_completion_blocks_unreachable_action_then_allows_repair(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path)},
        ui=ConsoleUI(),
    )

    plan = registry.execute(
        "create_task_plan",
        context,
        {"items": [{"id": "build", "title": "Build flow"}]},
    )
    context.state.update(plan.state_updates)
    start = registry.execute(
        "upsert_step",
        context,
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
    )
    assert start.is_error is False
    context.state.update(start.state_updates)

    blocked = registry.execute(
        "upsert_step",
        context,
        {"step": {"id": "welcome", "kind": "send_text", "text": "Salom"}},
    )
    assert blocked.is_error is True
    assert "new action steps must include `incoming" in blocked.content

    repaired = registry.execute(
        "upsert_step",
        context,
        {"step": {"id": "welcome", "kind": "send_text", "text": "Salom"}, "incoming": {"from": "start"}},
    )
    assert repaired.is_error is False
    context.state.update(repaired.state_updates)

    completed = registry.execute(
        "update_task_plan_item",
        context,
        {"item_id": "build", "status": "completed"},
    )
    assert completed.is_error is False
    assert '"source": "start"' in flow_path.read_text(encoding="utf-8")


def test_upsert_step_can_create_action_with_incoming_edge(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true},
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": []
}
""".strip(),
        encoding="utf-8",
    )
    events = []

    class CaptureUI(ConsoleUI):
        def flow_event(self, event_type, payload):
            events.append((event_type, payload))

    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=CaptureUI(),
    )

    result = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
            "incoming": {"from": "start"},
        },
    )

    assert result.is_error is False
    saved = flow_path.read_text(encoding="utf-8")
    assert '"id": "welcome"' in saved
    assert '"source": "start"' in saved
    assert '"target": "welcome"' in saved
    assert [event[0] for event in events] == ["node_upserted", "edge_upserted"]
    assert events[1][1]["route"] == {"from": "start", "to": "welcome"}


def test_upsert_step_rejects_incoming_target() -> None:
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    try:
        registry.execute(
            "upsert_step",
            context,
            {
                "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
                "incoming": {"from": "start", "to": "welcome"},
            },
        )
    except Exception as exc:
        assert "incoming must not include `to`" in str(exc)
    else:
        raise AssertionError("Expected incoming.to validation error")


def test_upsert_step_accepts_step_and_incoming_json_strings(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true},
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": []
}
""".strip(),
        encoding="utf-8",
    )
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    result = registry.execute(
        "upsert_step",
        context,
        {
            "step": '{"id":"welcome","kind":"send_text","text":"Salom"}',
            "incoming": '{"from":"start"}',
        },
    )

    assert result.is_error is False
    saved = flow_path.read_text(encoding="utf-8")
    assert '"id": "welcome"' in saved
    assert '"source": "start"' in saved


def test_upsert_step_button_incoming_updates_source_button_next(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    events = []

    class CaptureUI(ConsoleUI):
        def flow_event(self, event_type, payload):
            events.append((event_type, payload))

    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=CaptureUI(),
    )

    for payload in (
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
        {
            "step": {
                "id": "menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Register", "type": "callback", "value": "register"}]],
                },
            },
            "incoming": {"from": "start"},
        },
    ):
        result = registry.execute("upsert_step", context, payload)
        assert result.is_error is False
        context.state.update(result.state_updates)

    target = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "registration_start", "kind": "send_text", "text": "Ismingiz?"},
            "incoming": {"from": "menu", "via": "button", "button_text": "Register"},
        },
    )

    assert target.is_error is False
    saved = flow_path.read_text(encoding="utf-8")
    assert '"source": "menu"' in saved
    assert '"target": "registration_start"' in saved
    assert '"sourceHandle": "target-handler-menu-Register"' in saved
    assert '"sourceHandle": "default"' not in saved
    edge_events = [event for event in events if event[0] == "edge_upserted"]
    assert edge_events[-1][1]["button"] == {
        "from": "menu",
        "to": "registration_start",
        "button_text": "Register",
        "via": "button",
    }
    assert edge_events[-1][1]["edge"]["sourceHandle"] == "target-handler-menu-Register"


def test_upsert_step_blocks_default_incoming_from_keyboard_source(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )
    for payload in (
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
        {
            "step": {
                "id": "menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Register", "type": "callback", "value": "register"}]],
                },
            },
            "incoming": {"from": "start"},
        },
    ):
        result = registry.execute("upsert_step", context, payload)
        assert result.is_error is False
        context.state.update(result.state_updates)

    blocked = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "registration_start", "kind": "send_text", "text": "Ismingiz?"},
            "incoming": {"from": "menu"},
        },
    )
    explicit_route = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "auto_next", "kind": "send_text", "text": "Auto"},
            "incoming": {"from": "menu", "via": "route"},
        },
    )

    assert blocked.is_error is True
    assert 'via: "button"' in blocked.content
    assert 'via: "route"' in blocked.content
    assert explicit_route.is_error is False


def test_upsert_step_button_incoming_requires_existing_button_text(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )
    for payload in (
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
        {
            "step": {
                "id": "menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Register", "type": "callback", "value": "register"}]],
                },
            },
            "incoming": {"from": "start"},
        },
    ):
        result = registry.execute("upsert_step", context, payload)
        assert result.is_error is False
        context.state.update(result.state_updates)

    missing = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "registration_start", "kind": "send_text", "text": "Ismingiz?"},
            "incoming": {"from": "menu", "via": "button", "button_text": "Missing"},
        },
    )

    assert missing.is_error is True
    assert "button_text not found" in missing.content


def test_upsert_step_blocks_global_callback_triggers_for_keyboard_buttons(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    for kind, payload in (
        ("callback_query_trigger", {"value": "register"}),
        ("callback_button_trigger", {"selected_callbacks": ["register"]}),
    ):
        result = registry.execute(
            "upsert_step",
            context,
            {
                "step": {
                    "id": f"{kind}_global",
                    "kind": kind,
                    "global": True,
                    **payload,
                },
                "trigger": True,
            },
        )

        assert result.is_error is True
        assert "Global callback trigger blocked" in result.content
        assert 'incoming: {from: "menu_step", via: "button", button_text: "Button text"}' in result.content


def test_upsert_step_allows_non_global_callback_trigger_with_incoming(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    for payload in (
        {"step": {"id": "start", "kind": "command_trigger", "command": "/start"}, "trigger": True},
        {"step": {"id": "ask", "kind": "send_text", "text": "Press callback later"}, "incoming": {"from": "start"}},
        {
            "step": {"id": "wait_callback", "kind": "callback_query_trigger", "value": "confirm"},
            "trigger": True,
            "incoming": {"from": "ask"},
        },
    ):
        result = registry.execute("upsert_step", context, payload)
        assert result.is_error is False
        context.state.update(result.state_updates)

    saved = flow_path.read_text(encoding="utf-8")
    assert '"id": "wait_callback"' in saved


def test_upsert_step_trigger_can_use_optional_incoming(tmp_path: Path) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        """
{
  "nodes": [
    {
      "id": "start",
      "type": "CommandTriggerNode",
      "data": {"command": "/start", "global": true},
      "position": {"x": 0, "y": 0}
    }
  ],
  "edges": []
}
""".strip(),
        encoding="utf-8",
    )
    registry = build_tool_registry(dsl_only=True)
    context = ToolContext(
        workspace_root=tmp_path,
        state={"target_flow_json": str(flow_path), "allow_unplanned_edits": True},
        ui=ConsoleUI(),
    )

    result = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "next_command", "kind": "command_trigger", "command": "/next"},
            "trigger": True,
            "incoming": {"from": "start"},
        },
    )

    assert result.is_error is False
    saved = flow_path.read_text(encoding="utf-8")
    assert '"id": "next_command"' in saved
    assert '"target": "next_command"' in saved


def test_upsert_step_requires_strict_payload_shape() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )
    try:
        registry.execute(
            "upsert_step",
            context,
            {
                "trigger": "command_trigger",
                "step": {"id": "start_trigger", "kind": "command_trigger", "command": "/start"},
            },
        )
    except Exception as exc:
        message = str(exc)
        assert "boolean" in message.lower()
    else:
        raise AssertionError("Expected strict validation error for invalid trigger payload")


def test_upsert_step_supports_single_step_with_incoming(tmp_path: Path) -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "target_flow_json": "flow.json",
            "target_dsl_path": "agent/workflows/nonexistent.flow.yaml",
            "current_plan": {"items": [{"id": "create_nodes", "title": "Create nodes", "status": "in_progress"}]},
            "working_dsl": {
                "flow": {
                    "meta": {"name": "demo", "version": 1, "tags": []},
                    "triggers": [],
                    "steps": [],
                    "routes": [],
                }
            },
        },
        ui=ConsoleUI(),
    )
    trigger_result = registry.execute(
        "upsert_step",
        context,
        {"step": {"id": "start_trigger", "kind": "command_trigger", "command": "/start"}, "trigger": True},
    )
    assert trigger_result.is_error is False
    context.state = {**context.state, **trigger_result.state_updates}

    result = registry.execute(
        "upsert_step",
        context,
        {
            "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
            "incoming": {"from": "start_trigger"},
        },
    )
    assert result.is_error is False
    flow = result.state_updates["working_dsl"]["flow"]
    assert [item["id"] for item in flow["triggers"]] == ["start_trigger"]
    assert [item["id"] for item in flow["steps"]] == ["welcome"]
    assert flow["routes"] == [{"from": "start_trigger", "to": "welcome"}]


def test_connect_steps_supports_bulk_routes(tmp_path: Path) -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "target_flow_json": "flow.json",
            "target_dsl_path": "agent/workflows/nonexistent.flow.yaml",
            "current_plan": {"items": [{"id": "connect_routes", "title": "Connect routes", "status": "in_progress"}]},
            "working_dsl": {
                "flow": {
                    "meta": {"name": "demo", "version": 1, "tags": []},
                    "triggers": [{"id": "start_trigger", "kind": "command_trigger", "command": "/start"}],
                    "steps": [
                        {"id": "welcome", "kind": "send_text", "text": "Salom"},
                        {"id": "ask_name", "kind": "send_text", "text": "Ismingizni kiriting"},
                    ],
                    "routes": [],
                }
            },
        },
        ui=ConsoleUI(),
    )
    result = registry.execute(
        "connect_steps",
        context,
        {
            "routes": [
                {"from": "start_trigger", "to": "welcome"},
                {"from": "welcome", "to": "ask_name"},
            ]
        },
    )
    assert result.is_error is False
    assert result.state_updates["working_dsl"]["flow"]["routes"] == [
        {"from": "start_trigger", "to": "welcome"},
        {"from": "welcome", "to": "ask_name"},
    ]


def test_apply_flow_patch_is_not_exposed() -> None:
    registry = build_tool_registry()
    assert "apply_flow_patch" not in registry.names()


def make_runtime(responses):
    return AgentRuntime(
        workspace_root=REPO_ROOT,
        target_flow_json="assets/flow.json",
        target_dsl_path="agent/workflows/main.flow.yaml",
        model=FakeModel(responses),
        registry=build_tool_registry(),
        ui=ConsoleUI(),
    )
