from __future__ import annotations

from pathlib import Path

from bm_flow_agent.dsl.catalog import CATALOG, describe_kind, example_for_field, native_kinds, runtime_node_types
from bm_flow_agent.dsl.compiler import compile_dsl_document
from bm_flow_agent.dsl.importer import import_flow_json_to_dsl
from bm_flow_agent.dsl.models import DSLDocument, FlowDocument, FlowMeta, StepSpec
from bm_flow_agent.prompts import build_system_prompt
from bm_flow_agent.tools import build_tool_registry
from bm_flow_agent.tools.base import ToolContext
from bm_flow_agent.ui import ConsoleUI


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_describe_step_kind_reports_capabilities() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )
    result = registry.execute("describe_step_kind", context, {"kind": "send_poll"})
    assert result.is_error is False
    assert "Kind: send_poll" in result.content
    assert "SendPollNode" in result.content
    assert "question (optional) -> runtime `question`" in result.content
    assert "options (optional) -> runtime `options`" in result.content


def test_describe_step_kind_explains_video_fields_and_routing() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )
    result = registry.execute("describe_step_kind", context, {"kind": "send_video"})
    assert result.is_error is False
    assert "video_url (optional) -> runtime `videoUrl`" in result.content
    assert "AI prompt: Use `video_url`" in result.content
    assert "Example: `video_url: https://example.com/video.mp4`" in result.content
    assert "caption (optional) -> runtime `caption`" in result.content
    assert "Keyboard buttons may declare `next`" in result.content


def test_explain_engine_flow_model_reports_trigger_priority() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )

    result = registry.execute("explain_engine_flow_model", context, {})

    assert result.is_error is False
    assert "Global triggers priority 1" in result.content
    assert "Waiting triggers priority 2" in result.content
    assert "Root triggers priority 3" in result.content
    assert "Action node cannot start a flow" in result.content
    assert "Inline button `value` is matched" in result.content
    assert "do not create a global callback trigger" in result.content


def test_describe_step_kind_explains_trigger_and_action_engine_behavior() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )

    trigger = registry.execute("describe_step_kind", context, {"kind": "message_trigger"})
    action = registry.execute("describe_step_kind", context, {"kind": "send_text"})

    assert trigger.is_error is False
    assert "root trigger" in trigger.content
    assert "waiting trigger" in trigger.content
    assert action.is_error is False
    assert "Action node cannot start a flow" in action.content
    assert "If this action has keyboard buttons with `next`" in action.content
    assert "no global callback trigger is needed" in action.content
    assert 'via: "button", button_text: "Button text"' in action.content

    callback = registry.execute("describe_step_kind", context, {"kind": "callback_query_trigger"})
    assert callback.is_error is False
    assert "Do not use this as a global trigger for static inline keyboard menu buttons" in callback.content


def test_describe_step_kind_accepts_multiple_kinds() -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=REPO_ROOT,
        state={"target_flow_json": "assets/flow.json", "target_dsl_path": "agent/workflows/main.flow.yaml"},
        ui=ConsoleUI(),
    )
    result = registry.execute("describe_step_kind", context, {"kinds": ["send_video", "send_poll"]})
    assert result.is_error is False
    assert "Kind: send_video" in result.content
    assert "Kind: send_poll" in result.content
    assert "---" in result.content


def test_system_prompt_pushes_capability_lookup_before_authoring() -> None:
    prompt = build_system_prompt(
        {
            "target_flow_json": "assets/flow.json",
            "target_dsl_path": "agent/workflows/main.flow.yaml",
            "repo_catalog": {
                "supported_node_types": ["SendTextMessageNode"],
                "native_dsl_kinds": ["send_text", "send_video"],
            },
        }
    )
    assert "first call `describe_step_kind` with `kinds: [...]`" in prompt
    assert "Prefer `describe_step_kind` over guessing field names" in prompt
    assert "Global triggers priority 1" in prompt
    assert "Waiting triggers priority 2" in prompt
    assert "Root triggers priority 3" in prompt
    assert "Action node cannot start a flow" in prompt
    assert "Waiting trigger is reached from an action and pauses flow" in prompt
    assert "Do not create global callback triggers for keyboard buttons" in prompt
    assert "button `value` is matched by the engine automatically" in prompt
    assert 'incoming: {from: "menu_step", via: "button", button_text: "Button text"}' in prompt
    assert "runtime JSON" not in prompt


def test_every_catalog_field_has_ai_prompt_and_example() -> None:
    for capability in CATALOG:
        description = describe_kind(capability.kind)
        assert description is not None
        for field in description["fields"]:
            assert field["ai_prompt"]
            assert "Example:" in field["ai_prompt"]
            assert example_for_field(field["name"]) != '"value"'


def test_all_native_kinds_compile_and_import_without_raw_nodes() -> None:
    triggers: list[StepSpec] = []
    steps: list[StepSpec] = []

    for index, kind in enumerate(native_kinds()):
        step_id = f"{kind}_{index}"
        payload = minimal_step_payload(kind, step_id)
        step = StepSpec.model_validate(payload)
        if kind.endswith("_trigger"):
            triggers.append(step)
        else:
            steps.append(step)

    document = DSLDocument.model_validate(
        {
            "flow": FlowDocument(
                meta=FlowMeta(name="full-native"),
                triggers=triggers,
                steps=steps,
                routes=[],
            ).model_dump(exclude_none=True, by_alias=True)
        }
    )

    compiled = compile_dsl_document(document)
    node_types = {node["type"] for node in compiled["nodes"]}
    assert node_types == set(runtime_node_types())

    imported = import_flow_json_to_dsl(compiled, name="roundtrip")
    imported_kinds = {step.kind for step in [*imported.flow.triggers, *imported.flow.steps]}
    assert "raw_node" not in imported_kinds
    assert imported_kinds == set(native_kinds())


def minimal_step_payload(kind: str, step_id: str) -> dict[str, object]:
    common = {"id": step_id, "kind": kind}
    if kind == "command_trigger":
        return {**common, "command": "/start"}
    if kind == "message_trigger":
        return {**common, "message_type": "text", "filter": "any", "value": ""}
    if kind == "callback_query_trigger":
        return {**common, "filter": "any", "value": ""}
    if kind == "callback_button_trigger":
        return {**common, "selected_callbacks": ["open_menu"]}
    if kind == "reply_button_trigger":
        return {**common, "buttons": [{"text": "Menu"}]}
    if kind == "external_webhook_trigger":
        return {**common, "provider": "stripe"}
    if kind == "cron_trigger":
        return {**common, "schedule": "0 9 * * *", "enabled": True, "target_chat_ids": [1]}
    if kind == "send_text":
        return {**common, "text": "hello"}
    if kind == "send_photo":
        return {**common, "photo_url": "https://example.com/photo.jpg"}
    if kind == "send_video":
        return {**common, "video_url": "https://example.com/video.mp4"}
    if kind == "send_audio":
        return {**common, "audio_url": "https://example.com/audio.mp3"}
    if kind == "send_file":
        return {**common, "file_url": "https://example.com/file.pdf"}
    if kind == "send_animation":
        return {**common, "animation_url": "https://example.com/anim.gif"}
    if kind == "send_voice":
        return {**common, "voice_url": "https://example.com/voice.ogg"}
    if kind == "send_video_note":
        return {**common, "video_note_url": "https://example.com/note.mp4"}
    if kind == "send_location":
        return {**common, "latitude": 41.31, "longitude": 69.24}
    if kind == "send_contact":
        return {**common, "phone_number": "+998900000000", "first_name": "Bot"}
    if kind == "send_poll":
        return {**common, "question": "Choose", "options": ["A", "B"]}
    if kind == "send_sticker":
        return {**common, "sticker_url": "https://example.com/sticker.webp"}
    if kind == "send_media_group":
        return {
            **common,
            "media_items": [
                {"type": "photo", "sourceType": "url", "url": "https://example.com/1.jpg"},
                {"type": "photo", "sourceType": "url", "url": "https://example.com/2.jpg"},
            ],
        }
    if kind == "send_venue":
        return {**common, "latitude": 41.31, "longitude": 69.24, "title": "Office", "address": "Main st"}
    if kind == "send_dice":
        return {**common, "emoji": "🎲"}
    if kind == "edit_message":
        return {**common, "text": "edited", "target_message_step": "send_text_7"}
    if kind == "edit_or_send_text":
        return {**common, "text": "maybe edit"}
    if kind == "delete_message":
        return {**common, "target_message_step": "send_text_7"}
    if kind == "forward_message":
        return {**common, "disable_notification": True}
    if kind == "copy_message":
        return {**common, "from_chat_id": 1, "message_id": 1}
    if kind == "pin_message":
        return {**common, "message_id": 1}
    if kind == "unpin_message":
        return {**common, "message_id": 1}
    if kind == "unpin_all_messages":
        return common
    if kind == "chat_action":
        return {**common, "action": "typing"}
    if kind == "callback_query_answer":
        return {**common, "text": "done"}
    if kind == "answer_inline_query":
        return {
            **common,
            "results": [{"type": "article", "title": "Result", "messageText": "Body"}],
        }
    if kind == "check_membership":
        return {**common, "channels": [{"value": "@channel"}]}
    if kind == "if_condition":
        return {**common, "branches": [{"type": "if", "conditions": []}, {"type": "else"}]}
    if kind == "else_if":
        return {**common, "conditions": []}
    if kind == "else":
        return common
    if kind == "delay":
        return {**common, "delay_seconds": 5}
    if kind == "scheduler":
        return {**common, "delay_seconds": 30, "target_node_id": "send_text_7"}
    if kind == "random":
        return {**common, "options": [{"label": "A"}, {"label": "B"}]}
    if kind == "variable":
        return {**common, "operation": "set", "variable_name": "counter", "value": 1}
    if kind == "for_loop":
        return {**common, "loop_mode": "range", "range_start": 1, "range_end": 3}
    if kind == "for_loop_continue":
        return {**common, "loop_id": "loop_1"}
    if kind == "http_request":
        return {**common, "url": "https://example.com", "method": "GET"}
    if kind == "send_to_admin":
        return common
    if kind == "custom_code":
        return {**common, "code": "exit('done')"}
    if kind == "state":
        return {**common, "state_key": "name"}
    if kind == "collection":
        return {**common, "collection_name": "movies"}
    if kind == "load_collection_item":
        return {**common, "collection_name": "movies", "context_key": "movie"}
    if kind == "load_collection_list":
        return {**common, "collection_name": "movies", "context_key": "movies"}
    if kind == "update_collection":
        return {**common, "collection_name": "movies"}
    if kind == "delete_collection":
        return {**common, "collection_name": "movies"}
    if kind == "download":
        return {**common, "key": "file_key"}
    if kind == "cron":
        return {**common, "schedule": "0 9 * * *", "enabled": True}
    if kind == "subflow":
        return {
            **common,
            "flow": {
                "nodes": [
                    {
                        "id": "sub_trigger",
                        "type": "CommandTriggerNode",
                        "data": {"command": "/x"},
                        "position": {"x": 0, "y": 0},
                    }
                ],
                "edges": [],
            },
        }
    if kind == "subflow_exit":
        return {**common, "output": "done"}
    raise AssertionError(f"Unhandled kind: {kind}")
