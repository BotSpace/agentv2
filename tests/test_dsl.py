from __future__ import annotations

import json
from pathlib import Path

from bm_flow_agent.dsl import (
    compile_dsl_document,
    create_empty_document,
    import_flow_json_to_dsl,
    validate_compiled_flow,
    validate_dsl_document,
)
from bm_flow_agent.dsl.models import RouteSpec, StepSpec
from bm_flow_agent.dsl.parser import replace_step, upsert_route


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_and_compile_roundtrip_examples() -> None:
    for relative_path in ("examples/main.json", "examples/trigger_state_example_flow.json"):
        payload = json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        document = import_flow_json_to_dsl(payload)
        compiled = compile_dsl_document(document)
        assert runtime_signature(payload) == runtime_signature(compiled)


def test_import_assets_flow_smoke() -> None:
    payload = json.loads((REPO_ROOT / "assets/flow.json").read_text(encoding="utf-8"))
    document = import_flow_json_to_dsl(payload)
    compiled = compile_dsl_document(document)
    flow = payload.get("flow", payload)
    assert len(compiled["nodes"]) == len(flow["nodes"])
    assert len(compiled["edges"]) == len(flow["edges"])


def test_compile_button_edges_and_raw_node_passthrough() -> None:
    document = create_empty_document("sample")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )
    replace_step(
        document,
        StepSpec.model_validate(
            {
                "id": "welcome",
                "kind": "send_text",
                "text": "Hi",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Next", "type": "callback", "value": "next", "next": "custom"}]],
                },
            }
        ),
    )
    replace_step(
        document,
        StepSpec.model_validate(
            {
                "id": "custom",
                "kind": "raw_node",
                "node_type": "DownloadNode",
                "data": {"url": "https://example.com"},
                "routes": [{"from": "custom", "to": "done", "on": "success"}],
            }
        ),
    )
    replace_step(document, StepSpec.model_validate({"id": "done", "kind": "send_text", "text": "Done"}))
    upsert_route(document, RouteSpec.model_validate({"from": "start", "to": "welcome"}))

    compiled = compile_dsl_document(document)
    assert any(edge.get("sourceHandle") == "target-handler-welcome-Next" for edge in compiled["edges"])
    assert any(node["type"] == "DownloadNode" and node["data"]["url"] == "https://example.com" for node in compiled["nodes"])


def test_validate_dsl_and_compiled_flow() -> None:
    document = create_empty_document("broken")
    replace_step(document, StepSpec.model_validate({"id": "msg", "kind": "send_text", "text": "hi"}))
    upsert_route(document, RouteSpec.model_validate({"from": "missing", "to": "msg"}))

    dsl_errors = validate_dsl_document(document)
    compiled_errors = validate_compiled_flow(compile_dsl_document(document))

    assert any("at least one trigger" in error for error in dsl_errors)
    assert any("route source not found" in error for error in dsl_errors)
    assert any("edge source missing node" in error for error in compiled_errors)


def test_validate_rejects_action_steps_unreachable_from_trigger() -> None:
    document = create_empty_document("unreachable")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )
    replace_step(document, StepSpec.model_validate({"id": "welcome", "kind": "send_text", "text": "Hi"}))

    errors = validate_dsl_document(document)

    assert any("action step is not reachable from any trigger: welcome (send_text)" in error for error in errors)


def test_validate_accepts_route_chain_from_trigger() -> None:
    document = create_empty_document("reachable")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )
    replace_step(document, StepSpec.model_validate({"id": "welcome", "kind": "send_text", "text": "Hi"}))
    replace_step(document, StepSpec.model_validate({"id": "done", "kind": "send_text", "text": "Done"}))
    upsert_route(document, RouteSpec.model_validate({"from": "start", "to": "welcome"}))
    upsert_route(document, RouteSpec.model_validate({"from": "welcome", "to": "done"}))

    errors = validate_dsl_document(document)

    assert not any("not reachable from any trigger" in error for error in errors)


def test_validate_accepts_keyboard_next_from_reachable_action() -> None:
    document = create_empty_document("keyboard")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )
    replace_step(
        document,
        StepSpec.model_validate(
            {
                "id": "menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Next", "type": "callback", "value": "next", "next": "details"}]],
                },
            }
        ),
    )
    replace_step(document, StepSpec.model_validate({"id": "details", "kind": "send_text", "text": "Details"}))
    upsert_route(document, RouteSpec.model_validate({"from": "start", "to": "menu"}))

    errors = validate_dsl_document(document)

    assert not any("not reachable from any trigger" in error for error in errors)


def test_validate_rejects_global_callback_trigger_for_static_button_value() -> None:
    document = create_empty_document("bad-callback")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )
    replace_step(
        document,
        StepSpec.model_validate(
            {
                "id": "menu",
                "kind": "send_text",
                "text": "Menu",
                "keyboard": {
                    "active": "inline",
                    "inline": [[{"text": "Register", "type": "callback", "value": "register"}]],
                },
            }
        ),
    )
    replace_step(
        document,
        StepSpec.model_validate(
            {
                "id": "register_global",
                "kind": "callback_query_trigger",
                "global": True,
                "value": "register",
            }
        ),
        as_trigger=True,
    )
    upsert_route(document, RouteSpec.model_validate({"from": "start", "to": "menu"}))

    errors = validate_dsl_document(document)

    assert any("global callback trigger duplicates a keyboard button value" in error for error in errors)
    assert any('incoming: {from: "menu_step", via: "button", button_text: "Button text"}' in error for error in errors)


def test_validate_accepts_trigger_only_flow() -> None:
    document = create_empty_document("trigger-only")
    replace_step(
        document,
        StepSpec.model_validate({"id": "start", "kind": "command_trigger", "command": "/start"}),
        as_trigger=True,
    )

    errors = validate_dsl_document(document)

    assert not any("not reachable from any trigger" in error for error in errors)


def runtime_signature(payload: dict) -> dict:
    flow = payload.get("flow", payload)
    nodes = []
    for node in flow["nodes"]:
        data = normalize_node_data(node["type"], node.get("data", {}))
        nodes.append({"id": node["id"], "type": node["type"], "data": data})
    edges = []
    for edge in flow["edges"]:
        item = {"source": edge["source"], "target": edge["target"]}
        if "sourceHandle" in edge:
            item["sourceHandle"] = edge["sourceHandle"]
        if "targetHandle" in edge:
            item["targetHandle"] = edge["targetHandle"]
        edges.append(item)
    return {
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(
            edges,
            key=lambda item: (item["source"], item["target"], item.get("sourceHandle", ""), item.get("targetHandle", "")),
        ),
    }


def normalize_node_data(node_type: str, data: dict) -> dict:
    payload = json.loads(json.dumps(data))
    keyboard = payload.get("keyboard")
    if isinstance(keyboard, dict):
        if keyboard.get("reply") == []:
            keyboard.pop("reply", None)
        if keyboard.get("inline") == []:
            keyboard.pop("inline", None)
    if node_type in {"CallbackQueryTriggerNode", "MessageTriggerNode"} and payload.get("context") == {"value": ""}:
        payload["context"] = {}
    return payload
