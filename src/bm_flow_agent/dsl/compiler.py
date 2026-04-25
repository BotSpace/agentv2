from __future__ import annotations

from copy import deepcopy
from typing import Any

from bm_flow_agent.dsl.catalog import KIND_TO_NODE_TYPE, get_capability_for_kind
from bm_flow_agent.dsl.layout import assign_positions
from bm_flow_agent.dsl.models import DSLDocument, RouteSpec, StepSpec
from bm_flow_agent.dsl.parser import collect_all_steps


def compile_dsl_document(document: DSLDocument) -> dict[str, Any]:
    steps = collect_all_steps(document)
    step_index = {step.id: step for step in steps}

    nodes = [compile_step_to_node(step) for step in steps]

    all_routes = list(document.flow.routes)
    for step in steps:
        all_routes.extend(step.routes)

    edges: list[dict[str, Any]] = []
    for route in all_routes:
        edges.append(compile_route(route, step_index))

    for step in steps:
        edges.extend(compile_implicit_edges(step))

    positions = assign_positions(nodes, edges, [step.id for step in document.flow.triggers])
    for node in nodes:
        node["position"] = positions[node["id"]]

    return {"nodes": nodes, "edges": dedupe_edges(edges)}


def compile_step_to_node(step: StepSpec) -> dict[str, Any]:
    if step.kind == "raw_node":
        node_type = step.node_type or "UnknownNode"
        data = deepcopy(step.data)
    else:
        node_type = KIND_TO_NODE_TYPE[canonical_kind(step.kind)]
        data = compile_step_data(step)
    return {
        "id": step.id,
        "type": node_type,
        "data": data,
        "position": {"x": 0, "y": 0},
    }


def compile_step_data(step: StepSpec) -> dict[str, Any]:
    capability = require_capability(step.kind)
    payload = deepcopy(step.data)

    for field in capability.fields:
        value = read_step_value(step, field.dsl_name)
        if value is None or (isinstance(value, (list, dict)) and not value):
            continue
        if field.dsl_name == "keyboard":
            set_nested(payload, field.export_path, sanitize_keyboard_for_runtime(value))
        elif field.dsl_name == "buttons" and step.kind == "reply_button_trigger":
            set_nested(payload, field.export_path, deepcopy(value))
        else:
            set_nested(payload, field.export_path, deepcopy(value))

    if step.kind == "callback_button_trigger":
        selected = payload.get("selectedCallbacks") or []
        if not selected and step.value is not None:
            payload["selectedCallbacks"] = [step.value]
    if step.kind == "message_trigger" and "context" not in payload:
        payload["context"] = {"value": ""}
    if step.kind == "callback_query_trigger" and "context" not in payload:
        payload["context"] = {"value": ""}
    if step.kind == "custom_code" and "jsCode" not in payload and step.code is not None:
        payload["jsCode"] = step.code

    return payload


def compile_route(route: RouteSpec, step_index: dict[str, StepSpec]) -> dict[str, Any]:
    source_step = step_index.get(route.source)
    source_handle = route.effective_source_handle
    if source_step:
        capability = get_capability_for_kind(source_step.kind)
        if capability and capability.output_routes and source_handle and not source_handle.startswith("output-"):
            source_handle = f"output-{source_handle}"

    edge = {
        "id": build_edge_id(route.source, route.target, source_handle, route.target_handle),
        "source": route.source,
        "target": route.target,
        "type": "floating",
    }
    if source_handle:
        edge["sourceHandle"] = source_handle
    if route.target_handle:
        edge["targetHandle"] = route.target_handle
    return edge


def compile_implicit_edges(step: StepSpec) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    capability = get_capability_for_kind(step.kind)
    if capability and capability.button_target_edges and step.keyboard:
        keyboard = deepcopy(step.keyboard if isinstance(step.keyboard, dict) else step.keyboard.model_dump(by_alias=True))
        for button in iter_keyboard_buttons(keyboard):
            target = button.get("next")
            if target:
                handle = f"target-handler-{step.id}-{button['text']}"
                edges.append(
                    {
                        "id": build_edge_id(step.id, target, handle, None),
                        "source": step.id,
                        "target": target,
                        "type": "floating",
                        "sourceHandle": handle,
                    }
                )
    if capability and capability.selected_message_target and step.target_message_step:
        edges.append(
            {
                "id": build_edge_id(step.id, step.target_message_step, None, "selected-message-target"),
                "source": step.id,
                "target": step.target_message_step,
                "type": "floating",
                "targetHandle": "selected-message-target",
            }
        )
    if step.kind == "raw_node":
        for route in step.routes:
            edges.append(
                {
                    "id": build_edge_id(
                        route.source,
                        route.target,
                        route.effective_source_handle,
                        route.target_handle,
                    ),
                    "source": route.source,
                    "target": route.target,
                    "type": "floating",
                    **({"sourceHandle": route.effective_source_handle} if route.effective_source_handle else {}),
                    **({"targetHandle": route.target_handle} if route.target_handle else {}),
                }
            )
    return edges


def canonical_kind(kind: str) -> str:
    capability = require_capability(kind)
    return capability.kind


def require_capability(kind: str):
    capability = get_capability_for_kind(kind)
    if capability is None:
        raise KeyError(kind)
    return capability


def read_step_value(step: StepSpec, field_name: str) -> Any:
    if hasattr(step, field_name):
        return getattr(step, field_name)
    extra = step.model_extra or {}
    return extra.get(field_name)


def set_nested(payload: dict[str, Any], path: str, value: Any) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def sanitize_keyboard_for_runtime(keyboard: Any) -> dict[str, Any]:
    if isinstance(keyboard, dict):
        payload = deepcopy(keyboard)
    else:
        payload = keyboard.model_dump(by_alias=True, exclude_none=True)
    for button in iter_keyboard_buttons(payload):
        button.pop("next", None)
    return payload


def iter_keyboard_buttons(keyboard: dict[str, Any]) -> list[dict[str, Any]]:
    buttons: list[dict[str, Any]] = []
    for grid_name in ("inline", "reply"):
        for row in keyboard.get(grid_name, []) or []:
            for button in row or []:
                if isinstance(button, dict):
                    buttons.append(button)
    return buttons


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for edge in edges:
        key = (
            edge["source"],
            edge["target"],
            edge.get("sourceHandle"),
            edge.get("targetHandle"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def build_edge_id(source: str, target: str, source_handle: str | None, target_handle: str | None) -> str:
    parts = [source, target, source_handle or "default", target_handle or "default"]
    return "edge__" + "__".join(part.replace(" ", "_") for part in parts)
