from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

from bm_flow_agent.dsl.catalog import TRIGGER_NODE_TYPES, get_capability_for_kind, get_capability_for_node_type
from bm_flow_agent.dsl.models import DSLDocument, RouteSpec, StepSpec
from bm_flow_agent.dsl.parser import create_empty_document


def import_flow_json_to_dsl(data: dict[str, Any], *, name: str | None = None) -> DSLDocument:
    flow = normalize_runtime_flow(data)
    document = create_empty_document(name=name or "imported-flow")
    document.flow.meta.name = name or document.flow.meta.name

    nodes = {node["id"]: deepcopy(node) for node in flow.get("nodes", [])}
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: Counter[str] = Counter()
    for edge in flow.get("edges", []):
        edges_by_source[edge["source"]].append(deepcopy(edge))
        incoming[edge["target"]] += 1

    visited: set[str] = set()
    top_level_routes: list[RouteSpec] = []

    for node_id, node in nodes.items():
        if node_id in visited:
            continue
        if node["type"] == "IfConditionNode" and not node.get("data", {}).get("branches"):
            merged = try_merge_if_chain(node_id, nodes, edges_by_source, visited)
            if merged is not None:
                step, routes = merged
                target_list = document.flow.triggers if incoming[node_id] == 0 else document.flow.steps
                target_list.append(step)
                top_level_routes.extend(routes)
                continue

        step, routes = import_node(node, edges_by_source.get(node_id, []))
        visited.add(node_id)
        target_list = document.flow.triggers if node["type"] in TRIGGER_NODE_TYPES and incoming[node_id] == 0 else document.flow.steps
        target_list.append(step)
        top_level_routes.extend(routes)

    document.flow.routes = top_level_routes
    return document


def normalize_runtime_flow(data: dict[str, Any]) -> dict[str, Any]:
    if "flow" in data and isinstance(data["flow"], dict):
        return data["flow"]
    return data


def import_node(node: dict[str, Any], edges: list[dict[str, Any]]) -> tuple[StepSpec, list[RouteSpec]]:
    node_type = node["type"]
    data = deepcopy(node.get("data", {}))
    capability = get_capability_for_node_type(node_type)

    if capability is None:
        return import_raw_node(node, edges)

    target_message_step = None
    regular_edges: list[dict[str, Any]] = []
    for edge in edges:
        if capability.selected_message_target and edge.get("targetHandle") == "selected-message-target":
            target_message_step = edge["target"]
            continue
        regular_edges.append(edge)

    keyboard = data.get("keyboard")
    if capability.button_target_edges and isinstance(keyboard, dict):
        hydrate_button_targets(node["id"], keyboard, regular_edges)
        regular_edges = [edge for edge in regular_edges if not is_button_edge(node["id"], edge)]

    payload: dict[str, Any] = {"id": node["id"], "kind": capability.kind}
    for field in capability.fields:
        value = None
        for path in field.import_candidates():
            extracted, found = pop_nested(data, path)
            if found:
                value = extracted
                break
        if value is not None:
            payload[field.dsl_name] = value

    if capability.kind == "callback_button_trigger":
        callbacks = payload.get("selected_callbacks", [])
        if isinstance(callbacks, list) and len(callbacks) == 1:
            payload["value"] = callbacks[0]
    if capability.kind == "message_trigger" and "value" not in payload:
        payload["value"] = ""
    if capability.kind == "callback_query_trigger" and "value" not in payload:
        payload["value"] = ""
    if capability.kind == "reply_button_trigger" and "buttons" not in payload:
        payload["buttons"] = []
    if target_message_step:
        payload["target_message_step"] = target_message_step

    payload["data"] = data
    step = StepSpec.model_validate(payload)
    routes = [
        route_from_edge(step.kind, edge)
        for edge in regular_edges
        if not should_skip_edge_for_supported_node(step.kind, edge)
    ]
    return step, routes


def import_raw_node(node: dict[str, Any], edges: list[dict[str, Any]]) -> tuple[StepSpec, list[RouteSpec]]:
    step = StepSpec.model_validate(
        {
            "id": node["id"],
            "kind": "raw_node",
            "node_type": node["type"],
            "data": deepcopy(node.get("data", {})),
            "routes": [route_from_edge("raw_node", edge) for edge in edges],
        }
    )
    return step, []


def route_from_edge(kind: str, edge: dict[str, Any]) -> RouteSpec:
    source_handle = edge.get("sourceHandle")
    on = source_handle
    kind_capability = get_capability_for_kind(kind) if kind != "raw_node" else None
    if kind_capability and kind_capability.output_routes and source_handle and source_handle.startswith("output-"):
        on = source_handle[len("output-") :]
        source_handle = None
    return RouteSpec.model_validate(
        {
            "from": edge["source"],
            "to": edge["target"],
            "on": on,
            "source_handle": source_handle,
            "target_handle": edge.get("targetHandle"),
        }
    )


def should_skip_edge_for_supported_node(kind: str, edge: dict[str, Any]) -> bool:
    capability = get_capability_for_kind(kind)
    if capability and capability.button_target_edges and is_button_edge(edge["source"], edge):
        return True
    if capability and capability.selected_message_target and edge.get("targetHandle") == "selected-message-target":
        return True
    return False


def is_button_edge(node_id: str, edge: dict[str, Any]) -> bool:
    source_handle = edge.get("sourceHandle", "")
    return source_handle.startswith(f"target-handler-{node_id}-")


def hydrate_button_targets(node_id: str, keyboard: dict[str, Any], edges: list[dict[str, Any]]) -> None:
    target_map = {}
    prefix = f"target-handler-{node_id}-"
    for edge in edges:
        source_handle = edge.get("sourceHandle", "")
        if source_handle.startswith(prefix):
            button_text = source_handle[len(prefix) :]
            target_map[button_text] = edge["target"]
    for grid_name in ("inline", "reply"):
        for row in keyboard.get(grid_name, []) or []:
            for button in row or []:
                if isinstance(button, dict) and button.get("text") in target_map:
                    button["next"] = target_map[button["text"]]


def pop_nested(payload: dict[str, Any], path: str) -> tuple[Any, bool]:
    parts = path.split(".")
    cursor = payload
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            return None, False
        parents.append((cursor, part))
        cursor = child
    last = parts[-1]
    if last not in cursor:
        return None, False
    value = cursor.pop(last)
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
    return value, True


def try_merge_if_chain(
    start_id: str,
    nodes: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    visited: set[str],
) -> tuple[StepSpec, list[RouteSpec]] | None:
    chain = [nodes[start_id]]
    current = nodes[start_id]

    while True:
        next_edges = [edge for edge in edges_by_source.get(current["id"], []) if edge.get("sourceHandle") == "next"]
        if len(next_edges) != 1:
            break
        next_node = nodes.get(next_edges[0]["target"])
        if not next_node or next_node["type"] not in {"ElseIfNode", "ElseNode"}:
            break
        chain.append(next_node)
        current = next_node
        if current["type"] == "ElseNode":
            break

    if len(chain) == 1:
        return None

    branches: list[dict[str, Any]] = []
    routes: list[RouteSpec] = []

    for index, chain_node in enumerate(chain):
        data = deepcopy(chain_node.get("data", {}))
        node_type = chain_node["type"]
        if node_type == "ElseNode":
            branches.append({"type": "else"})
            target_edge = next(
                (edge for edge in edges_by_source.get(chain_node["id"], []) if edge.get("sourceHandle") in {None, "", "true"}),
                None,
            )
        else:
            branch_type = "if" if index == 0 else "else_if"
            branches.append(
                {
                    "type": branch_type,
                    "operator": data.get("operator", "AND"),
                    "conditions": data.get("conditions", []),
                }
            )
            target_edge = next(
                (edge for edge in edges_by_source.get(chain_node["id"], []) if edge.get("sourceHandle") in {"true", ""}),
                None,
            )
        if target_edge:
            routes.append(
                RouteSpec.model_validate(
                    {"from": start_id, "to": target_edge["target"], "on": f"branch_{index}"}
                )
            )
        visited.add(chain_node["id"])

    step = StepSpec.model_validate(
        {
            "id": start_id,
            "kind": "if_condition",
            "branches": branches,
            "data": {},
        }
    )
    return step, routes
