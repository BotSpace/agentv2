from __future__ import annotations

from typing import Any

from bm_flow_agent.dsl.catalog import KIND_TO_NODE_TYPE, get_capability_for_kind
from bm_flow_agent.dsl.compiler import compile_dsl_document
from bm_flow_agent.dsl.models import DSLDocument
from bm_flow_agent.dsl.parser import collect_all_steps


GLOBAL_CALLBACK_TRIGGER_KINDS = {"callback_query_trigger", "callback_button_trigger"}


def validate_dsl_document(document: DSLDocument) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    trigger_ids = {step.id for step in document.flow.triggers}
    step_ids = set()

    for step in collect_all_steps(document):
        if step.id in seen_ids:
            errors.append(f"duplicate step id: {step.id}")
        seen_ids.add(step.id)
        step_ids.add(step.id)
        if step.kind != "raw_node" and step.kind not in KIND_TO_NODE_TYPE:
            errors.append(f"unsupported step kind: {step.kind}")

    for route in document.flow.routes:
        if route.source not in step_ids:
            errors.append(f"route source not found: {route.source}")
        if route.target not in step_ids:
            errors.append(f"route target not found: {route.target}")

    for step in collect_all_steps(document):
        capability = None if step.kind == "raw_node" else get_capability_for_kind(step.kind)
        if capability and capability.selected_message_target and step.target_message_step and step.target_message_step not in step_ids:
            errors.append(f"edit_message target_message_step not found: {step.target_message_step}")
        if capability and capability.button_target_edges and step.keyboard:
            keyboard = step.keyboard if isinstance(step.keyboard, dict) else step.keyboard.model_dump(by_alias=True)
            for grid in ("inline", "reply"):
                for row in keyboard.get(grid, []) or []:
                    for button in row or []:
                        if isinstance(button, dict):
                            next_step = button.get("next")
                            if next_step and next_step not in step_ids:
                                errors.append(f"button next target not found: {step.id} -> {next_step}")

    errors.extend(validate_global_callback_trigger_misuse(document))

    if not trigger_ids:
        errors.append("flow must contain at least one trigger")
    else:
        connectivity = analyze_dsl_connectivity(document)
        for step in connectivity["unreachable_steps"]:
            errors.append(
                "action step is not reachable from any trigger: "
                f"{step['id']} ({step['kind']}). Connect it from a trigger or from a reachable action step."
            )

    return errors


def validate_global_callback_trigger_misuse(document: DSLDocument) -> list[str]:
    button_values = collect_static_callback_button_values(document)
    if not button_values:
        return []

    errors: list[str] = []
    for trigger in document.flow.triggers:
        if trigger.kind not in GLOBAL_CALLBACK_TRIGGER_KINDS or trigger.global_flag is not True:
            continue
        for value in callback_trigger_values(trigger):
            if value not in button_values:
                continue
            for button in button_values[value]:
                errors.append(
                    "global callback trigger duplicates a keyboard button value: "
                    f"{trigger.id} ({trigger.kind}, value={value}) duplicates button "
                    f"{button['source']} / \"{button['text']}\". Do not create a global callback trigger "
                    "for static keyboard buttons; set that button `next` by creating the target action with "
                    '`incoming: {from: "menu_step", via: "button", button_text: "Button text"}`.'
                )
    return errors


def collect_static_callback_button_values(document: DSLDocument) -> dict[str, list[dict[str, str]]]:
    values: dict[str, list[dict[str, str]]] = {}
    for step in collect_all_steps(document):
        capability = None if step.kind == "raw_node" else get_capability_for_kind(step.kind)
        if not capability or not capability.button_target_edges or not step.keyboard:
            continue
        keyboard = step.keyboard if isinstance(step.keyboard, dict) else step.keyboard.model_dump(by_alias=True)
        for button in iter_keyboard_buttons(keyboard, grids=("inline",)):
            if button.get("type", "callback") != "callback":
                continue
            value = button.get("value")
            text = button.get("text")
            if isinstance(value, str) and value and isinstance(text, str):
                values.setdefault(value, []).append({"source": step.id, "text": text})
    return values


def callback_trigger_values(trigger) -> set[str]:
    values: set[str] = set()
    if isinstance(trigger.value, str) and trigger.value:
        values.add(trigger.value)
    for value in trigger.selected_callbacks or []:
        if isinstance(value, str) and value:
            values.add(value)
    return values


def validate_compiled_flow(document: DSLDocument | dict[str, Any]) -> list[str]:
    compiled = compile_dsl_document(document) if isinstance(document, DSLDocument) else document
    errors: list[str] = []

    nodes = compiled.get("nodes", [])
    edges = compiled.get("edges", [])
    node_ids = {node.get("id") for node in nodes}

    if not nodes:
        errors.append("compiled flow has no nodes")
    if not isinstance(edges, list):
        errors.append("compiled flow edges must be a list")

    for node in nodes:
        if "id" not in node or "type" not in node or "data" not in node:
            errors.append(f"compiled node missing required keys: {node}")
        if not isinstance(node.get("position"), dict):
            errors.append(f"compiled node missing position object: {node.get('id')}")

    for edge in edges:
        if edge.get("source") not in node_ids:
            errors.append(f"edge source missing node: {edge}")
        if edge.get("target") not in node_ids:
            errors.append(f"edge target missing node: {edge}")

    return errors


def analyze_dsl_connectivity(document: DSLDocument) -> dict[str, Any]:
    trigger_ids = {step.id for step in document.flow.triggers}
    step_index = {step.id: step for step in collect_all_steps(document)}
    adjacency = build_dsl_adjacency(document)
    reachable = reachable_from_triggers(trigger_ids, adjacency)
    action_steps = [step for step in document.flow.steps if step.id in step_index]
    unreachable_steps = [
        {"id": step.id, "kind": step.kind}
        for step in action_steps
        if step.id not in reachable
    ]
    root_action_steps = [
        {"id": step.id, "kind": step.kind}
        for step in action_steps
        if not incoming_sources(step.id, adjacency)
    ]
    return {
        "triggers": [{"id": step.id, "kind": step.kind} for step in document.flow.triggers],
        "reachable_steps": [
            {"id": step_id, "kind": step_index[step_id].kind}
            for step_id in sorted(reachable)
            if step_id in step_index
        ],
        "unreachable_steps": unreachable_steps,
        "root_action_steps": root_action_steps,
        "suggested_fix_notes": suggested_fix_notes(unreachable_steps, trigger_ids, reachable),
    }


def build_dsl_adjacency(document: DSLDocument) -> dict[str, set[str]]:
    steps = collect_all_steps(document)
    adjacency = {step.id: set() for step in steps}

    for route in document.flow.routes:
        adjacency.setdefault(route.source, set()).add(route.target)

    for step in steps:
        for route in step.routes:
            adjacency.setdefault(route.source, set()).add(route.target)
        capability = None if step.kind == "raw_node" else get_capability_for_kind(step.kind)
        if capability and capability.button_target_edges and step.keyboard:
            keyboard = step.keyboard if isinstance(step.keyboard, dict) else step.keyboard.model_dump(by_alias=True)
            for button in iter_keyboard_buttons(keyboard):
                next_step = button.get("next")
                if next_step:
                    adjacency.setdefault(step.id, set()).add(next_step)
        if capability and capability.selected_message_target and step.target_message_step:
            adjacency.setdefault(step.target_message_step, set()).add(step.id)

    return adjacency


def iter_keyboard_buttons(keyboard: dict[str, Any], grids: tuple[str, ...] = ("inline", "reply")):
    for grid in grids:
        for row in keyboard.get(grid, []) or []:
            for button in row or []:
                if isinstance(button, dict):
                    yield button


def reachable_from_triggers(trigger_ids: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    reachable: set[str] = set()
    stack = list(trigger_ids)
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(sorted(adjacency.get(node_id, set()) - reachable))
    return reachable


def incoming_sources(step_id: str, adjacency: dict[str, set[str]]) -> set[str]:
    return {source for source, targets in adjacency.items() if step_id in targets}


def suggested_fix_notes(
    unreachable_steps: list[dict[str, str]],
    trigger_ids: set[str],
    reachable: set[str],
) -> list[str]:
    if not unreachable_steps:
        return []
    source_hint = sorted(reachable - trigger_ids) or sorted(trigger_ids)
    source_text = source_hint[0] if source_hint else "a trigger"
    return [
        (
            f"Connect `{source_text}` to `{step['id']}` or connect `{step['id']}` "
            "from another already reachable step."
        )
        for step in unreachable_steps
    ]
