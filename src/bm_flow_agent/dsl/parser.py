from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bm_flow_agent.dsl.models import DSLDocument, FlowMeta, RouteSpec, StepSpec


def create_empty_document(name: str = "untitled", description: str | None = None) -> DSLDocument:
    return DSLDocument.model_validate(
        {
            "flow": {
                "meta": FlowMeta(name=name, description=description).model_dump(exclude_none=True),
                "triggers": [],
                "steps": [],
                "routes": [],
            }
        }
    )


def load_dsl_document(path: str | Path) -> DSLDocument:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return DSLDocument.model_validate(data)


def save_dsl_document(document: DSLDocument, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_dsl_to_yaml(document), encoding="utf-8")
    return target


def dump_dsl_to_yaml(document: DSLDocument) -> str:
    data = document.model_dump(by_alias=True, exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def collect_all_steps(document: DSLDocument) -> list[StepSpec]:
    return [*document.flow.triggers, *document.flow.steps]


def find_step(document: DSLDocument, step_id: str) -> StepSpec | None:
    for step in collect_all_steps(document):
        if step.id == step_id:
            return step
    return None


def replace_step(document: DSLDocument, step: StepSpec, *, as_trigger: bool | None = None) -> DSLDocument:
    target_is_trigger = as_trigger if as_trigger is not None else _is_trigger_kind(step.kind)
    new_triggers = [item for item in document.flow.triggers if item.id != step.id]
    new_steps = [item for item in document.flow.steps if item.id != step.id]
    if target_is_trigger:
        new_triggers.append(step)
    else:
        new_steps.append(step)
    document.flow.triggers = new_triggers
    document.flow.steps = new_steps
    return document


def remove_step(document: DSLDocument, step_id: str) -> DSLDocument:
    document.flow.triggers = [item for item in document.flow.triggers if item.id != step_id]
    document.flow.steps = [item for item in document.flow.steps if item.id != step_id]
    document.flow.routes = [
        route for route in document.flow.routes if route.source != step_id and route.target != step_id
    ]
    for step in collect_all_steps(document):
        step.routes = [route for route in step.routes if route.source != step_id and route.target != step_id]
        if step.target_message_step == step_id:
            step.target_message_step = None
        if step.keyboard:
            keyboard = _keyboard_to_dict(step.keyboard)
            _drop_button_targets(keyboard, step_id)
            step.keyboard = keyboard
    return document


def upsert_route(document: DSLDocument, route: RouteSpec) -> DSLDocument:
    document.flow.routes = [
        existing
        for existing in document.flow.routes
        if not (
            existing.source == route.source
            and existing.target == route.target
            and existing.effective_source_handle == route.effective_source_handle
            and existing.target_handle == route.target_handle
        )
    ]
    document.flow.routes.append(route)
    return document


def _drop_button_targets(keyboard: dict[str, Any], removed_step_id: str) -> None:
    for key in ("inline", "reply"):
        for row in keyboard.get(key, []) or []:
            for button in row or []:
                if isinstance(button, dict) and button.get("next") == removed_step_id:
                    button.pop("next", None)


def _keyboard_to_dict(keyboard: Any) -> dict[str, Any]:
    if keyboard is None:
        return {}
    if isinstance(keyboard, dict):
        return keyboard
    return keyboard.model_dump(by_alias=True, exclude_none=True)


def _is_trigger_kind(kind: str) -> bool:
    return kind.endswith("_trigger")
