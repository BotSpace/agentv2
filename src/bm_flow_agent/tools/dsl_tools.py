from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from bm_flow_agent.dsl.catalog import TRIGGER_KINDS, native_kinds
from bm_flow_agent.dsl.compiler import compile_dsl_document
from bm_flow_agent.dsl.importer import import_flow_json_to_dsl
from bm_flow_agent.dsl.models import DSLDocument, RouteSpec, StepSpec
from bm_flow_agent.dsl.parser import (
    collect_all_steps,
    create_empty_document,
    dump_dsl_to_yaml,
    find_step,
    remove_step as remove_step_from_document,
    replace_step,
    upsert_route,
)
from bm_flow_agent.dsl.validator import validate_compiled_flow, validate_dsl_document
from bm_flow_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult
from bm_flow_agent.tools.repo_tools import resolve_repo_path


GLOBAL_CALLBACK_TRIGGER_KINDS = {"callback_query_trigger", "callback_button_trigger"}


def build_yaml_flow_tools() -> list[AgentTool]:
    supported_kinds = ", ".join(native_kinds())

    class NoArgs(BaseModel):
        pass

    class UpsertStepArgs(BaseModel):
        step: dict[str, Any] | str = Field(
            description=(
                "Single full StepSpec object. Must include at least `id` and `kind`. "
                "Send it as an object, not as a quoted JSON string. "
                "Use `describe_step_kind(kind)` when you need allowed fields for a kind."
            ),
        )
        trigger: bool | None = Field(
            default=None,
            description="Boolean only. True stores the step in flow.triggers; false stores it in flow.steps.",
        )
        incoming: dict[str, Any] | str | None = Field(
            default=None,
            description=(
                "Incoming edge to create together with this step. Required when creating a NEW action step. "
                "It is a sibling argument of `step`, not a field inside StepSpec. Shape: "
                "{from: 'source_step_id', via?: 'route|button', button_text?: 'Button text', "
                "on?: 'handle', source_handle?: 'handle', target_handle?: 'handle'}. "
                "`from` is the existing trigger/action id that should lead into this step. "
                "Use via='button' with button_text when the step should run only after a keyboard button is pressed. "
                "Use via='route' for explicit normal automatic route from a keyboard message source. "
                "Do not include `to`; target is always step.id. "
                "Example: {step: {id: 'welcome', kind: 'send_text', text: 'Salom'}, incoming: {from: 'start'}}."
            ),
        )

        @model_validator(mode="before")
        @classmethod
        def parse_string_payloads(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            parsed = dict(data)
            if isinstance(parsed.get("step"), str):
                parsed["step"] = parse_object_string(parsed["step"], "step")
            if isinstance(parsed.get("incoming"), str):
                parsed["incoming"] = parse_object_string(parsed["incoming"], "incoming")
            return parsed

        @model_validator(mode="after")
        def validate_incoming(self) -> "UpsertStepArgs":
            if self.incoming and "to" in self.incoming:
                raise ValueError("incoming must not include `to`; target is always step.id.")
            if self.incoming:
                via = self.incoming.get("via")
                if via and via not in {"route", "button"}:
                    raise ValueError("incoming.via must be either `route` or `button`.")
                if via == "button" and not self.incoming.get("button_text"):
                    raise ValueError("incoming.button_text is required when incoming.via is `button`.")
            return self

    class RemoveStepArgs(BaseModel):
        step_id: str

    class ConnectArgs(BaseModel):
        source: str | None = None
        target: str | None = None
        on: str | None = None
        source_handle: str | None = None
        target_handle: str | None = None
        routes: list[dict[str, Any]] = Field(
            default_factory=list,
            description="Bulk form: [{from: 'a', to: 'b', on: 'true'}, ...].",
        )

        @model_validator(mode="after")
        def validate_payload(self) -> "ConnectArgs":
            has_single = self.source is not None and self.target is not None
            if not has_single and not self.routes:
                raise ValueError("Provide either single route fields (`source`, `target`) or `routes`.")
            return self

    class PatchArgs(BaseModel):
        step_id: str
        patch: dict[str, Any]

    class SaveFlowYamlArgs(BaseModel):
        yaml: str = Field(
            description=(
                "Complete YAML DSL document for the whole flow. "
                "The assistant must submit the entire flow, not a partial patch."
            )
        )

    def get_flow_yaml_tool(context: ToolContext, _: NoArgs) -> ToolExecutionResult:
        path = resolve_repo_path(context, context.state.get("target_flow_json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = import_flow_json_to_dsl(payload, name=path.stem)
        return ToolExecutionResult(content=dump_dsl_to_yaml(document))

    def upsert_step_tool(context: ToolContext, args: UpsertStepArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        document = load_flow_document_from_target(context)

        step = StepSpec.model_validate(args.step)
        trigger_error = global_callback_trigger_error(step)
        if trigger_error:
            return ToolExecutionResult(content=trigger_error, is_error=True)
        existing = find_step(document, step.id)
        target_is_trigger = args.trigger if args.trigger is not None else step.kind in TRIGGER_KINDS
        if (
            existing is None
            and not target_is_trigger
            and not args.incoming
            and not has_existing_incoming_reference(document, step.id)
        ):
            return ToolExecutionResult(
                content=(
                    "Action step upsert blocked: new action steps must include "
                    "`incoming: {from: 'reachable_source_step_id'}` so they are connected immediately. "
                    "Trigger steps can omit incoming because they start flows."
                ),
                is_error=True,
            )
        incoming_guard = validate_incoming_semantics(document, args.incoming, target=step.id)
        if incoming_guard:
            return incoming_guard

        replace_step(document, step, as_trigger=args.trigger)
        button_edge_summary: dict[str, Any] | None = None
        if args.incoming and args.incoming.get("via") == "button":
            button_edge_summary = apply_button_incoming(document, args.incoming, target=step.id)

        incoming_route = route_from_incoming(args.incoming, target=step.id)
        if incoming_route:
            upsert_route(document, incoming_route)

        result = save_document_to_target(context, document, {"upserted_step": step.id})
        if not result.is_error:
            emit_flow_event(
                context,
                "node_upserted",
                {
                    "step": step_summary(step),
                    "node": compiled_node_summary(document, step.id),
                },
            )
            if incoming_route:
                emit_flow_event(
                    context,
                    "edge_upserted",
                    {
                        "route": route_summary(incoming_route),
                        "edge": match_compiled_edge(incoming_route, compiled_edge_summaries(document)),
                    },
                )
            if button_edge_summary:
                emit_flow_event(
                    context,
                    "edge_upserted",
                    {
                        "button": button_edge_summary,
                        "edge": match_compiled_button_edge(button_edge_summary, compiled_edge_summaries(document)),
                    },
                )
        return result

    def remove_step_tool(context: ToolContext, args: RemoveStepArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        document = load_flow_document_from_target(context)
        removed_step = find_step(document, args.step_id)
        removed_routes = route_summaries(routes_for_step(document, args.step_id))
        remove_step_from_document(document, args.step_id)
        result = save_document_to_target(context, document, {"removed_step": args.step_id})
        if not result.is_error:
            emit_flow_event(
                context,
                "node_removed",
                {
                    "step_id": args.step_id,
                    "step": step_summary(removed_step) if removed_step else None,
                    "routes": removed_routes,
                },
            )
        return result

    def connect_steps_tool(context: ToolContext, args: ConnectArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        document = load_flow_document_from_target(context)
        connected: list[str] = []
        route_specs: list[RouteSpec] = []

        if args.source is not None and args.target is not None:
            route = RouteSpec.model_validate(
                {
                    "from": args.source,
                    "to": args.target,
                    "on": args.on,
                    "source_handle": args.source_handle,
                    "target_handle": args.target_handle,
                }
            )
            upsert_route(document, route)
            connected.append(f"{route.source}->{route.target}")
            route_specs.append(route)

        for route_payload in args.routes:
            route = RouteSpec.model_validate(route_payload)
            upsert_route(document, route)
            connected.append(f"{route.source}->{route.target}")
            route_specs.append(route)

        result = save_document_to_target(context, document, {"connected_routes": connected})
        if not result.is_error:
            edges = compiled_edge_summaries(document)
            for route in route_specs:
                emit_flow_event(
                    context,
                    "edge_upserted",
                    {
                        "route": route_summary(route),
                        "edge": match_compiled_edge(route, edges),
                    },
                )
        return result

    def patch_step_block_tool(context: ToolContext, args: PatchArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        document = load_flow_document_from_target(context)
        existing = find_step(document, args.step_id)
        if existing is None:
            return ToolExecutionResult(content=f"Step not found: {args.step_id}", is_error=True)
        merged = deep_merge(existing.model_dump(by_alias=True, exclude_none=True), deepcopy(args.patch))
        merged["id"] = args.step_id
        patched = StepSpec.model_validate(merged)
        replace_step(document, patched)
        result = save_document_to_target(context, document, {"patched_step": args.step_id})
        if not result.is_error:
            emit_flow_event(
                context,
                "node_patched",
                {
                    "step_id": args.step_id,
                    "patch": deepcopy(args.patch),
                    "step": step_summary(patched),
                    "node": compiled_node_summary(document, args.step_id),
                },
            )
        return result

    def save_flow_yaml_tool(context: ToolContext, args: SaveFlowYamlArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        try:
            before = load_flow_document_from_target(context)
            document = parse_yaml_document(args.yaml)
        except Exception as exc:
            return ToolExecutionResult(content=f"Invalid YAML flow: {exc}", is_error=True)
        result = save_document_to_target(context, document, {"saved": "full_yaml"})
        if not result.is_error:
            emit_flow_event(
                context,
                "flow_replaced",
                flow_diff_summary(before, document),
            )
        return result

    return [
        AgentTool(
            name="get_flow_yaml",
            description=(
                "Return the current bot flow as YAML. Use this before explaining or changing the flow. "
                "This tool never exposes runtime JSON."
            ),
            args_model=NoArgs,
            handler=get_flow_yaml_tool,
        ),
        AgentTool(
            name="upsert_step",
            description=(
                "Create or replace exactly one YAML StepSpec object and immediately save the flow. "
                "Do not send bulk `items`; call this tool once per step. "
                "When creating a NEW action step, `incoming` is required and must connect from an existing "
                "trigger or reachable action step; otherwise the tool rejects the call. "
                "`incoming` is a tool argument next to `step`, not a field inside the StepSpec. "
                "For keyboard branches, use `incoming: {from: 'menu_step', via: 'button', button_text: 'Button text'}`; "
                "this updates the source button `next` instead of creating an automatic route. "
                "If you intentionally want an automatic route from a keyboard message source, use `via: 'route'`. "
                "Never send `incoming.to`; the target is always `step.id`. "
                "Trigger kinds must live in flow.triggers (`trigger: true`); action steps live in flow.steps "
                "and must be connected from a trigger or another reachable action before the plan is completed. "
                f"Supported native kinds: {supported_kinds}."
            ),
            args_model=UpsertStepArgs,
            handler=upsert_step_tool,
        ),
        AgentTool(
            name="remove_step",
            description="Remove one step and immediately save the flow.",
            args_model=RemoveStepArgs,
            handler=remove_step_tool,
        ),
        AgentTool(
            name="connect_steps",
            description=(
                "Create one or many YAML routes between existing steps and immediately save the flow. "
                "Prefer `upsert_step` with `incoming` when creating a new action step. "
                "Use this for bulk routing, repairs, or connecting already-created steps."
            ),
            args_model=ConnectArgs,
            handler=connect_steps_tool,
        ),
        AgentTool(
            name="patch_step_block",
            description="Patch one existing YAML step with a partial block and immediately save the flow.",
            args_model=PatchArgs,
            handler=patch_step_block_tool,
        ),
        AgentTool(
            name="save_flow_yaml",
            description=(
                "Emergency full-replace tool: validate and save a complete YAML flow document. "
                "Prefer `upsert_step` and `connect_steps` for normal edits."
            ),
            args_model=SaveFlowYamlArgs,
            handler=save_flow_yaml_tool,
        ),
    ]


def global_callback_trigger_error(step: StepSpec) -> str | None:
    if step.kind not in GLOBAL_CALLBACK_TRIGGER_KINDS or step.global_flag is not True:
        return None
    return (
        "Global callback trigger blocked: do not create global callback_query_trigger or "
        "callback_button_trigger for inline keyboard buttons. Static keyboard buttons already "
        "work through their button `next` edge. Create the target action with "
        '`incoming: {from: "menu_step", via: "button", button_text: "Button text"}` instead.'
    )


def resolve_write_path(context: ToolContext, path: str | None) -> Path:
    if not path:
        raise FileNotFoundError("target path is required")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (context.workspace_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if context.workspace_root not in candidate.parents and candidate != context.workspace_root:
        raise ValueError(f"path escapes workspace: {path}")
    return candidate


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_yaml_document(raw_yaml: str) -> DSLDocument:
    import yaml

    payload = yaml.safe_load(raw_yaml) or {}
    return DSLDocument.model_validate(payload)


def parse_object_string(raw_value: str, field_name: str) -> dict[str, Any]:
    import yaml

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError(f"`{field_name}` string must parse to an object/dictionary.")
    return parsed


def load_flow_document_from_target(context: ToolContext) -> DSLDocument:
    working = context.state.get("working_dsl")
    if working:
        return DSLDocument.model_validate(working)
    path = resolve_repo_path(context, context.state.get("target_flow_json"))
    if not path.exists():
        return create_empty_document(path.stem)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return import_flow_json_to_dsl(payload, name=path.stem)


def save_document_to_target(
    context: ToolContext,
    document: DSLDocument,
    action_summary: dict[str, Any],
) -> ToolExecutionResult:
    try:
        compiled = compile_dsl_document(document)
    except Exception as exc:
        return ToolExecutionResult(content=f"Invalid YAML flow: {exc}", is_error=True)

    errors = [*validate_dsl_document(document), *validate_compiled_flow(compiled)]
    if errors:
        if allow_draft_save(context, errors):
            content = {
                **action_summary,
                "saved": False,
                "draft": True,
                "pending_validation_errors": errors,
                "nodes": len(compiled.get("nodes", [])),
                "edges": len(compiled.get("edges", [])),
            }
            return ToolExecutionResult(
                content=json.dumps(content, ensure_ascii=False),
                state_updates={"working_dsl": document.model_dump(by_alias=True, exclude_none=True)},
            )
        return ToolExecutionResult(
            content="\n".join(errors),
            state_updates={"working_dsl": document.model_dump(by_alias=True, exclude_none=True)},
            is_error=True,
        )

    path = resolve_write_path(context, context.state.get("target_flow_json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compiled, ensure_ascii=False, indent=2), encoding="utf-8")
    content = {
        **action_summary,
        "saved": True,
        "draft": False,
        "nodes": len(compiled.get("nodes", [])),
        "edges": len(compiled.get("edges", [])),
    }
    return ToolExecutionResult(
        content=json.dumps(content, ensure_ascii=False),
        state_updates={
            "working_dsl": document.model_dump(by_alias=True, exclude_none=True),
            "last_compile_result": {
                "compiled_flow": compiled,
                "validated": True,
                "validation_errors": [],
            },
        },
    )


def allow_draft_save(context: ToolContext, errors: list[str]) -> bool:
    if context.state.get("strict_final_save"):
        return False
    plan = context.state.get("current_plan")
    has_plan = isinstance(plan, dict) and isinstance(plan.get("items"), list) and plan["items"]
    if not has_plan and not context.state.get("allow_unplanned_edits"):
        return False
    return all(is_draft_save_error(error) for error in errors)


def is_draft_save_error(error: str) -> bool:
    return (
        error.startswith("button next target not found:")
        or error.startswith("route source not found:")
        or error.startswith("route target not found:")
        or error.startswith("edge source missing node:")
        or error.startswith("edge target missing node:")
        or error.startswith("edit_message target_message_step not found:")
        or error.startswith("action step is not reachable from any trigger:")
    )


def strict_save_working_dsl(context: ToolContext) -> ToolExecutionResult:
    document = load_flow_document_from_target(context)
    return save_document_to_target(
        ToolContext(
            workspace_root=context.workspace_root,
            state={
                **context.state,
                "allow_unplanned_edits": True,
                "strict_final_save": True,
            },
            ui=context.ui,
        ),
        document,
        {"saved": "plan_completed"},
    )


def require_task_plan(context: ToolContext) -> ToolExecutionResult | None:
    if context.state.get("allow_unplanned_edits"):
        return None
    plan = context.state.get("current_plan")
    if isinstance(plan, dict) and isinstance(plan.get("items"), list) and plan["items"]:
        return None
    return ToolExecutionResult(
        content=(
            "Flow edit blocked: create_task_plan must be called before changing YAML flow. "
            "First inspect/clarify requirements, then create a task plan, then execute plan items."
        ),
        is_error=True,
    )


def emit_flow_event(context: ToolContext, event_type: str, payload: dict[str, Any]) -> None:
    emitter = getattr(context.ui, "flow_event", None)
    if callable(emitter):
        emitter(event_type, payload)


def step_summary(step: StepSpec) -> dict[str, Any]:
    return step.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)


def route_summary(route: RouteSpec) -> dict[str, Any]:
    return route.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)


def route_summaries(routes: list[RouteSpec]) -> list[dict[str, Any]]:
    return [route_summary(route) for route in routes]


def route_from_incoming(incoming: dict[str, Any] | None, *, target: str) -> RouteSpec | None:
    if not incoming:
        return None
    if "to" in incoming:
        raise ValueError("incoming must not include `to`; target is always step.id.")
    if incoming.get("via") == "button":
        return None
    payload = {
        "from": incoming.get("from"),
        "to": target,
        "on": incoming.get("on"),
        "source_handle": incoming.get("source_handle"),
        "target_handle": incoming.get("target_handle"),
    }
    return RouteSpec.model_validate(payload)


def validate_incoming_semantics(
    document: DSLDocument,
    incoming: dict[str, Any] | None,
    *,
    target: str,
) -> ToolExecutionResult | None:
    if not incoming:
        return None
    source_id = incoming.get("from")
    if not source_id:
        return ToolExecutionResult(content="incoming.from is required.", is_error=True)
    if incoming.get("via") == "button":
        source = find_step(document, source_id)
        if source is None:
            return None
        button_text = incoming.get("button_text")
        if not find_keyboard_button(source, button_text):
            return ToolExecutionResult(
                content=f"button_text not found on source step `{source_id}`: {button_text}",
                is_error=True,
            )
        return None
    if incoming.get("via") in {None, ""} and source_has_keyboard(document, source_id):
        return ToolExecutionResult(
            content=(
                f"Incoming source `{source_id}` has keyboard buttons. A default route from this source would run "
                "the target action without waiting for a button press. If this target should run after a button, "
                'use `incoming: {from: "...", via: "button", button_text: "..."}`. '
                'If it should run automatically after the message is sent, use `incoming: {from: "...", via: "route"}`.'
            ),
            is_error=True,
        )
    return None


def apply_button_incoming(document: DSLDocument, incoming: dict[str, Any], *, target: str) -> dict[str, Any]:
    source_id = incoming["from"]
    button_text = incoming["button_text"]
    source = find_step(document, source_id)
    if source is None:
        raise ValueError(f"button source step not found: {source_id}")
    keyboard = keyboard_to_dict(source.keyboard)
    button = find_keyboard_button_in_keyboard(keyboard, button_text)
    if button is None:
        raise ValueError(f"button_text not found on source step `{source_id}`: {button_text}")
    button["next"] = target
    source.keyboard = keyboard
    return {"from": source_id, "to": target, "button_text": button_text, "via": "button"}


def find_keyboard_button(step: StepSpec, button_text: str | None) -> dict[str, Any] | None:
    if not button_text or not step.keyboard:
        return None
    return find_keyboard_button_in_keyboard(keyboard_to_dict(step.keyboard), button_text)


def find_keyboard_button_in_keyboard(keyboard: dict[str, Any], button_text: str | None) -> dict[str, Any] | None:
    if not button_text:
        return None
    for section in ("inline", "reply"):
        for row in keyboard.get(section) or []:
            for button in row or []:
                if isinstance(button, dict) and button.get("text") == button_text:
                    return button
    return None


def source_has_keyboard(document: DSLDocument, source_id: str) -> bool:
    source = find_step(document, source_id)
    if source is None or not source.keyboard:
        return False
    keyboard = keyboard_to_dict(source.keyboard)
    for section in ("inline", "reply"):
        if any(row for row in keyboard.get(section) or []):
            return True
    return False


def keyboard_to_dict(keyboard: Any) -> dict[str, Any]:
    return keyboard if isinstance(keyboard, dict) else keyboard.model_dump(by_alias=True, exclude_none=True)


def has_existing_incoming_reference(document: DSLDocument, step_id: str) -> bool:
    if any(route.target == step_id for route in document.flow.routes):
        return True
    for step in collect_all_steps(document):
        if any(route.target == step_id for route in step.routes):
            return True
        keyboard = step.keyboard
        if keyboard is None:
            continue
        keyboard_payload = keyboard_to_dict(keyboard)
        for section in ("inline", "reply"):
            rows = keyboard_payload.get(section) or []
            for row in rows:
                for button in row:
                    if isinstance(button, dict) and button.get("next") == step_id:
                        return True
    return False


def routes_for_step(document: DSLDocument, step_id: str) -> list[RouteSpec]:
    routes = [
        route
        for route in document.flow.routes
        if route.source == step_id or route.target == step_id
    ]
    for step in collect_all_steps(document):
        routes.extend(
            route
            for route in step.routes
            if route.source == step_id or route.target == step_id
        )
    return routes


def compiled_node_summary(document: DSLDocument, step_id: str) -> dict[str, Any] | None:
    compiled = compile_dsl_document(document)
    for node in compiled.get("nodes", []):
        if node.get("id") == step_id:
            return compact_node(node)
    return None


def compiled_edge_summaries(document: DSLDocument) -> list[dict[str, Any]]:
    compiled = compile_dsl_document(document)
    return [compact_edge(edge) for edge in compiled.get("edges", [])]


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": node.get("id"),
        "type": node.get("type"),
        "data": deepcopy(node.get("data", {})),
    }
    if "position" in node:
        payload["position"] = deepcopy(node["position"])
    return payload


def compact_edge(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(edge[key])
        for key in ("id", "source", "target", "sourceHandle", "targetHandle", "type")
        if key in edge
    }


def match_compiled_edge(route: RouteSpec, edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    for edge in edges:
        if edge.get("source") != route.source or edge.get("target") != route.target:
            continue
        if route.target_handle and edge.get("targetHandle") != route.target_handle:
            continue
        route_handle = route.effective_source_handle
        edge_handle = edge.get("sourceHandle")
        if route_handle and edge_handle not in {route_handle, f"output-{route_handle}"}:
            continue
        return edge
    return None


def match_compiled_button_edge(button_summary: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any] | None:
    handle = f"target-handler-{button_summary['from']}-{button_summary['button_text']}"
    for edge in edges:
        if (
            edge.get("source") == button_summary["from"]
            and edge.get("target") == button_summary["to"]
            and edge.get("sourceHandle") == handle
        ):
            return edge
    return None


def flow_diff_summary(before: DSLDocument, after: DSLDocument) -> dict[str, Any]:
    before_compiled = compile_dsl_document(before)
    after_compiled = compile_dsl_document(after)
    before_nodes = {node["id"]: compact_node(node) for node in before_compiled.get("nodes", [])}
    after_nodes = {node["id"]: compact_node(node) for node in after_compiled.get("nodes", [])}
    before_edges = {edge["id"]: compact_edge(edge) for edge in before_compiled.get("edges", [])}
    after_edges = {edge["id"]: compact_edge(edge) for edge in after_compiled.get("edges", [])}
    return {
        "before": {"nodes": len(before_nodes), "edges": len(before_edges)},
        "after": {"nodes": len(after_nodes), "edges": len(after_edges)},
        "added_nodes": [after_nodes[node_id] for node_id in sorted(after_nodes.keys() - before_nodes.keys())],
        "removed_node_ids": sorted(before_nodes.keys() - after_nodes.keys()),
        "added_edges": [after_edges[edge_id] for edge_id in sorted(after_edges.keys() - before_edges.keys())],
        "removed_edge_ids": sorted(before_edges.keys() - after_edges.keys()),
    }
