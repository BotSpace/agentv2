from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from bm_flow_agent.dsl.catalog import describe_kind, native_kinds, runtime_node_types
from bm_flow_agent.dsl import DSLDocument
from bm_flow_agent.dsl.validator import analyze_dsl_connectivity
from bm_flow_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult


def build_repo_catalog(
    workspace_root: Path,
    flow_path: str,
    dsl_path: str,
    *,
    dsl_only: bool = False,
) -> dict[str, Any]:
    supported = parse_supported_node_types(workspace_root)
    return {
        "workspace_root": str(workspace_root),
        "default_flow_path": flow_path,
        "default_dsl_path": dsl_path,
        "dsl_only": dsl_only,
        "supported_node_types": supported,
        "native_dsl_kinds": native_kinds(),
        "key_files": [
            "internal/engine/controller/controller.go",
            "internal/engine/executors/executor.go",
            "internal/engine/loader/loader.go",
            "docs/ARCHITECTURE.md",
            "docs/HANDLERS.md",
        ],
    }


def parse_supported_node_types(workspace_root: Path) -> list[str]:
    executor_path = workspace_root / "internal/engine/executors/executor.go"
    discovered = set(runtime_node_types())
    if executor_path.exists():
        text = executor_path.read_text(encoding="utf-8")
        discovered.update(re.findall(r'executor\.handlers\["([^"]+)"\]', text))
    return sorted(discovered)


def build_repo_tools() -> list[AgentTool]:
    class NoArgs(BaseModel):
        pass

    class ReadFileArgs(BaseModel):
        path: str
        start_line: int | None = None
        end_line: int | None = None

    class SearchArgs(BaseModel):
        query: str
        max_results: int = Field(default=10, ge=1, le=50)

    class KindArgs(BaseModel):
        kind: str | None = Field(
            default=None,
            description="Single native DSL kind to inspect, for example send_text or subflow.",
        )
        kinds: list[str] = Field(
            default_factory=list,
            description=(
                "Multiple native DSL kinds to inspect in one call. "
                "Use this before creating or updating several node types."
            ),
        )

        @model_validator(mode="after")
        def validate_payload(self) -> "KindArgs":
            if self.kind is None and not self.kinds:
                raise ValueError("Provide either `kind` or `kinds`.")
            return self

    def get_engine_overview(context: ToolContext, _: BaseModel) -> ToolExecutionResult:
        catalog = context.state.get("repo_catalog", {})
        supported = catalog.get("supported_node_types", [])
        summary = {
            "supported_node_count": len(supported),
            "supported_node_types": supported,
            "native_kind_count": len(context.state.get("repo_catalog", {}).get("native_dsl_kinds", [])),
            "native_dsl_kinds": context.state.get("repo_catalog", {}).get("native_dsl_kinds", []),
            "key_files": catalog.get("key_files", []),
            "default_flow_path": context.state.get("target_flow_json"),
            "default_dsl_path": context.state.get("target_dsl_path"),
        }
        return ToolExecutionResult(content=json.dumps(summary, ensure_ascii=False, indent=2))

    def list_supported_nodes(context: ToolContext, _: BaseModel) -> ToolExecutionResult:
        catalog = context.state.get("repo_catalog", {})
        payload = {
            "runtime_node_types": catalog.get("supported_node_types", []),
            "native_dsl_kinds": catalog.get("native_dsl_kinds", []),
        }
        return ToolExecutionResult(content=json.dumps(payload, ensure_ascii=False, indent=2))

    def describe_step_kind(context: ToolContext, args: KindArgs) -> ToolExecutionResult:
        requested = []
        if args.kind:
            requested.append(args.kind)
        requested.extend(args.kinds)

        unknown: list[str] = []
        chunks: list[str] = []
        for kind in dedupe_preserve_order(requested):
            description = describe_kind(kind)
            if description is None:
                unknown.append(kind)
                continue
            chunks.append(format_kind_description(description))

        if unknown:
            return ToolExecutionResult(
                content=f"Unknown step kind(s): {', '.join(unknown)}",
                is_error=True,
            )
        return ToolExecutionResult(content="\n\n---\n\n".join(chunks))

    def read_repo_file(context: ToolContext, args: ReadFileArgs) -> ToolExecutionResult:
        if context.state.get("repo_catalog", {}).get("dsl_only", False):
            lowered = args.path.lower()
            if lowered.endswith(".json") or lowered.startswith("assets/flow"):
                return ToolExecutionResult(
                    content=(
                        "Internal runtime files are hidden in chat mode. "
                        "Inspect the flow with `get_flow_yaml` instead."
                    ),
                    is_error=True,
                )
        path = resolve_repo_path(context, args.path)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max((args.start_line or 1) - 1, 0)
        end = args.end_line or len(lines)
        sliced = lines[start:end]
        numbered = [f"{idx + start + 1}: {line}" for idx, line in enumerate(sliced)]
        return ToolExecutionResult(content="\n".join(numbered))

    def search_repo_docs(context: ToolContext, args: SearchArgs) -> ToolExecutionResult:
        search_roots = ["README.md", "docs", "internal/engine", "examples"]
        cmd = ["rg", "-n", args.query, *search_roots]
        completed = subprocess.run(
            cmd,
            cwd=context.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = completed.stdout.strip().splitlines()
        results = output[: args.max_results]
        if not results:
            return ToolExecutionResult(content="No matches found.")
        return ToolExecutionResult(content="\n".join(results))

    def request_clarification(context: ToolContext, args: BaseModel) -> ToolExecutionResult:
        payload = args.model_dump(exclude_none=True)
        return ToolExecutionResult(content="Clarification requested.", interrupt_payload=payload)

    def analyze_flow_connectivity(context: ToolContext, _: BaseModel) -> ToolExecutionResult:
        document = load_current_flow_as_dsl(context)
        return ToolExecutionResult(
            content=json.dumps(analyze_dsl_connectivity(document), ensure_ascii=False, indent=2)
        )

    def explain_engine_flow_model(context: ToolContext, _: BaseModel) -> ToolExecutionResult:
        return ToolExecutionResult(content=engine_flow_model_text())

    class ClarificationOptionArgs(BaseModel):
        id: str
        label: str
        description: str | None = None

    class ClarificationQuestionArgs(BaseModel):
        id: str = Field(description="Stable short id for this question.")
        question: str = Field(description="Direct human-facing question.")
        details: str | None = Field(default=None, description="Optional context explaining why this answer is needed.")
        options: list[ClarificationOptionArgs] = Field(
            default_factory=list,
            description="Structured choices for this question.",
        )
        selection_type: str = Field(
            default="single",
            description="Use `single` for one option or `multiple` when several options may be selected.",
        )
        allow_free_text: bool = Field(
            default=True,
            description="Whether the user may answer this question with free text.",
        )

        @model_validator(mode="after")
        def validate_selection_type(self) -> "ClarificationQuestionArgs":
            if self.selection_type not in {"single", "multiple"}:
                raise ValueError("selection_type must be `single` or `multiple`.")
            return self

    class ClarificationArgs(BaseModel):
        question: str | None = Field(default=None, description="Direct human-facing question for legacy single-question calls.")
        details: str | None = Field(default=None, description="Optional context explaining why this is needed.")
        options: list[ClarificationOptionArgs] = Field(
            default_factory=list,
            description="Optional structured choices. Use this when clear choices exist.",
        )
        selection_type: str = Field(
            default="single",
            description="Legacy single-question selection mode: `single` or `multiple`.",
        )
        allow_free_text: bool = Field(
            default=True,
            description="Whether the user may answer with free text instead of an option.",
        )
        questions: list[ClarificationQuestionArgs] = Field(
            default_factory=list,
            description=(
                "Preferred batch form. Ask multiple independent questions in one interrupt. "
                "Each question supports its own options and single/multiple selection."
            ),
        )

        @model_validator(mode="after")
        def validate_payload(self) -> "ClarificationArgs":
            if self.selection_type not in {"single", "multiple"}:
                raise ValueError("selection_type must be `single` or `multiple`.")
            if self.questions:
                return self
            if not self.question:
                raise ValueError("Provide either `question` or `questions`.")
            return self

    tools = [
        AgentTool(
            name="get_engine_overview",
            description="Return a concise technical overview of the Botmother engine and supported node catalog.",
            args_model=NoArgs,
            handler=get_engine_overview,
        ),
        AgentTool(
            name="list_supported_nodes",
            description="List runtime node types and native DSL kinds currently supported by the agent.",
            args_model=NoArgs,
            handler=list_supported_nodes,
        ),
        AgentTool(
            name="describe_step_kind",
            description=(
                "Describe one or many native DSL kinds before authoring nodes. "
                "Use `kind` for one kind or `kinds` for multiple kinds. "
                "Returns AI-friendly field prompts, examples, route handles, engine behavior, and runtime node mapping."
            ),
            args_model=KindArgs,
            handler=describe_step_kind,
        ),
        AgentTool(
            name="explain_engine_flow_model",
            description=(
                "Explain the Botmother engine flow execution model in DSL-facing terms: trigger priority, "
                "global/root/waiting triggers, pause/resume, button edges, action reachability, and common patterns."
            ),
            args_model=NoArgs,
            handler=explain_engine_flow_model,
        ),
        AgentTool(
            name="search_repo_docs",
            description="Search repository docs and engine files with ripgrep.",
            args_model=SearchArgs,
            handler=search_repo_docs,
        ),
        AgentTool(
            name="read_repo_file",
            description="Read a repository file with optional line limits.",
            args_model=ReadFileArgs,
            handler=read_repo_file,
        ),
        AgentTool(
            name="request_clarification",
            description=(
                "Ask the human for clarification with LangGraph interrupt when requirements are incomplete, "
                "ambiguous, risky, or product choices are missing. Prefer the `questions` batch form when "
                "you need several answers at once. Use `selection_type: multiple` when several options can "
                "be selected, for example CRM features."
            ),
            args_model=ClarificationArgs,
            handler=request_clarification,
        ),
        AgentTool(
            name="analyze_flow_connectivity",
            description=(
                "Analyze the current YAML flow graph and report which action steps are reachable "
                "from triggers. Use this before completing a plan or after validation says an action "
                "step is not reachable from any trigger."
            ),
            args_model=NoArgs,
            handler=analyze_flow_connectivity,
        ),
    ]
    return tools


def resolve_repo_path(context: ToolContext, path: str | None) -> Path:
    if not path:
        raise FileNotFoundError("path is required")
    candidate = (context.workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if context.workspace_root not in candidate.parents and candidate != context.workspace_root:
        raise ValueError(f"path escapes workspace: {path}")
    return candidate


def load_working_dsl(context: ToolContext, path: str | None = None) -> DSLDocument:
    working = context.state.get("working_dsl")
    if working and path is None:
        return DSLDocument.model_validate(working)
    target = resolve_repo_path(context, path or context.state.get("target_dsl_path"))
    return DSLDocument.model_validate(yaml.safe_load(target.read_text(encoding="utf-8")) or {})


def load_current_flow_as_dsl(context: ToolContext) -> DSLDocument:
    working = context.state.get("working_dsl")
    if working:
        return DSLDocument.model_validate(working)
    from bm_flow_agent.dsl.importer import import_flow_json_to_dsl

    path = resolve_repo_path(context, context.state.get("target_flow_json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return import_flow_json_to_dsl(payload, name=path.stem)


def format_kind_description(description: dict[str, Any]) -> str:
    lines = [
        f"Kind: {description['kind']}",
        f"Runtime node: {description['runtime_node_type']}",
        f"Category: {'trigger' if description['is_trigger'] else 'step'}",
        f"Purpose: {description['purpose']}",
    ]

    aliases = description.get("aliases") or []
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")

    lines.append("")
    lines.append("Fields:")
    fields = description.get("fields") or []
    if not fields:
        lines.append("- This kind has no dedicated fields. It mainly relies on routing/default runtime behavior.")
    else:
        for field in fields:
            required = "required" if field.get("required") else "optional"
            lines.append(
                f"- {field['name']} ({required}) -> runtime `{field['runtime_key']}`: {field['description']}"
            )
            lines.append(f"  AI prompt: {field['ai_prompt']}")

    route_handles = description.get("route_handles") or []
    lines.append("")
    lines.append("Engine behavior:")
    lines.extend(engine_behavior_lines(description))
    lines.append("")
    lines.append("Routing:")
    if description["is_trigger"]:
        lines.append("- Trigger node flow entry point bo'lishi mumkin; odatda `upsert_step`da `incoming` shart emas.")
    else:
        lines.append(
            "- Action step root bo'lib qolmasligi kerak. Yangi action yaratganda "
            "`upsert_step` tool callida `step` bilan yonma-yon `incoming: {from: \"reachable_source_id\"}` ber. "
            "`incoming.from` qaysi trigger/actiondan shu nodega kelishini bildiradi; `incoming.to` yozilmaydi, "
            "target har doim `step.id` bo'ladi."
        )
    if route_handles:
        lines.append(f"- Special handles: {', '.join(route_handles)}")
    else:
        lines.append("- No special route handles. Normal next-edge routing is used.")

    if description.get("button_target_edges"):
        lines.append("- Keyboard buttons may declare `next`, which compiles into button target edges.")
    if description.get("selected_message_target"):
        lines.append("- This node can point to another message node through the selected-message target edge.")
    if description.get("output_routes"):
        lines.append("- Routes using `on` are compiled as `output-{name}` handles.")

    notes = description.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def engine_flow_model_text() -> str:
    return "\n".join(
        [
            "Botmother engine flow model (DSL-facing):",
            "",
            "Trigger priority in HandleUpdate:",
            "1. Global triggers priority 1: any trigger with `global: true` is checked first on every update. If it matches, current flow state is reset and traversal starts from that trigger.",
            "2. Waiting triggers priority 2: if the user is in waiting state, the engine checks button edges and trigger nodes reachable from the paused action.",
            "3. Root triggers priority 3: trigger nodes with no incoming edge start a flow when no global/waiting trigger handled the update.",
            "",
            "Trigger types:",
            "- Root trigger: a trigger with no incoming edge. It starts a flow from idle state.",
            "- Global trigger: a trigger with `global: true`. Use for reset-like commands such as `/start`, `/cancel`, `/menu`.",
            "- Waiting trigger: a trigger reached from an action. The action runs, the engine pauses there, then the next matching user update continues from the trigger.",
            "",
            "Action traversal:",
            "- Action node cannot start a flow. It must be reachable from a root/global trigger or another reachable action.",
            "- Normal action-to-action edges execute with BFS-style traversal.",
            "- If the next reachable nodes are only triggers, the engine pauses and waits for user input.",
            "",
            "Button edges:",
            "- Keyboard button `next` creates a button edge handled by callback/reply matching, not normal BFS traversal.",
            "- Inline button `value` is matched to the source keyboard button, then the engine follows the `target-handler-{nodeID}-{buttonText}` edge.",
            "- If a static menu button should open a screen/action, create the target action with `incoming.via=\"button\"`; do not create a global callback trigger.",
            "- `callback_query_trigger` and `callback_button_trigger` are for special callback/waiting flows that cannot be represented by a source keyboard button `next`.",
            "",
            "Common construction patterns:",
            "- New flow: root/global trigger -> first action.",
            "- Ask text input: send_text question -> message_trigger -> action that handles saved state.",
            "- Menu: send_text with keyboard button `next` targets -> action or waiting trigger branches.",
        ]
    )


def engine_behavior_lines(description: dict[str, Any]) -> list[str]:
    kind = str(description["kind"])
    is_trigger = bool(description["is_trigger"])
    if is_trigger:
        lines = [
            "- If this trigger has no incoming edge, the engine treats it as a root trigger and can start a flow from idle state.",
            "- If this trigger is reached from an action, it becomes a waiting trigger: the previous action runs, then the engine pauses and waits for the next matching update.",
        ]
        if kind in {"callback_query_trigger", "callback_button_trigger"}:
            lines.append(
                "- Do not use this as a global trigger for static inline keyboard menu buttons. "
                "Those buttons already route through their keyboard `next` button edge; create the target action "
                "with `incoming: {from: \"menu_step\", via: \"button\", button_text: \"Button text\"}` instead."
            )
        if kind == "command_trigger":
            lines.append("- Use `global: true` only when this command should interrupt/reset the current flow, such as `/start`, `/cancel`, or `/menu`.")
        if kind == "cron_trigger":
            lines.append("- Cron trigger should stay root-only; do not connect incoming edges to it.")
        if kind == "external_webhook_trigger":
            lines.append("- External webhook trigger starts from webhook requests instead of Telegram messages.")
        return lines

    lines = [
        "- Action node cannot start a flow by itself; it must be reachable from a trigger or another reachable action.",
        "- When creating this as a new action, call `upsert_step` with sibling `incoming: {from: \"reachable_source_id\"}`.",
    ]
    if description.get("button_target_edges"):
        lines.append("- If this action has keyboard buttons with `next`, those button edges run only after callback/reply matching, not as normal BFS edges.")
        lines.append("- Inline button `value` is matched by the engine against this source keyboard; no global callback trigger is needed for static menu buttons.")
        lines.append("- Prefer button `next` targets that point directly to the action that should run after the click.")
        lines.append("- When creating a new action for one of this node's buttons, use `incoming: {from: this_node_id, via: \"button\", button_text: \"Button text\"}` so the button gets `next` instead of a normal route.")
    if description.get("route_handles"):
        lines.append("- Use the listed special route handles with `connect_steps.on` when branching from this action.")
    if description.get("output_routes"):
        lines.append("- Named outputs route through `output-{name}` handles.")
    return lines


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
