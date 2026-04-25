from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import click
import typer
import yaml
from dotenv import load_dotenv
from langchain_aws import ChatAnthropicBedrock
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.types import Command

from bm_flow_agent.auth import decode_token, user_from_claims
from bm_flow_agent.dsl import (
    DSLDocument,
    compile_dsl_document,
    dump_dsl_to_yaml,
    import_flow_json_to_dsl,
    validate_compiled_flow,
    validate_dsl_document,
)
from bm_flow_agent.dsl.importer import normalize_runtime_flow
from bm_flow_agent.graph import (
    AgentRuntime,
    build_agent_graph,
    initial_state,
    latest_assistant_text,
)
from bm_flow_agent.tools import build_repo_catalog, build_tool_registry
from bm_flow_agent.ui import ConsoleUI

_original_make_metavar = click.core.Parameter.make_metavar


def _compat_make_metavar(self, ctx=None):  # pragma: no cover - compatibility shim
    return _original_make_metavar(self, ctx)


click.core.Parameter.make_metavar = _compat_make_metavar

app = typer.Typer(help="Botmother Flow Agent", add_completion=False)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_flow_path() -> str:
    return "assets/flow.json"


def default_dsl_path() -> str:
    return "agent/workflows/main.flow.yaml"


def default_recursion_limit() -> int:
    return 200


def create_model(model_name: str):
    if env_flag("IS_OLLAMA"):
        return ChatOllama(model=model_name, temperature=0)
    return ChatAnthropicBedrock(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.7,
    )


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@app.command(
    "inspect",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def inspect_flow(ctx: typer.Context) -> None:
    options = parse_cli_options(ctx.args, {"flow": default_flow_path()})
    flow = options["flow"]
    root = workspace_root()
    ui = ConsoleUI()
    path = (
        (root / flow).resolve()
        if not Path(flow).is_absolute()
        else Path(flow).resolve()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = normalize_runtime_flow(payload)
    node_types: dict[str, int] = {}
    for node in normalized.get("nodes", []):
        node_types[node["type"]] = node_types.get(node["type"], 0) + 1
    ui.status(f"Flow: {path.relative_to(root)}")
    ui.info(
        json.dumps(
            {
                "nodes": len(normalized.get("nodes", [])),
                "edges": len(normalized.get("edges", [])),
                "node_types": node_types,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command(
    "import",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def import_flow(ctx: typer.Context) -> None:
    options = parse_cli_options(ctx.args, {"flow": None}, required={"flow"})
    flow = options["flow"]
    root = workspace_root()
    source = (
        (root / flow).resolve()
        if not Path(flow).is_absolute()
        else Path(flow).resolve()
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    document = import_flow_json_to_dsl(payload, name=source.stem)
    typer.echo(dump_dsl_to_yaml(document))


@app.command(
    "compile",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def compile_flow(ctx: typer.Context) -> None:
    options = parse_cli_options(ctx.args, {"flow": None, "in": "-"}, required={"flow"})
    flow = options["flow"]
    source = options["in"] or "-"
    root = workspace_root()
    ui = ConsoleUI()
    document = load_yaml_from_source(root, source)
    dsl_errors = validate_dsl_document(document)
    if dsl_errors:
        ui.error("\n".join(dsl_errors))
        raise typer.Exit(code=1)
    compiled = compile_dsl_document(document)
    compiled_errors = validate_compiled_flow(compiled)
    if compiled_errors:
        ui.error("\n".join(compiled_errors))
        raise typer.Exit(code=1)
    target = root / flow if not Path(flow).is_absolute() else Path(flow)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(compiled, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ui.success(f"Saved YAML flow to {display_path(target, root)}")


@app.command(
    "validate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def validate_flow(ctx: typer.Context) -> None:
    options = parse_cli_options(
        ctx.args, {"flow": default_flow_path()}, required={"flow"}
    )
    flow = options["flow"]
    root = workspace_root()
    ui = ConsoleUI()
    source = root / flow if not Path(flow).is_absolute() else Path(flow)
    payload = json.loads(source.read_text(encoding="utf-8"))
    document = import_flow_json_to_dsl(payload, name=source.stem)
    errors = [
        *validate_dsl_document(document),
        *validate_compiled_flow(compile_dsl_document(document)),
    ]
    if errors:
        ui.error("\n".join(errors))
        raise typer.Exit(code=1)
    ui.success("Validation passed.")


@app.command(
    "chat", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def chat(ctx: typer.Context) -> None:
    options = parse_cli_options(
        ctx.args,
        {
            "flow": default_flow_path(),
            "model": "qwen3.5:cloud",
            "thread-id": None,
            "recursion-limit": str(default_recursion_limit()),
            "project-id": None,
            "jwt": None,
        },
    )
    run_chat_session(
        flow=options["flow"],
        model_name=options["model"],
        thread_id=options["thread-id"],
        recursion_limit=int(options["recursion-limit"] or default_recursion_limit()),
        project_id=options["project-id"],
        jwt_token=options["jwt"],
    )


@app.command(
    "api", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def api(ctx: typer.Context) -> None:
    options = parse_cli_options(
        ctx.args,
        {
            "flow": default_flow_path(),
            "model": "qwen3.5:cloud",
            "host": "127.0.0.1",
            "port": "8000",
            "recursion-limit": str(default_recursion_limit()),
            "database-url": None,
            "redis-url": None,
            "cors-origins": None,
        },
    )
    import uvicorn

    from bm_flow_agent.api import create_app

    server = create_app(
        workspace_root=workspace_root(),
        default_flow=options["flow"] or default_flow_path(),
        model_name=options["model"] or "qwen3.5:cloud",
        model_factory=create_model,
        recursion_limit=int(options["recursion-limit"] or default_recursion_limit()),
        database_url=options["database-url"],
        redis_url=options["redis-url"],
        cors_origins=parse_csv_option(options["cors-origins"]),
    )
    uvicorn.run(
        server,
        host=options["host"] or "127.0.0.1",
        port=int(options["port"] or "8000"),
    )


def run_chat_session(
    *,
    flow: str,
    model_name: str,
    thread_id: str | None = None,
    recursion_limit: int = 200,
    project_id: str | None = None,
    jwt_token: str | None = None,
) -> None:
    root = workspace_root()
    load_dotenv(root / ".env", override=False)
    auth_context = build_cli_auth_context(project_id=project_id, jwt_token=jwt_token)
    ui = ConsoleUI()
    registry = build_tool_registry(dsl_only=True)
    repo_catalog = build_repo_catalog(root, flow, "", dsl_only=True)
    repo_catalog["project_id"] = auth_context.get("project_id")
    repo_catalog["user_id"] = auth_context.get("user_id")
    runtime = AgentRuntime(
        workspace_root=root,
        target_flow_json=flow,
        target_dsl_path="",
        model=create_model(model_name),
        registry=registry,
        ui=ui,
    )
    graph = build_agent_graph(runtime)
    session_id = thread_id or str(uuid.uuid4())

    current_state = initial_state(
        workspace_root=root,
        target_flow_json=flow,
        target_dsl_path="",
        repo_catalog=repo_catalog,
    )
    current_state["project_id"] = auth_context.get("project_id")
    current_state["user_id"] = auth_context.get("user_id")
    current_state["auth_claims"] = auth_context.get("claims")
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    ui.info("Chat with Botmother Flow Agent (`exit` to quit)")
    if auth_context.get("project_id"):
        ui.status(f"Project: {auth_context['project_id']}")
    if auth_context.get("user_id"):
        ui.status(f"User: {auth_context['user_id']}")
    while True:
        user_input = ui.prompt()
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue
        result = graph.invoke(
            {
                **current_state,
                "messages": [HumanMessage(content=user_input)],
            },
            config=config,
        )
        while result.get("__interrupt__"):
            payload = interrupt_value(result["__interrupt__"][0])
            ui.clarification(payload)
            answer = collect_clarification_answer(ui, payload)
            result = graph.invoke(Command(resume=answer), config=config)
        current_state = {**current_state, **result}
        response = latest_assistant_text(result)
        if response:
            ui.assistant(response)


def interrupt_value(interrupt_obj: object) -> dict[str, object]:
    return getattr(interrupt_obj, "value", interrupt_obj)


def collect_clarification_answer(
    ui: ConsoleUI, payload: dict[str, object]
) -> str | dict[str, str]:
    return ui.select_clarification(payload)


def build_cli_auth_context(
    *,
    project_id: str | None = None,
    jwt_token: str | None = None,
) -> dict[str, object | None]:
    resolved_project_id = (
        project_id or os.getenv("AGENT_PROJECT_ID") or os.getenv("PROJECT_ID")
    )
    resolved_token = jwt_token or os.getenv("AGENT_JWT_TOKEN") or os.getenv("JWT_TOKEN")
    claims = None
    user_id = None
    if resolved_token:
        try:
            claims = decode_token(resolved_token)
            user_id = user_from_claims(claims).user_id
        except Exception:
            claims = {"token_present": True, "token_valid": False}
    return {
        "project_id": resolved_project_id,
        "jwt_token": resolved_token,
        "claims": claims,
        "user_id": user_id,
    }


def load_yaml_from_source(root: Path, source: str) -> DSLDocument:
    yaml_path = root / source if not Path(source).is_absolute() else Path(source)
    raw = sys.stdin.read() if source == "-" else yaml_path.read_text(encoding="utf-8")
    return DSLDocument.model_validate(yaml.safe_load(raw) or {})


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_cli_options(
    args: list[str],
    defaults: dict[str, Optional[str]],
    *,
    required: set[str] | None = None,
) -> dict[str, Optional[str]]:
    parsed = dict(defaults)
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            raise typer.BadParameter(f"Unexpected argument: {token}")
        key = token[2:]
        if key not in parsed:
            raise typer.BadParameter(f"Unknown option: {token}")
        if index + 1 >= len(args):
            raise typer.BadParameter(f"Missing value for {token}")
        parsed[key] = args[index + 1]
        index += 2
    for key in required or set():
        if not parsed.get(key):
            raise typer.BadParameter(f"Missing required option --{key}")
    return parsed


def parse_csv_option(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
