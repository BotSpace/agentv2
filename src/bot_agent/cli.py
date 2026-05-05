from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import click
import typer
from dotenv import load_dotenv
from langchain_aws import ChatAnthropicBedrock
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from bot_agent.auth import decode_token, user_from_claims
from bot_agent.graph import AgentRuntime, build_agent_graph, initial_state, latest_assistant_text
from bot_agent.tools import build_project_catalog, build_tool_registry
from bot_agent.ui import ConsoleUI

_original_make_metavar = click.core.Parameter.make_metavar


def _compat_make_metavar(self, ctx=None):  # pragma: no cover - compatibility shim
    try:
        return _original_make_metavar(self, ctx)
    except TypeError:
        return _original_make_metavar(self)


click.core.Parameter.make_metavar = _compat_make_metavar

app = typer.Typer(help="Bot Agent", add_completion=False)


@dataclass(frozen=True)
class ModelSettings:
    provider: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    request_timeout: float | None = None


def workspace_root() -> Path:
    return Path.cwd()


def default_project_dir() -> str:
    return "generated_bots/bot"


def default_recursion_limit() -> int:
    return 200


def create_model(model_name: str, settings: ModelSettings | None = None):
    resolved = merge_model_settings(settings, model_settings_from_env())
    provider = resolve_provider(resolved)
    if provider == "openai":
        return create_openai_model(model_name, resolved)
    if provider == "ollama":
        ensure_ollama_ready()
        return ChatOllama(model=model_name, temperature=0)
    ensure_bedrock_ready()
    return ChatAnthropicBedrock(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.7,
    )


def model_settings_from_env() -> ModelSettings:
    timeout_value = os.getenv("BOT_AGENT_REQUEST_TIMEOUT") or os.getenv("OPENAI_REQUEST_TIMEOUT")
    return ModelSettings(
        provider=os.getenv("BOT_AGENT_MODEL_PROVIDER"),
        api_base=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=float(timeout_value) if timeout_value else None,
    )


def merge_model_settings(
    explicit: ModelSettings | None,
    defaults: ModelSettings | None = None,
) -> ModelSettings:
    base = defaults or ModelSettings()
    if explicit is None:
        return base
    return ModelSettings(
        provider=explicit.provider if explicit.provider is not None else base.provider,
        api_base=explicit.api_base if explicit.api_base is not None else base.api_base,
        api_key=explicit.api_key if explicit.api_key is not None else base.api_key,
        request_timeout=(
            explicit.request_timeout
            if explicit.request_timeout is not None
            else base.request_timeout
        ),
    )


def resolve_provider(settings: ModelSettings) -> str:
    provider = (settings.provider or "").strip().lower()
    if provider:
        if provider in {"openai", "chatgpt"}:
            return "openai"
        if provider in {"ollama"}:
            return "ollama"
        if provider in {"bedrock", "anthropic-bedrock"}:
            return "bedrock"
        raise RuntimeError(
            "Unsupported model provider. Use one of: openai, ollama, bedrock."
        )
    if settings.api_base or settings.api_key:
        return "openai"
    if env_flag("IS_OLLAMA"):
        return "ollama"
    return "bedrock"


def create_openai_model(model_name: str, settings: ModelSettings):
    base_url = normalize_openai_base_url((settings.api_base or "").strip() or None)
    api_key = (settings.api_key or "").strip()
    ensure_openai_ready(api_key=api_key, base_url=base_url)
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key or "dummy",
        base_url=base_url,
        timeout=settings.request_timeout or 30,
        max_retries=0,
    )


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url
    if parsed.path in {"", "/"}:
        return urlunparse(parsed._replace(path="/v1"))
    return base_url.rstrip("/")


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_ollama_ready() -> None:
    if not ollama_server_reachable():
        raise RuntimeError(
            "Ollama backend is selected but the Ollama server is not reachable at 127.0.0.1:11434. "
            "Start it with `ollama serve`, then run `IS_OLLAMA=true bot-agent chat --model <model>`."
        )


def ensure_bedrock_ready() -> None:
    has_access_key = bool(os.getenv("AWS_ACCESS_KEY_ID"))
    has_secret_key = bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
    has_region = bool(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
    if has_access_key and has_secret_key and has_region:
        return
    if ollama_server_reachable():
        raise RuntimeError(
            "No AWS Bedrock credentials were found. Since Ollama is running locally, use "
            "`IS_OLLAMA=true bot-agent chat --model <model>` instead."
        )
    raise RuntimeError(
        "No model backend is configured. Either:\n"
        "1. start Ollama and run `IS_OLLAMA=true bot-agent chat --model <model>`\n"
        "2. set `BOT_AGENT_MODEL_PROVIDER=openai` with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`\n"
        "3. or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION for Bedrock."
    )


def ensure_openai_ready(*, api_key: str, base_url: str | None) -> None:
    if api_key:
        return
    if base_url:
        return
    raise RuntimeError(
        "OpenAI backend is selected but OPENAI_API_KEY is missing. "
        "Set `OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL` for a custom API endpoint."
    )


def ollama_server_reachable(host: str = "127.0.0.1", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@app.command("chat", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def chat(ctx: typer.Context) -> None:
    options = parse_cli_options(
        ctx.args,
        {
            "project-dir": default_project_dir(),
            "model": "qwen3.5:cloud",
            "provider": None,
            "api-base": None,
            "api-key": None,
            "request-timeout": None,
            "debug": None,
            "thread-id": None,
            "recursion-limit": str(default_recursion_limit()),
            "project-id": None,
            "jwt": None,
        },
    )
    run_chat_session(
        project_dir=options["project-dir"] or default_project_dir(),
        model_name=options["model"] or "qwen3.5:cloud",
        model_settings=ModelSettings(
            provider=options["provider"],
            api_base=options["api-base"],
            api_key=options["api-key"],
            request_timeout=float(options["request-timeout"]) if options["request-timeout"] else None,
        ),
        debug=option_or_env_flag(options["debug"], "BOT_AGENT_DEBUG"),
        thread_id=options["thread-id"],
        recursion_limit=int(options["recursion-limit"] or default_recursion_limit()),
        project_id=options["project-id"],
        jwt_token=options["jwt"],
    )


@app.command("api", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def api(ctx: typer.Context) -> None:
    options = parse_cli_options(
        ctx.args,
        {
            "project-dir": default_project_dir(),
            "model": "qwen3.5:cloud",
            "provider": None,
            "api-base": None,
            "api-key": None,
            "request-timeout": None,
            "host": "127.0.0.1",
            "port": "8000",
            "recursion-limit": str(default_recursion_limit()),
            "database-url": None,
            "redis-url": None,
            "cors-origins": None,
        },
    )
    import uvicorn

    from bot_agent.api import create_app

    model_settings = ModelSettings(
        provider=options["provider"],
        api_base=options["api-base"],
        api_key=options["api-key"],
        request_timeout=float(options["request-timeout"]) if options["request-timeout"] else None,
    )

    server = create_app(
        workspace_root=workspace_root(),
        default_project_dir=options["project-dir"] or default_project_dir(),
        model_name=options["model"] or "qwen3.5:cloud",
        model_factory=lambda name: create_model(name, model_settings),
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
    project_dir: str,
    model_name: str,
    model_settings: ModelSettings | None = None,
    debug: bool = False,
    thread_id: str | None = None,
    recursion_limit: int = 200,
    project_id: str | None = None,
    jwt_token: str | None = None,
) -> None:
    root = workspace_root()
    load_dotenv(root / ".env", override=False)
    auth_context = build_cli_auth_context(project_id=project_id, jwt_token=jwt_token)
    resolved_settings = merge_model_settings(model_settings, model_settings_from_env())
    ui = ConsoleUI(debug_enabled=debug)
    ui.debug(
        f"session_start model={model_name} provider={resolve_provider(resolved_settings)} project_dir={project_dir}"
    )
    runtime = AgentRuntime(
        workspace_root=root,
        model=create_model(model_name, resolved_settings),
        registry=build_tool_registry(),
        ui=ui,
    )
    graph = build_agent_graph(runtime)
    session_id = thread_id or str(uuid.uuid4())

    current_state = initial_state(
        workspace_root=root,
        project_dir=project_dir,
        project_catalog=build_project_catalog(root, project_dir),
    )
    current_state["project_id"] = auth_context.get("project_id")
    current_state["user_id"] = auth_context.get("user_id")
    current_state["auth_claims"] = auth_context.get("claims")
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
    }

    ui.info("Chat with Bot Agent (`exit` to quit)")
    ui.status(f"Project directory: {project_dir}")
    while True:
        user_input = ui.prompt()
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue
        ui.debug(f"user_input received len={len(user_input.strip())}")
        result = graph.invoke({**current_state, "messages": [HumanMessage(content=user_input)]}, config=config)
        while result.get("__interrupt__"):
            payload = interrupt_value(result["__interrupt__"][0])
            ui.debug(
                f"interrupt received tool={payload.get('tool_name')} question={payload.get('question')!r}"
            )
            ui.clarification(payload)
            answer = collect_clarification_answer(ui, payload)
            ui.debug(f"interrupt answer={answer!r}")
            result = graph.invoke(Command(resume=answer), config=config)
        current_state = {**current_state, **result}
        response = latest_assistant_text(result)
        if response:
            ui.debug(f"assistant_response len={len(response)}")
            ui.assistant(response)


def interrupt_value(interrupt_obj: object) -> dict[str, object]:
    value = getattr(interrupt_obj, "value", interrupt_obj)
    return value if isinstance(value, dict) else {"question": str(value)}


def collect_clarification_answer(
    ui: ConsoleUI, payload: dict[str, object]
) -> str | dict[str, object]:
    return ui.select_clarification(payload)


def build_cli_auth_context(
    *,
    project_id: str | None = None,
    jwt_token: str | None = None,
) -> dict[str, object | None]:
    resolved_project_id = project_id or os.getenv("AGENT_PROJECT_ID") or os.getenv("PROJECT_ID")
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


def option_or_env_flag(value: str | None, env_name: str) -> bool:
    if value is None:
        return env_flag(env_name)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
