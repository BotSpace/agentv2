from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from bot_agent.api import create_app
from bot_agent.auth import CurrentUser, get_current_user
from bot_agent.cli import (
    ModelSettings,
    app,
    default_request_timeout,
    effective_request_timeout,
    merge_model_settings,
    normalize_openai_base_url,
    run_chat_session,
    workspace_root,
)
from bot_agent.graph import (
    AgentRuntime,
    build_agent_graph,
    compact_error,
    initial_state,
    invoke_model_with_retry,
    is_smalltalk_message,
    latest_assistant_text,
    timeout_error_hint,
)
from bot_agent.prompts import build_system_prompt
from bot_agent.storage import PersistentRunStore
from bot_agent.tools import build_project_catalog, build_tool_registry
from bot_agent.tools.base import ToolContext
from bot_agent.ui import ConsoleUI
from typer.testing import CliRunner


class FakeBoundModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def invoke(self, _messages):
        response = self.responses[self.index]
        self.index += 1
        return response


class FakeModel:
    def __init__(self, responses):
        self.responses = responses

    def bind_tools(self, _tools):
        return FakeBoundModel(self.responses)

    def invoke(self, _messages):
        return self.responses[0]


class FlakyModel:
    def __init__(self, failures: int, response: AIMessage):
        self.failures = failures
        self.response = response
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary api failure")
        return self.response


def test_registry_contains_file_based_tools() -> None:
    names = set(build_tool_registry().names())
    assert {
        "inspect_bot_project",
        "scaffold_bot_project",
        "list_files",
        "read_file",
        "write_file",
        "replace_in_file",
        "make_directory",
        "run_command",
        "validate_bot_project",
        "create_task_plan",
        "update_task_plan_item",
        "request_clarification",
    }.issubset(names)
    assert "create_bot_spec" not in names
    assert "write_bot_project" not in names


def test_scaffold_bot_project_creates_default_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "bot_agent.tools.bot_tools.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "project_dir": "generated_bots/demo_bot",
            "bot_project_dir": "generated_bots/demo_bot",
            "current_plan": {"items": [{"id": "scaffold", "title": "Scaffold", "status": "in_progress"}]},
        },
        ui=ConsoleUI(),
    )

    result = registry.execute("scaffold_bot_project", context, {})

    assert result.is_error is False
    project_dir = tmp_path / "generated_bots" / "demo_bot"
    assert (project_dir / "go.mod").exists()
    assert (project_dir / "cmd" / "bot" / "main.go").exists()
    assert (project_dir / "internal" / "app" / "app.go").exists()
    assert (project_dir / "internal" / "config" / "config.go").exists()
    assert (project_dir / "internal" / "handlers" / "handler.go").exists()
    assert (project_dir / "internal" / "state" / "store.go").exists()
    assert (project_dir / "pkg" / "messages" / "messages.go").exists()
    assert (project_dir / ".env.example").exists()
    assert (project_dir / "README.md").exists()
    assert "module demo_bot" in (project_dir / "go.mod").read_text(encoding="utf-8")
    assert '"command": "go mod tidy"' in result.content
    main_go = (project_dir / "cmd" / "bot" / "main.go").read_text(encoding="utf-8")
    assert '"demo_bot/internal/app"' in main_go
    assert '"demo_bot/internal/config"' in main_go
    app_go = (project_dir / "internal" / "app" / "app.go").read_text(encoding="utf-8")
    assert "func (a *App) Run" in app_go
    assert "HandleCallbackQuery" in app_go
    assert "state.NewMemoryStore()" in app_go
    handler_go = (project_dir / "internal" / "handlers" / "handler.go").read_text(encoding="utf-8")
    assert "func (h *Handler) HandleMessage" in handler_go
    assert "func (h *Handler) HandleCallbackQuery" in handler_go
    assert "NewEditMessageTextAndMarkup" in handler_go
    assert "query.Message == nil" in handler_go
    assert "NewInlineKeyboardMarkup" in handler_go
    assert "*state.MemoryStore" in handler_go
    state_go = (project_dir / "internal" / "state" / "store.go").read_text(encoding="utf-8")
    assert "type UserState struct" in state_go
    assert "func (s *MemoryStore) SetStep" in state_go
    messages_go = (project_dir / "pkg" / "messages" / "messages.go").read_text(encoding="utf-8")
    assert "Assalomu alaykum!" in messages_go
    assert "func Menu() string" in messages_go
    assert "//go:embed" not in messages_go


def test_prompt_instructs_agent_to_scaffold_first() -> None:
    prompt = build_system_prompt({"project_dir": "generated_bots/demo"})
    assert "scaffold_bot_project" in prompt
    assert "Write clean code" in prompt
    assert "cmd/bot" in prompt
    assert "internal/app" in prompt
    assert "pkg" in prompt
    assert "generated bot projects may keep simple user-facing text in Go files" in prompt
    assert "Do not stop at only /start and /help" in prompt
    assert "education center" in prompt
    assert "inline buttons" in prompt
    assert "prefer editing the existing bot message" in prompt
    assert "EditMessageTextAndMarkup" in prompt
    assert "Default scaffold code after scaffold_bot_project" in prompt
    assert "working from the repository/workspace root" in prompt
    assert "cmd/bot/main.go" in prompt
    assert "tool layer will automatically map" in prompt
    assert "### cmd/bot/main.go" in prompt
    assert '"demo/internal/app"' in prompt
    assert "func (h *Handler) HandleCallbackQuery" in prompt
    assert "NewEditMessageTextAndMarkup" in prompt
    assert "NewInlineKeyboardMarkup" in prompt
    assert "Assalomu alaykum!" in prompt
    assert "internal/state" in prompt
    assert "current step/screen" in prompt


def test_write_and_read_file_tools_work(tmp_path: Path) -> None:
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "project_dir": "bot",
            "current_plan": {"items": [{"id": "build", "title": "Build", "status": "in_progress"}]},
        },
        ui=ConsoleUI(),
    )

    write = registry.execute("write_file", context, {"path": "main.go", "content": "package main\n"})
    read = registry.execute("read_file", context, {"path": "main.go"})

    assert write.is_error is False
    assert "1: package main" in read.content
    assert (tmp_path / "bot" / "main.go").exists()


def test_bot_file_tools_treat_dot_as_project_root(tmp_path: Path) -> None:
    registry = build_tool_registry()
    context = ToolContext(workspace_root=tmp_path, state={"project_dir": "generated_bots/my_bot"}, ui=ConsoleUI())
    project_dir = tmp_path / "generated_bots" / "my_bot"
    project_dir.mkdir(parents=True)
    (project_dir / "go.mod").write_text("module my_bot\n", encoding="utf-8")

    result = registry.execute("list_files", context, {"path": "."})

    assert result.is_error is False
    assert '"go.mod"' in result.content


def test_bot_file_tools_accept_already_prefixed_project_paths(tmp_path: Path) -> None:
    registry = build_tool_registry()
    context = ToolContext(workspace_root=tmp_path, state={"project_dir": "generated_bots/my_bot"}, ui=ConsoleUI())
    project_dir = tmp_path / "generated_bots" / "my_bot"
    project_dir.mkdir(parents=True)
    (project_dir / "go.mod").write_text("module my_bot\n", encoding="utf-8")

    result = registry.execute("read_file", context, {"path": "generated_bots/my_bot/go.mod"})

    assert result.is_error is False
    assert "1: module my_bot" in result.content


def test_replace_in_file_tool_updates_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "bot" / "main.go"
    target.parent.mkdir()
    target.write_text("package main\n", encoding="utf-8")
    registry = build_tool_registry()
    context = ToolContext(
        workspace_root=tmp_path,
        state={
            "project_dir": "bot",
            "current_plan": {"items": [{"id": "edit", "title": "Edit", "status": "in_progress"}]},
        },
        ui=ConsoleUI(),
    )

    result = registry.execute(
        "replace_in_file",
        context,
        {"path": "main.go", "old": "package main", "new": "package main\n\nimport \"fmt\""},
    )

    assert result.is_error is False
    assert "import \"fmt\"" in target.read_text(encoding="utf-8")


def test_run_command_tool_uses_requested_cwd(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("bot_agent.tools.bot_tools.subprocess.run", fake_run)
    registry = build_tool_registry()
    context = ToolContext(workspace_root=tmp_path, state={"project_dir": "bot"}, ui=ConsoleUI())

    result = registry.execute("run_command", context, {"command": "go test ./...", "cwd": "."})

    assert result.is_error is False
    assert calls == [("go test ./...", tmp_path / "bot")]


def test_validate_bot_project_runs_go_commands(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "bot"
    project_dir.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("bot_agent.tools.bot_tools.subprocess.run", fake_run)
    registry = build_tool_registry()
    context = ToolContext(workspace_root=tmp_path, state={"bot_project_dir": str(project_dir)}, ui=ConsoleUI())

    result = registry.execute("validate_bot_project", context, {})

    assert result.is_error is False
    assert calls == [("go test ./...", project_dir), ("go vet ./...", project_dir)]


def test_graph_waits_for_plan_completion() -> None:
    runtime = AgentRuntime(
        workspace_root=Path.cwd(),
        registry=build_tool_registry(),
        ui=ConsoleUI(),
        model=FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "create_task_plan", "args": {"items": [{"id": "build", "title": "Build bot"}]}, "id": "plan-1", "type": "tool_call"}],
                ),
                AIMessage(content="Too early."),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "update_task_plan_item", "args": {"item_id": "build", "status": "completed"}, "id": "plan-2", "type": "tool_call"}],
                ),
                AIMessage(content="Done."),
            ]
        ),
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=Path.cwd(),
        project_dir="generated_bots/demo",
        project_catalog=build_project_catalog(Path.cwd(), "generated_bots/demo"),
    )

    result = graph.invoke(
        {**state, "messages": [HumanMessage(content="make bot")]},
        config={"configurable": {"thread_id": "t-plan"}, "recursion_limit": 20},
    )

    assert latest_assistant_text(result) == "Done."


def test_model_invoke_retries_transient_errors() -> None:
    sleeps = []
    model = FlakyModel(failures=2, response=AIMessage(content="Recovered."))

    response = invoke_model_with_retry(
        model,
        [HumanMessage(content="build bot")],
        ConsoleUI(),
        max_attempts=20,
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    assert response.content == "Recovered."
    assert model.calls == 3
    assert sleeps == [1, 2]


def test_compact_error_includes_type_and_short_message() -> None:
    error = RuntimeError("api response parse failed\nmissing tool_call id")

    assert compact_error(error) == "RuntimeError: api response parse failed missing tool_call id"


def test_compact_error_truncates_long_messages() -> None:
    error = RuntimeError("x" * 300)

    text = compact_error(error, max_length=40)

    assert text.startswith("RuntimeError: ")
    assert text.endswith("...")
    assert len(text) == 40


def test_timeout_error_hint_suggests_larger_request_timeout() -> None:
    assert "--request-timeout 120" in timeout_error_hint(TimeoutError("Request timed out"))
    assert timeout_error_hint(RuntimeError("temporary api failure")) == ""


def test_graph_skips_tools_for_smalltalk() -> None:
    calls = []

    class UnexpectedModel(FakeModel):
        def invoke(self, _messages):
            calls.append("invoke")
            raise AssertionError("smalltalk should not call the model")

        def bind_tools(self, _tools):
            calls.append("bind_tools")
            return super().bind_tools(_tools)

    runtime = AgentRuntime(
        workspace_root=Path.cwd(),
        registry=build_tool_registry(),
        ui=ConsoleUI(),
        model=UnexpectedModel([AIMessage(content="should not be used")]),
    )
    graph = build_agent_graph(runtime)
    state = initial_state(
        workspace_root=Path.cwd(),
        project_dir="generated_bots/demo",
        project_catalog=build_project_catalog(Path.cwd(), "generated_bots/demo"),
    )

    result = graph.invoke(
        {**state, "messages": [HumanMessage(content="salom")]},
        config={"configurable": {"thread_id": "t-smalltalk"}, "recursion_limit": 10},
    )

    response = latest_assistant_text(result)
    assert "Qanday bot yaratish kerak?" in response
    assert "o'quv markaz" in response
    assert calls == ["bind_tools"]


def test_is_smalltalk_message_detects_greeting() -> None:
    assert is_smalltalk_message("salom") is True
    assert is_smalltalk_message("Assalomu alaykum") is True
    assert is_smalltalk_message("o'quv markaz uchun bot qilib ber") is False


def test_cli_chat_command_smoke(monkeypatch) -> None:
    called = {}

    def fake_session(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("bot_agent.cli.run_chat_session", fake_session)
    result = CliRunner().invoke(app, ["chat", "--project-dir", "generated_bots/demo", "--model", "fake"])

    assert result.exit_code == 0
    assert called["project_dir"] == "generated_bots/demo"
    assert called["model_name"] == "fake"


def test_cli_chat_command_accepts_openai_options(monkeypatch) -> None:
    called = {}

    def fake_session(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("bot_agent.cli.run_chat_session", fake_session)
    result = CliRunner().invoke(
        app,
        [
            "chat",
            "--project-dir",
            "generated_bots/demo",
            "--model",
            "gpt-4.1",
            "--provider",
            "openai",
            "--api-base",
            "https://example.test/v1",
            "--api-key",
            "secret",
        ],
    )

    assert result.exit_code == 0
    assert called["model_settings"] == ModelSettings(
        provider="openai",
        api_base="https://example.test/v1",
        api_key="secret",
    )


def test_merge_model_settings_fills_missing_values_from_env_defaults() -> None:
    merged = merge_model_settings(
        ModelSettings(request_timeout=15),
        ModelSettings(provider="openai", api_base="http://localhost:4141/", api_key="secret"),
    )

    assert merged == ModelSettings(
        provider="openai",
        api_base="http://localhost:4141/",
        api_key="secret",
        request_timeout=15,
    )


def test_chat_session_initializes_bot_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("bot_agent.cli.workspace_root", lambda: tmp_path)
    monkeypatch.setattr("bot_agent.cli.create_model", lambda _name, _settings=None: object())
    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr("bot_agent.cli.build_agent_graph", lambda runtime: FakeGraph())
    prompts = iter(["build support bot", "exit"])
    monkeypatch.setattr("bot_agent.ui.ConsoleUI.prompt", lambda self, label="You": next(prompts))

    run_chat_session(project_dir="generated_bots/demo", model_name="dummy")

    assert captured["state"]["project_dir"] == "generated_bots/demo"
    assert captured["state"]["bot_project_dir"] == "generated_bots/demo"


def test_workspace_root_uses_current_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert workspace_root() == tmp_path


def test_create_model_fails_fast_without_backend(monkeypatch) -> None:
    monkeypatch.delenv("BOT_AGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("IS_OLLAMA", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setattr("bot_agent.cli.ollama_server_reachable", lambda: False)

    from bot_agent.cli import create_model

    try:
        create_model("dummy")
    except RuntimeError as exc:
        assert "No model backend is configured" in str(exc)
    else:
        raise AssertionError("create_model should fail when no backend is configured")


def test_create_model_fails_fast_when_ollama_selected_but_offline(monkeypatch) -> None:
    monkeypatch.delenv("BOT_AGENT_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("IS_OLLAMA", "true")
    monkeypatch.setattr("bot_agent.cli.ollama_server_reachable", lambda: False)

    from bot_agent.cli import create_model

    try:
        create_model("dummy")
    except RuntimeError as exc:
        assert "Ollama backend is selected" in str(exc)
    else:
        raise AssertionError("create_model should fail when Ollama is selected but offline")


def test_create_model_uses_openai_with_custom_api(monkeypatch) -> None:
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("bot_agent.cli.ChatOpenAI", FakeChatOpenAI)

    from bot_agent.cli import create_model

    create_model(
        "gpt-4.1",
        ModelSettings(
            provider="openai",
            api_base="https://example.test/v1",
            api_key="secret",
        ),
    )

    assert captured["model"] == "gpt-4.1"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["api_key"] == "secret"
    assert captured["temperature"] == 0
    assert captured["timeout"] == default_request_timeout()
    assert captured["max_retries"] == 0


def test_effective_request_timeout_uses_explicit_value() -> None:
    assert effective_request_timeout(ModelSettings(request_timeout=15)) == 15


def test_create_model_allows_keyless_custom_openai_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("bot_agent.cli.ChatOpenAI", FakeChatOpenAI)

    from bot_agent.cli import create_model

    create_model(
        "gpt-4.1",
        ModelSettings(provider="openai", api_base="https://example.test/v1"),
    )

    assert captured["api_key"] == "dummy"


def test_normalize_openai_base_url_adds_v1_for_root_endpoint() -> None:
    assert normalize_openai_base_url("http://localhost:4141/") == "http://localhost:4141/v1"
    assert normalize_openai_base_url("http://localhost:4141") == "http://localhost:4141/v1"
    assert normalize_openai_base_url("http://localhost:4141/v1") == "http://localhost:4141/v1"


def test_create_model_fails_fast_when_openai_selected_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    from bot_agent.cli import create_model

    try:
        create_model("gpt-4.1", ModelSettings(provider="openai"))
    except RuntimeError as exc:
        assert "OPENAI_API_KEY is missing" in str(exc)
    else:
        raise AssertionError("create_model should fail when OpenAI is selected without credentials")


def test_api_writes_file_based_bot_project(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "bot_agent.tools.bot_tools.subprocess.run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    store = PersistentRunStore(database_url=f"sqlite:///{tmp_path / 'agent.db'}", redis_url="")
    api = create_app(
        workspace_root=tmp_path,
        default_project_dir="generated_bots/bot",
        model_name="fake",
        store=store,
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "create_task_plan", "args": {"items": [{"id": "build", "title": "Build bot"}]}, "id": "plan-1", "type": "tool_call"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "make_directory", "args": {"path": "generated_bots/demo"}, "id": "mkdir-1", "type": "tool_call"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "generated_bots/demo/main.go", "content": "package main\n"},
                            "id": "write-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "validate_bot_project", "args": {}, "id": "validate-1", "type": "tool_call"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "update_task_plan_item", "args": {"item_id": "build", "status": "completed"}, "id": "plan-2", "type": "tool_call"}],
                ),
                AIMessage(content="Ready."),
            ]
        ),
    )
    api.dependency_overrides[get_current_user] = lambda: CurrentUser(user_id="u1", claims={"sub": "u1"})
    client = TestClient(api)

    created = client.post("/projects/p1/runs", json={"prompt": "create demo bot", "project_dir": "generated_bots/demo"}).json()
    request_id = created["request_id"]
    status = None
    for _ in range(50):
        status = client.get(f"/projects/p1/runs/{request_id}").json()
        if status["status"] == "completed":
            break

    assert status is not None
    assert status["status"] == "completed"
    assert (tmp_path / "generated_bots" / "demo" / "main.go").exists()
