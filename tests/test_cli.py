from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bm_flow_agent.cli import app, collect_clarification_answer
from bm_flow_agent.ui import ConsoleUI


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


def test_import_compile_validate_commands(tmp_path: Path) -> None:
    compiled_path = tmp_path / "flow.json"

    imported = RUNNER.invoke(
        app,
        ["import", "--flow", str(REPO_ROOT / "examples" / "main.json")],
    )
    assert imported.exit_code == 0, imported.output
    assert "flow:" in imported.output

    validated = RUNNER.invoke(app, ["validate", "--flow", str(REPO_ROOT / "examples" / "main.json")])
    assert validated.exit_code == 0, validated.output

    compiled = RUNNER.invoke(
        app,
        ["compile", "--flow", str(compiled_path), "--in", "-"],
        input=imported.output,
    )
    assert compiled.exit_code == 0, compiled.output
    payload = json.loads(compiled_path.read_text(encoding="utf-8"))
    assert "nodes" in payload and "edges" in payload


def test_inspect_command() -> None:
    result = RUNNER.invoke(app, ["inspect", "--flow", str(REPO_ROOT / "assets" / "flow.json")])
    assert result.exit_code == 0
    assert "node_types" in result.output


def test_chat_command_smoke(monkeypatch) -> None:
    called = {}

    def fake_chat_session(
        *,
        flow: str,
        model_name: str,
        thread_id: str | None = None,
        recursion_limit: int = 0,
        project_id: str | None = None,
        jwt_token: str | None = None,
    ) -> None:
        called["args"] = {
            "flow": flow,
            "model_name": model_name,
            "thread_id": thread_id,
            "recursion_limit": recursion_limit,
            "project_id": project_id,
            "jwt_token": jwt_token,
        }

    monkeypatch.setattr("bm_flow_agent.cli.run_chat_session", fake_chat_session)
    result = RUNNER.invoke(app, ["chat", "--flow", "assets/flow.json", "--project-id", "p1", "--jwt", "token"])
    assert result.exit_code == 0
    assert called["args"]["flow"] == "assets/flow.json"
    assert called["args"]["recursion_limit"] == 200
    assert called["args"]["project_id"] == "p1"
    assert called["args"]["jwt_token"] == "token"


def test_create_model_uses_ollama_when_is_ollama_enabled(monkeypatch) -> None:
    created = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            created["ollama"] = kwargs

    class FakeBedrock:
        def __init__(self, **kwargs):
            created["bedrock"] = kwargs

    monkeypatch.setenv("IS_OLLAMA", "true")
    monkeypatch.setattr("bm_flow_agent.cli.ChatOllama", FakeOllama)
    monkeypatch.setattr("bm_flow_agent.cli.ChatAnthropicBedrock", FakeBedrock)

    model = __import__("bm_flow_agent.cli", fromlist=["create_model"]).create_model("qwen3.5:cloud")

    assert isinstance(model, FakeOllama)
    assert created["ollama"] == {"model": "qwen3.5:cloud", "temperature": 0}
    assert "bedrock" not in created


def test_create_model_uses_bedrock_by_default(monkeypatch) -> None:
    created = {}

    class FakeOllama:
        def __init__(self, **kwargs):
            created["ollama"] = kwargs

    class FakeBedrock:
        def __init__(self, **kwargs):
            created["bedrock"] = kwargs

    monkeypatch.delenv("IS_OLLAMA", raising=False)
    monkeypatch.setattr("bm_flow_agent.cli.ChatOllama", FakeOllama)
    monkeypatch.setattr("bm_flow_agent.cli.ChatAnthropicBedrock", FakeBedrock)

    model = __import__("bm_flow_agent.cli", fromlist=["create_model"]).create_model("ignored-for-bedrock")

    assert isinstance(model, FakeBedrock)
    assert created["bedrock"] == {
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "temperature": 0.7,
    }
    assert "ollama" not in created


def test_chat_does_not_create_yaml_file(tmp_path: Path, monkeypatch) -> None:
    flow_path = tmp_path / "flow.json"
    dsl_path = tmp_path / "main.flow.yaml"
    flow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "start",
                        "type": "CommandTriggerNode",
                        "data": {"command": "/start", "global": True, "withArgs": False},
                        "position": {"x": 0, "y": 0},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("bm_flow_agent.cli.workspace_root", lambda: tmp_path)
    monkeypatch.setattr("bm_flow_agent.cli.create_model", lambda model_name: object())

    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr("bm_flow_agent.cli.build_agent_graph", lambda runtime: FakeGraph())
    prompts = iter(["salom", "exit"])
    monkeypatch.setattr("bm_flow_agent.ui.ConsoleUI.prompt", lambda self, label="You": next(prompts))

    from bm_flow_agent.cli import run_chat_session

    run_chat_session(flow=str(flow_path), model_name="dummy")

    assert not dsl_path.exists()
    assert captured["state"]["target_flow_json"] == str(flow_path)
    assert "working_dsl" not in captured["state"]


def test_chat_loads_project_context_from_env_file(tmp_path: Path, monkeypatch) -> None:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    (tmp_path / ".env").write_text("AGENT_PROJECT_ID=project-from-env\nAGENT_USER_ID=ignored-user\n", encoding="utf-8")
    monkeypatch.setattr("bm_flow_agent.cli.workspace_root", lambda: tmp_path)
    monkeypatch.setattr("bm_flow_agent.cli.create_model", lambda model_name: object())
    monkeypatch.delenv("AGENT_PROJECT_ID", raising=False)
    monkeypatch.delenv("AGENT_USER_ID", raising=False)
    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr("bm_flow_agent.cli.build_agent_graph", lambda runtime: FakeGraph())
    prompts = iter(["salom", "exit"])
    monkeypatch.setattr("bm_flow_agent.ui.ConsoleUI.prompt", lambda self, label="You": next(prompts))

    from bm_flow_agent.cli import run_chat_session

    run_chat_session(flow=str(flow_path), model_name="dummy")

    assert captured["state"]["project_id"] == "project-from-env"
    assert captured["state"]["user_id"] is None


def test_console_ui_emits_output(capsys) -> None:
    ui = ConsoleUI()
    ui.tool_call("save_flow_yaml", {"yaml": "flow: ..."})
    captured = capsys.readouterr().out
    assert "save_flow_yaml" in captured


def test_console_ui_prints_clarification_options(capsys) -> None:
    ui = ConsoleUI()
    ui.clarification(
        {
            "question": "Bot qaysi tilda?",
            "options": [
                {"id": "uz", "label": "O'zbek"},
                {"id": "ru", "label": "Rus"},
            ],
        }
    )
    captured = capsys.readouterr().out
    assert "Bot qaysi tilda?" in captured


def test_collect_clarification_answer_maps_number_to_option(monkeypatch) -> None:
    ui = ConsoleUI()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")

    answer = collect_clarification_answer(
        ui,
        {
            "options": [
                {"id": "uz", "label": "O'zbek"},
                {"id": "ru", "label": "Rus"},
            ]
        },
    )

    assert answer == {"answer": "Rus", "option_id": "ru"}


def test_collect_clarification_answer_accepts_free_text(monkeypatch) -> None:
    ui = ConsoleUI()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "custom talab")

    answer = collect_clarification_answer(
        ui,
        {
            "allow_free_text": True,
            "options": [
                {"id": "uz", "label": "O'zbek"},
            ],
        },
    )

    assert answer == "custom talab"


def test_collect_clarification_answer_supports_multiple_selection(monkeypatch) -> None:
    ui = ConsoleUI()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1,3")

    answer = collect_clarification_answer(
        ui,
        {
            "selection_type": "multiple",
            "options": [
                {"id": "students", "label": "Talabalar"},
                {"id": "payments", "label": "To'lovlar"},
                {"id": "reports", "label": "Hisobotlar"},
            ],
        },
    )

    assert answer == {
        "answer": "Talabalar, Hisobotlar",
        "option_ids": ["students", "reports"],
    }


def test_collect_clarification_answer_supports_batched_questions(monkeypatch) -> None:
    ui = ConsoleUI()
    answers = iter(["1", "1,2"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    answer = collect_clarification_answer(
        ui,
        {
            "questions": [
                {
                    "id": "language",
                    "question": "Til?",
                    "options": [
                        {"id": "uz", "label": "O'zbek"},
                        {"id": "ru", "label": "Rus"},
                    ],
                },
                {
                    "id": "features",
                    "question": "Funksiyalar?",
                    "selection_type": "multiple",
                    "options": [
                        {"id": "students", "label": "Talabalar"},
                        {"id": "payments", "label": "To'lovlar"},
                    ],
                },
            ]
        },
    )

    assert answer == {
        "answers": [
            {"question_id": "language", "answer": "O'zbek", "option_id": "uz"},
            {
                "question_id": "features",
                "answer": "Talabalar, To'lovlar",
                "option_ids": ["students", "payments"],
            },
        ]
    }
