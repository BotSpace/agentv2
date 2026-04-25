from __future__ import annotations

import json
import time
from pathlib import Path

import jwt
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from langchain_core.messages import AIMessage

from bm_flow_agent.api import create_app
from bm_flow_agent.auth import CurrentUser, get_current_user
from bm_flow_agent.storage import PersistentRunStore


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


def test_api_run_streams_tool_and_flow_change_events(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_task_plan",
                            "args": {"items": [{"id": "edit_flow", "title": "Edit flow"}]},
                            "id": "plan-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "upsert_step",
                            "args": {
                                "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
                                "trigger": False,
                                "incoming": {"from": "start"},
                            },
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_task_plan_item",
                            "args": {"item_id": "edit_flow", "status": "completed"},
                            "id": "plan-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Flow updated."),
            ]
        ),
    )
    client = make_client(app)

    created = client.post("/projects/p1/runs", json={"prompt": "add welcome"}).json()
    assert created["status"] == "queued"
    request_id = created["request_id"]
    wait_for_status(client, request_id, "completed")

    response = client.get(f"/projects/p1/runs/{request_id}/events")
    events = parse_sse(response.text)
    event_types = [event["type"] for event in events]

    assert "run_started" in event_types
    assert "plan_created" in event_types
    assert "plan_item_updated" in event_types
    assert "plan_completed" in event_types
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "node_upserted" in event_types
    assert "edge_upserted" in event_types
    assert "assistant_message" in event_types
    assert "run_completed" in event_types

    tool_call_events = [event for event in events if event["type"] == "tool_call"]
    assert tool_call_events[0]["description"] == "Agent ish rejasini tuzmoqda: 1 ta vazifa."
    assert any(
        event["description"] == "Agent node qo'shmoqda yoki yangilamoqda: welcome (send_text), start dan ulanadi."
        for event in tool_call_events
    )

    node_event = next(event for event in events if event["type"] == "node_upserted")
    assert node_event["description"] == "Node qo'shildi yoki yangilandi: welcome (send_text)."
    assert node_event["payload"]["description"] == node_event["description"]
    assert node_event["payload"]["step"] == {"id": "welcome", "kind": "send_text", "text": "Salom"}
    assert node_event["payload"]["node"]["id"] == "welcome"
    assert node_event["payload"]["node"]["type"] == "SendTextMessageNode"
    assert node_event["payload"]["node"]["data"] == {"messageText": "Salom"}

    edge_event = next(event for event in events if event["type"] == "edge_upserted")
    assert edge_event["description"] == "Edge ulandi: start -> welcome."
    assert edge_event["payload"]["description"] == edge_event["description"]
    assert edge_event["payload"]["route"] == {"from": "start", "to": "welcome"}
    assert edge_event["payload"]["edge"]["source"] == "start"
    assert edge_event["payload"]["edge"]["target"] == "welcome"


def test_api_connect_steps_bulk_emits_one_event_per_edge(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_task_plan",
                            "args": {"items": [{"id": "edit_flow", "title": "Edit flow"}]},
                            "id": "plan-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "connect_steps",
                            "args": {
                                "routes": [
                                    {"from": "start", "to": "welcome"},
                                    {"from": "welcome", "to": "ask_name"},
                                ]
                            },
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "upsert_step",
                            "args": {
                                "step": {"id": "welcome", "kind": "send_text", "text": "Salom"},
                                "trigger": False,
                            },
                            "id": "tool-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "upsert_step",
                            "args": {
                                "step": {"id": "ask_name", "kind": "send_text", "text": "Ismingiz?"},
                                "trigger": False,
                            },
                            "id": "tool-3",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_task_plan_item",
                            "args": {"item_id": "edit_flow", "status": "completed"},
                            "id": "plan-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        ),
    )
    client = make_client(app)

    request_id = client.post("/projects/p1/runs", json={"prompt": "add flow"}).json()["request_id"]
    wait_for_status(client, request_id, "completed")
    events = parse_sse(client.get(f"/projects/p1/runs/{request_id}/events").text)

    edge_events = [event for event in events if event["type"] == "edge_upserted"]
    assert len(edge_events) == 2
    assert [event["payload"]["edge"]["target"] for event in edge_events] == ["welcome", "ask_name"]


def test_api_clarification_pause_and_resume(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_clarification",
                            "args": {
                                "question": "Qaysi collection?",
                                "options": [
                                    {"id": "movies", "label": "Movies"},
                                    {"id": "series", "label": "Series"},
                                ],
                            },
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Movies collection ishlataman."),
            ]
        ),
    )
    client = make_client(app)

    request_id = client.post("/projects/p1/runs", json={"prompt": "make bot"}).json()["request_id"]
    status = wait_for_status(client, request_id, "waiting_for_clarification")
    assert status["pending_clarification"]["question"] == "Qaysi collection?"
    assert status["pending_clarification"]["options"][0]["id"] == "movies"

    clarify = client.post(
        f"/projects/p1/runs/{request_id}/clarify",
        json={"answer": "Movies", "option_id": "movies"},
    )
    assert clarify.status_code == 200
    wait_for_status(client, request_id, "completed")
    events = parse_sse(client.get(f"/projects/p1/runs/{request_id}/events").text)
    messages = client.get("/projects/p1/messages").json()

    assert any(event["type"] == "clarification_required" for event in events)
    assert events[-1]["type"] == "run_completed"
    assert any(
        message["role"] == "assistant"
        and "Agent aniqlashtirish uchun savol berdi:" in message["content"]
        and "Qaysi collection?" in message["content"]
        and "Movies" in message["content"]
        for message in messages
    )
    assert any(
        message["role"] == "user"
        and message["content"] == "Foydalanuvchi javob berdi:\nQaysi collection? — Movies"
        for message in messages
    )


def test_api_clarification_accepts_batched_answers(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_clarification",
                            "args": {
                                "questions": [
                                    {
                                        "id": "language",
                                        "question": "Til?",
                                        "options": [{"id": "uz", "label": "O'zbek"}],
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
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Batch answers accepted."),
            ]
        ),
    )
    client = make_client(app)

    request_id = client.post("/projects/p1/runs", json={"prompt": "make crm"}).json()["request_id"]
    status = wait_for_status(client, request_id, "waiting_for_clarification")
    assert status["pending_clarification"]["questions"][1]["selection_type"] == "multiple"

    clarify = client.post(
        f"/projects/p1/runs/{request_id}/clarify",
        json={
            "answer": "batch",
            "answers": [
                {"question_id": "language", "answer": "O'zbek", "option_id": "uz"},
                {
                    "question_id": "features",
                    "answer": "Talabalar, To'lovlar",
                    "option_ids": ["students", "payments"],
                },
            ],
        },
    )
    assert clarify.status_code == 200
    wait_for_status(client, request_id, "completed")
    messages = client.get("/projects/p1/messages").json()
    assistant_clarification = next(
        message
        for message in messages
        if message["role"] == "assistant"
        and "Agent quyidagi savollarni aniqlashtirish uchun so'radi:" in message["content"]
    )
    user_clarification = next(
        message
        for message in messages
        if message["role"] == "user"
        and "Foydalanuvchi quyidagi javoblarni berdi:" in message["content"]
    )

    assert "Til?" in assistant_clarification["content"]
    assert "Funksiyalar?" in assistant_clarification["content"]
    assert "Talabalar" in assistant_clarification["content"]
    assert "Til? — O'zbek" in user_clarification["content"]
    assert "Funksiyalar? — Talabalar, To'lovlar" in user_clarification["content"]
    assert "batch clarification" not in user_clarification["content"]


def test_api_active_runs_returns_waiting_progress_and_events(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_clarification",
                            "args": {
                                "question": "Qaysi til?",
                                "options": [{"id": "uz", "label": "O'zbek"}],
                            },
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Davom etdim."),
            ]
        ),
    )
    client = make_client(app)

    request_id = client.post("/projects/p1/runs", json={"prompt": "make bot"}).json()["request_id"]
    wait_for_status(client, request_id, "waiting_for_clarification")
    active = client.get("/projects/p1/runs/active?include_events=true&event_limit=10").json()

    assert [run["request_id"] for run in active] == [request_id]
    assert active[0]["status"] == "waiting_for_clarification"
    assert active[0]["pending_clarification"]["question"] == "Qaysi til?"
    assert active[0]["last_event_seq"] >= 1
    assert any(event["type"] == "clarification_required" for event in active[0]["events"])

    client.post(
        f"/projects/p1/runs/{request_id}/clarify",
        json={"answer": "O'zbek", "option_id": "uz"},
    )
    wait_for_status(client, request_id, "completed")
    assert client.get("/projects/p1/runs/active").json() == []


def test_api_active_runs_survive_new_app_instance(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'agent.db'}"
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=database_url,
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_clarification",
                            "args": {"question": "Davom etamizmi?"},
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="OK."),
            ]
        ),
    )
    client = make_client(app)
    request_id = client.post("/projects/p1/runs", json={"prompt": "start"}).json()["request_id"]
    wait_for_status(client, request_id, "waiting_for_clarification")

    restarted = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=database_url,
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="unused")]),
    )
    restarted_client = make_client(restarted)
    active = restarted_client.get("/projects/p1/runs/active").json()

    assert [run["request_id"] for run in active] == [request_id]
    assert active[0]["pending_clarification"]["question"] == "Davom etamizmi?"


def test_api_events_keep_tool_call_descriptions_after_reconnect(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'agent.db'}"
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=database_url,
        redis_url="",
        model_factory=lambda _name: FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_task_plan",
                            "args": {"items": [{"id": "edit_flow", "title": "Edit flow"}]},
                            "id": "plan-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_task_plan_item",
                            "args": {"item_id": "edit_flow", "status": "completed"},
                            "id": "plan-2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Reja tayyor."),
            ]
        ),
    )
    client = make_client(app)
    request_id = client.post("/projects/p1/runs", json={"prompt": "plan"}).json()["request_id"]
    wait_for_status(client, request_id, "completed")

    restarted = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=database_url,
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="unused")]),
    )
    restarted_client = make_client(restarted)
    events = parse_sse(restarted_client.get(f"/projects/p1/runs/{request_id}/events").text)
    tool_event = next(event for event in events if event["type"] == "tool_call")

    assert tool_event["description"] == "Agent ish rejasini tuzmoqda: 1 ta vazifa."
    assert tool_event["payload"]["description"] == tool_event["description"]


def test_api_startup_recovers_queued_and_running_runs(monkeypatch, tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    store = PersistentRunStore(database_url=f"sqlite:///{tmp_path / 'agent.db'}", redis_url="")
    queued = store.create(
        project_id="p1",
        user_id="test-user",
        prompt="queued",
        flow=str(flow_path),
    )
    running = store.create(
        project_id="p1",
        user_id="test-user",
        prompt="running",
        flow=str(flow_path),
    )
    store.set_status(running.request_id, "running")
    started = []

    def fake_start_agent_worker(**kwargs):
        started.append(kwargs)

    monkeypatch.setattr("bm_flow_agent.api.start_agent_worker", fake_start_agent_worker)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        store=store,
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="unused")]),
    )

    with TestClient(app):
        pass

    assert {item["request_id"] for item in started} == {queued.request_id, running.request_id}
    assert any(not item["resume_from_checkpoint"] for item in started)
    assert any(item["resume_from_checkpoint"] for item in started)


def test_api_requires_valid_bearer_token(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Done.")]),
    )
    client = TestClient(app)

    missing = client.post("/projects/p1/runs", json={"prompt": "hello"})
    invalid = client.post(
        "/projects/p1/runs",
        headers={"Authorization": "Bearer invalid"},
        json={"prompt": "hello"},
    )

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_api_accepts_rs256_jwt(monkeypatch, tmp_path: Path) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    token = jwt.encode({"sub": "jwt-user"}, private_pem, algorithm="RS256")
    monkeypatch.setenv("AGENT_JWT_PUBLIC_KEY", public_pem)
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Done.")]),
    )
    client = TestClient(app)

    response = client.post(
        "/projects/p1/runs",
        headers={"Authorization": f"Bearer {token}"},
        json={"prompt": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"]
    assert "chat_id" not in response.json()


def test_api_project_messages_are_project_scoped(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Assistant final.")]),
    )
    client = make_client(app)

    created = client.post("/projects/p1/runs", json={"prompt": "first prompt"}).json()
    wait_for_status(client, created["request_id"], "completed")
    second = client.post(
        "/projects/p1/runs",
        json={"prompt": "second prompt"},
    ).json()
    wait_for_status(client, second["request_id"], "completed")

    messages = client.get("/projects/p1/messages").json()
    other_project_run = client.get(f"/projects/p2/runs/{created['request_id']}")
    other_project_messages = client.get("/projects/p2/messages").json()

    assert [message["role"] for message in messages].count("user") == 2
    assert any(message["content"] == "Assistant final." for message in messages)
    assert other_project_run.status_code == 404
    assert other_project_messages == []


def test_old_routes_are_not_registered(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Done.")]),
    )
    client = TestClient(app)

    assert client.post("/runs", json={"prompt": "hello"}).status_code == 404
    assert client.get("/runs/abc").status_code == 404
    assert client.get("/runs/abc/events").status_code == 404
    assert client.post("/runs/abc/clarify", json={"answer": "x"}).status_code == 404
    assert client.get("/projects/p1/chats").status_code == 404
    assert client.get("/projects/p1/chats/abc/messages").status_code == 404


def test_api_allows_local_frontend_cors_preflight(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Done.")]),
    )
    client = TestClient(app)

    response = client.options(
        "/projects/p1/runs",
        headers={
            "Origin": "http://127.0.0.1:3001",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3001"


def test_run_create_rejects_chat_id(tmp_path: Path) -> None:
    flow_path = write_start_flow(tmp_path)
    app = create_app(
        workspace_root=tmp_path,
        default_flow=str(flow_path),
        model_name="fake",
        database_url=f"sqlite:///{tmp_path / 'agent.db'}",
        redis_url="",
        model_factory=lambda _name: FakeModel([AIMessage(content="Done.")]),
    )
    client = make_client(app)

    response = client.post("/projects/p1/runs", json={"prompt": "hello", "chat_id": "old"})

    assert response.status_code == 422


def write_start_flow(tmp_path: Path) -> Path:
    flow_path = tmp_path / "flow.json"
    flow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "start",
                        "type": "CommandTriggerNode",
                        "data": {"command": "/start", "global": True},
                        "position": {"x": 0, "y": 0},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return flow_path


def make_client(app) -> TestClient:
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id="test-user",
        claims={"sub": "test-user"},
    )
    return TestClient(app)


def wait_for_status(client: TestClient, request_id: str, expected: str) -> dict:
    for _ in range(80):
        status = client.get(f"/projects/p1/runs/{request_id}").json()
        if status["status"] == expected:
            return status
        if status["status"] == "failed":
            raise AssertionError(status["error"])
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {expected}")


def parse_sse(raw: str) -> list[dict]:
    events = []
    for block in raw.strip().split("\n\n"):
        if not block:
            continue
        data_line = next(line for line in block.splitlines() if line.startswith("data: "))
        events.append(json.loads(data_line.removeprefix("data: ")))
    return events
