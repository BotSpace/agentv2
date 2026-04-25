from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from bm_flow_agent.auth import CurrentUser, get_current_user
from bm_flow_agent.graph import (
    AgentRuntime,
    build_agent_graph,
    initial_state,
    latest_assistant_text,
)
from bm_flow_agent.storage import PersistentRunStore
from bm_flow_agent.tools import build_repo_catalog, build_tool_registry
from bm_flow_agent.ui import EventUI


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    flow: str | None = None
    thread_id: str | None = None


class RunCreateResponse(BaseModel):
    request_id: str
    status: str


class RunStatusResponse(BaseModel):
    request_id: str
    status: str
    final_response: str | None = None
    error: str | None = None
    pending_clarification: dict[str, Any] | None = None
    current_plan: dict[str, Any] | None = None


class ActiveRunResponse(BaseModel):
    request_id: str
    status: str
    prompt: str
    flow: str
    thread_id: str
    pending_clarification: dict[str, Any] | None = None
    current_plan: dict[str, Any] | None = None
    last_event_seq: int
    created_at: str
    updated_at: str
    events: list[dict[str, Any]] | None = None


class ClarifyRequest(BaseModel):
    answer: str = Field(min_length=1)
    option_id: str | None = None
    option_ids: list[str] | None = None
    answers: list[dict[str, Any]] | None = None


class ProjectMessage(BaseModel):
    id: int
    run_id: str | None = None
    role: str
    content: str
    payload: dict[str, Any] | None = None
    created_at: str


def create_app(
    *,
    workspace_root: Path,
    default_flow: str,
    model_name: str,
    model_factory: Callable[[str], Any],
    recursion_limit: int = 200,
    database_url: str | None = None,
    redis_url: str | None = None,
    store: PersistentRunStore | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    store = store or PersistentRunStore(database_url=database_url, redis_url=redis_url)
    checkpointer = create_api_checkpointer(database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store._loop = asyncio.get_running_loop()
        for run in store.list_active_runs(recoverable_only=True):
            start_agent_worker(
                store=store,
                request_id=run["request_id"],
                workspace_root=workspace_root,
                flow=run["flow"],
                model_name=model_name,
                model_factory=model_factory,
                recursion_limit=recursion_limit,
                checkpointer=checkpointer,
                resume_from_checkpoint=run["status"] == "running",
            )
        try:
            yield
        finally:
            close_api_checkpointer(checkpointer)

    app = FastAPI(title="Botmother Flow Agent API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or default_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.run_store = store
    app.state.agent_checkpointer = checkpointer

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/projects/{project_id}/runs", response_model=RunCreateResponse)
    async def create_run(
        project_id: str,
        request: RunCreateRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> RunCreateResponse:
        store._loop = asyncio.get_running_loop()
        flow = request.flow or default_flow
        record = store.create(
            project_id=str(project_id),
            user_id=user.user_id,
            prompt=request.prompt,
            flow=flow,
            thread_id=request.thread_id,
        )
        start_agent_worker(
            store=store,
            request_id=record.request_id,
            workspace_root=workspace_root,
            flow=flow,
            model_name=model_name,
            model_factory=model_factory,
            recursion_limit=recursion_limit,
            checkpointer=app.state.agent_checkpointer,
        )
        return RunCreateResponse(
            request_id=record.request_id,
            status="queued",
        )

    @app.get("/projects/{project_id}/runs/active", response_model=list[ActiveRunResponse])
    async def list_active_runs(
        project_id: str,
        include_events: bool = False,
        event_limit: int = 50,
        user: CurrentUser = Depends(get_current_user),
    ) -> list[dict[str, Any]]:
        return store.list_active_runs(
            project_id=project_id,
            user_id=user.user_id,
            include_events=include_events,
            event_limit=event_limit,
        )

    @app.get("/projects/{project_id}/runs/{request_id}", response_model=RunStatusResponse)
    async def get_run(
        project_id: str,
        request_id: str,
        user: CurrentUser = Depends(get_current_user),
    ) -> RunStatusResponse:
        record = require_record(store, request_id, project_id=project_id, user_id=user.user_id)
        return RunStatusResponse(
            request_id=record.request_id,
            status=record.status,
            final_response=record.final_response,
            error=record.error,
            pending_clarification=record.pending_clarification,
            current_plan=record.current_plan,
        )

    @app.get("/projects/{project_id}/runs/{request_id}/events")
    async def stream_events(
        project_id: str,
        request_id: str,
        user: CurrentUser = Depends(get_current_user),
    ) -> StreamingResponse:
        require_record(store, request_id, project_id=project_id, user_id=user.user_id)
        return StreamingResponse(
            store.stream(request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/projects/{project_id}/runs/{request_id}/clarify")
    async def clarify_run(
        project_id: str,
        request_id: str,
        request: ClarifyRequest,
        user: CurrentUser = Depends(get_current_user),
    ) -> dict[str, str]:
        record = require_record(store, request_id, project_id=project_id, user_id=user.user_id)
        if record.status != "waiting_for_clarification":
            raise HTTPException(status_code=409, detail="Run is not waiting for clarification")
        if request.answers:
            answer_payload: Any = {"answers": request.answers}
        elif request.option_ids:
            answer_payload = {"answer": request.answer, "option_ids": request.option_ids}
        elif request.option_id:
            answer_payload = {"answer": request.answer, "option_id": request.option_id}
        else:
            answer_payload = request.answer
        store.store_clarification_answer(request_id, answer_payload)
        if store.is_active_record(request_id):
            record.clarification_queue.put(answer_payload)
        else:
            store.clear_pending_clarification(request_id)
            start_agent_worker(
                store=store,
                request_id=request_id,
                workspace_root=workspace_root,
                flow=record.flow,
                model_name=model_name,
                model_factory=model_factory,
                recursion_limit=recursion_limit,
                checkpointer=app.state.agent_checkpointer,
                resume_from_checkpoint=True,
                resume_answer=answer_payload,
            )
        return {"status": "resuming"}

    @app.get("/projects/{project_id}/messages", response_model=list[ProjectMessage])
    async def list_project_messages(
        project_id: str,
        limit: int = 50,
        offset: int = 0,
        user: CurrentUser = Depends(get_current_user),
    ) -> list[dict[str, Any]]:
        return store.list_messages(
            project_id=project_id,
            user_id=user.user_id,
            limit=max(1, min(limit, 200)),
            offset=max(0, offset),
        )

    return app


def create_api_checkpointer(database_url: str | None) -> Any | None:
    url = database_url or os.getenv("AGENT_DATABASE_URL")
    if not url or not url.startswith(("postgresql://", "postgresql+psycopg://")):
        return None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpointing requires `langgraph-checkpoint-postgres`. "
            "Install agent dependencies before running API with Postgres."
        ) from exc

    normalized_url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    manager = PostgresSaver.from_conn_string(normalized_url)
    checkpointer = manager.__enter__() if hasattr(manager, "__enter__") else manager
    try:
        setattr(checkpointer, "_bm_manager", manager)
    except Exception:
        pass
    setup = getattr(checkpointer, "setup", None)
    if callable(setup):
        setup()
    return checkpointer


def close_api_checkpointer(checkpointer: Any | None) -> None:
    manager = getattr(checkpointer, "_bm_manager", None)
    if manager is not None and hasattr(manager, "__exit__"):
        manager.__exit__(None, None, None)


def run_agent_request(
    store: PersistentRunStore,
    request_id: str,
    workspace_root: Path,
    flow: str,
    model_name: str,
    model_factory: Callable[[str], Any],
    recursion_limit: int,
    checkpointer: Any | None = None,
    *,
    resume_from_checkpoint: bool = False,
    resume_answer: Any | None = None,
) -> None:
    try:
        record = store.require(request_id)
        store.set_status(request_id, "running")
        if not resume_from_checkpoint:
            store.emit(
                request_id,
                "run_started",
                {"flow": flow, "thread_id": record.thread_id, "model": model_name},
            )

        registry = build_tool_registry(dsl_only=True)
        repo_catalog = build_repo_catalog(workspace_root, flow, "", dsl_only=True)
        ui = EventUI(lambda event_type, payload: store.emit(request_id, event_type, payload))
        runtime = AgentRuntime(
            workspace_root=workspace_root,
            target_flow_json=flow,
            target_dsl_path="",
            model=model_factory(model_name),
            registry=registry,
            ui=ui,
        )
        graph = build_agent_graph(runtime, checkpointer=checkpointer)
        state = initial_state(
            workspace_root=workspace_root,
            target_flow_json=flow,
            target_dsl_path="",
            repo_catalog=repo_catalog,
        )
        config = {
            "configurable": {"thread_id": record.thread_id},
            "recursion_limit": recursion_limit,
        }
        if resume_from_checkpoint:
            if not checkpoint_exists(graph, config):
                raise RuntimeError("Cannot resume run: LangGraph checkpoint not found.")
            command = Command(resume=resume_answer) if resume_answer is not None else None
            result = graph.invoke(command, config=config)
        else:
            result = graph.invoke(
                {**state, "messages": [HumanMessage(content=record.prompt)]},
                config=config,
            )
        while result.get("__interrupt__"):
            payload = interrupt_value(result["__interrupt__"][0])
            store.set_pending_clarification(request_id, payload)
            answer = record.clarification_queue.get()
            store.clear_pending_clarification(request_id)
            result = graph.invoke(Command(resume=answer), config=config)

        final_response = latest_assistant_text(result)
        if final_response:
            ui.assistant(final_response)
        store.complete(request_id, final_response)
    except Exception as exc:  # pragma: no cover - exercised through API integration tests
        store.fail(request_id, str(exc))


def start_agent_worker(
    *,
    store: PersistentRunStore,
    request_id: str,
    workspace_root: Path,
    flow: str,
    model_name: str,
    model_factory: Callable[[str], Any],
    recursion_limit: int,
    checkpointer: Any | None = None,
    resume_from_checkpoint: bool = False,
    resume_answer: Any | None = None,
) -> threading.Thread:
    worker = threading.Thread(
        target=run_agent_request,
        kwargs={
            "store": store,
            "request_id": request_id,
            "workspace_root": workspace_root,
            "flow": flow,
            "model_name": model_name,
            "model_factory": model_factory,
            "recursion_limit": recursion_limit,
            "checkpointer": checkpointer,
            "resume_from_checkpoint": resume_from_checkpoint,
            "resume_answer": resume_answer,
        },
        daemon=True,
    )
    worker.start()
    return worker


def checkpoint_exists(graph: Any, config: dict[str, Any]) -> bool:
    try:
        snapshot = graph.get_state(config)
    except Exception:
        return False
    return bool(
        getattr(snapshot, "values", None)
        or getattr(snapshot, "next", None)
        or getattr(snapshot, "tasks", None)
        or getattr(snapshot, "interrupts", None)
    )


def interrupt_value(interrupt_obj: object) -> dict[str, Any]:
    value = getattr(interrupt_obj, "value", interrupt_obj)
    return value if isinstance(value, dict) else {"question": str(value)}


def default_cors_origins() -> list[str]:
    raw = os.getenv("AGENT_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://127.0.0.1:3001",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


def require_record(
    store: PersistentRunStore,
    request_id: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
):
    record = store.get(request_id, project_id=project_id, user_id=user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return record
