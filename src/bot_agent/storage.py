from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import redis
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from bot_agent.events import (
    RunRecord,
    apply_record_event,
    enrich_event_payload,
    format_sse,
)


metadata = MetaData()
TERMINAL_STATUSES = {"completed", "failed"}
ACTIVE_STATUSES = {"queued", "running", "waiting_for_clarification"}
RECOVERABLE_STATUSES = {"queued", "running"}

agent_runs = Table(
    "agent_runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("project_id", String(128), nullable=False, index=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("prompt", Text, nullable=False),
    Column("project_dir", Text, nullable=False),
    Column("thread_id", String(128), nullable=False),
    Column("status", String(64), nullable=False),
    Column("final_response", Text, nullable=True),
    Column("error", Text, nullable=True),
    Column("pending_clarification", JSON, nullable=True),
    Column("current_plan", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

agent_events = Table(
    "agent_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("project_id", String(128), nullable=False, index=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("seq", Integer, nullable=False),
    Column("type", String(128), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("run_id", "seq", name="uq_agent_events_run_seq"),
)

agent_messages = Table(
    "agent_messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=True, index=True),
    Column("project_id", String(128), nullable=False, index=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("role", String(64), nullable=False),
    Column("content", Text, nullable=False),
    Column("payload", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def default_database_url() -> str:
    return os.getenv("AGENT_DATABASE_URL", "sqlite:///./agent_runs.db")


def default_redis_url() -> str | None:
    return os.getenv("AGENT_REDIS_URL", "redis://localhost:6379/0")


def format_clarification_request_message(payload: dict[str, Any]) -> str:
    questions = payload.get("questions")
    if isinstance(questions, list) and questions:
        lines = ["Agent quyidagi savollarni aniqlashtirish uchun so'radi:"]
        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue
            lines.extend(format_question_lines(question, prefix=f"{index}. "))
        return "\n".join(lines)
    lines = ["Agent aniqlashtirish uchun savol berdi:"]
    lines.extend(format_question_lines(payload))
    return "\n".join(lines)


def format_question_lines(question: dict[str, Any], *, prefix: str = "") -> list[str]:
    lines = [f"{prefix}{question.get('question') or question.get('id') or 'Savol'}"]
    details = question.get("details")
    if details:
        lines.append(f"   Izoh: {details}")
    options = question.get("options")
    if isinstance(options, list) and options:
        option_lines = []
        for option in options:
            if not isinstance(option, dict):
                continue
            label = option.get("label") or option.get("id")
            description = option.get("description")
            if not label:
                continue
            option_line = f"{label}"
            if description:
                option_line += f" - {description}"
            option_lines.append(option_line)
        if option_lines:
            lines.append("   Variantlar: " + "; ".join(option_lines))
    return lines


def format_clarification_answer_message(
    answer: Any,
    pending_clarification: dict[str, Any] | None,
) -> str:
    if isinstance(answer, dict) and isinstance(answer.get("answers"), list):
        return format_batched_clarification_answer(answer["answers"], pending_clarification)
    question_text = "Aniqlashtirish savoli"
    if isinstance(pending_clarification, dict):
        question_text = str(pending_clarification.get("question") or question_text)
    answer_text = format_single_answer(answer, pending_clarification)
    return f"Foydalanuvchi javob berdi:\n{question_text} — {answer_text}"


def format_batched_clarification_answer(
    answers: list[Any],
    pending_clarification: dict[str, Any] | None,
) -> str:
    questions = question_lookup(pending_clarification)
    lines = ["Foydalanuvchi quyidagi javoblarni berdi:"]
    for index, answer in enumerate(answers, start=1):
        if not isinstance(answer, dict):
            lines.append(f"{index}. {answer}")
            continue
        question_id = str(answer.get("question_id") or index)
        question = questions.get(question_id, {})
        question_text = question.get("question") or question_id
        answer_text = answer.get("answer")
        if not answer_text:
            answer_text = labels_for_answer_ids(answer, question) or "Javob berildi"
        lines.append(f"{index}. {question_text} — {answer_text}")
    return "\n".join(lines)


def format_single_answer(answer: Any, pending_clarification: dict[str, Any] | None) -> str:
    question = pending_clarification if isinstance(pending_clarification, dict) else {}
    if isinstance(answer, dict):
        explicit_answer = answer.get("answer")
        labels = labels_for_answer_ids(answer, question)
        if explicit_answer and labels and str(explicit_answer) != labels:
            return f"{explicit_answer} ({labels})"
        return str(explicit_answer or labels or answer)
    return str(answer)


def labels_for_answer_ids(answer: dict[str, Any], question: dict[str, Any]) -> str:
    options = question.get("options")
    if not isinstance(options, list):
        return ""
    labels_by_id = {
        str(option.get("id")): str(option.get("label") or option.get("id"))
        for option in options
        if isinstance(option, dict) and option.get("id") is not None
    }
    option_ids = answer.get("option_ids")
    if isinstance(option_ids, list):
        labels = [labels_by_id.get(str(option_id), str(option_id)) for option_id in option_ids]
        return ", ".join(labels)
    option_id = answer.get("option_id")
    if option_id is not None:
        return labels_by_id.get(str(option_id), str(option_id))
    return ""


def question_lookup(pending_clarification: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(pending_clarification, dict):
        return {}
    questions = pending_clarification.get("questions")
    if not isinstance(questions, list):
        return {}
    return {
        str(question.get("id")): question
        for question in questions
        if isinstance(question, dict) and question.get("id") is not None
    }


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url or default_database_url()
    kwargs: dict[str, Any] = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


class RedisBroker:
    def __init__(self, redis_url: str | None) -> None:
        self._client = None
        if redis_url:
            try:
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._client = client
            except redis.RedisError:
                self._client = None

    def publish_event(self, request_id: str, event: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            self._client.publish(f"agent:runs:{request_id}:events", json.dumps(event, ensure_ascii=False))
        except redis.RedisError:
            return


class PersistentRunStore:
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        database_url: str | None = None,
        redis_url: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.engine = engine or make_engine(database_url)
        metadata.create_all(self.engine)
        self._loop = loop
        self._lock = threading.Lock()
        self._active: dict[str, RunRecord] = {}
        self._redis = RedisBroker(redis_url if redis_url is not None else default_redis_url())

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    def create(
        self,
        *,
        project_id: str,
        user_id: str,
        prompt: str,
        project_dir: str,
        thread_id: str | None = None,
    ) -> RunRecord:
        request_id = str(uuid.uuid4())
        resolved_thread_id = thread_id or request_id
        timestamp = now_utc()
        with self.engine.begin() as conn:
            conn.execute(
                insert(agent_runs).values(
                    id=request_id,
                    project_id=project_id,
                    user_id=user_id,
                    prompt=prompt,
                    project_dir=project_dir,
                    thread_id=resolved_thread_id,
                    status="queued",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            conn.execute(
                insert(agent_messages).values(
                    run_id=request_id,
                    project_id=project_id,
                    user_id=user_id,
                    role="user",
                    content=prompt,
                    payload={"prompt": prompt},
                    created_at=timestamp,
                )
            )
        record = RunRecord(
            request_id=request_id,
            prompt=prompt,
            project_dir=project_dir,
            thread_id=resolved_thread_id,
        )
        record.project_id = project_id  # type: ignore[attr-defined]
        record.user_id = user_id  # type: ignore[attr-defined]
        with self._lock:
            self._active[request_id] = record
        return record

    def get(self, request_id: str, *, project_id: str | None = None, user_id: str | None = None) -> RunRecord | None:
        with self._lock:
            active = self._active.get(request_id)
        if active and matches_scope(active, project_id, user_id):
            return active
        row = self._run_row(request_id, project_id=project_id, user_id=user_id)
        if row is None:
            return None
        return self._record_from_row(row)

    def require(self, request_id: str) -> RunRecord:
        record = self.get(request_id)
        if record is None:
            raise KeyError(request_id)
        return record

    def is_active_record(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._active

    def deactivate(self, request_id: str) -> None:
        with self._lock:
            self._active.pop(request_id, None)

    def emit(self, request_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        record = self.require(request_id)
        event_payload = enrich_event_payload(event_type, payload)
        with record.lock:
            event = self._append_event(record, event_type, event_payload)
            record.events.append(event)
            apply_record_event(record, event_type, event["payload"])
            self._persist_event_side_effects(record, event_type, event_payload)
            subscribers = list(record.subscribers)
        self._redis.publish_event(request_id, event)
        for subscriber in subscribers:
            self.loop.call_soon_threadsafe(subscriber.put_nowait, event)
        return event

    def set_status(self, request_id: str, status: str) -> None:
        record = self.require(request_id)
        with record.lock:
            record.status = status
        self._update_run(request_id, status=status)

    def complete(self, request_id: str, final_response: str) -> None:
        record = self.require(request_id)
        with record.lock:
            record.status = "completed"
            record.final_response = final_response
            record.pending_clarification = None
        self._update_run(
            request_id,
            status="completed",
            final_response=final_response,
            pending_clarification=None,
        )
        self.emit(request_id, "run_completed", {"final_response": final_response})
        self.deactivate(request_id)

    def fail(self, request_id: str, error: str) -> None:
        record = self.require(request_id)
        with record.lock:
            record.status = "failed"
            record.error = error
            record.pending_clarification = None
        self._update_run(request_id, status="failed", error=error, pending_clarification=None)
        self.emit(request_id, "run_failed", {"error": error})
        self.deactivate(request_id)

    def set_pending_clarification(self, request_id: str, payload: dict[str, Any]) -> None:
        record = self.require(request_id)
        with record.lock:
            record.status = "waiting_for_clarification"
            record.pending_clarification = payload
        self._update_run(
            request_id,
            status="waiting_for_clarification",
            pending_clarification=payload,
        )
        self.emit(request_id, "clarification_required", payload)

    def clear_pending_clarification(self, request_id: str) -> None:
        record = self.require(request_id)
        with record.lock:
            record.pending_clarification = None
            record.status = "running"
        self._update_run(request_id, status="running", pending_clarification=None)

    def store_clarification_answer(self, request_id: str, answer: Any) -> None:
        record = self.require(request_id)
        content = format_clarification_answer_message(answer, record.pending_clarification)
        with self.engine.begin() as conn:
            conn.execute(
                insert(agent_messages).values(
                    run_id=request_id,
                    project_id=getattr(record, "project_id"),
                    user_id=getattr(record, "user_id"),
                    role="user",
                    content=str(content),
                    payload={"clarification_answer": answer},
                    created_at=now_utc(),
                )
            )

    async def stream(self, request_id: str) -> AsyncIterator[str]:
        record = self.require(request_id)
        subscriber: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with record.lock:
            record.subscribers.append(subscriber)
        next_index = 0
        try:
            while True:
                buffered = self.list_events(request_id, after_seq=next_index)
                if buffered:
                    next_index = max(event["seq"] for event in buffered)
                for event in buffered:
                    yield format_sse(event)
                terminal = self.is_terminal(request_id)
                if terminal:
                    break
                try:
                    await asyncio.wait_for(subscriber.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield format_sse(
                        {
                            "request_id": request_id,
                            "seq": next_index + 1,
                            "type": "heartbeat",
                            "payload": {},
                        }
                    )
        finally:
            with record.lock:
                if subscriber in record.subscribers:
                    record.subscribers.remove(subscriber)

    def list_events(self, request_id: str, *, after_seq: int = 0) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(agent_events)
                .where(and_(agent_events.c.run_id == request_id, agent_events.c.seq > after_seq))
                .order_by(agent_events.c.seq)
            ).mappings()
            return [
                event_from_row(row)
                for row in rows
            ]

    def list_recent_events(self, request_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        capped_limit = max(1, min(limit, 200))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(agent_events)
                .where(agent_events.c.run_id == request_id)
                .order_by(agent_events.c.seq.desc())
                .limit(capped_limit)
            ).mappings()
            events = [
                event_from_row(row)
                for row in rows
            ]
        return list(reversed(events))

    def list_active_runs(
        self,
        *,
        project_id: str | None = None,
        user_id: str | None = None,
        recoverable_only: bool = False,
        include_events: bool = False,
        event_limit: int = 50,
    ) -> list[dict[str, Any]]:
        statuses = RECOVERABLE_STATUSES if recoverable_only else ACTIVE_STATUSES
        clauses = [agent_runs.c.status.in_(sorted(statuses))]
        if project_id is not None:
            clauses.append(agent_runs.c.project_id == project_id)
        if user_id is not None:
            clauses.append(agent_runs.c.user_id == user_id)
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(agent_runs)
                .where(and_(*clauses))
                .order_by(agent_runs.c.updated_at.desc(), agent_runs.c.created_at.desc())
            ).mappings()
            runs = []
            for row in rows:
                last_event_seq = conn.execute(
                    select(func.coalesce(func.max(agent_events.c.seq), 0)).where(
                        agent_events.c.run_id == row["id"]
                    )
                ).scalar_one()
                runs.append(active_run_summary(row, int(last_event_seq or 0)))
        if include_events:
            for run in runs:
                run["events"] = self.list_recent_events(run["request_id"], limit=event_limit)
        return runs

    def list_messages(
        self,
        *,
        project_id: str,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(agent_messages)
                .where(
                    and_(
                        agent_messages.c.project_id == project_id,
                        agent_messages.c.user_id == user_id,
                    )
                )
                .order_by(agent_messages.c.created_at, agent_messages.c.id)
                .limit(limit)
                .offset(offset)
            ).mappings()
            return [
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "payload": row["payload"],
                    "created_at": row["created_at"].isoformat(),
                }
                for row in rows
            ]

    def is_terminal(self, request_id: str) -> bool:
        row = self._run_row(request_id)
        return row is not None and row["status"] in {"completed", "failed"}

    def _append_event(self, record: RunRecord, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.engine.begin() as conn:
            current_seq = conn.execute(
                select(func.max(agent_events.c.seq)).where(agent_events.c.run_id == record.request_id)
            ).scalar_one_or_none()
            seq = int(current_seq or 0) + 1
            conn.execute(
                insert(agent_events).values(
                    run_id=record.request_id,
                    project_id=getattr(record, "project_id"),
                    user_id=getattr(record, "user_id"),
                    seq=seq,
                    type=event_type,
                    payload=payload,
                    created_at=now_utc(),
                )
            )
        return {
            "request_id": record.request_id,
            "seq": seq,
            "type": event_type,
            "description": payload.get("description"),
            "payload": payload,
        }

    def _persist_event_side_effects(self, record: RunRecord, event_type: str, payload: dict[str, Any]) -> None:
        updates: dict[str, Any] = {}
        if event_type in {"plan_created", "plan_item_updated", "plan_completed"}:
            updates["current_plan"] = record.current_plan
        if event_type == "assistant_message":
            content = str(payload.get("content", ""))
            self._insert_message(record, "assistant", content, payload)
        if event_type == "clarification_required":
            self._insert_message(
                record,
                "assistant",
                format_clarification_request_message(payload),
                payload,
            )
        if updates:
            self._update_run(record.request_id, **updates)

    def _insert_message(self, record: RunRecord, role: str, content: str, payload: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            timestamp = now_utc()
            conn.execute(
                insert(agent_messages).values(
                    run_id=record.request_id,
                    project_id=getattr(record, "project_id"),
                    user_id=getattr(record, "user_id"),
                    role=role,
                    content=content,
                    payload=payload,
                    created_at=timestamp,
                )
            )

    def _update_run(self, request_id: str, **values: Any) -> None:
        values["updated_at"] = now_utc()
        with self.engine.begin() as conn:
            conn.execute(update(agent_runs).where(agent_runs.c.id == request_id).values(**values))

    def _run_row(self, request_id: str, *, project_id: str | None = None, user_id: str | None = None):
        clauses = [agent_runs.c.id == request_id]
        if project_id is not None:
            clauses.append(agent_runs.c.project_id == project_id)
        if user_id is not None:
            clauses.append(agent_runs.c.user_id == user_id)
        with self.engine.begin() as conn:
            return conn.execute(select(agent_runs).where(and_(*clauses))).mappings().first()

    def _record_from_row(self, row) -> RunRecord:
        record = RunRecord(
            request_id=row["id"],
            prompt=row["prompt"],
            project_dir=row["project_dir"],
            thread_id=row["thread_id"],
            status=row["status"],
            final_response=row["final_response"],
            error=row["error"],
            pending_clarification=row["pending_clarification"],
            current_plan=row["current_plan"],
        )
        record.project_id = row["project_id"]  # type: ignore[attr-defined]
        record.user_id = row["user_id"]  # type: ignore[attr-defined]
        with self._lock:
            active = self._active.get(row["id"])
            if active:
                record.clarification_queue = active.clarification_queue
                record.subscribers = active.subscribers
                self._active[row["id"]] = record
        return record


def active_run_summary(row: Any, last_event_seq: int) -> dict[str, Any]:
    return {
        "request_id": row["id"],
        "status": row["status"],
        "prompt": row["prompt"],
        "project_dir": row["project_dir"],
        "thread_id": row["thread_id"],
        "pending_clarification": row["pending_clarification"],
        "current_plan": row["current_plan"],
        "last_event_seq": last_event_seq,
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def event_from_row(row: Any) -> dict[str, Any]:
    payload = enrich_event_payload(row["type"], row["payload"] or {})
    return {
        "request_id": row["run_id"],
        "seq": row["seq"],
        "type": row["type"],
        "description": payload.get("description"),
        "payload": payload,
    }


def matches_scope(record: RunRecord, project_id: str | None, user_id: str | None) -> bool:
    if project_id is not None and getattr(record, "project_id", None) != project_id:
        return False
    if user_id is not None and getattr(record, "user_id", None) != user_id:
        return False
    return True
