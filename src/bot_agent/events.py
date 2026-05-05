from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any


TERMINAL_STATUSES = {"completed", "failed"}


@dataclass
class RunRecord:
    request_id: str
    prompt: str
    project_dir: str
    thread_id: str
    status: str = "queued"
    final_response: str | None = None
    error: str | None = None
    pending_clarification: dict[str, Any] | None = None
    current_plan: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    clarification_queue: queue.Queue[Any] = field(default_factory=queue.Queue)
    task: asyncio.Task[Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def format_sse(event: dict[str, Any]) -> str:
    return (
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


def enrich_event_payload(event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    enriched = dict(payload or {})
    enriched.setdefault("description", describe_event(event_type, enriched))
    return enriched


def describe_event(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "run_started":
        return "bot agent ishni boshladi."
    if event_type == "assistant_message":
        return "Agent javob yubordi."
    if event_type == "tool_call":
        return describe_tool_call(payload)
    if event_type == "tool_result":
        return "Tool natijasi qaytdi."
    if event_type == "clarification_required":
        return "Agent foydalanuvchidan aniqlashtirish so'radi."
    if event_type == "plan_created":
        items = payload.get("items")
        count = len(items) if isinstance(items, list) else 0
        return f"Ish rejasi yaratildi: {count} ta vazifa."
    if event_type == "plan_item_updated":
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        title = item.get("title") or item.get("id") or "vazifa"
        status = item.get("status")
        return f"Reja bandi yangilandi: {title}{f' -> {status}' if status else ''}."
    if event_type == "plan_completed":
        return "Ish rejasi yakunlandi."
    if event_type == "bot_project_written":
        files = payload.get("files")
        count = len(files) if isinstance(files, list) else 0
        return f"bot loyihasi yozildi: {count} ta fayl."
    if event_type == "bot_project_validated":
        return "bot loyihasi tekshiruvdan o'tdi."
    if event_type == "file_written":
        return "Fayl yozildi yoki yangilandi."
    if event_type == "directory_created":
        return "Papka yaratildi."
    if event_type == "command_ran":
        return "Buyruq ishga tushirildi."
    if event_type == "run_completed":
        return "Agent ishni yakunladi."
    if event_type == "run_failed":
        return "Agent ishida xatolik yuz berdi."
    if event_type == "heartbeat":
        return "Aloqa faol."
    return f"Event: {event_type}."


def describe_tool_call(payload: dict[str, Any]) -> str:
    name = str(payload.get("name") or "tool")
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    if name == "get_project_overview":
        return "Agent loyiha sozlamalarini ko'rmoqda."
    if name == "read_repo_file":
        path = args.get("path")
        return f"Agent repo faylini o'qimoqda{f': {path}' if path else ''}."
    if name == "search_repo":
        query = args.get("query")
        return f"Agent repo ichidan qidirmoqda{f': {query}' if query else ''}."
    if name == "request_clarification":
        count = len(args.get("questions", [])) if isinstance(args.get("questions"), list) else 1
        return f"Agent foydalanuvchidan {count} ta savolga aniqlik kiritishni so'ramoqda."
    if name == "create_task_plan":
        count = len(args.get("items", [])) if isinstance(args.get("items"), list) else 0
        return f"Agent ish rejasini tuzmoqda{f': {count} ta vazifa' if count else ''}."
    if name == "update_task_plan_item":
        item_id = args.get("item_id")
        status = args.get("status")
        return f"Agent reja bandini yangilamoqda{f': {item_id}' if item_id else ''}{f' -> {status}' if status else ''}."
    if name == "inspect_bot_project":
        return "Agent bot loyiha papkasini tekshirmoqda."
    if name == "list_files":
        path = args.get("path")
        return f"Agent fayllarni ko'rmoqda{f': {path}' if path else ''}."
    if name == "read_file":
        path = args.get("path")
        return f"Agent faylni o'qimoqda{f': {path}' if path else ''}."
    if name == "write_file":
        path = args.get("path")
        return f"Agent fayl yozmoqda{f': {path}' if path else ''}."
    if name == "replace_in_file":
        path = args.get("path")
        return f"Agent fayldagi matnni almashtirmoqda{f': {path}' if path else ''}."
    if name == "make_directory":
        path = args.get("path")
        return f"Agent papka yaratayapti{f': {path}' if path else ''}."
    if name == "run_command":
        command = args.get("command")
        return f"Agent buyruq ishga tushirmoqda{f': {command}' if command else ''}."
    if name == "validate_bot_project":
        return "Agent bot loyihasini go test va go vet bilan tekshirmoqda."
    return f"Tool chaqirildi: {name}."


def apply_record_event(record: RunRecord, event_type: str, payload: dict[str, Any]) -> None:
    if event_type == "plan_created":
        record.current_plan = {"items": [dict(item) for item in payload.get("items", [])]}
        return
    if event_type == "plan_item_updated":
        items = payload.get("items")
        if isinstance(items, list):
            record.current_plan = {"items": [dict(item) for item in items if isinstance(item, dict)]}
        return
    if event_type == "plan_completed":
        record.current_plan = {"items": [dict(item) for item in payload.get("items", [])]}
