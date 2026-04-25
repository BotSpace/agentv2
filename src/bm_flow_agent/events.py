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
    flow: str
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
        return "Iltimos kuting, agent ishni boshladi."
    if event_type == "assistant_message":
        return "Agent javob yubordi."
    if event_type == "tool_call":
        return describe_tool_call(payload)
    if event_type == "tool_result":
        return "Tool natijasi qaytdi."
    if event_type == "node_upserted":
        step = payload.get("step") if isinstance(payload.get("step"), dict) else {}
        node = payload.get("node") if isinstance(payload.get("node"), dict) else {}
        node_id = step.get("id") or node.get("id") or "noma'lum"
        kind = step.get("kind") or node.get("type")
        return f"Node qo'shildi yoki yangilandi: {node_id}{f' ({kind})' if kind else ''}."
    if event_type == "node_patched":
        step_id = payload.get("step_id", "noma'lum")
        return f"Node qisman yangilandi: {step_id}."
    if event_type == "node_removed":
        step_id = payload.get("step_id", "noma'lum")
        routes = payload.get("routes")
        route_count = len(routes) if isinstance(routes, list) else 0
        suffix = f" U bilan bog'liq {route_count} ta route ham olib tashlandi." if route_count else ""
        return f"Node olib tashlandi: {step_id}.{suffix}"
    if event_type == "edge_upserted":
        button = payload.get("button") if isinstance(payload.get("button"), dict) else {}
        if button:
            source = button.get("from") or "noma'lum"
            target = button.get("to") or "noma'lum"
            button_text = button.get("button_text") or "noma'lum"
            return f'Button edge ulandi: {source} / "{button_text}" -> {target}.'
        route = payload.get("route") if isinstance(payload.get("route"), dict) else {}
        edge = payload.get("edge") if isinstance(payload.get("edge"), dict) else {}
        source = route.get("from") or edge.get("source") or "noma'lum"
        target = route.get("to") or edge.get("target") or "noma'lum"
        handle = route.get("on") or route.get("source_handle") or edge.get("sourceHandle")
        return f"Edge ulandi: {source} -> {target}{f' ({handle})' if handle else ''}."
    if event_type == "flow_replaced":
        before = payload.get("before") if isinstance(payload.get("before"), dict) else {}
        after = payload.get("after") if isinstance(payload.get("after"), dict) else {}
        return (
            "Flow to'liq almashtirildi: "
            f"{before.get('nodes', 0)} ta node -> {after.get('nodes', 0)} ta node, "
            f"{before.get('edges', 0)} ta edge -> {after.get('edges', 0)} ta edge."
        )
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
    if name == "get_engine_overview":
        return "Agent engine imkoniyatlarini o'rganmoqda."
    if name == "list_supported_nodes":
        return "Agent qo'llab-quvvatlanadigan node turlarini tekshirmoqda."
    if name == "describe_step_kind":
        kinds = args.get("kinds") or args.get("kind")
        return f"Agent node maydonlarini o'rganmoqda: {format_value(kinds)}."
    if name == "search_repo_docs":
        query = args.get("query") or args.get("pattern")
        return f"Agent loyiha hujjatlaridan qidirmoqda{f': {query}' if query else ''}."
    if name == "read_repo_file":
        path = args.get("path")
        return f"Agent repo faylini o'qimoqda{f': {path}' if path else ''}."
    if name == "request_clarification":
        count = len(args.get("questions", [])) if isinstance(args.get("questions"), list) else 1
        return f"Agent foydalanuvchidan {count} ta savolga aniqlik kiritishni so'ramoqda."
    if name == "analyze_flow_connectivity":
        return "Agent action node'lar triggerlardan boshlanishini tekshirmoqda."
    if name == "create_task_plan":
        count = len(args.get("items", [])) if isinstance(args.get("items"), list) else 0
        return f"Agent ish rejasini tuzmoqda{f': {count} ta vazifa' if count else ''}."
    if name == "update_task_plan_item":
        item_id = args.get("item_id")
        status = args.get("status")
        return f"Agent reja bandini yangilamoqda{f': {item_id}' if item_id else ''}{f' -> {status}' if status else ''}."
    if name == "get_flow_yaml":
        return "Agent hozirgi flowni YAML ko'rinishida o'qimoqda."
    if name == "upsert_step":
        step = args.get("step") if isinstance(args.get("step"), dict) else {}
        step_id = step.get("id")
        kind = step.get("kind")
        incoming = args.get("incoming") if isinstance(args.get("incoming"), dict) else {}
        source = incoming.get("from")
        suffix = f", {source} dan ulanadi" if source else ""
        return f"Agent node qo'shmoqda yoki yangilamoqda{format_id_kind(step_id, kind)}{suffix}."
    if name == "remove_step":
        step_id = args.get("step_id")
        return f"Agent node olib tashlamoqda{f': {step_id}' if step_id else ''}."
    if name == "connect_steps":
        routes = args.get("routes")
        if isinstance(routes, list) and routes:
            return f"Agent {len(routes)} ta edgeni ulayapti."
        source = args.get("source")
        target = args.get("target")
        return f"Agent edgeni ulayapti{f': {source} -> {target}' if source and target else ''}."
    if name == "patch_step_block":
        step_id = args.get("step_id")
        return f"Agent node ma'lumotini qisman yangilamoqda{f': {step_id}' if step_id else ''}."
    if name == "save_flow_yaml":
        return "Agent butun YAML flowni saqlamoqda."
    return f"Tool chaqirildi: {name}."


def format_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value) if value is not None else "noma'lum"


def format_id_kind(step_id: Any, kind: Any) -> str:
    if step_id and kind:
        return f": {step_id} ({kind})"
    if step_id:
        return f": {step_id}"
    if kind:
        return f": {kind}"
    return ""


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
