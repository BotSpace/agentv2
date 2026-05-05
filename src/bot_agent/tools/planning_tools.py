from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from bot_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult


PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]


def build_planning_tools() -> list[AgentTool]:
    class PlanItemArgs(BaseModel):
        id: str = Field(description="Stable short snake_case id for the task item.")
        title: str = Field(description="Human-readable task title.")
        details: str | None = None

    class CreateTaskPlanArgs(BaseModel):
        items: list[PlanItemArgs] = Field(
            min_length=1,
            description=(
                "Ordered implementation plan. For new product bots, use 5-8 items covering "
                "scaffold, domain sections, inline buttons/callbacks, copy, README, and validation."
            ),
        )

    class UpdateTaskPlanItemArgs(BaseModel):
        item_id: str
        status: PlanStatus
        summary: str | None = None

    def create_task_plan(context: ToolContext, args: CreateTaskPlanArgs) -> ToolExecutionResult:
        plan = {
            "items": [
                {
                    **item.model_dump(exclude_none=True),
                    "status": "pending",
                }
                for item in args.items
            ]
        }
        emit_agent_event(context, "plan_created", plan)
        return ToolExecutionResult(
            content=json.dumps(plan, ensure_ascii=False),
            state_updates={"current_plan": plan},
        )

    def update_task_plan_item(context: ToolContext, args: UpdateTaskPlanItemArgs) -> ToolExecutionResult:
        plan = normalize_plan(context.state.get("current_plan"))
        if not plan["items"]:
            return ToolExecutionResult(
                content="No current task plan exists. Call create_task_plan first.",
                is_error=True,
            )
        item = next((entry for entry in plan["items"] if entry["id"] == args.item_id), None)
        if item is None:
            return ToolExecutionResult(content=f"Plan item not found: {args.item_id}", is_error=True)
        item["status"] = args.status
        if args.summary:
            item["summary"] = args.summary
        payload = {"item": item, "items": plan["items"]}
        emit_agent_event(context, "plan_item_updated", payload)
        if is_plan_completed(plan):
            emit_agent_event(context, "plan_completed", plan)
        return ToolExecutionResult(
            content=json.dumps(payload, ensure_ascii=False),
            state_updates={"current_plan": plan},
        )

    return [
        AgentTool(
            name="create_task_plan",
            description=(
                "Create an ordered task plan before writing or changing bot project files. "
                "For new product bots, avoid tiny plans; cover scaffold, features, UI buttons, copy, and validation."
            ),
            args_model=CreateTaskPlanArgs,
            handler=create_task_plan,
        ),
        AgentTool(
            name="update_task_plan_item",
            description="Update one task plan item status. Mark every item completed before final response.",
            args_model=UpdateTaskPlanItemArgs,
            handler=update_task_plan_item,
        ),
    ]


def normalize_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {"items": []}
    items = plan.get("items")
    if not isinstance(items, list):
        return {"items": []}
    return {"items": [dict(item) for item in items if isinstance(item, dict)]}


def is_plan_completed(plan: dict[str, Any] | None) -> bool:
    normalized = normalize_plan(plan)
    items = normalized["items"]
    return bool(items) and all(item.get("status") == "completed" for item in items)


def has_incomplete_plan(plan: dict[str, Any] | None) -> bool:
    normalized = normalize_plan(plan)
    items = normalized["items"]
    return bool(items) and any(item.get("status") != "completed" for item in items)


def emit_agent_event(context: ToolContext, event_type: str, payload: dict[str, Any]) -> None:
    emitter = getattr(context.ui, "agent_event", None)
    if callable(emitter):
        emitter(event_type, payload)
