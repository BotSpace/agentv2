from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from bm_flow_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult


PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]


def build_planning_tools() -> list[AgentTool]:
    class PlanItemArgs(BaseModel):
        id: str = Field(description="Stable short snake_case id for the task item.")
        title: str = Field(description="Human-readable task title.")
        details: str | None = Field(default=None, description="Optional extra task context.")

    class CreateTaskPlanArgs(BaseModel):
        items: list[PlanItemArgs] = Field(
            min_length=1,
            description="Ordered list of task items to complete before final response.",
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
        emit_plan_event(context, "plan_created", plan)
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
            return ToolExecutionResult(
                content=f"Plan item not found: {args.item_id}",
                is_error=True,
            )
        item["status"] = args.status
        if args.summary:
            item["summary"] = args.summary
        payload = {"item": item, "items": plan["items"]}
        emit_plan_event(context, "plan_item_updated", payload)
        state_updates = {"current_plan": plan}
        if is_plan_completed(plan):
            save_result = save_completed_plan_flow(context)
            if save_result.is_error:
                item["status"] = "blocked"
                if not item.get("summary"):
                    item["summary"] = save_result.content
                emit_plan_event(context, "plan_item_updated", {"item": item, "items": plan["items"]})
                return ToolExecutionResult(
                    content=save_result.content,
                    state_updates={"current_plan": plan, **save_result.state_updates},
                    is_error=True,
                )
            state_updates.update(save_result.state_updates)
            emit_plan_event(context, "plan_completed", plan)
        return ToolExecutionResult(
            content=json.dumps(payload, ensure_ascii=False),
            state_updates=state_updates,
        )

    return [
        AgentTool(
            name="create_task_plan",
            description=(
                "Create an ordered task plan before making flow changes or doing multi-step work. "
                "Each item needs {id, title, details?}. The plan is shown to the user and API events."
            ),
            args_model=CreateTaskPlanArgs,
            handler=create_task_plan,
        ),
        AgentTool(
            name="update_task_plan_item",
            description=(
                "Update one task plan item status as work progresses. "
                "Use statuses pending, in_progress, completed, or blocked. "
                "Mark every item completed before final response."
            ),
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


def emit_plan_event(context: ToolContext, event_type: str, payload: dict[str, Any]) -> None:
    emitter = getattr(context.ui, "flow_event", None)
    if callable(emitter):
        emitter(event_type, payload)


def save_completed_plan_flow(context: ToolContext) -> ToolExecutionResult:
    from bm_flow_agent.tools.dsl_tools import strict_save_working_dsl

    if not context.state.get("working_dsl"):
        return ToolExecutionResult(content="No draft flow changes to save.")
    return strict_save_working_dsl(context)
