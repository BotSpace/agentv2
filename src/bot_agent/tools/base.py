from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from bot_agent.ui import ConsoleUI


@dataclass
class ToolContext:
    workspace_root: Path
    state: dict[str, Any]
    ui: ConsoleUI


@dataclass
class ToolExecutionResult:
    content: str
    state_updates: dict[str, Any] = field(default_factory=dict)
    interrupt_payload: dict[str, Any] | None = None
    is_error: bool = False


@dataclass
class AgentTool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], ToolExecutionResult]

    def invoke(self, context: ToolContext, args: dict[str, Any]) -> ToolExecutionResult:
        parsed = self.args_model.model_validate(args)
        return self.handler(context, parsed)

    def to_langchain_tool(self) -> StructuredTool:
        def _stub(**_: Any) -> str:
            return "handled by bot_agent"

        return StructuredTool.from_function(
            func=_stub,
            name=self.name,
            description=self.description,
            args_schema=self.args_model,
        )


class ToolRegistry:
    def __init__(self, tools: list[AgentTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.langchain_tools = [tool.to_langchain_tool() for tool in tools]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def execute(self, name: str, context: ToolContext, args: dict[str, Any]) -> ToolExecutionResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecutionResult(content=f"Unknown tool: {name}", is_error=True)
        return tool.invoke(context, args)
