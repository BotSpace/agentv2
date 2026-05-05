from bot_agent.tools.base import ToolRegistry
from bot_agent.tools.bot_tools import build_bot_tools
from bot_agent.tools.planning_tools import build_planning_tools
from bot_agent.tools.repo_tools import build_project_catalog, build_repo_tools


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            *build_repo_tools(),
            *build_planning_tools(),
            *build_bot_tools(),
        ]
    )


__all__ = ["ToolRegistry", "build_project_catalog", "build_tool_registry"]
