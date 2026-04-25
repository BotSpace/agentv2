from bm_flow_agent.tools.base import ToolRegistry
from bm_flow_agent.tools.dsl_tools import build_yaml_flow_tools
from bm_flow_agent.tools.planning_tools import build_planning_tools
from bm_flow_agent.tools.repo_tools import build_repo_catalog, build_repo_tools


def build_tool_registry(*, dsl_only: bool = False) -> ToolRegistry:
    return ToolRegistry(
        [
            *build_repo_tools(),
            *build_planning_tools(),
            *build_yaml_flow_tools(),
        ]
    )


__all__ = ["ToolRegistry", "build_repo_catalog", "build_tool_registry"]
