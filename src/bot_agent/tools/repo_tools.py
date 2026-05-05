from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from bot_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult


def build_project_catalog(workspace_root: Path, project_dir: str) -> dict[str, Any]:
    return {
        "workspace_root": str(workspace_root),
        "default_project_dir": project_dir,
        "path_rule": (
            "Bot project tools resolve relative paths against default_project_dir. "
            "Use `.` for the bot project root and paths like `cmd/bot/main.go` for files inside it."
        ),
        "generated_files": [
            "go.mod",
            "cmd/bot/main.go",
            "internal/app/app.go",
            "internal/config/config.go",
            "internal/handlers/handler.go",
            "internal/state/store.go",
            "pkg/messages/messages.go",
            ".env.example",
            "README.md",
        ],
        "telegram_library": "github.com/go-telegram-bot-api/telegram-bot-api/v5",
    }


def build_repo_tools() -> list[AgentTool]:
    class NoArgs(BaseModel):
        pass

    class ReadFileArgs(BaseModel):
        path: str
        start_line: int | None = None
        end_line: int | None = None

    class SearchArgs(BaseModel):
        query: str
        max_results: int = Field(default=10, ge=1, le=50)

    class ClarificationOptionArgs(BaseModel):
        id: str
        label: str
        description: str | None = None

    class ClarificationQuestionArgs(BaseModel):
        id: str = Field(description="Stable short id for this question.")
        question: str = Field(description="Direct human-facing question.")
        details: str | None = None
        options: list[ClarificationOptionArgs] = Field(default_factory=list)
        selection_type: str = Field(default="single")
        allow_free_text: bool = True

        @model_validator(mode="after")
        def validate_selection_type(self) -> "ClarificationQuestionArgs":
            if self.selection_type not in {"single", "multiple"}:
                raise ValueError("selection_type must be `single` or `multiple`.")
            return self

    class ClarificationArgs(BaseModel):
        question: str | None = None
        details: str | None = None
        options: list[ClarificationOptionArgs] = Field(default_factory=list)
        selection_type: str = "single"
        allow_free_text: bool = True
        questions: list[ClarificationQuestionArgs] = Field(default_factory=list)

        @model_validator(mode="after")
        def validate_payload(self) -> "ClarificationArgs":
            if self.selection_type not in {"single", "multiple"}:
                raise ValueError("selection_type must be `single` or `multiple`.")
            if self.questions:
                return self
            if not self.question:
                raise ValueError("Provide either `question` or `questions`.")
            return self

    def get_project_overview(context: ToolContext, _: NoArgs) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=json.dumps(context.state.get("project_catalog", {}), ensure_ascii=False, indent=2)
        )

    def read_repo_file(context: ToolContext, args: ReadFileArgs) -> ToolExecutionResult:
        path = resolve_repo_path(context, args.path)
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max((args.start_line or 1) - 1, 0)
        end = args.end_line or len(lines)
        sliced = lines[start:end]
        numbered = [f"{idx + start + 1}: {line}" for idx, line in enumerate(sliced)]
        return ToolExecutionResult(content="\n".join(numbered))

    def search_repo(context: ToolContext, args: SearchArgs) -> ToolExecutionResult:
        completed = subprocess.run(
            ["rg", "-n", args.query, "."],
            cwd=context.workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        results = completed.stdout.strip().splitlines()[: args.max_results]
        if not results:
            return ToolExecutionResult(content="No matches found.")
        return ToolExecutionResult(content="\n".join(results))

    def request_clarification(_: ToolContext, args: ClarificationArgs) -> ToolExecutionResult:
        return ToolExecutionResult(
            content="Clarification requested.",
            interrupt_payload=args.model_dump(exclude_none=True),
        )

    return [
        AgentTool(
            name="get_project_overview",
            description="Return the configured bot generator project overview.",
            args_model=NoArgs,
            handler=get_project_overview,
        ),
        AgentTool(
            name="read_repo_file",
            description="Read a UTF-8 file from this repository with optional line bounds.",
            args_model=ReadFileArgs,
            handler=read_repo_file,
        ),
        AgentTool(
            name="search_repo",
            description="Search repository files with ripgrep.",
            args_model=SearchArgs,
            handler=search_repo,
        ),
        AgentTool(
            name="request_clarification",
            description="Ask the user one or more structured clarification questions.",
            args_model=ClarificationArgs,
            handler=request_clarification,
        ),
    ]


def resolve_repo_path(context: ToolContext, path: str | None) -> Path:
    if not path:
        raise FileNotFoundError("path is required")
    root = context.workspace_root.resolve()
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"path escapes workspace: {path}")
    return candidate
