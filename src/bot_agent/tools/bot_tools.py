from __future__ import annotations

import json
import os
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, Field

from bot_agent.tools.base import AgentTool, ToolContext, ToolExecutionResult


def build_bot_tools() -> list[AgentTool]:
    class NoArgs(BaseModel):
        pass

    class PathArgs(BaseModel):
        path: str = Field(description="Relative or absolute path inside the workspace.")

    class ReadFileArgs(BaseModel):
        path: str
        start_line: int | None = None
        end_line: int | None = None

    class WriteFileArgs(BaseModel):
        path: str
        content: str

    class ReplaceInFileArgs(BaseModel):
        path: str
        old: str
        new: str
        replace_all: bool = True

    class RunCommandArgs(BaseModel):
        command: str
        cwd: str | None = None

    class ScaffoldBotProjectArgs(BaseModel):
        overwrite: bool = Field(
            default=False,
            description="When true, replace existing default scaffold files with a fresh base version.",
        )

    def inspect_bot_project(context: ToolContext, _: NoArgs) -> ToolExecutionResult:
        project_dir = bot_project_dir(context)
        files = list_files_under(project_dir) if project_dir.exists() else []
        return ToolExecutionResult(
            content=json.dumps(
                {
                    "project_dir": display_path(project_dir, context.workspace_root),
                    "exists": project_dir.exists(),
                    "files": files,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def scaffold_bot_project(context: ToolContext, args: ScaffoldBotProjectArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        project_dir = bot_project_dir(context)
        project_dir.mkdir(parents=True, exist_ok=True)
        module_name = sanitize_go_module_name(project_dir.name or "bot")
        files = default_bot_project_files(module_name)
        written_files: list[str] = []
        skipped_files: list[str] = []
        for relative_path, content in files.items():
            target = project_dir / relative_path
            if target.exists() and not args.overwrite:
                skipped_files.append(display_path(target, context.workspace_root))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            payload = {
                "path": display_path(target, context.workspace_root),
                "bytes": len(content.encode("utf-8")),
            }
            emit_agent_event(context, "file_written", payload)
            written_files.append(payload["path"])
        emit_agent_event(
            context,
            "directory_created",
            {"path": display_path(project_dir, context.workspace_root)},
        )
        tidy_result = subprocess.run(
            "go mod tidy",
            cwd=project_dir,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=go_command_env(),
        )
        tidy_payload = {
            "command": "go mod tidy",
            "cwd": display_path(project_dir, context.workspace_root),
            "exit_code": tidy_result.returncode,
            "stdout": tidy_result.stdout.strip(),
            "stderr": tidy_result.stderr.strip(),
        }
        emit_agent_event(context, "command_ran", tidy_payload)
        payload = {
            "project_dir": display_path(project_dir, context.workspace_root),
            "module_name": module_name,
            "written_files": written_files,
            "skipped_files": skipped_files,
            "go_mod_tidy": tidy_payload,
        }
        return ToolExecutionResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            state_updates={"bot_files": written_files},
            is_error=tidy_result.returncode != 0,
        )

    def list_files(context: ToolContext, args: PathArgs) -> ToolExecutionResult:
        target = resolve_workspace_path(context, args.path)
        if not target.exists():
            return ToolExecutionResult(content=f"path does not exist: {args.path}", is_error=True)
        if target.is_file():
            return ToolExecutionResult(content=json.dumps({"path": args.path, "files": [target.name]}, ensure_ascii=False))
        return ToolExecutionResult(
            content=json.dumps(
                {"path": args.path, "files": list_files_under(target)},
                ensure_ascii=False,
                indent=2,
            )
        )

    def read_file(context: ToolContext, args: ReadFileArgs) -> ToolExecutionResult:
        target = resolve_workspace_path(context, args.path)
        if not target.exists() or not target.is_file():
            return ToolExecutionResult(content=f"file not found: {args.path}", is_error=True)
        lines = target.read_text(encoding="utf-8").splitlines()
        start = max((args.start_line or 1) - 1, 0)
        end = args.end_line or len(lines)
        numbered = [f"{idx + start + 1}: {line}" for idx, line in enumerate(lines[start:end])]
        return ToolExecutionResult(content="\n".join(numbered))

    def write_file(context: ToolContext, args: WriteFileArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        target = resolve_workspace_path(context, args.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")
        payload = {"path": display_path(target, context.workspace_root), "bytes": len(args.content.encode("utf-8"))}
        emit_agent_event(context, "file_written", payload)
        return ToolExecutionResult(content=json.dumps(payload, ensure_ascii=False, indent=2))

    def replace_in_file(context: ToolContext, args: ReplaceInFileArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        target = resolve_workspace_path(context, args.path)
        if not target.exists() or not target.is_file():
            return ToolExecutionResult(content=f"file not found: {args.path}", is_error=True)
        original = target.read_text(encoding="utf-8")
        if args.old not in original:
            return ToolExecutionResult(content="target text not found", is_error=True)
        updated = original.replace(args.old, args.new) if args.replace_all else original.replace(args.old, args.new, 1)
        target.write_text(updated, encoding="utf-8")
        payload = {"path": display_path(target, context.workspace_root), "replaced": "all" if args.replace_all else "first"}
        emit_agent_event(context, "file_written", payload)
        return ToolExecutionResult(content=json.dumps(payload, ensure_ascii=False, indent=2))

    def make_directory(context: ToolContext, args: PathArgs) -> ToolExecutionResult:
        blocked = require_task_plan(context)
        if blocked:
            return blocked
        target = resolve_workspace_path(context, args.path)
        target.mkdir(parents=True, exist_ok=True)
        payload = {"path": display_path(target, context.workspace_root)}
        emit_agent_event(context, "directory_created", payload)
        return ToolExecutionResult(content=json.dumps(payload, ensure_ascii=False, indent=2))

    def run_command(context: ToolContext, args: RunCommandArgs) -> ToolExecutionResult:
        cwd = resolve_workspace_path(context, args.cwd or context.state.get("project_dir") or ".")
        completed = subprocess.run(
            args.command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            env=go_command_env() if args.command.strip().startswith("go ") else None,
        )
        payload = {
            "command": args.command,
            "cwd": display_path(cwd, context.workspace_root),
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        emit_agent_event(context, "command_ran", payload)
        return ToolExecutionResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            is_error=completed.returncode != 0,
        )

    def validate_bot_project(context: ToolContext, _: NoArgs) -> ToolExecutionResult:
        project_dir = bot_project_dir(context)
        if not project_dir.exists():
            return ToolExecutionResult(content="bot project directory does not exist.", is_error=True)
        results = []
        for command in ("go test ./...", "go vet ./..."):
            completed = subprocess.run(
                command,
                cwd=project_dir,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                env=go_command_env(),
            )
            results.append(
                {
                    "command": command,
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
            if completed.returncode != 0:
                payload = {"ok": False, "results": results}
                emit_agent_event(context, "command_ran", results[-1])
                return ToolExecutionResult(
                    content=json.dumps(payload, ensure_ascii=False, indent=2),
                    state_updates={"bot_validation": payload},
                    is_error=True,
                )
        payload = {"ok": True, "results": results}
        emit_agent_event(context, "bot_project_validated", payload)
        return ToolExecutionResult(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            state_updates={"bot_validation": payload},
        )

    return [
        AgentTool(
            name="inspect_bot_project",
            description="Inspect the configured bot project directory and list existing files.",
            args_model=NoArgs,
            handler=inspect_bot_project,
        ),
        AgentTool(
            name="scaffold_bot_project",
            description=(
                "Create the default bot project scaffold with cmd, internal, and pkg packages before "
                "making custom edits."
            ),
            args_model=ScaffoldBotProjectArgs,
            handler=scaffold_bot_project,
        ),
        AgentTool(
            name="list_files",
            description="List files under a directory or show the file name when the path is a file.",
            args_model=PathArgs,
            handler=list_files,
        ),
        AgentTool(
            name="read_file",
            description="Read a UTF-8 file with optional line bounds.",
            args_model=ReadFileArgs,
            handler=read_file,
        ),
        AgentTool(
            name="write_file",
            description="Create or fully overwrite a UTF-8 file.",
            args_model=WriteFileArgs,
            handler=write_file,
        ),
        AgentTool(
            name="replace_in_file",
            description="Replace existing text inside a UTF-8 file.",
            args_model=ReplaceInFileArgs,
            handler=replace_in_file,
        ),
        AgentTool(
            name="make_directory",
            description="Create a directory path inside the workspace.",
            args_model=PathArgs,
            handler=make_directory,
        ),
        AgentTool(
            name="run_command",
            description="Run a shell command inside the workspace or a chosen subdirectory.",
            args_model=RunCommandArgs,
            handler=run_command,
        ),
        AgentTool(
            name="validate_bot_project",
            description="Run go test ./... and go vet ./... in the bot project directory.",
            args_model=NoArgs,
            handler=validate_bot_project,
        ),
    ]


def require_task_plan(context: ToolContext) -> ToolExecutionResult | None:
    if context.state.get("allow_unplanned_edits"):
        return None
    plan = context.state.get("current_plan")
    if isinstance(plan, dict) and isinstance(plan.get("items"), list) and plan["items"]:
        return None
    return ToolExecutionResult(
        content=(
            "Project write blocked: create_task_plan must be called before writing bot files. "
            "First inspect requirements, then create a task plan, then edit files."
        ),
        is_error=True,
    )


def default_bot_project_files(module_name: str) -> dict[str, str]:
    return {
        output_path: render_scaffold_template(
            template_path,
            module_name=module_name,
            title_name=title_case_name(module_name),
        )
        for output_path, template_path in scaffold_template_map().items()
    }


def scaffold_template_map() -> dict[str, str]:
    return {
        "go.mod": "go.mod.tmpl",
        ".env.example": ".env.example.tmpl",
        "README.md": "README.md.tmpl",
        "cmd/bot/main.go": "cmd/bot/main.go.tmpl",
        "internal/app/app.go": "internal/app/app.go.tmpl",
        "internal/config/config.go": "internal/config/config.go.tmpl",
        "internal/handlers/handler.go": "internal/handlers/handler.go.tmpl",
        "pkg/messages/messages.go": "pkg/messages/messages.go.tmpl",
        "pkg/messages/templates/start.tmpl": "pkg/messages/templates/start.tmpl",
        "pkg/messages/templates/help.tmpl": "pkg/messages/templates/help.tmpl",
        "pkg/messages/templates/unknown_command.tmpl": "pkg/messages/templates/unknown_command.tmpl",
    }


def render_scaffold_template(template_path: str, *, module_name: str, title_name: str) -> str:
    template_root = files("bot_agent").joinpath("templates", "bot_project")
    content = template_root.joinpath(template_path).read_text(encoding="utf-8")
    return (
        content.replace("{{module_name}}", module_name)
        .replace("{{title_name}}", title_name)
    )


def sanitize_go_module_name(raw: str) -> str:
    lowered = "".join(char.lower() if char.isalnum() else "_" for char in raw.strip())
    collapsed = "_".join(part for part in lowered.split("_") if part)
    return collapsed or "bot"


def title_case_name(raw: str) -> str:
    return " ".join(part.capitalize() for part in raw.replace("_", " ").split())


def bot_project_dir(context: ToolContext) -> Path:
    raw = context.state.get("bot_project_dir") or context.state.get("project_dir") or "generated_bots/bot"
    return resolve_workspace_path(context, str(raw))


def resolve_workspace_path(context: ToolContext, raw: str) -> Path:
    path = Path(str(raw))
    root = context.workspace_root.resolve()
    candidate = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"path escapes workspace: {raw}")
    return candidate


def list_files_under(path: Path) -> list[str]:
    return [
        str(item.relative_to(path))
        for item in sorted(path.rglob("*"))
        if item.is_file() and not ignored_generated_path(item)
    ]


def ignored_generated_path(path: Path) -> bool:
    return any(part in {".git", "vendor", "__pycache__"} for part in path.parts)


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def emit_agent_event(context: ToolContext, event_type: str, payload: dict[str, object]) -> None:
    emitter = getattr(context.ui, "agent_event", None)
    if callable(emitter):
        emitter(event_type, payload)


def go_command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GOCACHE", str(Path(tempfile.gettempdir()) / "bot-agent-go-build-cache"))
    return env
