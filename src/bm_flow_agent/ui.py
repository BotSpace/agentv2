from __future__ import annotations

import json
import sys
import termios
import tty
from typing import Any, Callable

from termcolor import colored


class ConsoleUI:
    def info(self, message: str) -> None:
        print(colored(message, "cyan"))

    def status(self, message: str) -> None:
        print(colored(message, "blue"))

    def success(self, message: str) -> None:
        print(colored(message, "green"))

    def warning(self, message: str) -> None:
        print(colored(message, "yellow"))

    def error(self, message: str) -> None:
        print(colored(message, "red"))

    def assistant(self, message: str) -> None:
        print(f"{colored('Agent', 'green', attrs=['bold'])}: {message}")

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True)
        print(f"{colored('tool', 'magenta')}: {name}({payload})")

    def flow_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "plan_created":
            print(colored("Plan:", "cyan", attrs=["bold"]))
            for index, item in enumerate(payload.get("items", []), start=1):
                print(colored(f"  {index}. [{item.get('status', 'pending')}] {item.get('title', item.get('id'))}", "cyan"))
        elif event_type == "plan_item_updated":
            item = payload.get("item", {})
            print(colored(f"Plan item: [{item.get('status')}] {item.get('title', item.get('id'))}", "cyan"))
        elif event_type == "plan_completed":
            print(colored("Plan completed.", "green"))

    def clarification(self, payload: dict[str, Any]) -> None:
        question = payload.get("question", "I need more detail.")
        details = payload.get("details")
        print(colored(f"Clarification: {question}", "yellow", attrs=["bold"]))
        if details:
            print(colored(details, "yellow"))

    def prompt(self, label: str = "You") -> str:
        return input(f"{colored(label, 'blue', attrs=['bold'])}: ")

    def select_clarification(self, payload: dict[str, Any]) -> str | dict[str, Any]:
        questions = payload.get("questions") or []
        if isinstance(questions, list) and questions:
            return self._select_question_batch(questions)
        options = payload.get("options") or []
        if not isinstance(options, list) or not options:
            return self.prompt("You").strip()
        if sys.stdin.isatty() and sys.stdout.isatty():
            return self._interactive_select(payload, options, selection_type=str(payload.get("selection_type", "single")))
        return self._numbered_select(payload, options)

    def _select_question_batch(self, questions: list[Any]) -> dict[str, Any]:
        answers = []
        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue
            print(colored(f"\nSavol {index}: {question.get('question', question.get('id', ''))}", "yellow", attrs=["bold"]))
            details = question.get("details")
            if details:
                print(colored(str(details), "yellow"))
            options = question.get("options") or []
            if sys.stdin.isatty() and sys.stdout.isatty() and isinstance(options, list) and options:
                answer = self._interactive_select(
                    question,
                    options,
                    selection_type=str(question.get("selection_type", "single")),
                )
            elif isinstance(options, list) and options:
                answer = self._numbered_select(question, options)
            else:
                answer = self.prompt("Javob").strip()
            answers.append({"question_id": str(question.get("id", index)), **normalize_answer_payload(answer)})
        return {"answers": answers}

    def _numbered_select(self, payload: dict[str, Any], options: list[Any]) -> str | dict[str, Any]:
        for index, option in enumerate(options, start=1):
            if not isinstance(option, dict):
                continue
            label = option.get("label", option.get("id", str(index)))
            description = option.get("description")
            line = f"  {index}. {label}"
            if description:
                line += f" - {description}"
            print(colored(line, "yellow"))
        multiple = payload.get("selection_type") == "multiple"
        prompt = "Tanlang"
        if multiple:
            prompt += " (masalan: 1,3,4)"
        if payload.get("allow_free_text", True):
            prompt += " yoki javob yozing"
        raw_answer = input(f"{colored(prompt, 'blue', attrs=['bold'])}: ").strip()
        return map_clarification_selection(raw_answer, options, multiple=multiple)

    def _interactive_select(
        self,
        payload: dict[str, Any],
        options: list[Any],
        *,
        selection_type: str = "single",
    ) -> str | dict[str, Any]:
        choices = [option for option in options if isinstance(option, dict)]
        custom_index = len(choices) if payload.get("allow_free_text", True) else None
        multiple = selection_type == "multiple"
        checked: set[int] = set()
        selected = 0
        line_count = 0
        if multiple:
            print(colored("↑/↓ yurish, Space tanlash, Enter yakunlash. Raqamlar ham ishlaydi.", "cyan"))
        else:
            print(colored("↑/↓ bilan tanlang, Enter bosing. Raqam ham ishlaydi.", "cyan"))
        while True:
            if line_count:
                sys.stdout.write(f"\x1b[{line_count}F")
            rendered = []
            for index, option in enumerate(choices):
                pointer = "❯" if index == selected else " "
                if multiple:
                    marker = "☑" if index in checked else "☐"
                else:
                    marker = "●" if index == selected else "○"
                label = option.get("label", option.get("id", str(index + 1)))
                description = option.get("description")
                line = f"{pointer} {marker} {index + 1}. {label}"
                if description:
                    line += f" - {description}"
                rendered.append(colored(line, "green" if index == selected else "yellow", attrs=["bold"] if index == selected else None))
            if custom_index is not None:
                pointer = "❯" if selected == custom_index else " "
                marker = "●" if selected == custom_index else "○"
                rendered.append(colored(f"{pointer} {marker} Boshqa javob yozish", "green" if selected == custom_index else "yellow", attrs=["bold"] if selected == custom_index else None))
            for line in rendered:
                sys.stdout.write("\x1b[2K" + line + "\n")
            sys.stdout.flush()
            line_count = len(rendered)
            key = read_key()
            max_index = len(choices) if custom_index is not None else len(choices) - 1
            if key in {"up", "k"}:
                selected = max_index if selected <= 0 else selected - 1
            elif key in {"down", "j"}:
                selected = 0 if selected >= max_index else selected + 1
            elif key.isdigit() and 1 <= int(key) <= len(choices):
                selected = int(key) - 1
                if multiple:
                    toggle_checked(checked, selected)
                    continue
                sys.stdout.write("\n")
                return option_answer(choices[selected])
            elif key == " " and multiple and selected < len(choices):
                toggle_checked(checked, selected)
            elif key in {"enter", "\n", "\r"}:
                sys.stdout.write("\n")
                if multiple:
                    if custom_index is not None and selected == custom_index and not checked:
                        return self.prompt("Javob").strip()
                    if not checked and selected < len(choices):
                        checked.add(selected)
                    return multiple_option_answer([choices[index] for index in sorted(checked)])
                if custom_index is not None and selected == custom_index:
                    return self.prompt("Javob").strip()
                return option_answer(choices[selected])
            elif key in {"c", "C"} and custom_index is not None:
                sys.stdout.write("\n")
                return self.prompt("Javob").strip()


class EventUI:
    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self._emit = emit

    def info(self, message: str) -> None:
        self._emit("tool_result", {"message": message})

    def status(self, message: str) -> None:
        self._emit("tool_result", {"content": message, "is_error": False})

    def success(self, message: str) -> None:
        self._emit("tool_result", {"message": message, "is_error": False})

    def warning(self, message: str) -> None:
        self._emit("tool_result", {"content": message, "is_error": True})

    def error(self, message: str) -> None:
        self._emit("run_failed", {"error": message})

    def assistant(self, message: str) -> None:
        self._emit("assistant_message", {"content": message})

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._emit("tool_call", {"name": name, "args": args})

    def flow_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._emit(event_type, payload)

    def clarification(self, payload: dict[str, Any]) -> None:
        self._emit("clarification_required", payload)

    def prompt(self, label: str = "You") -> str:
        raise RuntimeError(f"EventUI cannot prompt for {label}")


def map_clarification_selection(raw_answer: str, options: list[Any], *, multiple: bool = False) -> str | dict[str, Any]:
    if multiple:
        indexes = parse_selection_indexes(raw_answer)
        if not indexes:
            return raw_answer
        choices = [option for option in options if isinstance(option, dict)]
        selected = [choices[index] for index in indexes if 0 <= index < len(choices)]
        if not selected:
            return raw_answer
        return multiple_option_answer(selected)
    if not raw_answer.isdigit():
        return raw_answer
    index = int(raw_answer) - 1
    choices = [option for option in options if isinstance(option, dict)]
    if index < 0 or index >= len(choices):
        return raw_answer
    return option_answer(choices[index])


def option_answer(option: dict[str, Any]) -> dict[str, str]:
    label = str(option.get("label") or option.get("id") or "")
    option_id = str(option.get("id") or label)
    return {"answer": label, "option_id": option_id}


def multiple_option_answer(options: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(option.get("label") or option.get("id") or "") for option in options]
    ids = [str(option.get("id") or label) for option, label in zip(options, labels)]
    return {"answer": ", ".join(labels), "option_ids": ids}


def normalize_answer_payload(answer: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    return {"answer": answer}


def parse_selection_indexes(raw_answer: str) -> list[int]:
    normalized = raw_answer.replace(" ", "")
    if not normalized:
        return []
    separators = [",", ";", "+"]
    values = [normalized]
    for separator in separators:
        if separator in normalized:
            values = normalized.split(separator)
            break
    indexes: list[int] = []
    for value in values:
        if not value.isdigit():
            return []
        indexes.append(int(value) - 1)
    return indexes


def toggle_checked(checked: set[int], index: int) -> None:
    if index in checked:
        checked.remove(index)
    else:
        checked.add(index)


def read_key() -> str:
    fd = sys.stdin.fileno()
    settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":
            sequence = sys.stdin.read(2)
            if sequence == "[A":
                return "up"
            if sequence == "[B":
                return "down"
            return "escape"
        if char in {"\r", "\n"}:
            return "enter"
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
