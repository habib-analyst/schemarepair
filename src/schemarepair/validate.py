"""Validation of raw model output against a Pydantic schema.

The acceptance test is the one real pipelines use: the raw text, taken
as-is, must parse as JSON and satisfy the schema. No extraction, no
clean-up — a markdown fence around perfect JSON is a failure here, because
``model_validate_json(raw)`` fails on it in production too. Lenient parsing
(``parse_lenient``) exists for *diagnosis* (taxonomy) and as an internal
step of repair strategies, never as the acceptance test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError


@dataclass
class Validation:
    ok: bool
    obj: BaseModel | None = None
    parsed: dict | None = None  # parsed JSON value, if parsing succeeded
    parse_error: str | None = None
    errors: list[dict] = field(default_factory=list)  # pydantic error dicts

    def error_summary(self, limit: int = 8) -> list[str]:
        out = []
        if self.parse_error:
            out.append(f"JSON parse error: {self.parse_error}")
        for e in self.errors[:limit]:
            loc = ".".join(str(p) for p in e.get("loc", ())) or "<root>"
            out.append(f"field '{loc}': {e.get('msg', e.get('type', 'invalid'))}")
        return out


def extract_json_span(text: str) -> str | None:
    """Return the outermost {...} span of *text* (string-aware), or None.

    If the object is truncated (no matching close brace), returns the span
    from the first '{' to the end of the text so downstream stages can still
    classify the failure as truncation.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def parse_raw(text: str) -> tuple[dict | None, str | None]:
    """Strictly parse *text* as one JSON object. (value, error_message)."""
    stripped = (text or "").strip()
    if not stripped:
        return None, "empty output"
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} (line {exc.lineno} col {exc.colno})"
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def parse_lenient(text: str) -> tuple[dict | None, str | None]:
    """Parse via span extraction (diagnosis / repair internals only)."""
    span = extract_json_span(text or "")
    if span is None:
        return None, "no JSON object found in output"
    try:
        value = json.loads(span)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} (line {exc.lineno} col {exc.colno})"
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def validate_parsed(parsed: dict, schema_cls: type[BaseModel], *,
                    strict: bool = False) -> Validation:
    try:
        obj = schema_cls.model_validate(parsed, strict=strict)
    except ValidationError as exc:
        return Validation(ok=False, parsed=parsed, errors=list(exc.errors()))
    return Validation(ok=True, obj=obj, parsed=parsed)


def validate(text: str, schema_cls: type[BaseModel], *,
             strict: bool = False) -> Validation:
    """Validate raw output text against a schema (strict text parse)."""
    parsed, err = parse_raw(text)
    if parsed is None:
        return Validation(ok=False, parse_error=err)
    return validate_parsed(parsed, schema_cls, strict=strict)
