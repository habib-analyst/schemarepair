"""Deterministic local repair of malformed JSON-ish model output.

A single-pass, schema-aware repair pipeline. Every transformation is logged
so a repair is auditable: you can see exactly which fixes were applied to a
given output. The pipeline never invents values; the only semantic changes
are type coercions toward the schema (e.g. "42" -> 42 for an int field,
"yes" -> true for a bool field) and dropping fields the schema forbids.

Stages:
  1. strip markdown fences
  2. extract the outermost object span
  3. scanner pass: single-quoted strings -> double-quoted, Python literals
     (True/False/None) -> JSON literals, // and /* */ comments removed
  4. balance truncated output (close open string / brackets)
  5. quote unquoted keys, remove trailing commas
  6. json.loads
  7. schema-guided cleanup: drop extra fields, coerce values per field type,
     match enums case-insensitively
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .validate import extract_json_span

_LITERAL_MAP = {"True": "true", "False": "false", "None": "null"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class RepairOutcome:
    obj: dict | None
    fixes: list[str] = field(default_factory=list)
    repaired_text: str | None = None


def _strip_fences(text: str, fixes: list[str]) -> str:
    stripped = text.strip()
    if "```" in stripped:
        stripped = re.sub(r"```[A-Za-z]*\s*", "", stripped).strip()
        fixes.append("strip_markdown_fence")
    return stripped


def _scan(text: str, fixes: list[str]) -> str:
    """Normalise quotes/literals/comments and report bracket balance."""
    out: list[str] = []
    i, n = 0, len(text)
    stack: list[str] = []
    changed_quotes = changed_literals = changed_comments = False
    while i < n:
        ch = text[i]
        # comments (outside strings)
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            changed_comments = True
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            changed_comments = True
            continue
        if ch in "\"'":
            quote = ch
            if quote == "'":
                changed_quotes = True
            buf = ['"']
            i += 1
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    buf.append(text[i:i + 2])
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                if c == '"' and quote == "'":
                    buf.append('\\"')
                    i += 1
                    continue
                buf.append(c)
                i += 1
            else:
                fixes.append("close_unterminated_string")
            buf.append('"')
            out.append("".join(buf))
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
            i += 1
            continue
        if ch in "}]":
            if stack:
                stack.pop()
            out.append(ch)
            i += 1
            continue
        m = _WORD.match(text, i)
        if m:
            word = m.group(0)
            if word in _LITERAL_MAP:
                out.append(_LITERAL_MAP[word])
                changed_literals = True
            else:
                out.append(word)
            i = m.end()
            continue
        out.append(ch)
        i += 1
    if changed_quotes:
        fixes.append("single_to_double_quotes")
    if changed_literals:
        fixes.append("python_literals_to_json")
    if changed_comments:
        fixes.append("strip_comments")
    result = "".join(out)
    if stack:
        closers = "".join("}" if c == "{" else "]" for c in reversed(stack))
        result += closers
        fixes.append("close_truncated_brackets")
    return result


_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _quote_keys(text: str, fixes: list[str]) -> str:
    new = _KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', text)
    if new != text:
        fixes.append("quote_unquoted_keys")
    return new


def _drop_trailing_commas(text: str, fixes: list[str]) -> str:
    new = _TRAILING_COMMA_RE.sub(r"\1", text)
    if new != text:
        fixes.append("remove_trailing_commas")
    return new


def _coerce_value(value: Any, annotation: Any) -> tuple[Any, bool]:
    """Best-effort coercion of *value* toward a field annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:  # e.g. list[str] -> leave to pydantic
        return value, False
    name = getattr(annotation, "__name__", str(annotation))
    if name == "int" and isinstance(value, str):
        s = value.replace(",", "").replace(" ", "").strip()
        try:
            return int(float(s)) if ("." in s or "e" in s.lower()) else int(s), True
        except ValueError:
            return value, False
    if name == "float" and isinstance(value, str):
        s = value.replace(",", "").replace(" ", "").strip()
        s = re.sub(r"^[^0-9+\-.]+", "", s)
        s = re.sub(r"[^0-9eE+\-.]+$", "", s)
        try:
            return float(s), True
        except ValueError:
            return value, False
    if name == "int" and isinstance(value, float) and value.is_integer():
        return int(value), True
    if name == "float" and isinstance(value, int):
        return float(value), True
    if name == "bool" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "y", "1"):
            return True, True
        if low in ("false", "no", "n", "0"):
            return False, True
    if name == "str" and isinstance(value, (int, float)):
        return str(value), True
    return value, False


def _schema_cleanup(obj: dict, schema_cls: type[BaseModel],
                    fixes: list[str]) -> dict:
    fields = schema_cls.model_fields
    cleaned = dict(obj)
    for key in list(cleaned):
        if key not in fields:
            del cleaned[key]
            fixes.append(f"drop_extra_field:{key}")
    import enum as _enum

    for fname, finfo in fields.items():
        if fname not in cleaned:
            continue
        value = cleaned[fname]
        ann = finfo.annotation
        if isinstance(ann, type) and issubclass(ann, _enum.Enum) \
                and isinstance(value, str):
            for member in ann:
                if isinstance(member.value, str) and \
                        member.value.lower() == value.strip().lower():
                    if member.value != value:
                        cleaned[fname] = member.value
                        fixes.append(f"enum_case_fix:{fname}")
                    break
            continue
        new_value, changed = _coerce_value(value, ann)
        if changed:
            cleaned[fname] = new_value
            fixes.append(f"coerce:{fname}")
    return cleaned


def repair(text: str, schema_cls: type[BaseModel]) -> RepairOutcome:
    """Deterministically repair *text* toward *schema_cls*. No model calls."""
    fixes: list[str] = []
    work = _strip_fences(text, fixes)
    span = extract_json_span(work)
    if span is None:
        return RepairOutcome(obj=None, fixes=fixes)
    if span.strip() != work:
        fixes.append("extract_json_span")
    work = _scan(span, fixes)
    work = _quote_keys(work, fixes)
    work = _drop_trailing_commas(work, fixes)
    try:
        obj = json.loads(work)
    except json.JSONDecodeError:
        return RepairOutcome(obj=None, fixes=fixes, repaired_text=work)
    if not isinstance(obj, dict):
        return RepairOutcome(obj=None, fixes=fixes, repaired_text=work)
    obj = _schema_cleanup(obj, schema_cls, fixes)
    return RepairOutcome(obj=obj, fixes=fixes, repaired_text=work)
