"""Validation-failure taxonomy for raw model output.

Every malformed output is labelled with one or more error classes, plus a
single *primary* class (the first failure a pipeline would hit, by the
priority order below). Syntactic classes are detected on the raw text;
semantic classes (missing / type / enum / constraint / extra) come from
Pydantic's structured errors and are only assigned when the JSON parses —
an unparseable output's semantic state is unknowable, and the taxonomy says
so instead of guessing.

Class list:
  empty_output, not_json, markdown_fence, prose_wrapping, truncated_json,
  trailing_comma, single_quotes, unquoted_keys, python_literals,
  parse_error_other, missing_field, extra_field, type_mismatch,
  enum_violation, constraint_violation
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from .validate import extract_json_span, parse_lenient, validate_parsed

_TRAILING_COMMA = re.compile(r",\s*[}\]]")
_UNQUOTED_KEY = re.compile(r'(?:^|[{,]\s*)[A-Za-z_][A-Za-z0-9_]*\s*:')
_PY_LITERAL = re.compile(r":\s*(True|False|None)\b")
_SINGLE_QUOTED = re.compile(r"(?:^|[{,]\s*)'[^']*'\s*:")

# primary-class priority: earliest pipeline stage first
_PRIORITY = [
    "empty_output", "not_json", "truncated_json", "markdown_fence",
    "prose_wrapping", "single_quotes", "unquoted_keys", "python_literals",
    "trailing_comma", "parse_error_other", "missing_field", "type_mismatch",
    "enum_violation", "constraint_violation", "extra_field",
]

_TYPE_ERRORS = {
    "int_parsing", "int_type", "float_parsing", "float_type", "bool_parsing",
    "bool_type", "string_type", "list_type", "model_type", "dict_type",
    "json_invalid", "decimal_parsing", "date_parsing", "datetime_parsing",
}


def classify(text: str, schema_cls: type[BaseModel]) -> dict:
    """Return {classes: [...], primary: str|None, is_valid: bool}."""
    classes: set[str] = set()
    if not text or not text.strip():
        return {"classes": ["empty_output"], "primary": "empty_output",
                "is_valid": False}
    if "```" in text:
        classes.add("markdown_fence")
    span = extract_json_span(text)
    if span is None:
        classes.add("not_json")
        primary = _pick(classes)
        return {"classes": sorted(classes), "primary": primary,
                "is_valid": False}
    if text.strip() != span.strip():
        classes.add("prose_wrapping")
    if _TRAILING_COMMA.search(span):
        classes.add("trailing_comma")
    if _SINGLE_QUOTED.search(span):
        classes.add("single_quotes")
    if _UNQUOTED_KEY.search(span) and '"' not in span.split(":", 1)[0]:
        classes.add("unquoted_keys")
    if _PY_LITERAL.search(span):
        classes.add("python_literals")

    parsed, parse_err = parse_lenient(text)
    if parsed is None:
        # unbalanced = truncated; balanced-but-unparseable = other syntax
        opens = span.count("{") + span.count("[")
        closes = span.count("}") + span.count("]")
        if opens > closes or (parse_err and "Unterminated" in parse_err):
            classes.add("truncated_json")
        else:
            classes.add("parse_error_other")
        return {"classes": sorted(classes), "primary": _pick(classes),
                "is_valid": False}

    result = validate_parsed(parsed, schema_cls)
    if result.ok:
        # Content is fine; acceptance still fails if the raw text needed
        # extraction (fence / prose), which the classes above already record.
        from .validate import validate as _strict_validate

        return {"classes": sorted(classes), "primary": _pick(classes),
                "is_valid": _strict_validate(text, schema_cls).ok}
    for err in result.errors:
        etype = err.get("type", "")
        if etype == "missing":
            classes.add("missing_field")
        elif etype == "extra_forbidden":
            classes.add("extra_field")
        elif etype == "literal_error" or "enum" in etype:
            classes.add("enum_violation")
        elif etype in _TYPE_ERRORS:
            classes.add("type_mismatch")
        else:
            classes.add("constraint_violation")
    return {"classes": sorted(classes), "primary": _pick(classes),
            "is_valid": False}


def _pick(classes: set[str]) -> str | None:
    for c in _PRIORITY:
        if c in classes:
            return c
    return sorted(classes)[0] if classes else None


def taxonomy_counts(texts: list[str],
                    schema_cls: type[BaseModel]) -> dict:
    """Aggregate class counts (multi-label) and primary-class counts."""
    multi: Counter = Counter()
    primary: Counter = Counter()
    valid = 0
    for t in texts:
        r = classify(t, schema_cls)
        if r["is_valid"]:
            valid += 1
        for c in r["classes"]:
            multi[c] += 1
        if r["primary"]:
            primary[r["primary"]] += 1
    return {"n": len(texts), "valid": valid,
            "class_counts": dict(multi.most_common()),
            "primary_counts": dict(primary.most_common())}
