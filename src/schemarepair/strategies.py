"""Repair strategies, benchmarked head-to-head.

Every strategy receives the same failed output and the same budget
(``max_calls`` model calls) and returns a ``StrategyResult``. Strategies:

  naive_retry    re-ask the identical prompt; hope sampling differs
  feedback_retry re-ask with the validation errors appended
  local_repair   deterministic programmatic repair; zero model calls
  field_patch    re-ask ONLY for the broken/missing fields, then merge
  ladder         local_repair, then field_patch on what repair salvaged

Cost accounting (model calls, prompt/completion tokens, wall time) is part
of the result: a strategy that fixes 90% at 10x the tokens of one that
fixes 85% is not obviously better, and the benchmark reports both axes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from .llm import LLMClient
from .localrepair import repair as local_repair_fn
from .validate import Validation, parse_lenient, validate, validate_parsed

SYSTEM = (
    "You are a data extraction engine. You convert records into JSON that "
    "matches a given schema. Respond with a single JSON object only - no "
    "markdown, no explanation."
)


@dataclass
class RepairContext:
    schema_cls: type[BaseModel]
    schema_name: str
    record_text: str
    original_messages: list[dict[str, str]]
    initial_raw: str
    initial: Validation


@dataclass
class StrategyResult:
    strategy: str
    ok: bool
    obj: BaseModel | None = None
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    detail: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _finish(strategy: str, started: float, result: StrategyResult) -> StrategyResult:
    result.strategy = strategy
    result.wall_s = time.time() - started
    return result


def _regen(ctx: RepairContext, llm: LLMClient, messages: list[dict[str, str]],
           seed: int, result: StrategyResult,
           max_new_tokens: int = 256) -> Validation | None:
    resp = llm.complete(messages, seed=seed, max_new_tokens=max_new_tokens)
    result.model_calls += 1
    result.prompt_tokens += resp.prompt_tokens
    result.completion_tokens += resp.completion_tokens
    return validate(resp.text, ctx.schema_cls)


def naive_retry(ctx: RepairContext, llm: LLMClient, *,
                max_calls: int = 2, seed_base: int = 0) -> StrategyResult:
    started = time.time()
    result = StrategyResult(strategy="naive_retry", ok=False)
    for k in range(max_calls):
        v = _regen(ctx, llm, ctx.original_messages, seed_base + 101 + k, result)
        if v is not None and v.ok:
            result.ok, result.obj = True, v.obj
            result.detail = f"fixed on retry {k + 1}"
            break
    return _finish("naive_retry", started, result)


def feedback_retry(ctx: RepairContext, llm: LLMClient, *,
                   max_calls: int = 2, seed_base: int = 0) -> StrategyResult:
    started = time.time()
    result = StrategyResult(strategy="feedback_retry", ok=False)
    errors = ctx.initial.error_summary()
    user = ctx.original_messages[-1]["content"] + (
        "\n\nYour previous output was invalid.\nPrevious output:\n"
        + ctx.initial_raw[:1200]
        + "\nValidation errors:\n- " + "\n- ".join(errors)
        + "\nReturn a corrected JSON object only.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    for k in range(max_calls):
        v = _regen(ctx, llm, messages, seed_base + 211 + k, result)
        if v is not None and v.ok:
            result.ok, result.obj = True, v.obj
            result.detail = f"fixed on feedback retry {k + 1}"
            break
    return _finish("feedback_retry", started, result)


def local_repair(ctx: RepairContext, llm: LLMClient | None = None, **_: object) -> StrategyResult:
    started = time.time()
    result = StrategyResult(strategy="local_repair", ok=False)
    outcome = local_repair_fn(ctx.initial_raw, ctx.schema_cls)
    if outcome.obj is not None:
        try:
            obj = ctx.schema_cls.model_validate(outcome.obj)
            result.ok, result.obj = True, obj
            result.detail = "fixes: " + ",".join(outcome.fixes)
        except Exception:
            result.detail = "repaired JSON still invalid: " + ",".join(outcome.fixes)
    else:
        result.detail = "no parseable object after repair"
    return _finish("local_repair", started, result)


def _broken_fields(ctx: RepairContext, parsed: dict | None,
                   errors: list[dict]) -> list[str]:
    fields = list(ctx.schema_cls.model_fields)
    if parsed is None:
        return fields
    broken = set()
    for e in errors:
        loc = e.get("loc", ())
        if loc:
            broken.add(str(loc[0]))
    for f in fields:
        if f not in parsed:
            broken.add(f)
    return [f for f in fields if f in broken]


def _patch_round(ctx: RepairContext, llm: LLMClient, base: dict,
                 broken: list[str], seed: int,
                 result: StrategyResult) -> dict | None:
    subset = {f: s for f, s in
              ctx.schema_cls.model_json_schema()["properties"].items()
              if f in broken}
    user = (
        "Record:\n" + ctx.record_text
        + "\n\nReturn a JSON object containing ONLY these fields, with "
          "values taken from the record:\n"
        + json.dumps({"properties": subset}, indent=1)
        + "\nFields needed: " + ", ".join(broken)
        + "\nReturn only that JSON object.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    resp = llm.complete(messages, seed=seed, max_new_tokens=192)
    result.model_calls += 1
    result.prompt_tokens += resp.prompt_tokens
    result.completion_tokens += resp.completion_tokens
    patch, _ = parse_lenient(resp.text)
    if patch is None:
        return None
    merged = {k: v for k, v in base.items()
              if k in ctx.schema_cls.model_fields}
    for f in broken:
        if f in patch:
            merged[f] = patch[f]
    return merged


def field_patch(ctx: RepairContext, llm: LLMClient, *,
                max_calls: int = 2, seed_base: int = 0,
                start_text: str | None = None) -> StrategyResult:
    started = time.time()
    result = StrategyResult(strategy="field_patch", ok=False)
    text = start_text if start_text is not None else ctx.initial_raw
    parsed, _ = parse_lenient(text)
    base: dict = dict(parsed) if parsed else {}
    # Dropping forbidden extra fields is free: try the cleaned base first,
    # before spending any model calls.
    cleaned = {k: v for k, v in base.items()
               if k in ctx.schema_cls.model_fields}
    if parsed is not None:
        try:
            obj = ctx.schema_cls.model_validate(cleaned)
            result.ok, result.obj = True, obj
            result.detail = "valid after dropping extra fields (no patch)"
            return _finish("field_patch", started, result)
        except Exception:
            base = cleaned
    first = (validate_parsed(parsed, ctx.schema_cls) if parsed is not None
             else ctx.initial)
    broken = _broken_fields(ctx, parsed, first.errors)
    for k in range(max_calls):
        if not broken:
            break
        merged = _patch_round(ctx, llm, base, broken, seed_base + 307 + k,
                              result)
        if merged is None:
            continue
        base = merged
        try:
            obj = ctx.schema_cls.model_validate(base)
            result.ok, result.obj = True, obj
            result.detail = f"patched fields {broken} in {k + 1} round(s)"
            break
        except Exception as exc:
            errors = list(exc.errors()) if hasattr(exc, "errors") else []
            broken = _broken_fields(ctx, base, errors)
    return _finish("field_patch", started, result)


def ladder(ctx: RepairContext, llm: LLMClient, *,
           max_calls: int = 2, seed_base: int = 0) -> StrategyResult:
    started = time.time()
    first = local_repair(ctx)
    if first.ok:
        first.strategy = "ladder"
        first.detail = "local repair sufficed: " + first.detail
        first.wall_s = time.time() - started
        return first
    outcome = local_repair_fn(ctx.initial_raw, ctx.schema_cls)
    start_text = (json.dumps(outcome.obj) if outcome.obj is not None
                  else ctx.initial_raw)
    result = field_patch(ctx, llm, max_calls=max_calls, seed_base=seed_base,
                         start_text=start_text)
    result.strategy = "ladder"
    if result.ok:
        result.detail = "local repair + " + result.detail
    result.wall_s = time.time() - started
    return result


STRATEGIES = {
    "naive_retry": naive_retry,
    "feedback_retry": feedback_retry,
    "local_repair": local_repair,
    "field_patch": field_patch,
    "ladder": ladder,
}
