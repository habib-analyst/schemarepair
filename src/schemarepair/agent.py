"""StructuredOutputAgent: schema-enforced generation with repair.

The production-facing piece of schemarepair. Give it any ``LLMClient``, a
Pydantic schema, and a repair strategy; ``run`` returns a validated model
instance or raises after the strategy's budget is spent. Every validation
failure — the initial one and every failed repair attempt that surfaces one —
is appended to a JSONL failure log with its taxonomy classes, so failure
modes are measured in production, not guessed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .llm import LLMClient
from .strategies import STRATEGIES, RepairContext
from .taxonomy import classify
from .validate import Validation, validate


@dataclass
class AgentResult:
    obj: BaseModel
    raw: str
    initial_validation: Validation
    strategy_used: str
    repaired: bool
    total_model_calls: int
    total_tokens: int


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, validation: Validation):
        super().__init__(message)
        self.validation = validation


class StructuredOutputAgent:
    def __init__(self, llm: LLMClient, schema_cls: type[BaseModel], *,
                 strategy: str = "ladder", max_calls: int = 2,
                 log_path: str | Path | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; "
                             f"choose from {sorted(STRATEGIES)}")
        self.llm = llm
        self.schema_cls = schema_cls
        self.strategy = strategy
        self.max_calls = max_calls
        self.log_path = Path(log_path) if log_path else None

    def _log_failure(self, raw: str, validation: Validation, *,
                     stage: str, task_id: str | None) -> None:
        if self.log_path is None:
            return
        tax = classify(raw, self.schema_cls)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": task_id,
            "schema": self.schema_cls.__name__,
            "stage": stage,
            "strategy": self.strategy,
            "taxonomy": tax["classes"],
            "primary": tax["primary"],
            "parse_error": validation.parse_error,
            "errors": validation.error_summary(),
            "raw": raw[:2000],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def run(self, messages: list[dict[str, str]], *,
            record_text: str = "", task_id: str | None = None,
            seed: int = 0) -> AgentResult:
        resp = self.llm.complete(messages, seed=seed)
        initial = validate(resp.text, self.schema_cls)
        if initial.ok:
            return AgentResult(obj=initial.obj, raw=resp.text,
                               initial_validation=initial,
                               strategy_used="none", repaired=False,
                               total_model_calls=1,
                               total_tokens=resp.total_tokens)
        self._log_failure(resp.text, initial, stage="initial",
                          task_id=task_id)
        ctx = RepairContext(
            schema_cls=self.schema_cls,
            schema_name=self.schema_cls.__name__,
            record_text=record_text,
            original_messages=messages,
            initial_raw=resp.text,
            initial=initial)
        fn = STRATEGIES[self.strategy]
        if self.strategy == "local_repair":
            result = fn(ctx)
        else:
            result = fn(ctx, self.llm, max_calls=self.max_calls,
                        seed_base=seed)
        if result.ok:
            return AgentResult(obj=result.obj, raw=resp.text,
                               initial_validation=initial,
                               strategy_used=self.strategy, repaired=True,
                               total_model_calls=1 + result.model_calls,
                               total_tokens=resp.total_tokens
                               + result.total_tokens)
        self._log_failure(resp.text, initial, stage="repair_exhausted",
                          task_id=task_id)
        raise StructuredOutputError(
            f"output still invalid after {self.strategy} "
            f"({result.detail}); errors: {initial.error_summary()}",
            initial)
