import json

import pytest

from schemarepair.llm import LLMResponse
from schemarepair.schemas import PersonRecord
from schemarepair.strategies import (RepairContext, feedback_retry,
                                     field_patch, ladder, local_repair,
                                     naive_retry)
from schemarepair.tasks import Task
from schemarepair.validate import validate

GOOD = ('{"name": "Sara Iqbal", "age": 34, "email": "sara.iqbal@example.com", '
        '"is_member": true}')
BAD_FENCED = "```json\n" + GOOD[:-1] + ",}\n```"


class FakeLLM:
    """Scripted LLM: returns queued responses, records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, seed=0, max_new_tokens=192,
                 temperature=0.8):
        self.calls.append(messages)
        text = self.responses.pop(0) if self.responses else "garbage"
        if callable(text):
            text = text(messages)
        return LLMResponse(text=text, prompt_tokens=50,
                           completion_tokens=max(1, len(text) // 4))


def _ctx(raw, record="Full name: Sara Iqbal; age: 34 years; email address: "
                     "sara.iqbal@example.com; club member: yes"):
    task = Task(task_id="t", schema_name="person_record", record_text=record,
                ground_truth={})
    return RepairContext(schema_cls=PersonRecord, schema_name="person_record",
                         record_text=record, original_messages=task.messages(),
                         initial_raw=raw, initial=validate(raw, PersonRecord))


def test_local_repair_uses_no_model_calls():
    res = local_repair(_ctx(BAD_FENCED))
    assert res.ok
    assert res.model_calls == 0
    assert res.obj.name == "Sara Iqbal"


def test_naive_retry_succeeds_on_second_attempt():
    llm = FakeLLM(["still not json", GOOD])
    res = naive_retry(_ctx("not json"), llm, max_calls=2)
    assert res.ok
    assert res.model_calls == 2
    assert res.obj.age == 34


def test_naive_retry_respects_budget():
    llm = FakeLLM(["nope", "nope", GOOD])
    res = naive_retry(_ctx("not json"), llm, max_calls=2)
    assert not res.ok
    assert res.model_calls == 2


def test_feedback_retry_sends_errors():
    llm = FakeLLM([GOOD])
    res = feedback_retry(_ctx('{"name": "Sara Iqbal"}'), llm, max_calls=1)
    assert res.ok
    user = llm.calls[0][-1]["content"]
    assert "Validation errors" in user
    assert "missing" in user.lower() or "field" in user.lower()


def test_field_patch_asks_only_for_broken_fields():
    raw = ('{"name": "Sara Iqbal", "age": "thirty-four", '
           '"is_member": true}')  # email missing, age wrong type
    seen = {}

    def patch_reply(messages):
        seen["user"] = messages[-1]["content"]
        return '{"age": 34, "email": "sara.iqbal@example.com"}'

    llm = FakeLLM([patch_reply])
    res = field_patch(_ctx(raw), llm, max_calls=1)
    assert res.ok
    assert res.obj.email == "sara.iqbal@example.com"
    assert "Fields needed: age, email" in seen["user"]
    assert res.model_calls == 1


def test_field_patch_drops_extra_fields_for_free():
    raw = ('{"name": "Sara Iqbal", "age": 34, "email": "s@example.com", '
           '"is_member": true, "hack": 1}')
    llm = FakeLLM([])  # no patch call should be needed... extra is droppable
    res = field_patch(_ctx(raw), llm, max_calls=1)
    # extra_forbidden means the field set is 'broken' at root; patch round
    # merges base without extras, so zero or one calls both acceptable --
    # what matters: result validates and the extra key is gone.
    assert res.ok
    assert not hasattr(res.obj, "hack")


def test_ladder_uses_local_repair_first():
    def explode(messages):  # pragma: no cover
        raise AssertionError("model must not be called")

    llm = FakeLLM([explode])
    res = ladder(_ctx(BAD_FENCED), llm, max_calls=1)
    assert res.ok
    assert res.model_calls == 0


def test_results_carry_token_accounting():
    llm = FakeLLM([GOOD])
    res = naive_retry(_ctx("junk"), llm, max_calls=1)
    assert res.prompt_tokens == 50
    assert res.completion_tokens > 0
    assert res.total_tokens == res.prompt_tokens + res.completion_tokens
