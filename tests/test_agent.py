import json

import pytest

from schemarepair.agent import StructuredOutputAgent, StructuredOutputError
from schemarepair.llm import LLMResponse
from schemarepair.schemas import PersonRecord

GOOD = ('{"name": "Sara Iqbal", "age": 34, "email": "sara.iqbal@example.com", '
        '"is_member": true}')
BAD_FENCED = "```json\n" + GOOD[:-1] + ",}\n```"
MSGS = [{"role": "user", "content": "extract the record"}]


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, *, seed=0, max_new_tokens=192,
                 temperature=0.8):
        text = self.responses.pop(0) if self.responses else "garbage"
        return LLMResponse(text=text, prompt_tokens=40,
                           completion_tokens=10)


def test_agent_returns_valid_directly():
    agent = StructuredOutputAgent(FakeLLM([GOOD]), PersonRecord)
    result = agent.run(MSGS)
    assert result.obj.age == 34
    assert not result.repaired
    assert result.total_model_calls == 1


def test_agent_repairs_and_logs_failure(tmp_path):
    log = tmp_path / "failures.jsonl"
    agent = StructuredOutputAgent(FakeLLM([BAD_FENCED]), PersonRecord,
                                  strategy="ladder", log_path=log)
    result = agent.run(MSGS, task_id="t-1")
    assert result.repaired
    assert result.obj.name == "Sara Iqbal"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["stage"] == "initial"
    assert records[0]["task_id"] == "t-1"
    assert "markdown_fence" in records[0]["taxonomy"]


def test_agent_raises_after_budget_and_logs(tmp_path):
    log = tmp_path / "failures.jsonl"
    agent = StructuredOutputAgent(FakeLLM(["junk", "junk", "junk"]),
                                  PersonRecord, strategy="naive_retry",
                                  max_calls=2, log_path=log)
    with pytest.raises(StructuredOutputError):
        agent.run(MSGS)
    stages = [json.loads(line)["stage"]
              for line in log.read_text().splitlines()]
    assert stages == ["initial", "repair_exhausted"]


def test_agent_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        StructuredOutputAgent(FakeLLM([]), PersonRecord, strategy="magic")
