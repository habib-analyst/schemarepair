# schemarepair

**A structured-output agent, and the first head-to-head benchmark of what to
do when the model's JSON is broken.**

Every agent pipeline hits the same moment: the LLM returns something that
almost parses, validation throws, and you have to choose a recovery strategy.
Teams choose by folklore — "just retry", "send the error back", "run it
through a JSON fixer". This project measures the choice. On 80 real outputs
from a local open-weight model (Qwen2.5-0.5B-Instruct, CPU), 55 failed
schema validation; five repair strategies were then run against the *same*
55 failures with the same call budget:

| Strategy | Fixed | Fix rate | Model calls | Tokens / fix | End-to-end validity |
|---|---|---|---|---|---|
| **field_patch** (re-ask only for broken fields) | 47/55 | **85.5%** | 32 | **166** | **90.0%** |
| **ladder** (local repair, then field patch) | 47/55 | **85.5%** | 32 | **166** | **90.0%** |
| **local_repair** (deterministic, no model) | 33/55 | 60.0% | **0** | **0** | 72.5% |
| naive_retry (resample the same prompt) | 13/55 | 23.6% | 102 | 3,314 | 47.5% |
| feedback_retry (resample with the error) | 2/55 | 3.6% | 108 | 29,628 | 33.8% |

Two results should change defaults:

1. **Retrying is nearly useless; patching is not.** Regenerating the whole
   answer — with or without the validation error — mostly reproduces the
   same failure (this model wraps JSON in markdown fences on ~93% of
   failures, and being told so doesn't stop it). Asking only for the broken
   fields fixed 85.5% of failures at ~166 tokens per fix, roughly **20x
   cheaper per fix than naive retry and ~179x cheaper than feedback
   retry**.
2. **Error feedback made things worse than blind resampling** (3.6% vs
   23.6%): the feedback prompt anchors the model to its broken output.
   Folklore says "send the error back"; on this model, that is the worst
   strategy measured.

And a free one: a deterministic local repair pass (fence stripping, quote /
comma / literal fixes, schema-guided coercion) rescued 60% of failures at
zero model calls — there is no reason to spend a token before running it.

Full method, taxonomy, and honest limits: [PAPER.md](PAPER.md).
Raw data: [results/results.json](results/results.json) and the corpus in
[results/corpus.jsonl](results/corpus.jsonl).

## The agent

```python
from pydantic import BaseModel
from schemarepair import StructuredOutputAgent

class Person(BaseModel):
    name: str
    age: int
    email: str

agent = StructuredOutputAgent(llm, Person, strategy="ladder",
                              log_path="failures.jsonl")
person = agent.run(messages, record_text=record).obj   # validated, or raises
```

- Any LLM works: anything implementing `complete(messages, seed=...)`
  (a transformers-local client is included; bring your own API wrapper).
- Every validation failure is appended to a JSONL log with its taxonomy
  classes — failure modes are measured in production, not guessed.
- Strategies are pluggable: `naive_retry`, `feedback_retry`,
  `local_repair`, `field_patch`, `ladder`.

## What counts as "valid"

The acceptance test is the one production uses: the raw text, as-is, must
parse as JSON and satisfy the Pydantic schema (`model_validate_json`
semantics). A markdown fence around perfect JSON is a failure — because it
is one in your pipeline too. Under this test the model's initial validity
was **31.3%** (25/80); under Pydantic strict mode, **15.0%**. Prompting
mattered more than anything else measured: outputs under an explicit
JSON-only system prompt were valid 60% of the time, versus **2.5%** under a
generic "helpful assistant" prompt.

## Failure taxonomy (55 failures, multi-label)

| Class | Count | | Class | Count |
|---|---|---|---|---|
| markdown_fence | 51 | | type_mismatch | 4 |
| prose_wrapping | 51 | | truncated_json | 4 |
| extra_field | 12 | | enum_violation | 2 |
| missing_field | 11 | | constraint_violation | 1 |

Conformance is not correctness: outputs that *did* validate still had a
mean field accuracy of only 0.94 against the source records (a flipped
latitude sign validates fine). The benchmark therefore scores repaired
outputs on field accuracy too — field_patch's fixes average 0.93.

## Reproduce

```bash
pip install -e ".[llm,dev]"
pytest                                   # 34 tests, no model needed
python -m schemarepair.benchmark \
    --model-dir /path/to/Qwen2.5-0.5B-Instruct \
    --out results --per-schema 10 --max-calls 2
```

Corpus generation is seeded and the corpus is checked in, so the strategy
phase can be re-run on identical failures with `--skip-generate`.

## Limits (summary)

One small model, one run per strategy, 55 failures, synthetic programmatic
records, and a deliberately strict acceptance test. Effect sizes here are
large (85.5% vs 3.6%), but the ranking — especially feedback < naive —
should be re-measured before generalising to frontier models. See
PAPER.md §5 for the full threat list.

## License

MIT — Habib Ur Rehman.
