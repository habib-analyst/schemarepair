# When the JSON Is Broken: A Head-to-Head Benchmark of Repair Strategies for Structured LLM Output

**Habib Ur Rehman** — project paper for `schemarepair` (2026-10-26)

## Abstract

Agents that consume LLM output as structured data must decide what to do
when validation fails. The common choices — retry the prompt, retry with
the validation error appended, repair the text deterministically, or
re-ask only for the broken fields — are usually adopted by folklore, not
measurement. We benchmark five recovery strategies head-to-head on the
same 55 real validation failures produced by a small local open-weight
model (Qwen2.5-0.5B-Instruct, CPU-only) across 8 Pydantic schemas and 80
extraction tasks. Field-level patching fixed 85.5% of failures at 166
tokens per fix; deterministic local repair fixed 60.0% at zero model
calls; naive retry fixed 23.6% at 3,314 tokens per fix; and retry with
error feedback — the most commonly recommended strategy — fixed only
3.6%, at 29,628 tokens per fix, *worse than blind resampling*. We also
release a validation-failure taxonomy of the corpus (markdown fences and
prose wrapping dominate: 51/55 each) and show that conformance and
correctness diverge: schema-valid outputs still averaged 0.94 field
accuracy against the source records. All code, the corpus, and raw
results are in this repository.

## 1. Introduction

Structured output is the load-bearing interface of agent systems: tool
calls, extraction pipelines, and workflow steps all consume JSON that must
satisfy a schema. Small and mid-size models fail this contract constantly
— in our corpus, 69% of outputs failed a standard `model_validate_json`
acceptance test. The engineering question is not how to prevent every
failure but what to do *after* one, because the strategies differ by
orders of magnitude in cost and, as we show, in effectiveness.

Contributions:

1. **A strategy benchmark, not another repair library.** Five recovery
   strategies compared on identical failures, identical call budgets,
   with fix rate, model calls, tokens per fix, and the field accuracy of
   salvaged outputs.
2. **A failure taxonomy** measured on real outputs, separating syntactic
   classes (fences, prose, truncation, quote/comma defects) from semantic
   ones (missing, extra, type, enum, constraint).
3. **A production agent** (`StructuredOutputAgent`) implementing the
   winning strategies with per-failure JSONL logging classified by the
   same taxonomy.

## 2. Related work

Repair *libraries* are plentiful: `json-repair`, `repairjson`, and
`outputguard` (deterministic fixers, some with retry-prompt helpers), and
schema-aware coercers such as `coerce-json`. They ship a combined pipeline
and report, at most, repair speed or recovery on hand-built corpora; none
compares independent recovery strategies against each other on real model
outputs with cost accounting. JSONSchemaBench benchmarks *constrained
decoding* — preventing malformed output during generation — a different
(and complementary) layer. LLMStructBench taxonomises model extraction
errors but evaluates no repair. The closest project, a "five-rung repair
ladder" for small local models, reports numbers from a *simulated*
provider; no real model was benchmarked. Our benchmark uses real outputs
throughout, and compares strategies as alternatives rather than as rungs
of one pipeline (we additionally measure the ladder composition).

## 3. Method

### 3.1 Tasks and corpus

Eight Pydantic v2 schemas (`extra="forbid"`): person record, product item,
appointment, invoice (nested line items), sensor reading, book record,
support ticket, geo place. Each task presents a ground-truth record
rendered as messy source text — booleans as yes/no, money with currency
symbols, coordinates with hemisphere letters, ratings as "4.5 out of 5",
populations with thousands separators — and asks for one JSON object.
Records are generated programmatically from fixed pools with a fixed seed
(20261026), so the corpus is reproducible exactly. 10 tasks per schema,
80 total.

Two prompting regimes alternate across tasks: **strict** (explicit
JSON-only system prompt with the schema) and **plain** (a generic
"helpful assistant" system prompt, schema in the user turn) — the two ways
real applications actually call models.

Generation: Qwen2.5-0.5B-Instruct via transformers on CPU (2 cores),
temperature 0.8, top-p 0.95, repetition penalty 1.08, max 256 new tokens,
deterministic per-task seeds. The corpus (raw outputs, ground truth,
token counts) is checked in at `results/corpus.jsonl`.

### 3.2 Acceptance test

An output is *valid* iff the raw text, as-is, parses as JSON and satisfies
the schema — `model_validate_json` semantics. Lenient extraction is never
part of acceptance; it appears only inside repair strategies and in
diagnosis (taxonomy), because that is where it appears in real systems.
We also report validity under Pydantic strict mode.

### 3.3 Strategies (budget: ≤ 2 model calls each)

- **naive_retry** — resend the identical prompt; accept the first valid
  sample.
- **feedback_retry** — resend with the broken output and the validation
  error summary appended; accept the first valid sample.
- **local_repair** — deterministic pipeline, zero model calls: strip
  fences, extract the object span, single-pass normalisation (single→
  double quotes, Python literals, comment removal, truncation balancing),
  quote unquoted keys, drop trailing commas, then schema-guided cleanup
  (drop forbidden extras, coerce values toward field types, case-fix
  enums). Every applied fix is logged.
- **field_patch** — parse leniently; identify broken/missing fields from
  the validation errors; drop forbidden extras for free; make one call
  asking *only* for the broken fields (with the source record); merge;
  re-validate; at most 2 rounds. If the cleaned parse already validates,
  zero calls are spent.
- **ladder** — local_repair, then field_patch on whatever it salvaged.

### 3.4 Metrics

Per strategy, over the 55 initial failures: **fix rate**; **end-to-end
validity** (valid outputs over the full 80 after the strategy is applied
to failures); **model calls** and **tokens** (counted with the model's own
tokenizer), including tokens per fix; and **field accuracy** of fixed
outputs — the fraction of top-level ground-truth fields exactly matched —
because a conformant output with wrong values is a silent failure, not a
win.

### 3.5 Taxonomy

Each failure is labelled multi-label. Syntactic classes are detected on
the raw text (fences, prose wrapping, trailing commas, single quotes,
unquoted keys, Python literals, truncation, emptiness); semantic classes
come from Pydantic's structured errors on the leniently parsed object
(missing, extra, type, enum, constraint). A single *primary* class per
failure follows pipeline order (earliest failure first). Semantic classes
are only assigned when the JSON parses under extraction — an unparseable
output's semantic state is unknowable, and the taxonomy says so rather
than guessing.

## 4. Results

### 4.1 Initial validity

Of 80 outputs, **25 (31.3%)** passed acceptance; 12 (15.0%) passed strict
mode. The prompting regime dominated: strict prompts yielded 24/40 valid
(60.0%), plain prompts 1/40 (2.5%). Per schema, the nested invoice was
hardest (1/10 valid) and flat records easiest (4/10). Even valid outputs
were not fully correct: mean field accuracy 0.942 (e.g., a Sydney latitude
emitted as +33.8688 instead of −33.8688 — schema-valid, factually wrong).

### 4.2 Taxonomy of the 55 failures

Multi-label counts: markdown_fence 51, prose_wrapping 51, extra_field 12,
missing_field 11, type_mismatch 4, truncated_json 4, enum_violation 2,
constraint_violation 1. Primary classes: markdown_fence 47, truncated_json
4, type_mismatch 3, missing_field 1. The typical failure of a small model
is not broken JSON syntax — the JSON inside the fence is usually
well-formed — it is *packaging*: fences and prose around it, plus a tail
of hallucinated extra fields (12) and omitted fields (11).

### 4.3 Strategy comparison

| Strategy | Fixed | Fix rate | Calls (total) | Tokens (total) | Tokens/fix | End-to-end | Field acc. of fixes |
|---|---|---|---|---|---|---|---|
| field_patch | 47 | **85.5%** | 32 | 7,798 | **166** | **90.0%** | 0.930 |
| ladder | 47 | **85.5%** | 32 | 7,798 | **166** | **90.0%** | 0.930 |
| local_repair | 33 | 60.0% | 0 | 0 | 0 | 72.5% | 0.919 |
| naive_retry | 13 | 23.6% | 102 | 43,078 | 3,314 | 47.5% | 0.946 |
| feedback_retry | 2 | 3.6% | 108 | 59,256 | 29,628 | 33.8% | 0.875 |

Field patching dominates on every axis simultaneously: highest fix rate,
fewest calls (0.58 per failure — most failures need none, because the
lenient parse plus extra-field drop already validates, and the rest need
one small call), and the lowest token cost by a wide margin. Its fixes
also preserve values well (0.930 field accuracy).

**Ladder ≡ field_patch on this corpus.** The ladder's patch stage faced
exactly the semantic failures field_patch faced: local repair's fix set
(33) is a subset of what field_patch resolves for free via its lenient
parse, and where both spend calls, the patch prompts — which depend only
on the record and the broken-field set, not on the salvaged text — are
identical, as therefore are the seeded samples. The ladder would only
diverge on corpora where deterministic coercion changes *which* fields
are broken before the patch stage.

**Feedback hurt.** Retry with the validation error appended fixed 2/55 —
fewer than blind resampling (13/55) — while burning the most tokens
(feedback prompts carry the broken output and error list, and the model
tends to echo the fenced format it is shown). Under a strict acceptance
test, a model that has a entrenched output format does not talk its way
out of it. We flag this as the result most in need of replication on
other models (§5), but the direction is unambiguous here: on this model,
error feedback is strictly worse than doing the dumb thing, and both are
far worse than not regenerating at all.

**The residual 8 failures** (fixed by no strategy) are cases where values
are absent or wrong at the content level — e.g., a boolean emitted as
`null`, or fields whose values cannot be recovered from the output and
were not recovered by a targeted patch call within budget. Deterministic
repair correctly refuses to invent them.

## 5. Threats to validity — honestly

- **One small model.** All numbers are Qwen2.5-0.5B-Instruct on CPU.
  Frontier models fence less, follow "no markdown" instructions better,
  and may respond to error feedback very differently. The benchmark is
  built to be re-run (seeded corpus checked in; any `LLMClient` works);
  rankings, especially feedback-vs-naive, must be re-measured per model.
- **Strict acceptance is a choice.** Judging retries by raw-text validity
  is what production does, but a pipeline that extracts JSON before
  validating would see higher retry fix rates (and, correspondingly,
  would already be running a piece of local repair). The strategy ranking
  between regeneration and patching is robust to this choice; the
  absolute retry numbers are not.
- **Small n, single run.** 55 failures, one seeded run per strategy.
  Fix-rate differences of the headline size (85.5% vs 3.6%) are far
  outside noise; differences like ladder-vs-field_patch (identical here)
  and the field-accuracy gaps (~0.92–0.95) are not precisely estimated.
- **Synthetic records.** Ground-truth records are programmatic, with
  planted format traps (units, hemispheres, separators). Real records are
  messier in different ways; the taxonomy's head (fences/prose) matches
  widely reported production experience, but the tail may not.
- **Taxonomy heuristics.** Syntactic detectors are regex/heuristic and
  can mislabel edge cases (e.g., a `, key:` sequence inside a string
  value can resemble an unquoted key). Semantic classes depend on
  Pydantic's error typing.
- **Sampling variance in generation.** Corpus outputs are one sample per
  task at temperature 0.8 with fixed seeds; another seed set shifts the
  failure mix somewhat. Wall-clock times were measured on a shared,
  contended VM and are not reported as results for that reason.

## 6. Conclusion

When a small model breaks your schema — and it will, roughly two times in
three — do not ask it again, and do not read it its errors. Parse what it
gave you, drop what the schema forbids, and ask only for the fields that
are actually broken: on this benchmark that recovers 85.5% of failures for
about 166 tokens each, versus 3.6% for the error-feedback retry that
conventional wisdom recommends. A deterministic repair pass in front costs
nothing and clears the syntactic majority. The code, corpus, and full
results in this repository let anyone re-run the comparison on their own
model before trusting any of these numbers — including ours.

## References (selected)

- json-repair (Python), repairjson (Rust core) — deterministic JSON repair
  libraries.
- outputguard — validate/repair/retry toolkit with 15 repair strategies.
- JSONSchemaBench — benchmark of constrained-decoding frameworks for
  structured outputs (arXiv 2501.10868).
- LLMStructBench — benchmarking LLM structured data extraction (arXiv).
- Qwen2.5-0.5B-Instruct — Alibaba Qwen team, open-weight model used for
  all generation.
