"""Repair-strategy benchmark: generation + head-to-head evaluation.

Phase 1 generates one raw output per task from a local model and stores the
corpus (so every strategy is judged on identical failures). Phase 2 scores
the corpus: initial validity (lax and strict), the failure taxonomy, then
each strategy's fix rate, model-call/token cost, and the field accuracy of
what it salvaged.

Usage:
    python -m schemarepair.benchmark --model-dir /path/to/model \
        --out results_dir [--per-schema 24] [--max-calls 2]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .llm import LocalLLM
from .strategies import STRATEGIES, RepairContext
from .tasks import build_tasks
from .taxonomy import classify
from .validate import validate


def field_accuracy(obj, ground_truth: dict) -> float:
    """Fraction of top-level ground-truth fields exactly matched."""
    if obj is None or not ground_truth:
        return 0.0
    dumped = obj.model_dump(mode="json")
    hits = sum(1 for k, v in ground_truth.items() if dumped.get(k) == v)
    return hits / len(ground_truth)


def generate_corpus(llm: LocalLLM, per_schema: int, corpus_path: Path,
                    temperature: float = 0.8) -> list[dict]:
    tasks = build_tasks(per_schema)
    rows = []
    with corpus_path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks):
            seed = 10_000 + i
            variant = "strict" if i % 2 == 0 else "plain"
            resp = llm.complete(task.messages(variant), seed=seed,
                                max_new_tokens=256, temperature=temperature)
            row = {
                "task_id": task.task_id,
                "schema_name": task.schema_name,
                "variant": variant,
                "record_text": task.record_text,
                "ground_truth": task.ground_truth,
                "raw": resp.text,
                "gen_prompt_tokens": resp.prompt_tokens,
                "gen_completion_tokens": resp.completion_tokens,
                "seed": seed,
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[gen] {i + 1}/{len(tasks)} {task.task_id} "
                  f"({resp.completion_tokens} completion tokens)", flush=True)
    return rows


def run_benchmark(llm: LocalLLM, rows: list[dict], max_calls: int,
                  seed_base: int = 777_000,
                  partial_path: Path | None = None) -> dict:
    from .schemas import SCHEMAS

    n = len(rows)
    valid_lax = valid_strict = 0
    tax_multi: Counter = Counter()
    tax_primary: Counter = Counter()
    per_schema: dict[str, dict] = {}
    per_variant: dict[str, dict] = {}
    initial_acc: list[float] = []
    failures: list[tuple[dict, RepairContext]] = []

    for i, row in enumerate(rows):
        schema_cls = SCHEMAS[row["schema_name"]]
        v = validate(row["raw"], schema_cls)
        vs = validate(row["raw"], schema_cls, strict=True)
        ps = per_schema.setdefault(row["schema_name"],
                                   {"n": 0, "valid": 0})
        ps["n"] += 1
        pv = per_variant.setdefault(row.get("variant", "strict"),
                                    {"n": 0, "valid": 0})
        pv["n"] += 1
        if v.ok:
            valid_lax += 1
            ps["valid"] += 1
            pv["valid"] += 1
            initial_acc.append(field_accuracy(v.obj, row["ground_truth"]))
        else:
            tax = classify(row["raw"], schema_cls)
            for c in tax["classes"]:
                tax_multi[c] += 1
            if tax["primary"]:
                tax_primary[tax["primary"]] += 1
            ctx = RepairContext(
                schema_cls=schema_cls, schema_name=row["schema_name"],
                record_text=row["record_text"],
                original_messages=_messages_for(row), initial_raw=row["raw"],
                initial=v)
            failures.append((row, ctx))
        if vs.ok:
            valid_strict += 1

    print(f"[bench] corpus {n}: valid lax {valid_lax}, strict {valid_strict}, "
          f"failures {len(failures)}", flush=True)

    summary = {
        "corpus_size": n,
        "initial_validity": {
            "valid_lax": valid_lax,
            "rate_lax": round(valid_lax / n, 4),
            "valid_strict": valid_strict,
            "rate_strict": round(valid_strict / n, 4),
            "field_accuracy_of_valid":
                round(sum(initial_acc) / len(initial_acc), 4)
                if initial_acc else None,
        },
        "per_schema_validity": per_schema,
        "per_variant_validity": per_variant,
        "taxonomy_failures_only": {
            "class_counts": dict(tax_multi.most_common()),
            "primary_counts": dict(tax_primary.most_common()),
        },
    }

    strat_stats: dict[str, dict] = {}
    for name, fn in STRATEGIES.items():
        fixed = 0
        calls = tokens = 0
        wall = 0.0
        accs: list[float] = []
        for j, (row, ctx) in enumerate(failures):
            if name == "local_repair":
                res = fn(ctx)
            else:
                res = fn(ctx, llm, max_calls=max_calls,
                         seed_base=seed_base + j * 131)
            calls += res.model_calls
            tokens += res.total_tokens
            wall += res.wall_s
            if res.ok:
                fixed += 1
                accs.append(field_accuracy(res.obj, row["ground_truth"]))
            if (j + 1) % 20 == 0 or j + 1 == len(failures):
                print(f"[bench] {name}: {j + 1}/{len(failures)} failures, "
                      f"fixed so far {fixed}", flush=True)
        attempted = len(failures)
        strat_stats[name] = {
            "attempted": attempted,
            "fixed": fixed,
            "fix_rate": round(fixed / attempted, 4) if attempted else None,
            "end_to_end_validity": round((valid_lax + fixed) / n, 4),
            "model_calls_total": calls,
            "mean_calls_per_failure": round(calls / attempted, 3)
            if attempted else 0,
            "tokens_total": tokens,
            "mean_tokens_per_failure": round(tokens / attempted, 1)
            if attempted else 0,
            "tokens_per_fix": round(tokens / fixed, 1) if fixed else None,
            "wall_s_total": round(wall, 1),
            "field_accuracy_fixed": round(sum(accs) / len(accs), 4)
            if accs else None,
        }
        if partial_path is not None:
            partial = dict(summary)
            partial["strategies"] = dict(strat_stats)
            partial["complete"] = False
            partial_path.write_text(json.dumps(partial, indent=1),
                                    encoding="utf-8")
            print(f"[bench] partial results written after {name}", flush=True)

    return {**summary, "strategies": strat_stats}


def _messages_for(row: dict) -> list[dict[str, str]]:
    from .tasks import Task

    task = Task(task_id=row["task_id"], schema_name=row["schema_name"],
                record_text=row["record_text"],
                ground_truth=row["ground_truth"])
    return task.messages(row.get("variant", "strict"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-schema", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse an existing corpus.jsonl in --out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.jsonl"

    started = time.time()
    llm = LocalLLM(args.model_dir)
    if args.skip_generate and corpus_path.exists():
        rows = [json.loads(line) for line in
                corpus_path.read_text(encoding="utf-8").splitlines() if line]
        print(f"[bench] reused corpus: {len(rows)} rows", flush=True)
    else:
        rows = generate_corpus(llm, args.per_schema, corpus_path,
                               temperature=args.temperature)
    results = run_benchmark(llm, rows, args.max_calls,
                            partial_path=out / "results.json")
    results["complete"] = True
    results["meta"] = {
        "model_dir": str(args.model_dir),
        "per_schema": args.per_schema,
        "max_calls_budget_per_strategy": args.max_calls,
        "temperature": args.temperature,
        "wall_s_total": round(time.time() - started, 1),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1),
                                      encoding="utf-8")
    print("[bench] wrote results.json", flush=True)
    print(json.dumps(results["initial_validity"], indent=1), flush=True)
    for name, s in results["strategies"].items():
        print(f"[bench] {name}: fix_rate={s['fix_rate']} "
              f"tokens_per_fix={s['tokens_per_fix']}", flush=True)


if __name__ == "__main__":
    main()
