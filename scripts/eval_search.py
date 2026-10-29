#!/usr/bin/env python3
"""
Eval harness for the NPU semantic search engine (scripts/npu_vault_search.py).

Runs a fixed query set (eval/queries.json) through the engine and computes
retrieval quality and latency metrics:

  * Recall@1 / Recall@3 / Recall@5 — at least one expected file in top-k;
  * MRR — mean reciprocal rank of the first expected file;
  * latency avg / p50 / p95 over the per-query median of `total_ms`
    (each query is run --runs times, the median run is taken);
  * informational: share of queries where the expected section matched.

The engine itself is NOT modified: it is imported as a module and called
through its public `search_vault()` API.

Examples:
    PY=/home/direkt/.local/share/npu-search/venv/bin/python
    $PY scripts/eval_search.py \
        --queries eval/queries.json \
        --out eval/results/baseline-main.json \
        --vault-path /home/direkt/Документы/learn \
        --label baseline-main
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_QUERIES = os.path.join(REPO_DIR, "eval", "queries.json")
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "eval", "results")
ENGINE_PATH = os.path.join(SCRIPT_DIR, "npu_vault_search.py")


def load_engine():
    """Import scripts/npu_vault_search.py as a module (engine stays untouched)."""
    spec = importlib.util.spec_from_file_location("npu_vault_search", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["npu_vault_search"] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' equivalent)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def section_match(expected_sections: list[str], result_sections: list[str]) -> bool | None:
    """Informational: does any returned section match any expected section?

    Returns None when the query carries no expected_sections (not applicable).
    """
    if not expected_sections:
        return None
    for got in result_sections:
        got_norm = (got or "").strip().lower()
        for want in expected_sections:
            want_norm = want.strip().lower()
            if want_norm and (want_norm in got_norm or got_norm in want_norm):
                return True
    return False


def first_hit_rank(expected_files: list[str], ranked_files: list[str]) -> int | None:
    expected = set(expected_files)
    for i, name in enumerate(ranked_files, start=1):
        if name in expected:
            return i
    return None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def run_eval(args) -> dict:
    engine = load_engine()

    with open(args.queries, encoding="utf-8") as fh:
        payload = json.load(fh)
    queries = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(queries, list) or not queries:
        raise SystemExit(f"no queries found in {args.queries}")

    top_k = args.top_k
    runs = max(1, args.runs)

    print(
        f"[eval] {len(queries)} queries x {runs} runs, top_k={top_k}, "
        f"device={args.device}, vault={args.vault_path or '(default)'}",
        file=sys.stderr,
    )

    per_query = []
    for qi, item in enumerate(queries, start=1):
        query = item["query"]
        expected_files = item.get("expected_files", [])
        expected_sections = item.get("expected_sections", [])
        category = item.get("category", "uncategorized")

        run_times: list[float] = []
        run_results: list[list[dict]] = []
        for _ in range(runs):
            started = time.time()
            results = engine.search_vault(
                query,
                top_k=top_k,
                device=args.device,
                vault_path=args.vault_path,
                model_dir=args.model_dir,
            )
            wall_ms = round((time.time() - started) * 1000, 1)
            run_times.append(
                float(results[0]["total_ms"]) if results else wall_ms
            )
            run_results.append(results)

        median_total = statistics.median(run_times)
        median_idx = min(
            range(runs), key=lambda i: abs(run_times[i] - median_total)
        )
        results = run_results[median_idx]
        ranked_files = [r["filename"] for r in results]
        result_sections = [r.get("section", "") for r in results]
        hit_rank = first_hit_rank(expected_files, ranked_files)

        per_query.append({
            "query": query,
            "category": category,
            "expected_files": expected_files,
            "expected_sections": expected_sections,
            "hit_rank": hit_rank,
            "recall_at_1": 1.0 if hit_rank is not None and hit_rank <= 1 else 0.0,
            "recall_at_3": 1.0 if hit_rank is not None and hit_rank <= 3 else 0.0,
            "recall_at_5": 1.0 if hit_rank is not None and hit_rank <= 5 else 0.0,
            "reciprocal_rank": (1.0 / hit_rank) if hit_rank else 0.0,
            "section_match": section_match(expected_sections, result_sections),
            "total_ms_median": median_total,
            "total_ms_runs": run_times,
            "device": results[0].get("device", "") if results else "",
            "top": [
                {
                    "rank": r.get("rank"),
                    "filename": r.get("filename"),
                    "section": r.get("section"),
                    "score": r.get("score"),
                }
                for r in results
            ],
        })
        print(
            f"[eval] {qi}/{len(queries)} done ({median_total:.0f} ms, "
            f"hit_rank={hit_rank})",
            file=sys.stderr,
        )

    def aggregate(entries: list[dict]) -> dict:
        n = len(entries) or 1
        latencies = [e["total_ms_median"] for e in entries]
        section_entries = [e for e in entries if e["section_match"] is not None]
        return {
            "queries": len(entries),
            "recall_at_1": sum(e["recall_at_1"] for e in entries) / n,
            "recall_at_3": sum(e["recall_at_3"] for e in entries) / n,
            "recall_at_5": sum(e["recall_at_5"] for e in entries) / n,
            "mrr": sum(e["reciprocal_rank"] for e in entries) / n,
            "latency_ms": {
                "avg": sum(latencies) / n,
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "min": min(latencies),
                "max": max(latencies),
            },
            "section_match_rate": (
                sum(1 for e in section_entries if e["section_match"]) / len(section_entries)
                if section_entries else None
            ),
            "section_match_applicable": len(section_entries),
        }

    metrics = aggregate(per_query)
    by_category: dict[str, dict] = {}
    for cat in sorted({e["category"] for e in per_query}):
        by_category[cat] = aggregate([e for e in per_query if e["category"] == cat])

    devices = sorted({e["device"] for e in per_query if e["device"]})
    report = {
        "label": args.label or "eval",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "engine": ENGINE_PATH,
        "queries_file": os.path.abspath(args.queries),
        "vault_path": args.vault_path or "(engine default)",
        "config": {
            "top_k": top_k,
            "device_requested": args.device,
            "model_dir": args.model_dir or "(engine default)",
            "runs_per_query": runs,
        },
        "device_used": devices,
        "metrics": metrics,
        "metrics_by_category": by_category,
        "per_query": per_query,
    }
    return report


def print_report(report: dict) -> None:
    m = report["metrics"]
    lat = m["latency_ms"]
    print()
    print(f"=== Eval: {report['label']} ===")
    print(f"queries: {m['queries']}   runs/query: {report['config']['runs_per_query']}   "
          f"top_k: {report['config']['top_k']}   device: {', '.join(report['device_used']) or '?'}")
    print()
    print(f"{'Metric':<28}{'Value':>10}")
    print("-" * 38)
    print(f"{'Recall@1':<28}{m['recall_at_1']:>10.4f}")
    print(f"{'Recall@3':<28}{m['recall_at_3']:>10.4f}")
    print(f"{'Recall@5':<28}{m['recall_at_5']:>10.4f}")
    print(f"{'MRR':<28}{m['mrr']:>10.4f}")
    print(f"{'Latency avg (ms)':<28}{lat['avg']:>10.1f}")
    print(f"{'Latency p50 (ms)':<28}{lat['p50']:>10.1f}")
    print(f"{'Latency p95 (ms)':<28}{lat['p95']:>10.1f}")
    if m["section_match_rate"] is not None:
        print(f"{'Section match (info)':<28}{m['section_match_rate']:>10.4f}   "
              f"({m['section_match_applicable']} applicable)")
    print()

    print(f"{'Category':<16}{'N':>3}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'p50ms':>8}")
    print("-" * 59)
    for cat, cm in report["metrics_by_category"].items():
        print(f"{cat:<16}{cm['queries']:>3}{cm['recall_at_1']:>8.3f}"
              f"{cm['recall_at_3']:>8.3f}{cm['recall_at_5']:>8.3f}"
              f"{cm['mrr']:>8.3f}{cm['latency_ms']['p50']:>8.0f}")
    print()

    print(f"{'#':>3}  {'hit':>4}{'RR':>7}{'sec':>5}{'ms':>7}  query")
    print("-" * 78)
    for i, e in enumerate(report["per_query"], start=1):
        hit = str(e["hit_rank"]) if e["hit_rank"] else "-"
        sec = {True: "ok", False: "no", None: "-"}[e["section_match"]]
        q = e["query"] if len(e["query"]) <= 48 else e["query"][:45] + "..."
        print(f"{i:>3}  {hit:>4}{e['reciprocal_rank']:>7.2f}{sec:>5}"
              f"{e['total_ms_median']:>7.0f}  {q}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NPU semantic search quality")
    parser.add_argument("--queries", default=DEFAULT_QUERIES,
                        help="Path to queries JSON (default: eval/queries.json)")
    parser.add_argument("--out", default="",
                        help="Output JSON path (default: eval/results/<label>.json)")
    parser.add_argument("--vault-path", default="",
                        help="Vault root (default: engine resolution)")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default 5)")
    parser.add_argument("--device", default="auto",
                        help="Inference device: auto / NPU / GPU / CPU (default auto)")
    parser.add_argument("--model-dir", default="",
                        help="Model directory override (default: engine resolution)")
    parser.add_argument("--label", default="", help="Run label stored in the report")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per query, median total_ms is taken (default 3)")
    args = parser.parse_args()

    if not os.path.isfile(args.queries):
        raise SystemExit(f"queries file not found: {args.queries}")

    report = run_eval(args)

    out_path = args.out
    if not out_path:
        label = args.label or "eval"
        out_path = os.path.join(DEFAULT_RESULTS_DIR, f"{label}.json")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print_report(report)
    print(f"[eval] report written: {out_path}")


if __name__ == "__main__":
    main()
