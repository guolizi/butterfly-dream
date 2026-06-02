#!/usr/bin/env python3
"""Butterfly Dream 轻量评测集 — recall@k / precision 量化指标。

用法：
    python3 eval/run_eval.py                  # 全量跑
    python3 eval/run_eval.py --name "中文"    # 只跑名字含"中文"的场景
    python3 eval/run_eval.py --json           # JSON 输出（方便对比历史）

输出：
    per-scenario 的 recall@k / precision / F1
    汇总平均 recall@k 和 precision
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# 确保能找到插件
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from butterfly_dream.store import MemoryStore
from butterfly_dream.retrieval import ThreeDimRetriever


def _fact_dict(fact) -> dict:
    """Normalize a fact result (could be dict or sqlite3.Row-like)."""
    if isinstance(fact, dict):
        return fact
    return {key: fact[key] for key in fact.keys()}


def _content_contains(fact: dict, substring: str) -> bool:
    return substring.lower() in (fact.get("content") or "").lower()


def _any_expected_in_results(results: list, expected_substrings: list[str]) -> list[bool]:
    """For each expected substring, check if it appears in ANY result."""
    found = []
    for exp in expected_substrings:
        found.append(any(_content_contains(r, exp) for r in results))
    return found


def _precision(results: list, expected: list[str], unexpected: list[str]) -> float:
    """Fraction of returned results that are relevant (match expected, not unexpected)."""
    if not results:
        return 0.0
    relevant = 0
    for r in results:
        is_relevant = False
        for exp in expected:
            if _content_contains(r, exp):
                is_relevant = True
                break
        if not is_relevant:
            pass  # irrelevant
        # Check if it's one of the explicitly unexpected items
        is_unexpected = any(_content_contains(r, u) for u in unexpected)
        if is_relevant and not is_unexpected:
            relevant += 1
        elif not is_relevant and not is_unexpected:
            # Neutral result - neither expected nor unexpected
            # Don't count for or against precision (or count as irrelevant)
            pass
    return relevant / len(results) if results else 0.0


def run_scenario(scenario: dict, tmp_db: str) -> dict:
    """Run a single scenario; return metrics dict."""
    store = MemoryStore(tmp_db)
    retriever = ThreeDimRetriever(store)

    # Phase 1: Setup — insert facts
    fact_id_map = {}  # content -> fact_id
    for step in scenario.get("setup", []):
        result = store.add_fact(
            content=step["content"],
            category=step.get("category", "general"),
            tags=step.get("tags", ""),
            importance=step.get("importance", 5.0),
            is_persistent=step.get("is_persistent", False),
            dedup_threshold=step.get("dedup_threshold", 0.0),
            entities=step.get("entities"),
        )
        fact_id_map[step["content"]] = result["fact_id"]

    # Phase 2: Queries
    results_summary = []
    for q in scenario.get("queries", []):
        qtype = q.get("type", "search")
        params = q.get("params", {})
        expected = q.get("expected", [])
        unexpected = q.get("unexpected", [])

        raw_results = []

        if qtype == "search":
            raw_results = retriever.search(
                query=q["query"],
                scenario=params.get("scenario", "balanced"),
                persistent_only=params.get("persistent_only", False),
                limit=10,
            )
        elif qtype == "probe":
            raw_results = store.get_entity_facts(q["query"])
        elif qtype == "timeline":
            raw_results = store.get_entity_timeline(
                q["query"],
                min_importance=params.get("min_importance", 0.0),
            )
        elif qtype == "contradict":
            # 遍历所有事实两两检测矛盾
            all_facts = store.list_facts(limit=100)

            pairs = []
            for i, fa in enumerate(all_facts):
                for fb in all_facts[i + 1:]:
                    combined = MemoryStore._combine_fact_content(fa["content"], fb["content"])
                    if "[冲突]" in combined or "[conflict]" in combined:
                        pairs.append((fa, fb))
            # 返回冲突对中的所有事实（去重）
            seen_ids = set()
            for a, b in pairs:
                if a["fact_id"] not in seen_ids:
                    raw_results.append(a)
                    seen_ids.add(a["fact_id"])
                if b["fact_id"] not in seen_ids:
                    raw_results.append(b)
                    seen_ids.add(b["fact_id"])
        elif qtype == "dedup_check":
            # 检查 setup 后事实总条数是否符合预期
            count = store.count_facts()
            expected_count = expected.get("exact_count", 0) if isinstance(expected, dict) else len(expected)
            ok = count == expected_count
            raw_results = [{"content": f"count={count} (expected={expected_count})"}]
            if ok:
                # Pretend expected found as a clean result
                expected = [f"count={expected_count}"]
                unexpected = []
        elif qtype == "summarize":
            raw_results = [store.get_entity_summary(q["query"])]

        # Compute metrics
        recall_at_k = {}
        for k in q.get("recall_at_k", []):
            top_k = raw_results[:k]
            found = _any_expected_in_results(top_k, expected)
            if len(expected) > 0:
                recall = sum(found) / len(expected)
            else:
                recall = 0.0
            recall_at_k[str(k)] = round(recall, 4)

        # Precision (at full results, capped at top 10)
        top_10 = raw_results[:10]
        prec = round(_precision(top_10, expected, unexpected), 4)

        # F1 (harmonic mean of recall@5 and precision)
        rec5 = recall_at_k.get("5", recall_at_k.get("3", recall_at_k.get("1", 0.0)))
        f1 = round(2 * prec * rec5 / (prec + rec5 + 1e-10), 4)

        results_summary.append({
            "query": q.get("query", ""),
            "type": qtype,
            "recall_at_k": recall_at_k,
            "precision": prec,
            "f1": f1,
            "n_results": len(raw_results),
            "expected_found": sum(_any_expected_in_results(raw_results, expected)),
            "expected_total": len(expected),
        })

    return {
        "name": scenario["name"],
        "n_queries": len(scenario["queries"]),
        "queries": results_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream 评测集")
    parser.add_argument("--name", default="", help="只跑名字包含此关键词的场景")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    scenarios_path = script_dir / "scenarios.json"
    if not scenarios_path.exists():
        print(f"❌ 找不到 {scenarios_path}")
        sys.exit(1)

    with open(scenarios_path, encoding="utf-8") as f:
        all_scenarios = json.load(f)

    if args.name:
        all_scenarios = [s for s in all_scenarios if args.name in s["name"]]
        if not all_scenarios:
            print(f"❌ 没有名字包含 '{args.name}' 的场景")
            sys.exit(1)

    # Run all scenarios
    all_results = []
    for scenario in all_scenarios:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db = tmp.name
        try:
            result = run_scenario(scenario, tmp_db)
            all_results.append(result)
        finally:
            os.unlink(tmp_db)

    # Compute averages
    total_recall = {}
    total_precision = []
    total_f1 = []
    n_q = 0
    for scenario in all_results:
        for q in scenario["queries"]:
            for k, v in q["recall_at_k"].items():
                total_recall[k] = total_recall.get(k, 0.0) + v
            total_precision.append(q["precision"])
            total_f1.append(q["f1"])
            n_q += 1

    avg_recall = {k: round(v / n_q, 4) for k, v in total_recall.items()}
    avg_precision = round(sum(total_precision) / n_q, 4) if n_q else 0.0
    avg_f1 = round(sum(total_f1) / n_q, 4) if n_q else 0.0

    # Output
    output = {
        "summary": {
            "scenarios": len(all_results),
            "queries": n_q,
            "avg_recall_at_k": avg_recall,
            "avg_precision": avg_precision,
            "avg_f1": avg_f1,
        },
        "results": all_results,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Pretty print
    print(f"\n{'='*70}")
    print(f"  🦋 Butterfly Dream 评测报告")
    print(f"  场景数: {len(all_results)}  |  查询数: {n_q}")
    print(f"{'='*70}\n")

    for scenario in all_results:
        print(f"  📁 {scenario['name']}")
        print(f"  {'─'*60}")
        for q in scenario["queries"]:
            rk = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(q["recall_at_k"].items()))
            print(f"    [{q['type']:12s}] {q['query'][:40]:40s}  {rk}  P={q['precision']:.3f}  F1={q['f1']:.3f}")
        print()

    print(f"  {'═'*60}")
    rk_str = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(avg_recall.items()))
    print(f"  平均:  {rk_str}  P={avg_precision:.3f}  F1={avg_f1:.3f}")
    print(f"  {'═'*60}\n")


if __name__ == "__main__":
    main()
