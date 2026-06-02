#!/usr/bin/env python3
"""Butterfly Dream 轻量评测集 — 综合记忆评测框架。

融合业界最优实践：
  - recall@k / precision / F1（基线）
  - 遗忘曲线（LongMemEval, MEMOBENCH）
  - 联想记忆（MEMOBENCH）
  - 记忆编辑成功率（MemoryBench, MemGPT）
  - 同义改写鲁棒性（MemoryBench）
  - 延迟监控（MemGPT, AgentBench）
  - 大规模压力测试（LongMemEval）
  - 跨会话检索（LoCoMo, LoTa-Bench）

用法：
    python3 eval/run_eval.py                  # 全量跑
    python3 eval/run_eval.py --name "中文"    # 只跑名字含"中文"的场景
    python3 eval/run_eval.py --json           # JSON 输出（方便对比历史）

输出：
    per-scenario 的 recall@k / precision / F1 / 延迟 / 额外指标
    汇总平均 recall@k 和 precision
"""

import argparse
import json
import os
import sys
import time
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
    """Fraction of returned results that are relevant."""
    if not results:
        return 0.0
    relevant = 0
    for r in results:
        is_relevant = False
        for exp in expected:
            if _content_contains(r, exp):
                is_relevant = True
                break
        is_unexpected = any(_content_contains(r, u) for u in unexpected)
        if is_relevant and not is_unexpected:
            relevant += 1
    return relevant / len(results) if results else 0.0


def _apply_extra_setup(store: MemoryStore, extra_setup: list) -> dict:
    """Apply extra setup facts, returning a mapping of content→fact_id."""
    fact_id_map = {}
    for step in extra_setup:
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
    return fact_id_map


def _search_query(retriever: ThreeDimRetriever, store: MemoryStore,
                   query: str, qtype: str, params: dict) -> list:
    """Execute a single query and return raw results."""
    if qtype == "search":
        return retriever.search(
            query=query,
            scenario=params.get("scenario", "balanced"),
            persistent_only=params.get("persistent_only", False),
            limit=10,
        )
    elif qtype == "probe":
        return store.get_entity_facts(query)
    elif qtype == "timeline":
        return store.get_entity_timeline(
            query,
            min_importance=params.get("min_importance", 0.0),
        )
    elif qtype == "contradict":
        all_facts = store.list_facts(limit=100)
        pairs = []
        for i, fa in enumerate(all_facts):
            for fb in all_facts[i + 1:]:
                combined = MemoryStore._combine_fact_content(fa["content"], fb["content"])
                if "[冲突]" in combined or "[conflict]" in combined:
                    pairs.append((fa, fb))
        seen_ids = set()
        results = []
        for a, b in pairs:
            if a["fact_id"] not in seen_ids:
                results.append(a)
                seen_ids.add(a["fact_id"])
            if b["fact_id"] not in seen_ids:
                results.append(b)
                seen_ids.add(b["fact_id"])
        return results
    elif qtype == "dedup_check":
        count = store.count_facts()
        expected_count = params.get("expected_count", 0)
        ok = count == expected_count
        return [{"content": f"count={count} (expected={expected_count})", "_ok": ok}]
    elif qtype == "summarize":
        return [store.get_entity_summary(query)]
    return []


def run_scenario(scenario: dict, tmp_db: str) -> dict:
    """Run a single scenario; return metrics dict."""
    store = MemoryStore(tmp_db)
    retriever = ThreeDimRetriever(store)

    # Phase 1: Setup — insert facts
    for step in scenario.get("setup", []):
        store.add_fact(
            content=step["content"],
            category=step.get("category", "general"),
            tags=step.get("tags", ""),
            importance=step.get("importance", 5.0),
            is_persistent=step.get("is_persistent", False),
            dedup_threshold=step.get("dedup_threshold", 0.0),
            entities=step.get("entities"),
        )

    # Phase 2: Queries
    results_summary = []
    forgetting_curve = []  # For aggregation

    for q in scenario.get("queries", []):
        qtype = q.get("type", "search")
        params = q.get("params", {})
        expected = q.get("expected", [])
        unexpected = q.get("unexpected", [])

        # Apply extra setup before this query (forging curve distractors, edits, etc.)
        extra = q.get("extra_setup", [])
        if extra:
            _apply_extra_setup(store, extra)

        # Time the query
        t_start = time.perf_counter()
        raw_results = _search_query(retriever, store, q.get("query", ""), qtype, params)
        t_elapsed = round(time.perf_counter() - t_start, 4)

        # Compute recall_at_k
        recall_at_k = {}
        for k in q.get("recall_at_k", []):
            top_k = raw_results[:k]
            found = _any_expected_in_results(top_k, expected)
            if len(expected) > 0:
                recall = sum(found) / len(expected)
            else:
                recall = 0.0
            recall_at_k[str(k)] = round(recall, 4)

        # Precision (at top 10)
        top_10 = raw_results[:10]
        prec = round(_precision(top_10, expected, unexpected), 4)

        # F1 (harmonic mean of recall@5 and precision)
        rec5 = recall_at_k.get("5", recall_at_k.get("3", recall_at_k.get("1", 0.0)))
        f1 = round(2 * prec * rec5 / (prec + rec5 + 1e-10), 4)

        entry = {
            "query": q.get("query", ""),
            "type": qtype,
            "recall_at_k": recall_at_k,
            "precision": prec,
            "f1": f1,
            "latency_ms": round(t_elapsed * 1000, 2),
            "n_results": len(raw_results),
            "expected_found": sum(_any_expected_in_results(raw_results, expected)),
            "expected_total": len(expected),
        }

        # Handle dedup_check specially
        if qtype == "dedup_check":
            entry["dedup_ok"] = raw_results[0].get("_ok", False) if raw_results else False

        # Track forgetting curve data
        n_distractors = len(extra)
        if len(expected) > 0:
            recall_1 = recall_at_k.get("1", 0.0)
            forgetting_curve.append({
                "distractors": n_distractors,
                "recall@1": recall_1,
                "query": q.get("query", ""),
            })

        results_summary.append(entry)

    # Compute forgetting curve aggregate for this scenario
    fc_agg = None
    if len(forgetting_curve) >= 2:
        first_r1 = forgetting_curve[0]["recall@1"]
        last_r1 = forgetting_curve[-1]["recall@1"]
        decay_rate = round((first_r1 - last_r1) / max(first_r1, 0.001), 4) if first_r1 > 0 else 0.0
        fc_agg = {
            "points": forgetting_curve,
            "initial_recall@1": first_r1,
            "final_recall@1": last_r1,
            "decay_rate": decay_rate,
        }

    result = {
        "name": scenario["name"],
        "n_queries": len(scenario["queries"]),
        "queries": results_summary,
    }
    if fc_agg:
        result["forgetting_curve"] = fc_agg

    return result


def compute_summary(all_results: list) -> dict:
    """Compute aggregate summary across all scenarios."""
    total_recall = {}
    total_precision = []
    total_f1 = []
    total_latency = []
    n_q = 0

    # Forgetting curve across all scenarios
    all_fc_points = []
    # Paraphrase consistency: grouped by scenario
    paraphrase_scores = []

    for scenario in all_results:
        # Track forgetting curve points
        if "forgetting_curve" in scenario:
            all_fc_points.extend(scenario["forgetting_curve"]["points"])

        # Track paraphrase scenarios (name contains "改写" or "paraphrase")
        if "改写" in scenario["name"] or "paraphrase" in scenario["name"].lower():
            r1_values = [q["recall_at_k"].get("1", 0) for q in scenario["queries"] if q["type"] == "search"]
            if r1_values:
                mean_r1 = sum(r1_values) / len(r1_values)
                variance = sum((v - mean_r1) ** 2 for v in r1_values) / len(r1_values)
                paraphrase_scores.append({
                    "scenario": scenario["name"],
                    "mean_recall@1": round(mean_r1, 4),
                    "variance": round(variance, 4),
                    "consistency_score": round(1.0 - min(variance, 1.0), 4),
                })

        for q in scenario["queries"]:
            for k, v in q["recall_at_k"].items():
                total_recall[k] = total_recall.get(k, 0.0) + v
            total_precision.append(q["precision"])
            total_f1.append(q["f1"])
            total_latency.append(q["latency_ms"])
            n_q += 1

    avg_recall = {k: round(v / n_q, 4) for k, v in total_recall.items()}
    avg_precision = round(sum(total_precision) / n_q, 4) if n_q else 0.0
    avg_f1 = round(sum(total_f1) / n_q, 4) if n_q else 0.0
    avg_latency = round(sum(total_latency) / n_q, 2) if n_q else 0.0

    # Forgetting curve summary
    fc_summary = None
    if all_fc_points:
        # Group by distractor count
        by_dist = {}
        for p in all_fc_points:
            d = p["distractors"]
            if d not in by_dist:
                by_dist[d] = []
            by_dist[d].append(p["recall@1"])
        fc_summary = {
            "curve": [
                {"distractors": d, "avg_recall@1": round(sum(vs)/len(vs), 4)}
                for d, vs in sorted(by_dist.items())
            ],
            "total_points": len(all_fc_points),
        }

    return {
        "scenarios": len(all_results),
        "queries": n_q,
        "avg_recall_at_k": avg_recall,
        "avg_precision": avg_precision,
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "forgetting_curve": fc_summary,
        "paraphrase_consistency": paraphrase_scores if paraphrase_scores else None,
    }


def pretty_print(all_results: list, summary: dict):
    """Print a nicely formatted report."""
    print(f"\n{'='*75}")
    print(f"  🦋 Butterfly Dream 综合记忆评测报告")
    print(f"  场景数: {summary['scenarios']}  |  查询数: {summary['queries']}")
    print(f"{'='*75}\n")

    for scenario in all_results:
        print(f"  📁 {scenario['name']}")
        print(f"  {'─'*65}")
        for q in scenario["queries"]:
            rk = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(q["recall_at_k"].items()))
            lat = f"  ⏱{q['latency_ms']:6.1f}ms"
            tag = ""
            if q.get("dedup_ok") is not None:
                tag = f"  {'✅' if q['dedup_ok'] else '❌'}dedup"
            print(f"    [{q['type']:12s}] {q['query'][:35]:35s}  {rk}{lat}{tag}")
        print()

    # Forgetting curve
    if summary.get("forgetting_curve"):
        curve = summary["forgetting_curve"]["curve"]
        print(f"  🌀 遗忘曲线 (跨场景聚合)")
        print(f"  {'─'*65}")
        for pt in curve:
            bar_len = int(pt["avg_recall@1"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {pt['distractors']:3d} 干扰后  {bar}  {pt['avg_recall@1']:.3f}")
        print()

    # Paraphrase consistency
    if summary.get("paraphrase_consistency"):
        print(f"  🔄 同义改写鲁棒性")
        print(f"  {'─'*65}")
        for ps in summary["paraphrase_consistency"]:
            print(f"    {ps['scenario'][:35]:35s}  R@1均值={ps['mean_recall@1']:.3f}  "
                  f"一致性={ps['consistency_score']:.3f}  方差={ps['variance']:.4f}")
        print()

    print(f"  {'═'*65}")
    rk_str = "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(summary["avg_recall_at_k"].items()))
    print(f"  平均:  {rk_str}  P={summary['avg_precision']:.3f}  "
          f"F1={summary['avg_f1']:.3f}  ⏱{summary['avg_latency_ms']}ms")
    print(f"  {'═'*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream 综合评测集")
    parser.add_argument("--name", default="", help="只跑名字包含此关键词的场景")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--extra-scenarios", default="", help="额外的 JSON 场景文件路径")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    scenarios_path = script_dir / "scenarios.json"
    if not scenarios_path.exists():
        print(f"❌ 找不到 {scenarios_path}")
        sys.exit(1)

    with open(scenarios_path, encoding="utf-8") as f:
        all_scenarios = json.load(f)

    # Load extra scenarios if specified
    if args.extra_scenarios:
        extra_path = Path(args.extra_scenarios)
        if not extra_path.exists():
            print(f"❌ 找不到额外场景文件 {extra_path}")
            sys.exit(1)
        with open(extra_path, encoding="utf-8") as f:
            extra = json.load(f)
        all_scenarios.extend(extra)
        print(f"📎 加载额外场景: {extra_path} ({len(extra)} 个)")

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

    # Compute summary
    summary = compute_summary(all_results)

    # Output
    output = {"summary": summary, "results": all_results}

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    pretty_print(all_results, summary)


if __name__ == "__main__":
    main()
