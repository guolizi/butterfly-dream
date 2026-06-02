#!/usr/bin/env python3
"""🦋 Butterfly Dream 端到端提取评测 — 使用真实 LLM 调用。

设计原则：
- 黑盒端到端：真实对话 → 实际 LLM 提取 → 存储 → 自然语言查询 → 验证
- 可比性：支持不同 model/provider 对比，观察 LLM 对提取质量的影响
- 公平公正：查询使用自然语言，不迁就被测系统

评测流程：
  对话 → _run_llm_extraction(真实 LLM) → 存储事实 → 自然语言查询 → 验证结果

用法：
    python3 eval/test_extraction.py                              # 默认用 deepseek-v4-flash
    python3 eval/test_extraction.py --model deepseek-v4-flash     # 指定模型
    python3 eval/test_extraction.py --provider openai --model gpt-4o-mini
    python3 eval/test_extraction.py --compare                     # 对比多个模型
    python3 eval/test_extraction.py --name "偏好"                 # 只跑特定场景
    python3 eval/test_extraction.py --json                        # JSON 输出
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# 自动加载 Hermes .env（独立运行时需要 API key）
# ═══════════════════════════════════════════════════════════════

def _load_hermes_env():
    """Load ~/.hermes/.env into os.environ if not already set."""
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key not in os.environ:  # don't override existing env
                # Strip surrounding quotes if any
                val = val.strip().strip("\"'").strip()
                os.environ[key] = val

_load_hermes_env()

from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.retrieval import ThreeDimRetriever
from butterfly_dream.__init__ import _call_extraction_llm


# ═══════════════════════════════════════════════════════════════
# 对话场景定义
# ═══════════════════════════════════════════════════════════════

# golden_facts: 预期应提取出的「黄金事实」（用于参考统计，不做严格匹配）
# queries: (自然语言问句, [期望答案关键词])

SCENARIOS = [
    {
        "name": "用户偏好-编程语言偏好变化",
        "conversation": [
            {"role": "user", "content": "我最近在学习Rust，感觉比Python快很多"},
            {"role": "assistant", "content": "Rust的性能确实很出色，你之前主要用什么语言？"},
            {"role": "user", "content": "之前一直用Python写后端，上个月开始切换到Rust了"},
        ],
        "golden_facts": [
            "用户之前用Python写后端开发",
            "用户上个月从Python切换到Rust",
            "用户认为Rust比Python快很多",
        ],
        "queries": [
            ("用户之前用什么语言写后端？", ["Python"]),
            ("用户现在用什么语言？", ["Rust"]),
            ("用户为什么切换语言？", ["快"]),
        ],
    },
    {
        "name": "用户信息-工作与居住地",
        "conversation": [
            {"role": "user", "content": "我刚搬到东京，在Google开始新工作了"},
            {"role": "assistant", "content": "恭喜！东京是个很不错的城市。Google在东京有办公室吗？"},
            {"role": "user", "content": "有的，我在涩谷的办公室工作，团队主要做搜索相关"},
        ],
        "golden_facts": [
            "用户搬到了日本东京生活",
            "用户在Google东京办公室工作",
            "用户的团队在涩谷做搜索相关开发",
        ],
        "queries": [
            ("用户住在哪个城市？", ["东京"]),
            ("用户在哪个公司工作？", ["Google"]),
            ("用户的团队做什么？", ["搜索"]),
        ],
    },
    {
        "name": "项目信息-技术栈",
        "conversation": [
            {"role": "user", "content": "我们新项目准备用FastAPI写后端，前端用React"},
            {"role": "assistant", "content": "很好的技术选型。数据库选了什么？"},
            {"role": "user", "content": "PostgreSQL，部署在AWS上，用Docker管理"},
            {"role": "assistant", "content": "你们用Kubernetes吗？"},
            {"role": "user", "content": "暂时不用K8s，直接用ECS部署，以后可能会迁移"},
        ],
        "golden_facts": [
            "新项目后端使用FastAPI框架",
            "新项目前端使用React框架",
            "项目使用PostgreSQL数据库",
            "项目部署在AWS ECS上使用Docker",
        ],
        "queries": [
            ("项目后端用什么框架？", ["FastAPI"]),
            ("项目用什么数据库？", ["PostgreSQL"]),
            ("项目部署在哪里？", ["AWS"]),
        ],
    },
    {
        "name": "多轮对话-偏好积累",
        "conversation": [
            {"role": "user", "content": "我养了一只橘猫，超级可爱"},
            {"role": "assistant", "content": "橘猫很亲人！叫什么名字？"},
            {"role": "user", "content": "叫小胖，已经三岁了，最喜欢吃鱼"},
            {"role": "assistant", "content": "哈哈，橘猫果然爱吃。你还有其他宠物吗？"},
            {"role": "user", "content": "还有一条金毛，叫旺财，它们相处得很好"},
        ],
        "golden_facts": [
            "用户养了一只三岁的橘猫叫小胖",
            "用户还养了一条金毛犬叫旺财",
            "小胖最喜欢吃鱼",
        ],
        "queries": [
            ("用户养了什么宠物？", ["橘猫", "金毛"]),
            ("橘猫叫什么名字？", ["小胖"]),
            ("金毛叫什么名字？", ["旺财"]),
        ],
    },
    {
        "name": "跨会话-信息更新",
        "sessions": [
            {
                "conversation": [
                    {"role": "user", "content": "我目前在微软工作，做Azure相关的开发"},
                    {"role": "assistant", "content": "Azure是个很庞大的平台，你主要负责哪部分？"},
                    {"role": "user", "content": "主要是Kubernetes相关的服务"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "我刚跳槽到字节跳动了，现在做TikTok推荐系统"},
                    {"role": "assistant", "content": "从Azure到推荐系统，跨度挺大的"},
                    {"role": "user", "content": "是的，不过技术栈还是用Go和Rust"},
                ],
            },
        ],
        "golden_facts": [
            "用户在微软做Azure开发",
            "用户主要负责Kubernetes相关服务",
            "用户跳槽到了字节跳动做TikTok推荐系统",
            "用户的技术栈包含Go和Rust",
        ],
        "queries": [
            ("用户现在在哪个公司？", ["字节跳动"]),
            ("用户以前在哪个公司？", ["微软"]),
            ("用户用什么编程语言？", ["Go", "Rust"]),
            ("用户现在做什么方向？", ["TikTok", "推荐系统"]),
        ],
    },
]


# ═══════════════════════════════════════════════════════════════
# 评测运行器
# ═══════════════════════════════════════════════════════════════

def run_scenario(scenario: dict, provider: str, model: str) -> dict:
    """Run a single end-to-end extraction scenario with real LLM.

    1. Initialize ButterflyDreamMemoryProvider with temp DB
    2. Call real LLM extraction via _run_llm_extraction for each session
    3. Store extracted facts
    4. Query with natural language and verify results
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    config = {
        "db_path": db_path,
        "llm_extract": True,
        "extraction_model": {"provider": provider, "model": model},
        "circuit_breaker": {"max_failures": 10, "cooldown_seconds": 120},
        "trivial_filter": False,   # 关闭琐事过滤，让 LLM 自行判断
        "reflection": False,
    }

    provider_inst = ButterflyDreamMemoryProvider(config)
    provider_inst.initialize(session_id="eval-extraction")

    all_extracted = []
    total_ms = 0.0
    extraction_results = []

    # Determine sessions to process
    sessions = scenario.get("sessions", [{"conversation": scenario.get("conversation", [])}])

    for idx, session in enumerate(sessions, 1):
        conv = session["conversation"]
        t0 = time.perf_counter()
        extracted = provider_inst._run_llm_extraction(conv)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        extraction_results.append({
            "session": idx,
            "n_facts": len(extracted),
            "latency_ms": round(elapsed_ms, 2),
        })
        all_extracted.extend(extracted)

    # Get full fact details from DB (includes category, trust_score, etc.)
    all_facts = provider_inst._store.list_facts(limit=50) if provider_inst._store else []

    # Query and verify
    retriever = ThreeDimRetriever(provider_inst._store)
    query_results = []

    for query_str, expected_contain in scenario.get("queries", []):
        raw_results = retriever.search(query=query_str, scenario="chat", limit=5)
        n_found = 0
        for ec in expected_contain:
            if any(ec.lower() in (r.get("content") or "").lower() for r in raw_results):
                n_found += 1

        query_results.append({
            "query": query_str,
            "expected": expected_contain,
            "found": n_found,
            "expected_total": len(expected_contain),
            "ok": n_found >= len(expected_contain),
            "n_results": len(raw_results),
            "top_content": [r.get("content", "")[:60] for r in raw_results[:3]],
        })

    total_queries = len(query_results)
    passed_queries = sum(1 for q in query_results if q["ok"])

    # Compute recall@1
    recall_1_count = 0
    for q in query_results:
        rr = retriever.search(query=q["query"], scenario="chat", limit=5)
        if rr:
            content = (rr[0].get("content") or "").lower()
            if any(ec.lower() in content for ec in q["expected"]):
                recall_1_count += 1

    recall_at_1 = round(recall_1_count / total_queries, 4) if total_queries else 0.0
    precision = round(passed_queries / total_queries, 4) if total_queries else 0.0

    # Stats on extracted facts (use full DB records for category/importance)
    categories = {}
    importance_sum = 0
    for f in all_facts:
        cat = f.get("category", "general") or "general"
        categories[cat] = categories.get(cat, 0) + 1
        importance_sum += float(f.get("importance", 5.0))

    n_total = len(all_facts)
    avg_importance = round(importance_sum / n_total, 2) if n_total else 0.0

    # Build per-session fact details from DB
    # (we can't match IDs perfectly, so show all extracted facts together)
    fact_details = [
        {"content": f["content"][:80],
         "category": f.get("category", "general"),
         "importance": float(f.get("importance", 5)),
         "is_persistent": bool(f.get("is_persistent", False))}
        for f in all_facts
    ]

    provider_inst.shutdown()
    os.unlink(db_path)

    return {
        "name": scenario["name"],
        "total_queries": total_queries,
        "passed": passed_queries,
        "failed": total_queries - passed_queries,
        "recall_at_1": recall_at_1,
        "precision": precision,
        "extraction": {
            "total_facts": n_total,
            "avg_latency_ms": round(total_ms / len(sessions), 2) if sessions else 0,
            "total_latency_ms": round(total_ms, 2),
            "categories": categories,
            "avg_importance": avg_importance,
            "golden_facts_count": len(scenario.get("golden_facts", [])),
            "sessions": extraction_results,
            "facts": fact_details,
        },
        "details": query_results,
    }


# ═══════════════════════════════════════════════════════════════
# 模型对比 / 报告输出
# ═══════════════════════════════════════════════════════════════

_DEFAULT_MODELS = [
    {"provider": "deepseek", "model": "deepseek-v4-flash"},
    # 添加其他模型时在这里扩展，例如：
    # {"provider": "deepseek", "model": "deepseek-v4-pro"},
    # {"provider": "openai", "model": "gpt-4o-mini"},
]


def run_all_scenarios(provider: str, model: str, name_filter: str = "") -> list:
    """Run all matching scenarios with given provider/model."""
    selected = [s for s in SCENARIOS if not name_filter or name_filter in s["name"]]
    if not selected:
        print(f"❌ 没有名字包含 '{name_filter}' 的场景")
        sys.exit(1)
    return [run_scenario(s, provider, model) for s in selected]


def print_report(results: list, provider: str, model: str):
    """Pretty-print evaluation report."""
    total_q = sum(r["total_queries"] for r in results)
    total_p = sum(r["passed"] for r in results)
    total_f = sum(r["failed"] for r in results)
    avg_r1 = round(sum(r["recall_at_1"] for r in results) / len(results), 4)
    avg_prec = round(total_p / total_q, 4) if total_q else 0.0

    total_facts = sum(r["extraction"]["total_facts"] for r in results)
    avg_lat = round(sum(r["extraction"]["avg_latency_ms"] for r in results) / len(results), 2)
    avg_imp = round(sum(r["extraction"]["avg_importance"] for r in results) / len(results), 2)

    print(f"\n{'='*75}")
    print(f"  🦋 Butterfly Dream 端到端提取评测报告")
    print(f"  🤖 模型: {provider}/{model}")
    print(f"  场景数: {len(results)}  |  查询数: {total_q}  |  提取事实: {total_facts}")
    print(f"{'='*75}\n")

    for sc in results:
        mark = "✅" if sc["failed"] == 0 else "⚠️"
        ex = sc["extraction"]
        print(f"  {mark} {sc['name']}")
        print(f"  {'─'*65}")
        print(f"    查询: {sc['passed']}/{sc['total_queries']} 通过  "
              f"R@1={sc['recall_at_1']:.3f}  精度={sc['precision']:.3f}")
        print(f"    提取: {ex['total_facts']}条事实  "
              f"(黄金预期: {ex['golden_facts_count']}条)  "
              f"平均重要性: {ex['avg_importance']}")
        print(f"    延迟: {ex['total_latency_ms']}ms总  "
              f"{ex['avg_latency_ms']}ms/轮")
        if ex["categories"]:
            cats = ", ".join(f"{k}={v}" for k, v in sorted(ex["categories"].items()))
            print(f"    分类: {cats}")
        for d in sc["details"]:
            ok_mark = "✅" if d["ok"] else "❌"
            exp_str = ", ".join(d["expected"])
            print(f"    {ok_mark}  Q: {d['query'][:35]:35s}  "
                  f"期望: [{exp_str[:30]:30s}]  找到: {d['found']}/{d['expected_total']}")

        # Show extracted facts
        if ex.get("facts"):
            print(f"    📝  LLM 提取的事实:")
            for f in ex["facts"]:
                imp = int(f.get("importance", 5))
                imp_bar = "★" * imp + "☆" * (10 - imp)
                cat = f.get("category", "general") or "general"
                print(f"      [{cat}] {f['content'][:55]:55s}  "
                      f"重要={imp_bar}  {'🔒' if f.get('is_persistent') else '🔄'}")
        print()

    print(f"  {'═'*65}")
    print(f"  总计: {total_p}/{total_q} 通过  "
          f"R@1={avg_r1:.3f}  精度={avg_prec:.3f}")
    print(f"  事实: {total_facts}条总  "
          f"平均延迟: {avg_lat}ms/轮  "
          f"平均重要性: {avg_imp}")
    print(f"  {'═'*65}\n")


def compare_models(models: list, name_filter: str = ""):
    """Run all scenarios with multiple models and compare results."""
    all_results = {}
    for cfg in models:
        p, m = cfg["provider"], cfg["model"]
        print(f"\n  ⏳ 运行 {p}/{m} ...")
        results = run_all_scenarios(p, m, name_filter)
        all_results[f"{p}/{m}"] = results

    # Comparison table
    print(f"\n{'='*75}")
    print(f"  📊 模型对比总结")
    print(f"{'='*75}")
    print(f"  {'模型':<30s} {'查询通过':>10s} {'R@1':>7s} {'精度':>7s} {'事实':>7s} {'延迟ms':>8s}")
    print(f"  {'─'*70}")
    for label, results in all_results.items():
        total_q = sum(r["total_queries"] for r in results)
        total_p = sum(r["passed"] for r in results)
        avg_r1 = round(sum(r["recall_at_1"] for r in results) / len(results), 4)
        avg_prec = round(total_p / total_q, 4) if total_q else 0.0
        total_facts = sum(r["extraction"]["total_facts"] for r in results)
        avg_lat = round(sum(r["extraction"]["avg_latency_ms"] for r in results) / len(results), 2)
        print(f"  {label:<30s} {total_p}/{total_q:<7s} {avg_r1:<7.3f} {avg_prec:<7.3f} "
              f"{total_facts:<7d} {avg_lat:<8.2f}")
    print(f"  {'─'*70}\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🦋 Butterfly Dream 端到端提取评测 — 使用真实 LLM"
    )
    parser.add_argument("--name", default="", help="只跑名字包含此关键词的场景")
    parser.add_argument("--provider", default="deepseek", help="LLM provider (默认: deepseek)")
    parser.add_argument("--model", default="deepseek-v4-flash", help="LLM model (默认: deepseek-v4-flash)")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--compare", action="store_true",
                        help="对比多个模型（忽略 --provider/--model，使用内置列表）")
    parser.add_argument("--models", default="",
                        help="对比模式自定义模型列表，格式: provider1/model1,provider2/model2")
    parser.add_argument("--scenarios", default="",
                        help="从 JSON 文件加载外部场景（覆盖内置场景）")
    parser.add_argument("--length-group", action="store_true",
                        help="按长度分组输出统计（需外部场景有 _meta.length 字段）")
    args = parser.parse_args()

    # Load external scenarios if specified
    global SCENARIOS
    if args.scenarios:
        with open(args.scenarios, encoding="utf-8") as f:
            SCENARIOS = json.load(f)
        if not isinstance(SCENARIOS, list):
            print("❌ 外部场景文件必须是 JSON 数组")
            sys.exit(1)
        print(f"📦 从 {args.scenarios} 加载了 {len(SCENARIOS)} 个外部场景")

    if args.compare:
        models = _DEFAULT_MODELS
        if args.models:
            custom = []
            for entry in args.models.split(","):
                parts = entry.strip().split("/")
                if len(parts) == 2:
                    custom.append({"provider": parts[0], "model": parts[1]})
            if custom:
                models = custom
        compare_models(models, args.name)
        return

    results = run_all_scenarios(args.provider, args.model, args.name)

    if args.json:
        output = {
            "meta": {"provider": args.provider, "model": args.model},
            "results": results,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print_report(results, args.provider, args.model)


if __name__ == "__main__":
    main()
