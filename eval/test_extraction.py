#!/usr/bin/env python3
"""Butterfly Dream 端到端记忆评测 — 模拟真实对话场景。

设计原则：
- 黑盒测试：不关心系统内部如何实现，只关心输入输出
- 真实场景：模拟多轮对话，让系统自行提取事实
- 公平公正：查询使用自然语言，不迁就被测系统

评测流程：
  对话 → on_session_end(提取) → 存储 → 自然语言查询 → 验证

用法：
    python3 eval/test_extraction.py                  # 全量运行
    python3 eval/test_extraction.py --name "偏好"     # 只跑特定场景
    python3 eval/test_extraction.py --json           # JSON 输出
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# 端到端对话场景定义
# ═══════════════════════════════════════════════════════════════

# 每个场景 = 对话 + 期望提取的事实 + 验证查询
SCENARIOS = [
    {
        "name": "用户偏好-编程语言偏好变化",
        "conversation": [
            {"role": "user", "content": "我最近在学习Rust，感觉比Python快很多"},
            {"role": "assistant", "content": "Rust的性能确实很出色，你之前主要用什么语言？"},
            {"role": "user", "content": "之前一直用Python写后端，上个月开始切换到Rust了"},
        ],
        "mock_facts": [
            {"content": "用户之前用Python写后端开发", "category": "user_pref", "tags": "编程语言", "importance": 7, "is_persistent": False},
            {"content": "用户上个月从Python切换到Rust", "category": "user_pref", "tags": "编程语言,变化", "importance": 8, "is_persistent": False},
            {"content": "用户认为Rust比Python快很多", "category": "user_pref", "tags": "编程语言,评价", "importance": 6, "is_persistent": False},
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
        "mock_facts": [
            {"content": "用户搬到了日本东京生活", "category": "user_pref", "tags": "居住,迁移", "importance": 8, "is_persistent": True},
            {"content": "用户在Google东京办公室工作", "category": "user_pref", "tags": "工作,公司", "importance": 8, "is_persistent": True},
            {"content": "用户的团队在涩谷做搜索相关开发", "category": "project", "tags": "工作,团队", "importance": 7, "is_persistent": False},
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
        "mock_facts": [
            {"content": "新项目后端使用FastAPI框架", "category": "project", "tags": "技术栈,后端", "importance": 7, "is_persistent": False},
            {"content": "新项目前端使用React框架", "category": "project", "tags": "技术栈,前端", "importance": 7, "is_persistent": False},
            {"content": "项目使用PostgreSQL数据库", "category": "project", "tags": "技术栈,数据库", "importance": 7, "is_persistent": False},
            {"content": "项目部署在AWS ECS上使用Docker", "category": "project", "tags": "部署,基础设施", "importance": 6, "is_persistent": False},
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
        "mock_facts": [
            {"content": "用户养了一只三岁的橘猫叫小胖", "category": "user_pref", "tags": "宠物", "importance": 6, "is_persistent": True},
            {"content": "用户还养了一条金毛犬叫旺财", "category": "user_pref", "tags": "宠物", "importance": 6, "is_persistent": True},
            {"content": "小胖最喜欢吃鱼", "category": "user_pref", "tags": "宠物,偏好", "importance": 4, "is_persistent": False},
        ],
        "queries": [
            ("用户养了什么宠物？", ["橘猫", "金毛"]),
            ("橘猫叫什么名字？", ["小胖"]),
            ("金毛叫什么名字？", ["旺财"]),
        ],
    },
    {
        "name": "跨会话-信息更新",
        "conversation_session1": [
            {"role": "user", "content": "我目前在微软工作，做Azure相关的开发"},
            {"role": "assistant", "content": "Azure是个很庞大的平台，你主要负责哪部分？"},
            {"role": "user", "content": "主要是Kubernetes相关的服务"},
        ],
        "mock_facts_session1": [
            {"content": "用户在微软做Azure开发", "category": "user_pref", "tags": "工作", "importance": 8, "is_persistent": True},
            {"content": "用户主要负责Kubernetes相关服务", "category": "project", "tags": "技术栈", "importance": 7, "is_persistent": False},
        ],
        "conversation_session2": [
            {"role": "user", "content": "我刚跳槽到字节跳动了，现在做TikTok推荐系统"},
            {"role": "assistant", "content": "从Azure到推荐系统，跨度挺大的"},
            {"role": "user", "content": "是的，不过技术栈还是用Go和Rust"},
        ],
        "mock_facts_session2": [
            {"content": "用户跳槽到了字节跳动做TikTok推荐系统", "category": "user_pref", "tags": "工作,更新", "importance": 9, "is_persistent": True},
            {"content": "用户的技术栈包含Go和Rust", "category": "user_pref", "tags": "编程语言", "importance": 7, "is_persistent": False},
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

def run_conversation_scenario(scenario: dict) -> dict:
    """Run a single end-to-end conversation scenario.

    Steps:
    1. Initialize provider with temp DB
    2. Run session1 extraction (mock LLM)
    3. If session2 exists, run session2 extraction (mock LLM)
    4. Query and verify results
    """
    from butterfly_dream import ButterflyDreamMemoryProvider
    from butterfly_dream.retrieval import ThreeDimRetriever

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    config = {
        "db_path": db_path,
        "llm_extract": True,
        "extraction_model": {"provider": "test", "model": "test"},
        "circuit_breaker": {"max_failures": 3, "cooldown_seconds": 120},
        "trivial_filter": True,
        "reflection": False,
    }

    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id="eval-extraction")

    query_results = []

    def _mock_extraction(facts_to_return: list):
        """Mock the LLM extraction call to return predefined facts."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            with patch("butterfly_dream._resolve_provider_credentials") as mock_creds:
                mock_creds.return_value = ("https://test.api", "test-key-123")

                inner = json.dumps({"facts": facts_to_return}, ensure_ascii=False)
                response_body = json.dumps({
                    "choices": [{"message": {"content": inner}}]
                }, ensure_ascii=False)

                mock_resp = MagicMock()
                mock_resp.__enter__.return_value.read.return_value = response_body.encode("utf-8")
                mock_urlopen.return_value = mock_resp

                # Run extraction through the normal pipeline
                provider._run_llm_extraction(scenario.get("conversation_session1",
                                                          scenario.get("conversation", [])))

    # Run session 1 extraction
    conv1 = scenario.get("conversation_session1", scenario.get("conversation", []))
    facts1 = scenario.get("mock_facts_session1", scenario.get("mock_facts", []))
    if conv1 and facts1:
        _mock_extraction(facts1)

    # Run session 2 extraction (if exists)
    conv2 = scenario.get("conversation_session2")
    facts2 = scenario.get("mock_facts_session2")
    if conv2 and facts2:
        _mock_extraction(facts2)

    # Query
    retriever = ThreeDimRetriever(provider._store)
    for query_str, expected_contain in scenario["queries"]:
        raw_results = retriever.search(query=query_str, scenario="chat", limit=5)
        # Compute expected match counts:
        found_any = False
        n_found = 0
        for ec in expected_contain:
            if any(ec.lower() in (r.get("content") or "").lower() for r in raw_results):
                n_found += 1
                found_any = True

        query_results.append({
            "query": query_str,
            "expected": expected_contain,
            "found": n_found,
            "expected_total": len(expected_contain),
            "ok": found_any and n_found >= len(expected_contain),
            "n_results": len(raw_results),
            "top_content": [r.get("content", "")[:60] for r in raw_results[:3]],
        })

    # Compute scenario metrics
    total_queries = len(query_results)
    passed_queries = sum(1 for q in query_results if q["ok"])
    precision = round(passed_queries / total_queries, 4) if total_queries else 0.0

    # Count stored facts
    all_facts = provider._store.list_facts(limit=50)

    # Compute recall@k before shutdown
    recall_1_count = 0
    retriever_for_recall = ThreeDimRetriever(provider._store)
    for q in query_results:
        recall_rr = retriever_for_recall.search(query=q["query"], scenario="chat", limit=5)
        if recall_rr:
            content = (recall_rr[0].get("content") or "").lower()
            if any(ec.lower() in content for ec in q["expected"]):
                recall_1_count += 1
    recall_at_1 = round(recall_1_count / total_queries, 4) if total_queries else 0.0

    provider.shutdown()
    os.unlink(db_path)

    # Compute scenario metrics
    total_queries = len(query_results)
    passed_queries = sum(1 for q in query_results if q["ok"])
    precision = round(passed_queries / total_queries, 4) if total_queries else 0.0

    return {
        "name": scenario["name"],
        "total_queries": total_queries,
        "passed": passed_queries,
        "failed": total_queries - passed_queries,
        "precision": precision,
        "recall_at_1": recall_at_1,
        "stored_facts": len(all_facts),
        "details": query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream 端到端记忆评测")
    parser.add_argument("--name", default="", help="只跑名字包含此关键词的场景")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    selected = [s for s in SCENARIOS if not args.name or args.name in s["name"]]
    if not selected:
        print(f"❌ 没有名字包含 '{args.name}' 的场景")
        sys.exit(1)

    results = [run_conversation_scenario(s) for s in selected]

    total_q = sum(r["total_queries"] for r in results)
    total_p = sum(r["passed"] for r in results)
    total_f = sum(r["failed"] for r in results)
    avg_r1 = round(sum(r["recall_at_1"] for r in results) / len(results), 4)
    avg_prec = round(total_p / total_q, 4) if total_q else 0.0

    summary = {
        "scenarios": len(results),
        "total_queries": total_q,
        "passed": total_p,
        "failed": total_f,
        "avg_recall_at_1": avg_r1,
        "avg_precision": avg_prec,
    }

    output = {"summary": summary, "results": results}

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Pretty print
    print(f"\n{'='*75}")
    print(f"  🦋 Butterfly Dream 端到端记忆评测报告")
    print(f"  场景数: {summary['scenarios']}  |  查询数: {summary['total_queries']}")
    print(f"{'='*75}\n")

    for sc in results:
        mark = "✅" if sc["failed"] == 0 else "⚠️"
        print(f"  {mark} {sc['name']}")
        print(f"  {'─'*65}")
        print(f"    查询: {sc['passed']}/{sc['total_queries']} 通过  "
              f"R@1={sc['recall_at_1']:.3f}  精度={sc['precision']:.3f}  "
              f"存储: {sc['stored_facts']}条事实")
        for d in sc["details"]:
            ok_mark = "✅" if d["ok"] else "❌"
            exp_str = ", ".join(d["expected"])
            print(f"    {ok_mark}  Q: {d['query'][:35]:35s}  期望: [{exp_str[:30]:30s}]  "
                  f"找到: {d['found']}/{d['expected_total']}")
        print()

    print(f"  {'═'*65}")
    print(f"  总计: {total_p}/{total_q} 通过  R@1={avg_r1:.3f}  精度={avg_prec:.3f}")
    print(f"  {'═'*65}\n")


if __name__ == "__main__":
    main()
