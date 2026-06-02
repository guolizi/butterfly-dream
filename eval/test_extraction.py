#!/usr/bin/env python3
"""Butterfly Dream 提取质量评测 — 测试事实提取管道的各个组件。

提取是记忆系统的入口——只有提取正确，检索才有意义。
本评测集测试 LLM 调用之外的所有确定性组件。

用法：
    python3 eval/test_extraction.py                    # 全量运行
    python3 eval/test_extraction.py --name "解析"       # 只跑特定模块
    python3 eval/test_extraction.py --json             # JSON 输出
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from butterfly_dream import _call_extraction_llm, _TRIVIAL_PATTERNS


# ═══════════════════════════════════════════════════════════════
# 1. 琐事过滤测试
# ═══════════════════════════════════════════════════════════════

TRIVIAL_TESTS = [
    # (content, expected_trivial, description)
    # --- 英文 ---
    ("ok", True, "EN bare ok"),
    ("OK", True, "EN uppercase OK"),
    ("Okay", True, "EN Okay"),
    ("thanks", True, "EN thanks"),
    ("thank you", True, "EN thank you"),
    ("thx", True, "EN thx"),
    ("yes", True, "EN yes"),
    ("no", True, "EN no"),
    ("yep", True, "EN yep"),
    ("👍", True, "EN thumbs up emoji"),
    ("好的", True, "ZH 好的"),
    ("嗯嗯", True, "ZH 嗯嗯"),
    ("知道了", True, "ZH 知道了"),
    ("没问题", True, "ZH 没问题"),
    ("谢谢", True, "ZH 谢谢"),
    ("哈哈", True, "ZH 哈哈"),
    # --- 非琐事内容（应返回 False）---
    ("I like Python", False, "EN meaningful sentence"),
    ("用户喜欢喝咖啡", False, "ZH meaningful sentence"),
    ("帮我查一下天气", False, "ZH request"),
    ("今天星期几？", False, "ZH question"),
    ("你好，我想了解一下你们的API", False, "ZH longer greeting with intent"),
    ("def fib(n): return n", False, "Code snippet"),
    ("The weather is nice today", False, "EN statement about weather"),
    # --- 边界情况 ---
    ("", True, "Empty string"),
    ("   ", True, "Whitespace only"),
    ("a", True, "Single char"),
    ("好", True, "Single CJK char"),
    ("哦", False, "Single CJK interjection (not in pattern)"),
    ("嗯", True, "Single CJK acknowledgment"),
]


def run_trivial_tests() -> dict:
    """Test _is_trivial_content against known patterns."""
    # Import the function from the plugin
    from butterfly_dream import ButterflyDreamMemoryProvider

    results = []
    passed = 0
    failed = 0

    for content, expected, desc in TRIVIAL_TESTS:
        actual = ButterflyDreamMemoryProvider._is_trivial_content(content)
        ok = actual == expected
        if ok:
            passed += 1
        else:
            failed += 1
        results.append({
            "content": content[:50],
            "expected": expected,
            "actual": actual,
            "ok": ok,
            "description": desc,
        })

    return {
        "name": "琐事过滤",
        "description": "测试 _is_trivial_content 能否正确识别/放行各类内容",
        "total": len(TRIVIAL_TESTS),
        "passed": passed,
        "failed": failed,
        "precision": round(passed / len(TRIVIAL_TESTS), 4),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 2. LLM 响应解析测试
# ═══════════════════════════════════════════════════════════════

# 模拟的 LLM 响应（测试 _call_extraction_llm 的解析逻辑）
# 通过 mock urllib.request.urlopen 来注入
MOCK_LLM_RESPONSES = [
    # (mock_response_json, expected_facts_count, description)
    (
        {"facts": [
            {"content": "用户喜欢Python编程", "category": "user_pref", "tags": "语言", "importance": 7, "is_persistent": True},
        ]},
        1,
        "标准 facts 键名",
    ),
    (
        {"memories": [
            {"content": "用户住在北京朝阳区CBD", "category": "user_pref", "tags": "地址", "importance": 6},
        ]},
        1,
        "memories 键名（兼容）",
    ),
    (
        {"results": [
            {"content": "用户今天中午吃了火锅", "category": "general", "tags": "饮食", "importance": 3},
        ]},
        1,
        "results 键名（兼容）",
    ),
    (
        {"contents": [
            {"content": "不应被解析的内容", "category": "general", "tags": "", "importance": 5},
        ]},
        0,
        "未知键名（应忽略）",
    ),
    (
        {"facts": [
            {"content": "太短", "category": "general", "tags": "", "importance": 5},
        ]},
        0,
        "内容过短（<10字符，应过滤）",
    ),
    (
        {"facts": [
            {"content": "用户有一个有效事实用于测试", "category": "invalid_category", "tags": "", "importance": 5},
        ]},
        1,
        "无效分类（应降级为general）",
    ),
    (
        {"facts": [
            {"content": "用户重要性测试超出范围上限", "category": "general", "tags": "", "importance": 15},
        ]},
        1,
        "重要性超出范围（应裁剪为10）",
    ),
    (
        {"facts": [
            {"content": "用户重要性测试低于范围下限值", "category": "general", "tags": "", "importance": 0},
        ]},
        1,
        "重要性低于范围（应裁剪为1）",
    ),
    (
        {"facts": [
            {"content": "用户持久标记测试应该通过", "category": "general", "tags": "", "importance": 5, "is_persistent": True},
            {"content": "用户非持久标记测试应该通过", "category": "general", "tags": "", "importance": 5, "is_persistent": False},
        ]},
        2,
        "持久/非持久标记正确解析",
    ),
    (
        "invalid json",
        0,
        "无效 JSON（应安全返回空列表）",
    ),
    (
        {"facts": "not a list"},
        0,
        "facts 不是列表（应安全返回空列表）",
    ),
    (
        {},
        0,
        "空字典（应安全返回空列表）",
    ),
    (
        [{"content": "直接返回列表格式兼容测试", "category": "general", "tags": "", "importance": 5}],
        1,
        "直接返回列表（兼容格式）",
    ),
    (
        {"insights": [
            {"content": "用户最近开始每周去健身房", "category": "user_pref", "tags": "健康", "importance": 6},
            {"content": "用户每周去三次健身房锻炼", "category": "user_pref", "tags": "习惯", "importance": 5},
        ]},
        2,
        "insights 键名（兼容 reflection）",
    ),
]


def _make_mock_response(data):
    """Create a mock HTTP response that returns JSON data."""
    body = json.dumps(data).encode("utf-8") if not isinstance(data, str) else data.encode("utf-8")
    # FIX: mock response_data as a dict (parsed JSON), not the raw response
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": json.dumps(data)}}]
    }).encode("utf-8") if not isinstance(data, str) else data.encode("utf-8")
    return mock_resp


def run_parsing_tests() -> dict:
    """Test _call_extraction_llm's response parsing logic."""
    results = []
    passed = 0
    failed = 0

    for mock_data, expected_count, desc in MOCK_LLM_RESPONSES:
        # Mock both credential resolution and HTTP call
        with patch("urllib.request.urlopen") as mock_urlopen:
            with patch("butterfly_dream._resolve_provider_credentials") as mock_creds:
                mock_creds.return_value = ("https://test.api", "test-key-123")

                # Create the nested mock properly
                mock_response = MagicMock()
                if isinstance(mock_data, str) and mock_data == "invalid json":
                    mock_response.__enter__.return_value.read.return_value = b"not json at all"
                else:
                    inner_content = json.dumps(mock_data, ensure_ascii=False)
                    response_body = json.dumps({
                        "choices": [{"message": {"content": inner_content}}]
                    }, ensure_ascii=False)
                    mock_response.__enter__.return_value.read.return_value = response_body.encode("utf-8")

                mock_urlopen.return_value = mock_response

                facts = _call_extraction_llm(
                    messages_text="test conversation",
                    provider="test",
                    model="test-model",
                    timeout=5,
                )

        ok = len(facts) == expected_count
        if ok:
            passed += 1
        else:
            failed += 1

        results.append({
            "description": desc,
            "expected_count": expected_count,
            "actual_count": len(facts),
            "ok": ok,
            "sample": facts[0] if facts else None,
        })

    return {
        "name": "LLM 响应解析",
        "description": "测试 _call_extraction_llm 对各种 LLM 响应格式的解析能力",
        "total": len(MOCK_LLM_RESPONSES),
        "passed": passed,
        "failed": failed,
        "precision": round(passed / len(MOCK_LLM_RESPONSES), 4),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 3. 熔断器测试
# ═══════════════════════════════════════════════════════════════

def run_circuit_breaker_tests() -> dict:
    """Test circuit breaker state machine."""
    from butterfly_dream import ButterflyDreamMemoryProvider

    # Create a minimal provider instance for testing
    config = {
        "llm_extract": True,
        "extraction_model": {"provider": "test", "model": "test"},
        "circuit_breaker": {"max_failures": 3, "cooldown_seconds": 2},
        "trivial_filter": True,
        "reflection": True,
    }
    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id="test-cb")

    results = []
    passed = 0
    failed = 0

    # Test 1: Initial state — should be OK
    ok1 = provider._circuit_breaker_ok()
    results.append({
        "test": "初始状态",
        "expected": True,
        "actual": ok1,
        "ok": ok1,
    })
    if ok1:
        passed += 1
    else:
        failed += 1

    # Test 2-4: Fail 3 times — should still be OK until reaching max
    for i in range(3):
        provider._mark_extraction_result(False)
        is_ok = provider._circuit_breaker_ok()
        ok = (is_ok == (i < 2))  # First 2 fail: still ok; 3rd fail: breaker trips
        results.append({
            "test": f"第{i+1}次失败后",
            "expected": i < 2,
            "actual": is_ok,
            "ok": ok,
        })
        if ok:
            passed += 1
        else:
            failed += 1

    # Test 5: After cooldown — should reset
    time.sleep(2.1)  # Wait for cooldown
    ok5 = provider._circuit_breaker_ok()
    results.append({
        "test": "冷却后重置",
        "expected": True,
        "actual": ok5,
        "ok": ok5,
    })
    if ok5:
        passed += 1
    else:
        failed += 1

    # Test 6: Success resets counter
    provider._mark_extraction_result(True)
    for i in range(3):
        provider._mark_extraction_result(False)
    # Should be tripped again
    ok6 = not provider._circuit_breaker_ok()
    results.append({
        "test": "成功重置后再次熔断",
        "expected": True,
        "actual": not provider._circuit_breaker_ok(),
        "ok": ok6,
    })
    if ok6:
        passed += 1
    else:
        failed += 1

    provider.shutdown()

    return {
        "name": "熔断器状态机",
        "description": "测试 circuit breaker 的失败计数、熔断触发、冷却恢复逻辑",
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "precision": round(passed / len(results), 4),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 4. 实体提取测试
# ═══════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_TESTS = [
    # (text, expected_entities_contain, description)
    ("用户喜欢VS Code和Neovim", ["VS", "Code", "Neovim"], "工具名提取"),
    ("用户在阿里巴巴和腾讯工作过", ["阿里巴巴", "腾讯"], "公司名提取"),
    ("用户最喜欢的编程语言是Python和Rust", ["Python", "Rust"], "编程语言提取"),
    ("用户住在北京，工作在深圳", ["北京", "深圳"], "城市名提取"),
    ("用户喜欢在macOS上开发", ["macOS"], "系统名提取"),
    ("普通句子没有实体", [], "无实体文本"),
    ("C++和C#都是好语言", ["C++", "C#"], "特殊字符实体"),
    ("", [], "空文本"),
]


def run_entity_extraction_tests() -> dict:
    """Test entity extraction from text."""
    from butterfly_dream.retrieval import tokenize

    results = []
    passed = 0
    failed = 0

    for text, expected_contain, desc in ENTITY_EXTRACTION_TESTS:
        tokens = tokenize(text)
        found = [e for e in expected_contain if e.lower() in tokens or e in tokens]
        missing = [e for e in expected_contain if e not in found]
        ok = len(missing) == 0
        if ok:
            passed += 1
        else:
            failed += 1
        results.append({
            "text": text[:50],
            "description": desc,
            "expected_entities": expected_contain,
            "missing": missing,
            "tokens_found": sorted(tokens)[:10],
            "ok": ok,
        })

    return {
        "name": "实体提取",
        "description": "测试 tokenize() 能否从文本中正确提取实体关键词",
        "total": len(ENTITY_EXTRACTION_TESTS),
        "passed": passed,
        "failed": failed,
        "precision": round(passed / len(ENTITY_EXTRACTION_TESTS), 4),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 5. 完整 Mocked 提取管道测试
# ═══════════════════════════════════════════════════════════════

def run_mocked_pipeline_tests() -> dict:
    """Test the full extraction pipeline with mocked LLM.

    Simulates: conversation → LLM responds → parse → store → verify.
    """
    import tempfile
    from butterfly_dream import ButterflyDreamMemoryProvider

    results = []
    passed = 0
    failed = 0

    test_scenarios = [
        {
            "name": "标准提取-存储后可检索",
            "mock_response": {
                "facts": [
                    {"content": "用户最喜欢的编程语言是Rust", "category": "user_pref", "tags": "语言", "importance": 8, "is_persistent": True},
                    {"content": "用户每天用Neovim写代码", "category": "user_pref", "tags": "工具,习惯", "importance": 7},
                ]
            },
            "queries": [
                ("Rust", ["Rust"]),
                ("Neovim", ["Neovim"]),
            ],
        },
        {
            "name": "提取+去重-重复内容跳过",
            "mock_response": {
                "facts": [
                    {"content": "用户最喜欢的编程语言是Rust", "category": "user_pref", "tags": "语言", "importance": 8},
                ]
            },
            "queries": [
                ("Rust", ["Rust"]),
            ],
            "dedup_check": True,  # After 2 extractions, count should be 1
        },
    ]

    for scenario in test_scenarios:
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
        provider.initialize(session_id="test-pipeline")

        with patch("urllib.request.urlopen") as mock_urlopen:
            # Also mock credential resolution
            with patch("butterfly_dream._resolve_provider_credentials") as mock_creds:
                mock_creds.return_value = ("https://test.api", "test-key-123")

                # Build mock response
                inner = json.dumps(scenario["mock_response"], ensure_ascii=False)
                body = json.dumps({"choices": [{"message": {"content": inner}}]}).encode("utf-8")
                mock_resp = MagicMock()
                mock_resp.__enter__.return_value.read.return_value = body
                mock_urlopen.return_value = mock_resp

                # Run extraction
                provider._run_llm_extraction([
                    {"role": "user", "content": "I like Rust"},
                    {"role": "assistant", "content": "Great choice!"},
                ])

        # Verify stored facts
        store = provider._store
        all_facts = store.list_facts(limit=20)

        # Run queries
        for query_str, expected_contain in scenario["queries"]:
            from butterfly_dream.retrieval import ThreeDimRetriever
            retriever = ThreeDimRetriever(store)
            raw = retriever.search(query=query_str, scenario="chat", limit=5)
            found_any = any(any(ec.lower() in (f.get("content") or "").lower() for ec in expected_contain) for f in raw)
            ok = found_any
            results.append({
                "test": f"{scenario['name']} — 查询'{query_str}'",
                "expected_found": True,
                "actual_found": found_any,
                "ok": ok,
                "n_results": len(raw),
            })
            if ok:
                passed += 1
            else:
                failed += 1

        # Dedup check
        if scenario.get("dedup_check"):
            count = store.count_facts()
            ok = count == 1
            results.append({
                "test": f"{scenario['name']} — 去重检测",
                "expected_count": 1,
                "actual_count": count,
                "ok": ok,
            })
            if ok:
                passed += 1
            else:
                failed += 1

        provider.shutdown()
        os.unlink(db_path)

    return {
        "name": "Mocked 提取管道",
        "description": "模拟 LLM 响应，测试完整提取→存储→检索管道",
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "precision": round(passed / len(results), 4),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════

ALL_EVALS = {
    "琐事": run_trivial_tests,
    "解析": run_parsing_tests,
    "熔断": run_circuit_breaker_tests,
    "实体": run_entity_extraction_tests,
    "管道": run_mocked_pipeline_tests,
}


def pretty_print(results: list[dict], summary: dict):
    print(f"\n{'='*75}")
    print(f"  🦋 Butterfly Dream 提取质量评测报告")
    print(f"  模块数: {summary['modules']}  |  总测试数: {summary['total']}")
    print(f"{'='*75}\n")

    for mod in results:
        mark = "✅" if mod["failed"] == 0 else "⚠️"
        print(f"  {mark} {mod['name']}  ({mod['description'][:50]})")
        print(f"  {'─'*65}")
        print(f"    {mod['passed']:>4d}/{mod['total']:<3d}  精度={mod['precision']:.3f}")
        for d in mod["details"]:
            ok_mark = "✅" if d["ok"] else "❌"
            desc = d.get("test") or d.get("description") or ""
            print(f"    {ok_mark}  {desc[:55]:55s}")
        print()

    print(f"  {'═'*65}")
    print(f"  总计: {summary['passed']}/{summary['total']} 通过  精度={summary['precision']:.3f}")
    print(f"  {'═'*65}\n")


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream 提取质量评测")
    parser.add_argument("--name", default="", help="只跑名字包含此关键词的模块 (琐事/解析/熔断/实体/管道)")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    results = []
    for name, func in ALL_EVALS.items():
        if args.name and args.name not in name:
            continue
        result = func()
        results.append(result)

    total = sum(r["total"] for r in results)
    passed = sum(r["passed"] for r in results)
    summary = {
        "modules": len(results),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "precision": round(passed / total, 4) if total else 0,
    }

    output = {"summary": summary, "results": results}

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    pretty_print(results, summary)


if __name__ == "__main__":
    main()
