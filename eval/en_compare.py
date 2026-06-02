#!/usr/bin/env python3
"""🦋 Butterfly Dream 自然语言查询检索对比测试。

在相同 9 条事实集上，进行英文和中文自然语言查询验证。
与「原版 Holographic」（Hermes 主线 holographic 插件）的对比数据记录如下：

对比基准 (同 9 条事实, 8 英文 + 8 中文查询):

| 场景 | 原版 Holographic | Butterfly Dream |
|:----|:---------------:|:--------------:|
| 🔤 英文自然语言查询 (8 条) | **0/8 (0%)** | **6/8 (75%)** |
| 🀄 中文自然语言查询 (8 条) | **0/8 (0%)** | **8/8 (100%)** |

原版 Holographic 失败根因: FTS5 AND 语义 + 停用词问题导致全灭。
Butterfly Dream 的 OR 展开 + 前缀匹配 + jieba 分词是解决这个问题的关键。

用法:
    python3 eval/en_compare.py                          # 跑全部
    python3 eval/en_compare.py --language en             # 只跑英文
    python3 eval/en_compare.py --language zh             # 只跑中文
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.retrieval import ThreeDimRetriever


# ═══════════════════════════════════════════════════════════════
# 测试数据 — 9 条标准事实（3 EN + 3 ZH + 3 混合）
# ═══════════════════════════════════════════════════════════════

FACTS = [
    # 英文事实
    {"content": "User prefers VS Code for Python development", "tags": "editor,python", "category": "user_pref", "importance": 6},
    {"content": "User switched from VS Code to Neovim last week", "tags": "editor,change", "category": "user_pref", "importance": 8},
    {"content": "User knows Python and Rust programming languages", "tags": "skills", "category": "user_pref", "importance": 5},
    # 中文事实
    {"content": "用户居住在东京，在Google涩谷办公室工作", "tags": "工作,生活", "category": "user_pref", "importance": 8},
    {"content": "用户养了一只3岁的橘猫叫小胖，最喜欢吃鱼", "tags": "宠物,猫咪", "category": "user_pref", "importance": 6},
    {"content": "项目使用FastAPI后端、React前端、PostgreSQL数据库", "tags": "技术栈", "category": "project", "importance": 7},
    # 混合事实
    {"content": "用户常用 VS Code 和 Neovim 写 Python 和 Rust", "tags": "editor,language", "category": "user_pref", "importance": 7},
    {"content": "用户计划把单体应用迁移到 microservices, 用 Docker + Kubernetes 部署", "tags": "migration,cloud", "category": "project", "importance": 9},
    {"content": "TikTok推荐系统用 Go 和 Rust, 部署在 AWS EKS 上", "tags": "job,tech", "category": "project", "importance": 8},
]

EN_QUERIES = [
    ("Which editor does the user prefer for Python?", ["VS Code"]),
    ("What editor did the user switch to recently?", ["Neovim"]),
    ("What programming languages does the user know?", ["Python", "Rust"]),
    ("Where does the user live and work?", ["东京", "Google"]),
    ("What pet does the user have?", ["橘猫", "小胖"]),
    ("What tech stack does the project use?", ["FastAPI", "React", "PostgreSQL"]),
    ("What tools does the user use for development?", ["VS Code", "Neovim", "Python", "Rust"]),
    ("What is the migration plan for the monolith?", ["microservices", "Docker", "Kubernetes"]),
]

ZH_QUERIES = [
    ("用户用什么编辑器写Python？", ["VS Code"]),
    ("用户最近切换到了什么编辑器？", ["Neovim"]),
    ("用户会哪些编程语言？", ["Python", "Rust"]),
    ("用户住在哪里在哪里工作？", ["东京", "Google"]),
    ("用户养了什么宠物？", ["橘猫", "小胖"]),
    ("项目用了什么技术栈？", ["FastAPI", "React", "PostgreSQL"]),
    ("用户开发用什么工具和语言？", ["VS Code", "Neovim", "Python", "Rust"]),
    ("用户的迁移计划是什么？", ["microservices", "Docker", "Kubernetes"]),
]


def run_test(queries: list) -> dict:
    """Run Butterfly Dream on a set of queries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    config = {
        "db_path": db_path,
        "llm_extract": False,
        "trivial_filter": False,
        "reflection": False,
    }
    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id="en-compare")

    # Seed facts
    for f in FACTS:
        provider._handle_add(f)

    # Query
    retriever = ThreeDimRetriever(provider._store)
    results = []
    for query_str, expected in queries:
        raw_results = retriever.search(query=query_str, scenario="chat", limit=5)
        n_found = 0
        for ec in expected:
            if any(ec.lower() in (r.get("content") or "").lower() for r in raw_results):
                n_found += 1
        ok = n_found >= len(expected)
        results.append({
            "query": query_str,
            "expected": expected,
            "found": n_found,
            "expected_total": len(expected),
            "ok": ok,
            "n_results": len(raw_results),
            "top_hits": [(r.get("content") or "")[:60] for r in raw_results[:3]],
        })

    passed = sum(1 for r in results if r["ok"])
    os.unlink(db_path)
    return {"passed": passed, "total": len(results), "results": results}


def print_report(label: str, data: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  通过: {data['passed']}/{data['total']}")
    for r in data["results"]:
        mark = "✅" if r["ok"] else "❌"
        print(f"    {mark}  {r['query'][:45]:45s}  → {r['found']}/{r['expected_total']}")


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream 自然语言查询检索对比")
    parser.add_argument("--language", default="both", choices=["en", "zh", "both"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    all_results = {}

    if args.language in ("en", "both"):
        en_results = run_test(EN_QUERIES)
        all_results["en"] = en_results
        if not args.json:
            print_report("🔤 英文自然语言查询 (对比基准: 原版 Holographic 0/8)", en_results)

    if args.language in ("zh", "both"):
        zh_results = run_test(ZH_QUERIES)
        all_results["zh"] = zh_results
        if not args.json:
            print_report("🀄 中文自然语言查询 (对比基准: 原版 Holographic 0/8)", zh_results)

    if not args.json:
        print(f"\n{'='*60}")
        print("  对比基准 (同 9 条事实):")
        print(f"  {'场景':<35s} {'原版 Holographic':<18s} {'Butterfly Dream':<15s}")
        print(f"  {'─'*65}")
        for lang, label, prev in [("en", "英文查询", "0/8"), ("zh", "中文查询", "0/8")]:
            if lang in all_results:
                d = all_results[lang]
                print(f"  {label:<35s} {prev:<18s} {d['passed']}/{d['total']}")
        print(f"{'='*60}")
        print("  原版 Holographic 失败根因: FTS5 AND 语义 → 停用词全灭")
        print("  Butterfly Dream 改进: OR 展开 + 前缀匹配 + jieba 分词 → 任意词召回 + 三维精排")
        print(f"{'='*60}\n")

    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
