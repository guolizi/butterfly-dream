#!/usr/bin/env python3
"""🦋 生成多长度、多场景的对话评测集。

用 LLM 生成从短到长的真实对话，覆盖生活和工作场景，
每个对话内嵌特定事实，用于端到端记忆提取评测。

用法：
    python3 eval/gen_long_scenarios.py                                      # 生成全套
    python3 eval/gen_long_scenarios.py --length medium                      # 只生成长度
    python3 eval/gen_long_scenarios.py --domain life --count 2              # 指定场景
    python3 eval/gen_long_scenarios.py --output eval/scenarios_gen.json     # 输出路径
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════
# 自动加载 Hermes .env
# ═══════════════════════════════════════════════════════════════

def _load_env():
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.is_file():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key not in os.environ:
                    val = val.strip().strip("\"'").strip()
                    os.environ[key] = val

_load_env()


# ═══════════════════════════════════════════════════════════════
# LLM 调用
# ═══════════════════════════════════════════════════════════════

_PROVIDER = "deepseek"
_MODEL = "deepseek-v4-flash"


def _call_llm(messages: list, timeout: int = 120) -> str:
    """Call the LLM and return raw response content."""
    import urllib.request
    import urllib.error

    prefix = _PROVIDER.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    base_url = os.environ.get(f"{prefix}_BASE_URL", "https://api.deepseek.com/v1")

    if not api_key:
        print("❌ No API key found — set DEEPSEEK_API_KEY in env or ~/.hermes/.env")
        sys.exit(1)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": _MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════
# 场景生成配置
# ═══════════════════════════════════════════════════════════════

LENGTH_CONFIGS = [
    {
        "id": "short",
        "turns": "3-5 turns (6-10 messages)",
        "turns_desc": "short, 3-5 turns",
        "chars": "200-500 characters total",
        "n_facts": "2-3",
        "n_queries": "2-3",
    },
    {
        "id": "medium",
        "turns": "10-20 turns (20-40 messages)",
        "turns_desc": "medium-length, 10-20 turns",
        "chars": "2000-5000 characters total",
        "n_facts": "5-8",
        "n_queries": "4-6",
    },
    {
        "id": "long",
        "turns": "30-50 turns (60-100 messages)",
        "turns_desc": "long, 30-50 turns",
        "chars": "10000-30000 characters total",
        "n_facts": "10-15",
        "n_queries": "6-10",
    },
    {
        "id": "very-long",
        "turns": "50-100 turns (100-200 messages)",
        "turns_desc": "very long, 50-100 turns",
        "chars": "30000-80000 characters total",
        "n_facts": "15-25",
        "n_queries": "8-12",
    },
]

DOMAINS = [
    {
        "id": "work-software",
        "desc": "Software development project — architecture discussions, code reviews, deployment planning, bug debugging across multiple days/weeks",
        "fact_types": "tech stack choices, architecture decisions, user preferences for tools/languages, project milestones, team processes",
    },
    {
        "id": "work-product",
        "desc": "Product management — feature planning, user research, sprint retrospectives, stakeholder meetings, roadmap discussions",
        "fact_types": "product decisions, user personas, feature priorities, timeline commitments, stakeholder preferences",
    },
    {
        "id": "work-data",
        "desc": "Data analysis / ML project — dataset discussions, model selection, experiment results, deployment monitoring, metric tracking",
        "fact_types": "data sources, model choices, evaluation metrics, infrastructure preferences, experiment outcomes",
    },
    {
        "id": "life-travel",
        "desc": "Travel planning — destination research, itinerary building, budget discussions, accommodation options, activity preferences spanning multiple conversations",
        "fact_types": "travel preferences, budget constraints, must-visit attractions, accommodation style, dietary needs",
    },
    {
        "id": "life-health",
        "desc": "Health and fitness journey — workout routines, diet planning, progress tracking, medical consultations, habit formation",
        "fact_types": "fitness goals, dietary restrictions, medical conditions, exercise preferences, progress milestones",
    },
    {
        "id": "life-learning",
        "desc": "Learning journey — course selection, study progress, project practice, certification planning, skill development across weeks/months",
        "fact_types": "learning goals, course preferences, study schedule, completed milestones, preferred learning style",
    },
    {
        "id": "life-entertainment",
        "desc": "Entertainment and hobbies — movie/game/book recommendations, music preferences, hobby discussions, community activities",
        "fact_types": "entertainment preferences, favorite genres, subscription services, hobby details, social activities",
    },
    {
        "id": "life-home",
        "desc": "Home and daily life — home improvement, cooking, pet care, shopping decisions, family arrangements",
        "fact_types": "home preferences, cooking habits, pet details, shopping patterns, family info",
    },
]


# ═══════════════════════════════════════════════════════════════
# 场景生成 Prompt
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a data generation assistant. Create realistic, natural conversations between a User (human) and an Assistant (AI) that contain embedded facts for testing a memory extraction system.

Your output must be valid JSON with this exact structure:
```json
{
  "name": "short descriptive name of the scenario",
  "conversation": [
    {"role": "user", "content": "turn 1 message..."},
    {"role": "assistant", "content": "turn 1 response..."},
    ...
  ],
  "golden_facts": [
    "fact statement 1",
    "fact statement 2",
    ...
  ],
  "queries": [
    ["natural language query 1", ["expected_keyword1", "expected_keyword2"]],
    ["natural language query 2", ["expected_keyword1"]],
    ...
  ]
}
```

CRITICAL RULES:
1. Make conversations authentic — user should sound like a real person, not an LLM
2. For multi-turn conversations, spread the facts across the entire conversation naturally
3. Golden facts should be the EXACT information embedded — specific and concrete
4. Queries should be natural questions a human would ask, using natural language (MANDATORY: queries MUST be in Chinese, matching the extracted facts' language)
5. Each query expects finding the fact content via keyword substring match — choose keywords that uniquely appear in the extracted facts
6. **Use realistic mixed language**: The core conversation is in Chinese (中文), but naturally embed English technical/professional terms where appropriate — e.g. "这个API的response用Pydantic做validation", "我们试试用Gradient Boosting跑一下", "部署在AWS ECS上用Docker". English terms should be real professional vocabulary, not random English words.
7. Assitant responses should provide information or ask follow-ups that elicit user details"""


# ═══════════════════════════════════════════════════════════════
# 生成器
# ═══════════════════════════════════════════════════════════════

_EN_SYSTEM_PROMPT = """You are a data generation assistant. Create realistic, natural conversations between a User (human) and an Assistant (AI) that contain embedded facts for testing a memory extraction system.

Your output must be valid JSON with this exact structure:
```json
{
  "name": "short descriptive name of the scenario",
  "conversation": [...],
  "golden_facts": [...],
  "queries": [["natural language query", ["expected_keyword1"]], ...]
}
```

CRITICAL RULES:
1. Make conversations authentic in English
2. Spread facts naturally across the conversation
3. Golden facts must be specific and concrete, in English
4. Queries MUST be in English, matching the extracted facts' language
5. Choose keywords that uniquely appear in extracted facts
6. Assistant responses should be helpful and elicit user details"""


def generate_scenario(length_cfg: dict, domain_cfg: dict, language: str = "zh", retries: int = 3) -> dict:
    """Generate one scenario using the LLM."""
    user_prompt = f"""Generate a realistic conversation for a memory extraction benchmark.

Length: {length_cfg['turns_desc']} ({length_cfg['chars']})
Domain: {domain_cfg['desc']}
Types of facts to embed: {domain_cfg['fact_types']}
Number of facts: {length_cfg['n_facts']}
Number of queries: {length_cfg['n_queries']}

The conversation should feel natural and unfold over multiple turns. Don't make it obvious that facts are being placed — they should emerge naturally from the dialogue.

Output the JSON object with "name", "conversation", "golden_facts", and "queries" keys."""

    for attempt in range(retries):
        print(f"  ⏳ Attempt {attempt + 1}/{retries}...", end=" ", flush=True)
        t0 = time.perf_counter()
        sys_prompt = _SYSTEM_PROMPT if language == "zh" else _EN_SYSTEM_PROMPT
        raw = _call_llm([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
        elapsed = time.perf_counter() - t0
        print(f"{elapsed:.1f}s")

        if not raw:
            print("  ⚠️ Empty response, retrying...")
            continue

        # Extract JSON from response (might be in code blocks)
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw.strip()

        try:
            scenario = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find JSON by scanning for { }
            brace_start = json_str.find("{")
            brace_end = json_str.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    scenario = json.loads(json_str[brace_start:brace_end + 1])
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ JSON parse failed: {e}")
                    continue
            else:
                print("  ⚠️ No JSON found in response")
                continue

        # Validate required keys
        missing = [k for k in ("name", "conversation", "golden_facts", "queries") if k not in scenario]
        if missing:
            print(f"  ⚠️ Missing keys: {missing}")
            continue

        # Validate conversation format
        if not isinstance(scenario["conversation"], list):
            print("  ⚠️ conversation is not a list")
            continue
        for msg in scenario["conversation"]:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                print("  ⚠️ Invalid message format")
                continue

        # Add metadata
        scenario["_meta"] = {
            "length": length_cfg["id"],
            "domain": domain_cfg["id"],
            "n_messages": len(scenario["conversation"]),
            "total_chars": sum(len(m["content"]) for m in scenario["conversation"]),
            "n_golden": len(scenario["golden_facts"]),
            "n_queries": len(scenario["queries"]),
        }
        scenario["name"] = f"[{length_cfg['id']}][{domain_cfg['id']}] {scenario['name']}"
        return scenario

    print(f"  ❌ Failed after {retries} retries")
    return None


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="生成多长度多场景的对话评测集")
    parser.add_argument("--length", default="", choices=["short", "medium", "long", "very-long"],
                        help="只生成特定长度")
    parser.add_argument("--domain", default="", help="只生成特定领域 (work-software, life-travel, etc.)")
    parser.add_argument("--count", type=int, default=2, help="每个配置生成几个场景 (default: 2)")
    parser.add_argument("--language", default="zh",
                        choices=["zh", "en"],
                        help="对话语言: zh=中文(默认), en=纯英文")
    parser.add_argument("--output", default="eval/scenarios_gen.json",
                        help="输出路径 (default: eval/scenarios_gen.json)")
    parser.add_argument("--dry-run", action="store_true", help="只打印配置列表，不生成")
    args = parser.parse_args()

    lengths = [c for c in LENGTH_CONFIGS if not args.length or c["id"] == args.length]
    domains = [d for d in DOMAINS if not args.domain or d["id"] == args.domain]

    if not lengths:
        print(f"❌ Unknown length: {args.length}")
        sys.exit(1)
    if not domains:
        print(f"❌ Unknown domain: {args.domain}")
        sys.exit(1)

    # Display plan
    total = len(lengths) * len(domains) * args.count
    print(f"📋 生成计划: {len(lengths)} 长度 × {len(domains)} 场景 × {args.count} 个 = {total} 场景")
    for lc in lengths:
        print(f"  📏 {lc['id']}: {lc['turns_desc']} ({lc['chars']})")
    for dc in domains:
        print(f"  🏷️  {dc['id']}: {dc['desc'][:60]}...")

    if args.dry_run:
        return

    all_scenarios = []
    ok = 0
    fail = 0

    for lc in lengths:
        for dc in domains:
            for i in range(args.count):
                print(f"\n{'='*60}")
                print(f"  [{lc['id']}][{dc['id']}] #{i+1}/{args.count}")
                print(f"{'='*60}")
                scenario = generate_scenario(lc, dc, args.language)
                if scenario:
                    meta = scenario["_meta"]
                    print(f"  ✅ {meta['n_messages']}msgs, {meta['total_chars']}chars, "
                          f"{meta['n_golden']}facts, {meta['n_queries']}queries")
                    all_scenarios.append(scenario)
                    ok += 1
                else:
                    fail += 1

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ 完成: {ok} 成功, {fail} 失败")
    print(f"  💾 保存到: {output}")
    print(f"  📊 总计: {sum(s['_meta']['n_messages'] for s in all_scenarios)} 条消息, "
          f"{sum(s['_meta']['total_chars'] for s in all_scenarios)} 字符, "
          f"{sum(s['_meta']['n_golden'] for s in all_scenarios)} 黄金事实")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
