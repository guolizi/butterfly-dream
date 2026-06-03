#!/usr/bin/env python3
"""诊断脚本：跑1题 LongMemEval，打印全链路中间结果"""
import json, os, sys, time, tempfile
from pathlib import Path

def _load_hermes_env():
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file(): return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key not in os.environ:
                os.environ[key] = val.strip().strip("\"'").strip()

_load_hermes_env()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from butterfly_dream import ButterflyDreamMemoryProvider

# Load first question
data_dir = Path(__file__).resolve().parent / "data"
with open(data_dir / "longmemeval_oracle.json", encoding="utf-8") as f:
    dataset = json.load(f)

entry = dataset[0]
print("=" * 70)
print(f"📋 题目 ID: {entry['question_id']}")
print(f"📋 题目类型: {entry['question_type']}")
print(f"📋 问题: {entry['question']}")
print(f"📋 标准答案: {entry['answer']}")
print(f"📋 会话数: {len(entry['haystack_sessions'])}")
print("=" * 70)

# Show each session summary
for si, session in enumerate(entry['haystack_sessions']):
    n_turns = len(session)
    roles = [t['role'] for t in session]
    print(f"\n--- Session {si+1} ({n_turns} turns) ---")
    for ti, turn in enumerate(session):
        role = turn['role']
        content = turn['content']
        # Truncate long content
        display = content[:300] + "..." if len(content) > 300 else content
        print(f"  [{role}] {display}")

# Init provider
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
    db_path = tmp.name

config = {
    "db_path": db_path,
    "llm_extract": True,
    "extraction_model": {"provider": "openrouter", "model": "owl-alpha"},
    "trivial_filter": True,
    "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
    "reflection": False,
}
provider = ButterflyDreamMemoryProvider(config)
provider.initialize(session_id="longmemeval-debug")

print("\n" + "=" * 70)
print("🔍 开始提取...")
print("=" * 70)

sessions = entry["haystack_sessions"]

# Flatten all sessions into one message list (same fix as adapter)
all_messages = []
for session in sessions:
    for turn in session:
        all_messages.append({"role": turn["role"], "content": turn["content"]})

before = provider._store.count_facts() if provider._store else 0
provider.on_session_end(all_messages)

# Wait for extraction (max 30s)
for _ in range(60):
    time.sleep(0.5)
    after = provider._store.count_facts() if provider._store else 0
    if after > before:
        break

after = provider._store.count_facts() if provider._store else 0
new_facts = after - before
total_turns = sum(len(s) for s in sessions)
print(f"  拍平 {len(sessions)} sessions → {total_turns} turns → {new_facts} new facts (total={after})")

# Print ALL extracted facts
print("\n" + "=" * 70)
print("📝 所有提取出的事实:")
print("=" * 70)

all_facts = provider._store.list_facts(limit=500) if provider._store else []
for i, fact in enumerate(all_facts):
    trust = fact.get('trust', '?')
    trust_str = f"{trust:.2f}" if isinstance(trust, (int, float)) else str(trust)
    print(f"  [{i+1}] (trust={trust_str}) {fact.get('content', '')[:200]}")

# Search
print("\n" + "=" * 70)
print(f"🔍 检索: '{entry['question']}'")
print("=" * 70)

from butterfly_dream.retrieval import ThreeDimRetriever
retriever = ThreeDimRetriever(provider._store)
results = retriever.search(query=entry["question"], scenario="chat", limit=5)
for i, r in enumerate(results):
    score = r.get("score", r.get("relevance_score", "?"))
    print(f"  [{i+1}] (score={score}) {r.get('content', '')[:200]}")

if not results:
    print("  ⚠️  没有检索到任何结果!")

provider.shutdown()
os.unlink(db_path)
