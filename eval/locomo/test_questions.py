#!/usr/bin/env python3
"""Quick test: run retrieval + answer for specific LoCoMo questions."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.retrieval import ThreeDimRetriever
from eval_utils import get_model_config, call_llm, _load_hermes_env

_load_hermes_env()

# Load dataset
data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
with open(data_path) as f:
    data = json.load(f)
conv = [d for d in data if d["sample_id"] == "conv-26"][0]

# Target questions (1-indexed): 11-20 → 0-indexed: 10-19
target_indices = list(range(10, 20))

# Load existing DB
db_path = str(Path(__file__).resolve().parent.parent.parent / "eval" / "dbs" / "locomo" / "conv-26.db")
provider = ButterflyDreamMemoryProvider({
    "db_path": db_path, "llm_extract": False,
    "extraction_model": get_model_config("extraction"),
})
provider.initialize(session_id="locomo-conv-26-test")

n_facts = provider._store.count_facts() if provider._store else 0
print(f"📦 DB loaded: {n_facts} facts\n")

retriever = ThreeDimRetriever(provider._store)
log_entries = []

for qi in target_indices:
    qa = conv["qa"][qi]
    question = qa["question"]
    gold = str(qa.get("answer", qa.get("adversarial_answer", "")))
    category = qa["category"]

    print(f"{'='*70}")
    print(f"Q{qi+1} [cat={category}]")
    print(f"  Question: {question}")
    print(f"  Gold:     {gold}")
    print()

    # Retrieval
    results = retriever.search(query=question, scenario="chat", limit=20)
    print(f"  📖 Retrieved {len(results)} facts (top 10 used for answer):")
    for i, r in enumerate(results[:10]):
        print(f"    [{i+1}] (score={r['score']:.4f}) {r['content'][:90]}")
    print(f"    ... ({len(results)-10} more in log)")
    print()

    # Generate answer — use top 10 for context
    context = "\n".join(r["content"] for r in results[:10] if r.get("content"))
    prompt = f"""Based on the following memory context about a conversation between two people, answer the question.
Be concise and direct. If the context doesn't contain enough information, say "I don't have enough information."

Memory context:
{context}

Question: {question}

Answer:"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided memory context. Be concise."},
        {"role": "user", "content": prompt},
    ]
    answer = call_llm("answer", messages=messages, max_tokens=1024)
    print(f"  🤖 Answer: {answer}")

    # Judge
    judge_prompt = f"""Rate the following answer on a scale of 1-5 based on semantic equivalence with the reference.
Question: {question}
Reference answer: {gold}
Generated answer: {answer}
Scoring:
5 = Perfect match
4 = Mostly correct
3 = Partially correct
2 = Mostly wrong
1 = Completely wrong
Reply with ONLY the number."""
    judge_messages = [
        {"role": "system", "content": "You are a precise evaluator. Reply with only a number."},
        {"role": "user", "content": judge_prompt},
    ]
    raw = call_llm("judge", messages=judge_messages, temperature=0)
    import re
    m = re.search(r'[1-5]', raw) if raw else None
    score = int(m.group(0)) if m else 0
    mark = "✅" if score >= 4 else "❌"
    print(f"  {mark} Score: {score}/5")
    print()

    log_entries.append({
        "question_id": qi + 1,
        "category": category,
        "question": question,
        "gold": gold,
        "answer": answer,
        "score": score,
        "retrieved_facts": [
            {"rank": i + 1, "score": round(r["score"], 4), "content": r["content"]}
            for i, r in enumerate(results)
        ],
    })

    time.sleep(3)  # rate limit

provider.shutdown()

# Write detailed log with facts
log_path = Path(__file__).resolve().parent / "test_log.jsonl"
with open(log_path, "w", encoding="utf-8") as f:
    for entry in log_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"\n📝 Log saved: {log_path}")
print("🏁 Done!")
