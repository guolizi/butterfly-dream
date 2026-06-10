#!/usr/bin/env python3
"""Run locomo eval on cat1 questions only, reusing existing DB."""
import json, logging, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock Hermes agent modules (standalone context)
from unittest.mock import MagicMock
from abc import ABC, abstractmethod

class _MockMemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self): ...
    @abstractmethod
    def is_available(self): ...
    @abstractmethod
    def initialize(self, *a, **kw): ...
    @abstractmethod
    def shutdown(self): ...
    def system_prompt_block(self): return ''
    def prefetch(self, *a, **kw): return ''
    def sync_turn(self, *a, **kw): pass
    def get_tool_schemas(self): return []
    def handle_tool_call(self, *a, **kw): return ''
    def on_pre_compress(self, *a, **kw): return ''
    def on_session_end(self, *a, **kw): pass
    def on_memory_write(self, *a, **kw): pass

for _mod in ['agent', 'agent.memory_provider', 'hermes_cli', 'hermes_cli.config',
             'hermes_constants', 'tools', 'tools.registry']:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
if 'agent.memory_provider' in sys.modules:
    import agent.memory_provider as _amp
    _amp.MemoryProvider = _MockMemoryProvider

sys.modules['agent.memory_provider'].MemoryProvider = _MockMemoryProvider

from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.retrieval import ThreeDimRetriever
from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env, get_db_path

_load_hermes_env()
_logger = logging.getLogger("locomo_cat1")
_logger.setLevel(logging.INFO)

# ── Q&A ──
def answer_question(provider, question, category=0):
    t0 = time.perf_counter()
    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=20)
    search_time = time.perf_counter() - t0
    
    if not results:
        return ("No information available.", 0, [], search_time)
    
    # Build context
    context_parts = []
    retrieved_facts = []
    for r in results:
        fid = r["fact_id"]
        content = r.get("content", "")
        date = r.get("content_date", r.get("created_at", ""))[:10] if r.get("content_date", r.get("created_at", "")) else ""
        imp = r.get("importance", 5)
        if date:
            context_parts.append(f"[{date}] (imp={imp}) {content}")
        else:
            context_parts.append(f"(imp={imp}) {content}")
        retrieved_facts.append({
            "fact_id": fid, "content": content[:80],
            "relevance": round(r.get("_relevance", 0), 4),
            "score": round(r.get("score", 0), 4),
        })
    
    context = "\n".join(context_parts)
    answer = _generate_answer(question, context, category)
    return (answer, len(results), retrieved_facts, search_time)

def _generate_answer(question, context, category=0):
    """Few-shot prompt per category."""
    if category == 5:
        prompt = f"""Based only on the information below, answer the question.
If the information doesn't contain the answer, say "No information available."

Context:
{context}

Question: {question}
Answer:"""
    else:
        prompt = f"""Based on the following information, answer the question concisely.
If the information doesn't directly answer the question, say "No information available."

Context:
{context}

Question: {question}
Answer:"""
    
    raw = call_llm(
        role="answer",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    return raw.strip() if raw else ""

# ── Judge ──
def judge_answer(question, ground_truth, predicted, category=0):
    judge_prompt = f"""You are evaluating a question-answering system for a memory-augmented agent.

Question: {question}
Ground truth: {ground_truth}
Predicted answer: {predicted}

Rate the predicted answer from 1-5 based on how well it matches the ground truth:
1 = Completely wrong / hallucinated
2 = Partially wrong but tangentially related
3 = Partially correct (some key info present, some missing)
4 = Correct but slightly incomplete or imprecise
5 = Perfectly correct and complete

Return only a single number (1-5)."""
    
    raw = call_llm(
        role="judge",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    m = re.search(r'[1-5]', raw or '')
    return int(m.group()) if m else 1

# ── Main ──
def main():
    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    data = json.load(open(data_path))
    
    # Filter conv-26
    conv = [d for d in data if d["sample_id"] == "conv-26"]
    if not conv:
        print("❌ conv-26 not found")
        return
    conv = conv[0]
    qa_list = conv["qa"]
    sid = conv["sample_id"]
    
    # Filter cat1
    cat1_qas = [qa for qa in qa_list if qa.get("category") == 1]
    print(f"📋 {sid}: {len(qa_list)} total, {len(cat1_qas)} cat1 questions")
    
    # Load memory provider with existing DB
    q_db = str(Path(__file__).resolve().parent.parent / "dbs" / "locomo" / f"{sid}.db")
    if not os.path.exists(q_db):
        print(f"❌ DB not found: {q_db}")
        return
    
    print(f"♻️  Reusing DB: {q_db}")
    provider = ButterflyDreamMemoryProvider({
        "db_path": q_db, "llm_extract": True,
        "extraction_model": get_model_config("extraction"),
        "trivial_filter": True,
        "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
        "reflection": False,
    })
    provider.initialize(session_id=f"locomo-{sid}")
    
    results = []
    total_time = 0
    qa_count = 0
    
    for i, qa in enumerate(cat1_qas):
        qid = qa.get("question_id", f"Q{i}")
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)
        
        if not question:
            continue
        
        # Generate answer
        answer, n_facts, facts_retrieved, search_time = answer_question(provider, question, category)
        total_time += search_time
        
        # Judge
        score = judge_answer(question, str(ground_truth), answer, category)
        
        results.append({
            "question_id": qid,
            "category": category,
            "question": question,
            "ground_truth": str(ground_truth),
            "predicted": answer,
            "score": score,
            "n_facts": n_facts,
            "search_time_ms": round(search_time * 1000, 1),
        })
        
        qa_count += 1
        passed = "✓" if score >= 4 else ""
        print(f"  [{i+1}/{len(cat1_qas)}] Q{qid} score={score}{passed} ({n_facts} facts, {search_time*1000:.0f}ms)")
        
        if score < 3:
            print(f"    Q: {question[:60]}")
            print(f"    GT: {str(ground_truth)[:60]}")
            print(f"    PR: {answer[:60]}")
    
    # Summary
    by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for r in results:
        c = r["category"]
        by_cat[c]["total"] += 1
        by_cat[c]["scores"].append(r["score"])
        if r["score"] >= 4:
            by_cat[c]["correct"] += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Summary — cat1 ({len(results)} questions)")
    print(f"{'='*50}")
    for c in sorted(by_cat):
        s = by_cat[c]
        avg = sum(s["scores"]) / s["total"] if s["total"] else 0
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  cat{c}: avg={avg:.2f} acc={acc:.1f}% ({s['correct']}/{s['total']})")
    
    all_scores = [r["score"] for r in results]
    print(f"\n  Overall: avg={sum(all_scores)/len(all_scores):.2f} total_time={total_time:.1f}s")

if __name__ == "__main__":
    main()
