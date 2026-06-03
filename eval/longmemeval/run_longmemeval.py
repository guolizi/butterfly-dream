#!/usr/bin/env python3
"""Butterfly Dream adapter for LongMemEval benchmark.

流程：
  1. 加载 LongMemEval 数据集
  2. 对每个问题，把历史会话喂入 Butterfly Dream 提取事实
  3. 用自然语言问题检索相关事实
  4. 基于检索到的事实生成回答
  5. 输出 JSONL 供 evaluate_qa.py 评分

用法：
    python run_longmemeval.py --subset oracle --limit 50
    python run_longmemeval.py --subset oracle --limit 50 --model owl-alpha
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env

_load_hermes_env()

# Load Hermes .env (contains DEEPSEEK_API_KEY etc.)

def load_dataset(subset: str = "oracle") -> list:
    """Load LongMemEval dataset."""
    data_dir = Path(__file__).resolve().parent / "data"
    if subset == "oracle":
        path = data_dir / "longmemeval_oracle.json"
    elif subset == "s":
        path = data_dir / "longmemeval_s.json"
    else:
        raise ValueError(f"Unknown subset: {subset}")
    
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def process_sessions(provider: ButterflyDreamMemoryProvider, sessions: list):
    """Feed haystack sessions into Butterfly Dream for extraction.

    Processes each session separately so the extraction LLM handles
    manageable chunks and session boundaries are preserved. This is
    important for cross-session evaluation items.
    """
    for session in sessions:
        session_msgs = [{"role": turn["role"], "content": turn["content"]}
                        for turn in session]
        if not session_msgs:
            continue
        before = provider._store.count_facts() if provider._store else 0
        # Reset extraction index so each session is processed independently
        provider._last_extracted_idx = 0
        provider.on_session_end(session_msgs)
        time.sleep(1.0)  # rate limit between sessions
        # Wait for async extraction to finish (max 60s per session)
        for _ in range(120):
            time.sleep(0.5)
            if provider._store and provider._store.count_facts() > before:
                break


def answer_question(provider: ButterflyDreamMemoryProvider, question: str) -> str:
    """Search memory and generate an answer."""
    from butterfly_dream.retrieval import ThreeDimRetriever
    
    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=5)
    
    if not results:
        return "I don't have enough information to answer this question."
    
    # Build context from retrieved facts
    context_parts = []
    for r in results:
        content = r.get("content", "")
        if content:
            context_parts.append(content)
    
    context = "\n".join(context_parts)
    
    # Generate answer using LLM
    return _generate_answer(question, context)


def _generate_answer(question: str, context: str) -> str:
    """Use LLM to generate an answer based on retrieved context (via eval_utils.call_llm)."""
    prompt = f"""Based on the following memory context, answer the user's question.
If the context does not contain enough information to answer, say "I don't have enough information."
Be concise and direct.

Memory context:
{context}

Question: {question}

Answer:"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided memory context."},
        {"role": "user", "content": prompt},
    ]
    result = call_llm("answer", messages=messages)
    return result if result else "Unable to generate answer: no API key configured."


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × LongMemEval")
    parser.add_argument("--subset", default="oracle", choices=["oracle", "s"],
                        help="Dataset subset (oracle=few sessions, S=full ~40 sessions)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to process (0=all)")
    parser.add_argument("--output", default="",
                        help="Output JSONL path")
    parser.add_argument("--data", default="",
                        help="Direct path to a JSON dataset file (overrides --subset)")
    parser.add_argument("--model", default="glm-4.7-flash",
                        help="Extraction model")
    args = parser.parse_args()
    
    # Load dataset
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)
        print(f"📋 Loaded {len(data)} questions from {args.data}")
    else:
        data = load_dataset(args.subset)
        if args.limit > 0:
            data = data[:args.limit]
        print(f"📋 Loaded {len(data)} questions (subset={args.subset})")
    
    # Output path
    if not args.output:
        args.output = f"results_{args.subset}_bd.jsonl"
    output_path = Path(__file__).resolve().parent / args.output
    
    # Each question gets its own fresh provider (avoids _last_extracted_idx
    # accumulating across questions and skipping extraction).
    results = []
    total_time = 0
    extraction_errors = 0
    total_facts = 0
    
    for i, entry in enumerate(data):
        qid = entry["question_id"]
        question = entry["question"]
        answer = entry["answer"]
        sessions = entry["haystack_sessions"]
        
        t0 = time.perf_counter()
        
        # Fresh provider per question
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            q_db = tmp.name
        q_config = {
            "db_path": q_db,
            "llm_extract": True,
            "extraction_model": {"provider": "glm", "model": args.model},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        }
        q_provider = ButterflyDreamMemoryProvider(q_config)
        q_provider.initialize(session_id=f"longmemeval-{qid}")
        
        # Step 1: Process all history sessions
        try:
            process_sessions(q_provider, sessions)
        except Exception as e:
            extraction_errors += 1
            print(f"  ⚠️  [{i+1}/{len(data)}] Extraction error for {qid}: {e}")
        
        n_facts = q_provider._store.count_facts() if q_provider._store else 0
        total_facts += n_facts
        
        # Step 2: Answer the question
        try:
            hypothesis = answer_question(q_provider, question)
        except Exception as e:
            hypothesis = f"Error: {e}"
            print(f"  ⚠️  [{i+1}/{len(data)}] Answer error for {qid}: {e}")
        
        # Cleanup
        q_provider.shutdown()
        try:
            os.unlink(q_db)
        except OSError:
            pass
        
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        
        # Save result
        result = {
            "question_id": qid,
            "hypothesis": hypothesis,
        }
        results.append(result)
        
        # Progress
        avg_ms = (total_time / (i + 1)) * 1000
        print(f"  [{i+1}/{len(data)}] ⏱{avg_ms:.0f}ms/q  📝{n_facts} facts  ❌{extraction_errors} errors  {question[:50]}...")
    
    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Summary
    avg_ms = (total_time / len(data)) * 1000
    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(results)} questions processed")
    print(f"  📝 Total facts extracted: {total_facts}")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/question")
    print(f"  ❌  Extraction errors: {extraction_errors}")
    print(f"  📄 Results: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
