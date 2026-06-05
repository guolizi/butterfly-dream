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
from collections import defaultdict
from pathlib import Path

from butterfly_dream import ButterflyDreamMemoryProvider
from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env, get_db_path, set_run_dir, _RUNS_DIR

# Add eval/ to sys.path so eval_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

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


def process_sessions(provider: ButterflyDreamMemoryProvider, sessions: list,
                     haystack_dates: list | None = None):
    """Feed haystack sessions into Butterfly Dream for extraction.

    Processes each session separately so the extraction LLM handles
    manageable chunks and session boundaries are preserved. This is
    important for cross-session evaluation items.

    Args:
        haystack_dates: Optional list of session dates (parallel to sessions).
                        Injected into the first message of each session so
                        the extraction LLM can anchor relative time references.
    """
    for si, session in enumerate(sessions):
        sess_date = haystack_dates[si] if haystack_dates and si < len(haystack_dates) else ""
        session_msgs = []
        for turn in session:
            content = turn["content"]
            # Inject date into the first message (same pattern as LoCoMo adapter)
            if not session_msgs and sess_date:
                content = f"[Date: {sess_date}] {content}"
            session_msgs.append({"role": turn["role"], "content": content})
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


def answer_question(provider: ButterflyDreamMemoryProvider, question: str) -> tuple:
    """Search memory and generate an answer. Returns (answer, retrieved_facts)."""
    from butterfly_dream.retrieval import ThreeDimRetriever
    
    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=20)
    
    if not results:
        return ("I don't have enough information to answer this question.", [])
    
    # Use top 10 for LLM context (avoid noise from lower-ranked facts)
    context_parts = []
    for r in results[:10]:
        content = r.get("content", "")
        if content:
            context_parts.append(content)
    
    context = "\n".join(context_parts)
    # Log all top 20 retrieved facts
    retrieved_facts = [{"rank": i+1, "score": round(r["score"], 4), "content": r["content"]} for i, r in enumerate(results)]
    
    # Generate answer using LLM
    return (_generate_answer(question, context), retrieved_facts)


def _generate_answer(question: str, context: str) -> str:
    """Use LLM to generate an answer based on retrieved context (via eval_utils.call_llm)."""
    prompt = f"""Based on the following memory context, answer the user's question.
Use ALL relevant facts. Be specific — include names, dates, locations, and details.
If multiple facts relate to the question, combine them into a complete answer.

IMPORTANT rules for abstention:
1. If the question asks about a specific entity (person, company, place) and that entity does NOT appear in the memory context, say "I don't have enough information." — do NOT substitute with a different entity.
2. If the question asks about timing/ordering and the context lacks date information to determine it, say "I don't have enough information."
3. When in doubt, prefer "I don't have enough information" over a guess.

Memory context:
{context}

Question: {question}

Answer (be specific and complete):"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided memory context."},
        {"role": "user", "content": prompt},
    ]
    result = call_llm("answer", messages=messages, max_tokens=1024)
    return result if result else "Unable to generate answer: no API key configured."


def _create_run_dir(benchmark: str, tag: str) -> Path:
    """Create a timestamped run directory and return it."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"{ts}_{benchmark}_{tag}" if tag else f"{ts}_{benchmark}"
    run_dir = _RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_config(run_dir: Path, benchmark: str, args, extra: dict = None):
    """Write config.json with run metadata."""
    from datetime import datetime
    cfg = {
        "benchmark": benchmark,
        "timestamp": datetime.now().isoformat(),
        "args": {k: v for k, v in vars(args).items() if v},
        "model_config": get_model_config("all"),
    }
    if extra:
        cfg.update(extra)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _write_summary(run_dir: Path, benchmark: str, args, all_results: list,
                   total_time: float, n_items: int):
    """Write human-readable summary.txt."""
    from datetime import datetime
    n_total = len(all_results)
    avg_ms = (total_time / n_items) * 1000 if n_items else 0

    _ext_cfg = get_model_config("extraction")
    _eff_model = args.model or f"{_ext_cfg.get('provider','?')}/{_ext_cfg.get('model','?')}"
    lines = [
        f"Benchmark: {benchmark}",
        f"Time: {datetime.now().isoformat()}",
        f"Model: {_eff_model}",
        f"Subset: {getattr(args, 'subset', 'n/a')}",
        "",
        f"Total questions: {n_total}",
        f"Avg time: {avg_ms:.0f}ms/question",
        f"Total time: {total_time:.1f}s",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × LongMemEval")
    parser.add_argument("--subset", default="oracle", choices=["oracle", "s"],
                        help="Dataset subset (oracle=few sessions, S=full ~40 sessions)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to process (0=all)")
    parser.add_argument("--sample", default="",
                        help="Uniform sampling: 'N' or 'N-per-type' (e.g. '3-per-type')")
    parser.add_argument("--tag", default="",
                        help="Run tag for folder naming")
    parser.add_argument("--data", default="",
                        help="Direct path to a JSON dataset file (overrides --subset)")
    parser.add_argument("--model", default="",
                        help="Extraction model (overrides config)")
    args = parser.parse_args()
    
    # Create run directory
    run_dir = _create_run_dir("longmemeval", args.tag)
    set_run_dir(run_dir)
    print(f"📁 Run dir: {run_dir}")

    # Load dataset
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)
        print(f"📋 Loaded {len(data)} questions from {args.data}")
    else:
        data = load_dataset(args.subset)
        print(f"📋 Loaded {len(data)} questions (subset={args.subset})")

    # Uniform sampling (--sample 3-per-type or --sample 3)
    if args.sample:
        import random as _random
        _random.seed(42)
        n = int(args.sample.replace("-per-type", ""))
        by_type = defaultdict(list)
        for e in data:
            by_type[e["question_type"]].append(e)
        sampled = []
        for t in sorted(by_type):
            picked = _random.sample(by_type[t], min(n, len(by_type[t])))
            sampled.extend(picked)
        data = sampled
        print(f"🎲 Sampled {len(data)} questions ({n} per type, {len(by_type)} types)")
    elif args.limit > 0:
        data = data[:args.limit]

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
        dates = entry.get("haystack_dates")
        
        t0 = time.perf_counter()
        
        # Fresh provider per question
        q_db = str(get_db_path('longmemeval', qid))
        q_config = {
            "db_path": q_db,
            "llm_extract": True,
            "extraction_model": get_model_config("extraction") if not args.model else {"provider": "openrouter", "model": args.model},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        }
        q_provider = ButterflyDreamMemoryProvider(q_config)
        q_provider.initialize(session_id=f"longmemeval-{qid}")
        
        # Step 1: Process all history sessions
        try:
            process_sessions(q_provider, sessions, haystack_dates=dates)
        except Exception as e:
            extraction_errors += 1
            print(f"  ⚠️  [{i+1}/{len(data)}] Extraction error for {qid}: {e}")
        
        n_facts = q_provider._store.count_facts() if q_provider._store else 0
        total_facts += n_facts
        
        # Step 2: Answer the question
        try:
            hypothesis, retrieved_facts = answer_question(q_provider, question)
        except Exception as e:
            hypothesis = f"Error: {e}"
            retrieved_facts = []
            print(f"  ⚠️  [{i+1}/{len(data)}] Answer error for {qid}: {e}")
        
        # Cleanup
        q_provider.shutdown()
        time.sleep(3)
        print(f"  💾 DB saved: {q_db}")
        
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        
        # Save result
        result = {
            "question_id": qid,
            "hypothesis": hypothesis,
            "retrieved_facts": retrieved_facts,
        }
        results.append(result)
        
        # Progress
        avg_ms = (total_time / (i + 1)) * 1000
        print(f"  [{i+1}/{len(data)}] ⏱{avg_ms:.0f}ms/q  📝{n_facts} facts  ❌{extraction_errors} errors  {question[:50]}...")
    
    output_path = run_dir / "results.jsonl"
    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write config.json and summary.txt
    _write_config(run_dir, "longmemeval", args)
    _write_summary(run_dir, "longmemeval", args, results, total_time, len(data))

    # Summary
    avg_ms = (total_time / len(data)) * 1000 if data else 0
    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(results)} questions processed")
    print(f"  📝 Total facts extracted: {total_facts}")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/question")
    print(f"  ❌  Extraction errors: {extraction_errors}")
    print(f"  📁 Run dir: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
