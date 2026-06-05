#!/usr/bin/env python3
"""Butterfly Dream adapter for PersonaMem benchmark.

PersonaMem 测试 LLM 对用户偏好/人格的追踪能力（选择题格式）。
流程：
  1. 加载 PersonaMem 数据集 (questions CSV + shared_contexts JSONL)
  2. 对每个问题，把用户上下文喂入 Butterfly Dream 提取事实
  3. 用用户问题检索相关事实
  4. 基于检索到的事实，从选项中选出最佳答案
  5. 输出 JSONL 供评分

用法：
    python run_personamem.py --limit 20
    python run_personamem.py --type recall_user_shared_facts --limit 50
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
import tempfile
import time
from pathlib import Path

from butterfly_dream import ButterflyDreamMemoryProvider
from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env, get_db_path
from butterfly_dream import ButterflyDreamMemoryProvider
from eval_utils import set_run_dir, _RUNS_DIR

# Add eval/ to sys.path so eval_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

_load_hermes_env()

# Load Hermes .env

def load_questions(csv_path: str) -> list:
    """Load questions from CSV file."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_shared_contexts(jsonl_path: str) -> dict:
    """Load shared contexts from JSONL file. Returns {context_id: messages}."""
    contexts = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            for ctx_id, messages in data.items():
                contexts[ctx_id] = messages
    return contexts


def process_context(provider: ButterflyDreamMemoryProvider, messages: list, end_index: int):
    """Feed conversation context into Butterfly Dream for extraction.

    Flatten all messages up to end_index into one list, call on_session_end once.
    """
    # Slice context and flatten
    sliced = messages[:end_index] if end_index else messages
    all_messages = []
    for turn in sliced:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            all_messages.append({"role": role, "content": content})

    if not all_messages:
        return

    before = provider._store.count_facts() if provider._store else 0
    provider.on_session_end(all_messages)
    # Wait for async extraction (max 120s, check every 0.5s)
    for _ in range(240):
        time.sleep(0.5)
        if provider._store and provider._store.count_facts() > before:
            break


def _parse_options(raw: str) -> list:
    """Parse all_options field, handling both JSON double-quotes and Python single-quotes."""
    import re
    # Try standard JSON first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Normalize single quotes to double quotes (careful with embedded quotes)
    # Replace outer brackets and single-quoted strings
    normalized = raw.strip()
    if normalized.startswith("'") or normalized.startswith('"'):
        # Split by ', ' or ", " pattern to get individual options
        # Use regex to extract (a), (b), (c), (d) prefixed strings
        options = re.findall(r'\([a-d]\)[^\'"]*', normalized)
        if options:
            return [o.strip().rstrip(',').rstrip("'").rstrip('"') for o in options]
    # Last resort: split by option prefix
    options = re.findall(r'\([a-d]\)\s*.*?(?=\([a-d]\)|$)', normalized, re.DOTALL)
    return [o.strip() for o in options] if options else [normalized]
def answer_question(provider: ButterflyDreamMemoryProvider, question: str, options: list) -> tuple:
    """Search memory and pick the best option. Returns (answer, retrieved_facts)."""
    from butterfly_dream.retrieval import ThreeDimRetriever

    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=20)

    if not results:
        # No context found, pick first option as default
        return (options[0] if options else "", [])

    # Use top 10 for LLM context (avoid noise from lower-ranked facts)
    context_parts = [r.get("content", "") for r in results[:10] if r.get("content")]
    context = "\n".join(context_parts)
    # Log all top 20 retrieved facts
    retrieved_facts = [{"rank": i+1, "score": round(r["score"], 4), "content": r["content"]} for i, r in enumerate(results)]

    return (_generate_answer(question, context, options), retrieved_facts)


def _generate_answer(question: str, context: str, options: list) -> str:
    """Use LLM to pick the best option based on retrieved context (via eval_utils.call_llm)."""
    # Format options for the prompt
    options_text = "\n".join(f"{opt}" for opt in options)

    prompt = f"""Based on the following memory context about a user, select the most appropriate response to the user's message.

Memory context:
{context}

User's message: {question}

Options:
{options_text}

Reply with ONLY the letter prefix of the best option, e.g. "(a)". Do not explain."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that selects the best response option based on user memory. Reply with only the letter prefix."},
        {"role": "user", "content": prompt},
    ]
    choice = call_llm("answer", messages=messages)
    if not choice:
        return options[0] if options else ""
    # Extract letter prefix like "(a)" from the response
    import re
    m = re.search(r'\([a-z]\)', choice.lower())
    if m:
        return m.group(0)
    return choice


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
    from collections import defaultdict
    total_correct = sum(1 for r in all_results if r["is_correct"])
    avg_ms = (total_time / n_items) * 1000 if n_items else 0

    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in all_results:
        t = r["question_type"]
        by_type[t]["total"] += 1
        if r["is_correct"]:
            by_type[t]["correct"] += 1

    _ext_cfg = get_model_config("extraction")
    _eff_model = args.model or f"{_ext_cfg.get('provider','?')}/{_ext_cfg.get('model','?')}"
    lines = [
        f"Benchmark: {benchmark}",
        f"Time: {datetime.now().isoformat()}",
        f"Model: {_eff_model}",
        f"Size: {args.size}",
        f"Sample: {args.sample or 'all'}",
        f"Type filter: {args.type or 'all'}",
        f"Topic filter: {args.topic or 'all'}",
        "",
        f"Total questions: {len(all_results)}",
        f"Accuracy: {total_correct}/{len(all_results)} = {total_correct/len(all_results)*100:.1f}%",
        f"Avg time: {avg_ms:.0f}ms/question",
        "",
        "--- Per Question Type ---",
    ]
    for t, s in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
        acc = s["correct"] / s["total"] * 100
        lines.append(f"  {t[:45]:45s} {s['correct']}/{s['total']} = {acc:.0f}%")

    (run_dir / "summary.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × PersonaMem")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to process (0=all)")
    parser.add_argument("--sample", default="",
                        help="Uniform sampling: 'N' or 'N-per-type' (e.g. '3-per-type')")
    parser.add_argument("--type", default="",
                        help="Filter by question_type")
    parser.add_argument("--topic", default="",
                        help="Filter by topic")
    parser.add_argument("--tag", default="",
                        help="Run tag for folder naming (e.g. 'sample3', 'v2')")
    parser.add_argument("--model", default="",
                        help="Extraction model (overrides config)")
    parser.add_argument("--size", default="32k", choices=["32k", "128k", "1M"],
                        help="Context size version")
    args = parser.parse_args()

    # Create run directory
    run_dir = _create_run_dir("personamem", args.tag)
    set_run_dir(run_dir)
    print(f"📁 Run dir: {run_dir}")

    data_dir = Path(__file__).resolve().parent / "data"
    csv_path = data_dir / f"questions_{args.size}.csv"
    jsonl_path = data_dir / f"shared_contexts_{args.size}.jsonl"

    # Load data
    print(f"📋 Loading questions from {csv_path.name}...")
    questions = load_questions(str(csv_path))
    print(f"📋 Loading shared contexts from {jsonl_path.name}...")
    shared_contexts = load_shared_contexts(str(jsonl_path))
    print(f"   {len(questions)} questions, {len(shared_contexts)} contexts loaded")

    # Filter
    if args.type:
        questions = [q for q in questions if q["question_type"] == args.type]
        print(f"   Filtered to {len(questions)} questions (type={args.type})")
    if args.topic:
        questions = [q for q in questions if q["topic"] == args.topic]
        print(f"   Filtered to {len(questions)} questions (topic={args.topic})")

    # Uniform sampling (--sample 3-per-type or --sample 3)
    if args.sample:
        import random as _random
        _random.seed(42)
        n = int(args.sample.replace("-per-type", ""))
        by_type = defaultdict(list)
        for q in questions:
            by_type[q["question_type"]].append(q)
        sampled = []
        for t in sorted(by_type):
            picked = _random.sample(by_type[t], min(n, len(by_type[t])))
            sampled.extend(picked)
        questions = sampled
        print(f"🎲 Sampled {len(questions)} questions ({n} per type, {len(by_type)} types)")
    elif args.limit > 0:
        questions = questions[:args.limit]

    # Output
    output_path = run_dir / "results.jsonl"

    results = []
    total_time = 0
    correct = 0
    extraction_errors = 0
    total_facts = 0

    for i, q in enumerate(questions):
        qid = q["question_id"]
        question_text = q["user_question_or_message"]
        gold_letter = q["correct_answer"].strip()
        options = _parse_options(q["all_options"])
        ctx_id = q["shared_context_id"]
        end_idx = int(q["end_index_in_shared_context"])
        qtype = q["question_type"]
        topic = q["topic"]

        t0 = time.perf_counter()

        # Fresh provider per question
        q_db = str(get_db_path('personamem', qid))
        q_config = {
            "db_path": q_db,
            "llm_extract": True,
            "extraction_model": get_model_config("extraction") if not args.model else {"provider": "openrouter", "model": args.model},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        }
        q_provider = ButterflyDreamMemoryProvider(q_config)
        q_provider.initialize(session_id=f"personamem-{qid}")

        # Step 1: Process context
        try:
            if ctx_id in shared_contexts:
                process_context(q_provider, shared_contexts[ctx_id], end_idx)
            else:
                print(f"  ⚠️  [{i+1}] Context not found: {ctx_id}")
        except Exception as e:
            extraction_errors += 1
            print(f"  ⚠️  [{i+1}] Extraction error: {e}")

        n_facts = q_provider._store.count_facts() if q_provider._store else 0
        total_facts += n_facts

        # Step 2: Answer
        try:
            hypothesis, retrieved_facts = answer_question(q_provider, question_text, options)
        except Exception as e:
            hypothesis = options[0] if options else ""
            retrieved_facts = []
            print(f"  ⚠️  [{i+1}] Answer error: {e}")

        # Cleanup
        q_provider.shutdown()
        time.sleep(3)
        print(f"  💾 DB saved: {q_db}")

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        is_correct = hypothesis.lower() == gold_letter.lower()
        if is_correct:
            correct += 1

        result = {
            "question_id": qid,
            "question_type": qtype,
            "topic": topic,
            "hypothesis": hypothesis,
            "correct_answer": gold_letter,
            "is_correct": is_correct,
            "retrieved_facts": retrieved_facts,
        }
        results.append(result)

        # Progress
        acc = correct / (i + 1) * 100
        avg_ms = (total_time / (i + 1)) * 1000
        mark = "✅" if is_correct else "❌"
        print(f"  [{i+1}/{len(questions)}] {mark} {acc:.0f}%  ⏱{avg_ms:.0f}ms  📝{n_facts}f  {qtype[:30]}")

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write config.json and summary.txt
    _write_config(run_dir, "personamem", args, extra={
        "n_questions": len(questions),
        "size": args.size,
    })
    _write_summary(run_dir, "personamem", args, results, total_time, len(results))

    # Summary
    avg_ms = (total_time / len(results)) * 1000 if results else 0
    acc = correct / len(results) * 100 if results else 0

    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(results)} questions, {correct} correct")
    print(f"  📊 Accuracy: {acc:.1f}%")
    print(f"  📝 Total facts: {total_facts}")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/question")
    print(f"  ❌ Extraction errors: {extraction_errors}")
    print(f"  📁 Run dir: {run_dir}")
    print(f"{'='*60}")

    # Per-type breakdown
    from collections import defaultdict
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        t = r["question_type"]
        by_type[t]["total"] += 1
        if r["is_correct"]:
            by_type[t]["correct"] += 1

    print(f"\n  === 按题型 ===")
    for t, s in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
        acc = s["correct"] / s["total"] * 100
        print(f"    {t[:45]:45s} {s['correct']}/{s['total']} = {acc:.0f}%")


if __name__ == "__main__":
    main()
