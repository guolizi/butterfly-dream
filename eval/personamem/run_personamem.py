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
import tempfile
import time
from pathlib import Path

from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env

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
def answer_question(provider: ButterflyDreamMemoryProvider, question: str, options: list) -> str:
    """Search memory and pick the best option."""
    from butterfly_dream.retrieval import ThreeDimRetriever

    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=15)

    if not results:
        # No context found, pick first option as default
        return options[0] if options else ""

    context_parts = [r.get("content", "") for r in results if r.get("content")]
    context = "\n".join(context_parts)

    return _generate_answer(question, context, options)


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


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × PersonaMem")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to process (0=all)")
    parser.add_argument("--type", default="",
                        help="Filter by question_type")
    parser.add_argument("--topic", default="",
                        help="Filter by topic")
    parser.add_argument("--output", default="",
                        help="Output JSONL path")
    parser.add_argument("--model", default="glm-4.7-flash",
                        help="Extraction model")
    parser.add_argument("--size", default="32k", choices=["32k", "128k", "1M"],
                        help="Context size version")
    args = parser.parse_args()

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
    if args.limit > 0:
        questions = questions[:args.limit]

    # Output
    if not args.output:
        suffix = f"_{args.type}" if args.type else ""
        args.output = f"results_personamem{suffix}.jsonl"
    output_path = Path(__file__).resolve().parent / args.output

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
            hypothesis = answer_question(q_provider, question_text, options)
        except Exception as e:
            hypothesis = options[0] if options else ""
            print(f"  ⚠️  [{i+1}] Answer error: {e}")

        # Cleanup
        q_provider.shutdown()
        try:
            os.unlink(q_db)
        except OSError:
            pass

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

    # Summary
    avg_ms = (total_time / len(results)) * 1000 if results else 0
    acc = correct / len(results) * 100 if results else 0

    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(results)} questions, {correct} correct")
    print(f"  📊 Accuracy: {acc:.1f}%")
    print(f"  📝 Total facts: {total_facts}")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/question")
    print(f"  ❌ Extraction errors: {extraction_errors}")
    print(f"  📄 Results: {output_path}")
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
