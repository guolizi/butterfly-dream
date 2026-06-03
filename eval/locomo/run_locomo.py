#!/usr/bin/env python3
"""Butterfly Dream adapter for LoCoMo benchmark (ACL 2024).

LoCoMo: 10 个长对话，1986 道开放式问答，5 个类别。
流程：
  1. 加载 locomo10.json
  2. 对每个对话，把所有 session 拍平喂入 Butterfly Dream 提取事实
  3. 对每道 QA，检索相关事实并生成回答
  4. 用 DeepSeek judge 评分 (1-5 分)

用法：
    python run_locomo.py                          # 全量
    python run_locomo.py --limit 50               # 限制 QA 数
    python run_locomo.py --conv conv-26           # 只跑某个对话
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request
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


CAT_NAMES = {
    1: "single-session single-hop",
    2: "single-session multi-hop",
    3: "cross-session single-hop",
    4: "cross-session multi-hop",
    5: "temporal reasoning",
}


def load_dataset(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_session_names(conv: dict) -> list:
    """Return sorted session keys from a LoCoMo conversation dict."""
    return sorted(
        [k for k in conv.keys()
         if k.startswith("session_") and "_date" not in k
         and "_observation" not in k and "_summary" not in k],
        key=lambda k: int(k.split("_")[1])
    )


def flatten_session(conv: dict, session_name: str) -> list:
    """Convert one LoCoMo session into a message list for extraction.

    Maps speaker names to user/assistant and injects session date.
    """
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")
    sess_num = session_name.split("_")[1]
    date_key = f"session_{sess_num}_date_time"
    sess_date = conv.get(date_key, "")

    messages = []
    for i, turn in enumerate(conv[session_name]):
        speaker = turn["speaker"]
        if speaker == speaker_a:
            role = "user"
        elif speaker == speaker_b:
            role = "assistant"
        else:
            role = "user"
        content = turn["text"]
        if i == 0 and sess_date:
            content = f"[Date: {sess_date}] {content}"
        messages.append({"role": role, "content": content})
    return messages


def flatten_conversation(conv: dict) -> list:
    """Flatten all sessions into a single message list (legacy, for QA answering)."""
    all_messages = []
    for sname in get_session_names(conv):
        all_messages.extend(flatten_session(conv, sname))
    return all_messages


def process_conversation(provider: ButterflyDreamMemoryProvider, conv: dict):
    """Extract facts from a LoCoMo conversation, one session at a time.

    Each session is fed separately to on_session_end so the extraction LLM
    processes manageable chunks and preserves session boundaries.
    """
    session_names = get_session_names(conv)
    for sname in session_names:
        session_msgs = flatten_session(conv, sname)
        if not session_msgs:
            continue
        before = provider._store.count_facts() if provider._store else 0
        # Reset extraction index so each session is processed independently
        provider._last_extracted_idx = 0
        provider.on_session_end(session_msgs)
        # Wait for extraction to complete (max 60s per session)
        for _ in range(120):
            time.sleep(0.5)
            if provider._store and provider._store.count_facts() > before:
                break


def answer_question(provider: ButterflyDreamMemoryProvider, question: str) -> str:
    """Search memory and generate an answer."""
    from butterfly_dream.retrieval import ThreeDimRetriever

    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario="chat", limit=10)

    if not results:
        return "I don't have enough information to answer this question."

    context_parts = [r.get("content", "") for r in results if r.get("content")]
    context = "\n".join(context_parts)
    return _generate_answer(question, context)


def _generate_answer(question: str, context: str) -> str:
    """Use LLM to generate an answer."""
    from butterfly_dream.__init__ import _resolve_provider_credentials

    base_url, api_key = _resolve_provider_credentials("deepseek")
    if not api_key:
        return "Unable to generate answer."

    prompt = f"""Based on the following memory context about a conversation between two people, answer the question.
Be concise and direct. If the context doesn't contain enough information, say "I don't have enough information."

Memory context:
{context}

Question: {question}

Answer:"""

    url = f"{base_url}/chat/completions"
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided memory context. Be concise."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


def judge_answer(question: str, gold: str, hypothesis: str) -> int:
    """Use LLM to judge answer quality (1-5 scale)."""
    from butterfly_dream.__init__ import _resolve_provider_credentials

    base_url, api_key = _resolve_provider_credentials("deepseek")
    if not api_key:
        return 0

    prompt = f"""Rate the following answer on a scale of 1-5 based on semantic equivalence with the reference.

Question: {question}
Reference answer: {gold}
Generated answer: {hypothesis}

Scoring:
5 = Perfect match (semantically equivalent)
4 = Mostly correct (minor omission or extra detail)
3 = Partially correct (captures main idea but misses key details)
2 = Mostly wrong (some relevant info but fundamentally incorrect)
1 = Completely wrong or irrelevant

Reply with ONLY the number (1-5). No explanation."""

    url = f"{base_url}/chat/completions"
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "You are a precise evaluator. Reply with only a number."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        raw = data["choices"][0]["message"]["content"].strip()
        import re
        m = re.search(r'[1-5]', raw)
        return int(m.group(0)) if m else 0
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × LoCoMo")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max QA pairs to process (0=all)")
    parser.add_argument("--conv", default="",
                        help="Process only this conversation ID (e.g. conv-26)")
    parser.add_argument("--output", default="",
                        help="Output JSONL path")
    parser.add_argument("--model", default="deepseek-v4-flash",
                        help="Extraction model")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (faster, exact match only)")
    args = parser.parse_args()

    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    data = load_dataset(str(data_path))
    print(f"📋 Loaded {len(data)} conversations")

    if args.conv:
        data = [d for d in data if d["sample_id"] == args.conv]
        if not data:
            print(f"❌ Conversation {args.conv} not found")
            return

    if not args.output:
        args.output = "results_locomo.jsonl"
    output_path = Path(__file__).resolve().parent / args.output

    all_results = []
    total_time = 0
    qa_count = 0

    for conv_idx, sample in enumerate(data):
        sid = sample["sample_id"]
        conv = sample["conversation"]
        qa_list = sample["qa"]

        if args.limit > 0 and qa_count >= args.limit:
            break

        print(f"\n{'='*60}")
        print(f"💬 [{conv_idx+1}/{len(data)}] {sid} — {len(qa_list)} QA pairs")
        print(f"{'='*60}")

        # Extract per-session
        t0 = time.perf_counter()
        session_names = get_session_names(conv)
        total_turns = sum(len(conv[s]) for s in session_names)
        print(f"  📝 {len(session_names)} sessions, {total_turns} turns")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            q_db = tmp.name
        qp = ButterflyDreamMemoryProvider({
            "db_path": q_db, "llm_extract": True,
            "extraction_model": {"provider": "deepseek", "model": args.model},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        })
        qp.initialize(session_id=f"locomo-{sid}")

        try:
            process_conversation(qp, conv)
        except Exception as e:
            print(f"  ⚠️  Extraction error: {e}")

        n_facts = qp._store.count_facts() if qp._store else 0
        extract_time = time.perf_counter() - t0
        print(f"  ✅ Extracted {n_facts} facts in {extract_time:.1f}s")

        # Answer QAs
        conv_correct = 0
        conv_scores = []
        for qi, qa in enumerate(qa_list):
            if args.limit > 0 and qa_count >= args.limit:
                break

            question = qa["question"]
            gold = qa.get("answer", qa.get("adversarial_answer", ""))
            category = qa["category"]

            hypothesis = answer_question(qp, question)

            score = 0
            if not args.no_judge:
                score = judge_answer(question, gold, hypothesis)
            else:
                # Simple contains check
                score = 5 if gold.lower() in hypothesis.lower() else 1

            is_correct = score >= 4
            if is_correct:
                conv_correct += 1
            conv_scores.append(score)
            qa_count += 1

            mark = "✅" if is_correct else "❌"
            print(f"  [{qa_count}] {mark} cat={category} score={score} Q: {question[:60]}...")

            all_results.append({
                "sample_id": sid,
                "question_id": f"{sid}_q{qi}",
                "question": question,
                "gold": gold,
                "hypothesis": hypothesis,
                "category": category,
                "category_name": CAT_NAMES.get(category, ""),
                "score": score,
                "is_correct": is_correct,
            })

        # Cleanup provider
        qp.shutdown()
        try:
            os.unlink(q_db)
        except OSError:
            pass

        avg_score = sum(conv_scores) / len(conv_scores) if conv_scores else 0
        conv_time = time.perf_counter() - t0
        total_time += conv_time
        print(f"\n  📊 {sid}: {conv_correct}/{len(conv_scores)} correct, avg_score={avg_score:.2f}, {conv_time:.1f}s")

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total_correct = sum(1 for r in all_results if r["is_correct"])
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    avg_ms = (total_time / len(data)) * 1000 if data else 0

    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(all_results)} QA pairs")
    print(f"  📊 Accuracy (score≥4): {total_correct}/{len(all_results)} = {total_correct/len(all_results)*100:.1f}%")
    print(f"  📊 Average score: {avg_score:.2f}/5.0")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/conversation")
    print(f"  📄 Results: {output_path}")
    print(f"{'='*60}")

    # Per-category breakdown
    from collections import defaultdict
    by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for r in all_results:
        c = r["category"]
        by_cat[c]["total"] += 1
        if r["is_correct"]:
            by_cat[c]["correct"] += 1
        by_cat[c]["scores"].append(r["score"])

    print(f"\n  === 按类别 ===")
    for c in sorted(by_cat):
        s = by_cat[c]
        avg = sum(s["scores"]) / len(s["scores"])
        acc = s["correct"] / s["total"] * 100
        print(f"    [{c}] {CAT_NAMES.get(c, ''):30s} {s['correct']}/{s['total']} = {acc:.0f}%  avg={avg:.2f}")


if __name__ == "__main__":
    main()
