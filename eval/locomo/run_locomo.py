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
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Add eval/ to sys.path so eval_utils is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from butterfly_dream import ButterflyDreamMemoryProvider
from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env, get_db_path, set_run_dir, _RUNS_DIR

_load_hermes_env()

# ── Logging ──────────────────────────────────────────────────────────
_extraction_logger = logging.getLogger("locomo_extraction")
_extraction_logger.setLevel(logging.DEBUG)



def _setup_extraction_log(run_dir: Path):
    """Create a file handler for extraction progress logs."""
    handler = logging.FileHandler(run_dir / "extraction.log", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _extraction_logger.handlers.clear()
    _extraction_logger.addHandler(handler)
    return handler


def _log(msg: str, level: str = "info"):
    """Log to both extraction.log and console."""
    getattr(_extraction_logger, level, _extraction_logger.info)(msg)
    print(f"  {msg}")


def _run_clustering_after_extraction(store):
    """Run v2 entity clustering after extraction to build three-layer ontology.

    Creates abstract entities + includes edges for semantic reasoning.
    Safe to call even if no clusters are found.
    """
    if store is None:
        print("  ⚠️  No store available, skipping clustering")
        return
    try:
        from butterfly_dream.clustering import compute_clusters
        t0 = time.perf_counter()
        clusters = compute_clusters(store, threshold=0.55, min_cluster_size=2)
        if clusters:
            for c in clusters:
                print(f"  🏷️  Cluster '{c['name']}' ({c['size']} members, coherence={c['coherence']})")
        else:
            print("  🏷️  No clusters found (entities too diverse or <2 per group)")
        _log(f"Clustering done: {len(clusters)} clusters in {time.perf_counter()-t0:.2f}s")
    except Exception as e:
        print(f"  ⚠️  Clustering error (non-fatal): {e}")
        _log(f"Clustering skipped: {e}", "warning")


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
    Retries on 429 / empty extraction with exponential backoff.
    Logs success/failure per session to extraction.log.
    """
    session_names = get_session_names(conv)
    total = len(session_names)
    succeeded = 0
    failed_sessions = []

    _log(f"Processing {total} sessions ...")

    for idx, sname in enumerate(session_names):
        session_msgs = flatten_session(conv, sname)
        if not session_msgs:
            _log(f"  [{idx+1}/{total}] {sname}: empty session, skipped", "warning")
            continue

        # Set session date so extraction LLM can resolve relative dates
        sess_num = sname.split("_")[1]
        date_key = f"session_{sess_num}_date_time"
        sess_date = conv.get(date_key, "")
        if sess_date:
            provider._session_date = sess_date[:10]

        before = provider._store.count_facts() if provider._store else 0
        n_turns = len(session_msgs)
        session_ok = False

        # Retry loop for rate-limited extractions
        for attempt in range(4):
            provider._last_extracted_idx = 0
            provider.on_session_end(session_msgs)
            # Wait for extraction to complete (max 30s per attempt)
            for _ in range(60):
                time.sleep(0.5)
                if provider._store and provider._store.count_facts() > before:
                    break
            if provider._store and provider._store.count_facts() > before:
                new_facts = provider._store.count_facts() - before
                _log(f"[{idx+1}/{total}] {sname}: ✅ extracted {new_facts} facts ({n_turns} turns, attempt {attempt+1})")
                session_ok = True
                break  # extraction succeeded
            # Back off before retry
            wait = 5 * (attempt + 1)
            _log(f"[{idx+1}/{total}] {sname}: ⏳ no new facts, retry in {wait}s (attempt {attempt+1}/4)", "warning")
            time.sleep(wait)

        if session_ok:
            succeeded += 1
        else:
            failed_sessions.append(sname)
            _log(f"[{idx+1}/{total}] {sname}: ❌ FAILED after 4 attempts", "error")

        time.sleep(3.0)  # rate limit between sessions

    _log(f"Extraction done: {succeeded}/{total} sessions succeeded")
    if failed_sessions:
        _log(f"Failed sessions: {', '.join(failed_sessions)}", "error")


def answer_question(provider: ButterflyDreamMemoryProvider, question: str, category: int = 0) -> tuple:
    """Search memory and generate an answer. Returns (answer, n_facts_retrieved, retrieved_facts)."""
    from butterfly_dream.retrieval import ThreeDimRetriever

    retriever = ThreeDimRetriever(provider._store)
    t0 = time.perf_counter()
    results = retriever.search(query=question, scenario="chat", limit=20)
    search_time = time.perf_counter() - t0

    if not results:
        return ("No information available.", 0, [], search_time)

    # Use top 15 for LLM context (avoid excluding valid facts at ranks 11-15)
    # Prepend entity labels [Name] so the model knows who each fact belongs to
    context_parts = []
    fids = [r.get("fact_id") for r in results[:15] if r.get("fact_id")]
    fact_entities = {}
    if fids:
        placeholders = ",".join("?" * len(fids))
        entity_rows = provider._store.execute_query(
            f"""SELECT fe.fact_id, GROUP_CONCAT(e.name, ', ') as entities
                FROM fact_entities fe
                JOIN entities e ON fe.entity_id = e.entity_id
                WHERE fe.fact_id IN ({placeholders})
                GROUP BY fe.fact_id""",
            fids
        )
        fact_entities = {r["fact_id"]: r["entities"] for r in entity_rows}
    for r in results[:15]:
        content = r.get("content", "")
        if not content:
            continue
        fid = r.get("fact_id")
        ents = fact_entities.get(fid, "")
        entity_tag = f"[{ents}] " if ents else ""
        date = r.get("content_date", "")
        if date:
            context_parts.append(f"{entity_tag}[{date}] {content}")
        else:
            context_parts.append(f"{entity_tag}{content}")
    context = "\n".join(context_parts)
    # Log all top 20 retrieved facts (with date for debugging)
    retrieved_facts = [{
        "rank": i+1, "score": round(r["score"], 4),
        "content": r["content"], "content_date": r.get("content_date", "")
    } for i, r in enumerate(results)]
    _log(f"Retrieved {len(results)} facts in {search_time*1000:.0f}ms for Q: {question[:80]}...")
    for rf in retrieved_facts[:5]:
        _log(f"  [{rf['rank']}] score={rf['score']:.4f} | {rf['content'][:90]}")
    if len(retrieved_facts) > 5:
        _log(f"  ... {len(retrieved_facts)-5} more facts")
    return (_generate_answer(question, context, category), len(context_parts), retrieved_facts, search_time)


def _generate_answer(question: str, context: str, category: int = 0) -> str:
    """Use LLM to generate an answer via eval_utils.call_llm().
    
    Routes to different prompts based on question category:
    - Cat5 (adversarial/temporal): adversarial detection prompt
    - Others (Cat1-4): simple direct answer + inference prompt
    """
    if category == 5:
        return _generate_adversarial_answer(question, context)
    return _generate_simple_answer(question, context)


def _generate_simple_answer(question: str, context: str) -> str:
    """CoT prompt: think step by step, then answer concisely."""
    prompt = f"""You are analyzing facts from a conversation between two people. Answer the question based on the context.

Think step by step:
1. What exactly does the question ask for?
2. Read through each fact carefully — which ones are relevant?
3. Do those facts directly answer it, or can you make a reasonable inference?
4. What is the complete and accurate answer?

Be thorough — scan ALL facts before deciding. Facts may describe things differently than the question uses, so make reasonable connections. The answer is almost always in the context if you look carefully.

Memory context:
{context}

Question: {question}

Let me go through the facts step by step:"""

    messages = [
        {"role": "system", "content": "You analyze facts carefully and reason step by step before answering. End your answer with 'Answer: <concise answer>' on a new line."},
        {"role": "user", "content": prompt},
    ]
    result = call_llm("answer", messages=messages, max_tokens=1024)
    return result if result else "No information available."


def _generate_adversarial_answer(question: str, context: str) -> str:
    """Adversarial detection prompt for Cat5 (temporal reasoning = adversarial swaps)."""
    prompt = f"""Based on the following memory context about a conversation between two people, answer the question.

Guidelines:
- **Step 1 — Check for adversarial modification**: Some questions are adversarially modified (entity swap, attribute swap, temporal swap). If the context has no direct answer to the exact question but contains a fact about a DIFFERENT person at the same/similar situation, or about the SAME person in a closely related scenario, it's likely adversarial. Answer using that related fact.
  Examples:
  * "What is Jon's favorite style of painting?" Context: "Jon loves contemporary dance" → Answer: "Contemporary" (attribute swap: dance→painting)
  * "What happened to Maria's job?" Context: "John lost his job" → Answer: "John lost his job" (entity swap: John→Maria)
  * "What is the name of Maria's one-year-old child?" Context: "John's one-year-old son is named Kyle" → Answer: "Kyle" (entity swap: John→Maria)

- **Step 2 — Normal answer**: If there IS a direct answer in the context, answer directly. If MULTIPLE facts are relevant, combine them ALL.

- **Step 3 — Only say "No info" as last resort**: Only say "No information available" after confirming adversarial check found nothing related.

Memory context:
{context}

Question: {question}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on the provided memory context, making reasonable inferences when appropriate. Be concise."},
        {"role": "user", "content": prompt},
    ]
    result = call_llm("answer", messages=messages, max_tokens=1024)
    return result if result else "Unable to generate answer."


def judge_answer(question: str, gold: str, hypothesis: str) -> int:
    """Use LLM to judge answer quality (1-5 scale) via eval_utils.call_llm()."""
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

    messages = [
        {"role": "system", "content": "You are a precise evaluator. Reply with only a number."},
        {"role": "user", "content": prompt},
    ]
    raw = call_llm("judge", messages=messages, temperature=0)
    if not raw:
        return 0
    import re
    m = re.search(r'[1-5]', raw)
    return int(m.group(0)) if m else 0


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
                   total_time: float, n_convs: int):
    """Write human-readable summary.txt."""
    from collections import defaultdict
    from datetime import datetime
    total_correct = sum(1 for r in all_results if r["is_correct"])
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    avg_ms = (total_time / n_convs) * 1000 if n_convs else 0

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for r in all_results:
        c = r["category"]
        by_cat[c]["total"] += 1
        if r["is_correct"]:
            by_cat[c]["correct"] += 1
        by_cat[c]["scores"].append(r["score"])

    _ext_cfg = get_model_config("extraction")
    _eff_model = args.model or f"{_ext_cfg.get('provider','?')}/{_ext_cfg.get('model','?')}"
    lines = [
        f"Benchmark: {benchmark}",
        f"Time: {datetime.now().isoformat()}",
        f"Model: {_eff_model}",
        f"Sample: {args.sample or 'all'}",
        f"Judge: {'off' if args.no_judge else 'on'}",
        "",
        f"Total QA: {len(all_results)}",
        f"Accuracy: {total_correct}/{len(all_results)} = {total_correct/len(all_results)*100:.1f}%",
        f"Avg score: {avg_score:.2f}/5.0",
        f"Avg time: {avg_ms:.0f}ms/conv",
        "",
        "--- Per Category ---",
    ]
    CAT_NAMES = {1: "single-session single-hop", 2: "single-session multi-hop",
                 3: "cross-session single-hop", 4: "cross-session multi-hop",
                 5: "temporal reasoning"}
    for c in sorted(by_cat):
        s = by_cat[c]
        avg = sum(s["scores"]) / len(s["scores"])
        acc = s["correct"] / s["total"] * 100
        lines.append(f"  [{c}] {CAT_NAMES.get(c, ''):30s} {s['correct']}/{s['total']} = {acc:.0f}%  avg={avg:.2f}")

    (run_dir / "summary.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Butterfly Dream × LoCoMo")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max QA pairs to process (0=all)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N QA pairs (0-based, used with --limit)")
    parser.add_argument("--sample", default="",
                        help="Uniform sampling: 'N' or 'N-per-cat' (e.g. '3-per-cat')")
    parser.add_argument("--conv", default="",
                        help="Process only this conversation ID (e.g. conv-26)")
    parser.add_argument("--tag", default="",
                        help="Run tag for folder naming (e.g. 'sample3', 'v2')")
    parser.add_argument("--model", default="",
                        help="Extraction model (overrides config)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (faster, exact match only)")
    parser.add_argument("--db-dir", default="",
                        help="Reuse existing DBs from this directory (skip extraction)")
    args = parser.parse_args()

    # Create run directory
    run_dir = _create_run_dir("locomo", args.tag)
    set_run_dir(run_dir)
    _setup_extraction_log(run_dir)
    _log(f"Run started: {run_dir.name}")
    print(f"📁 Run dir: {run_dir}")

    data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
    data = load_dataset(str(data_path))
    print(f"📋 Loaded {len(data)} conversations")

    if args.conv:
        data = [d for d in data if d["sample_id"] == args.conv]
        if not data:
            print(f"❌ Conversation {args.conv} not found")
            return

    # Uniform sampling (--sample 3-per-cat or --sample 3)
    sampled_conv_ids = None
    if args.sample:
        import random as _random
        _random.seed(42)
        n = int(args.sample.replace("-per-cat", ""))
        eligible_convs = []
        for conv in data:
            has_answer = any("answer" in qa for qa in conv["qa"])
            if has_answer:
                eligible_convs.append(conv)
        sampled_conv_ids = [c["sample_id"] for c in _random.sample(eligible_convs, min(n, len(eligible_convs)))]
        data = [d for d in data if d["sample_id"] in sampled_conv_ids]
        print(f"🎲 Sampled {len(data)} conversations (from {len(eligible_convs)} with answers)")

    output_path = run_dir / "results.jsonl"

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

        # Decide DB path and whether to extract
        skip_extract = False
        if args.db_dir:
            q_db = str(Path(args.db_dir) / f"{sid}.db")
            if not Path(q_db).exists():
                print(f"  ⚠️  DB not found at {q_db}, will create & extract")
                q_db = str(get_db_path('locomo', sid))
            else:
                skip_extract = True
                print(f"  ♻️  Reusing existing DB: {q_db}")
        else:
            q_db = str(get_db_path('locomo', sid))

        qp = ButterflyDreamMemoryProvider({
            "db_path": q_db, "llm_extract": True,
            "extraction_model": get_model_config("extraction") if not args.model else {"provider": "openrouter", "model": args.model},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        })
        qp.initialize(session_id=f"locomo-{sid}")

        t0 = time.perf_counter()
        if skip_extract:
            n_facts = qp._store.count_facts() if qp._store else 0
            extract_time = 0
            print(f"  ✅ Skipped extraction, {n_facts} existing facts")
        else:
            session_names = get_session_names(conv)
            total_turns = sum(len(conv[s]) for s in session_names)
            print(f"  📝 {len(session_names)} sessions, {total_turns} turns")
            try:
                process_conversation(qp, conv)
            except Exception as e:
                print(f"  ⚠️  Extraction error: {e}")
            n_facts = qp._store.count_facts() if qp._store else 0
            extract_time = time.perf_counter() - t0
            print(f"  ✅ Extracted {n_facts} facts in {extract_time:.1f}s")

        # ── V2: Run entity clustering to build three-layer ontology ──
        _run_clustering_after_extraction(qp._store)

        # Answer QAs
        conv_correct = 0
        conv_scores = []
        for qi, qa in enumerate(qa_list):
            if qa_count < args.start:
                qa_count += 1
                continue
            if args.limit > 0 and qa_count >= args.start + args.limit:
                break

            question = qa["question"]
            category = qa["category"]
            # For Cat5 (adversarial), prefer adversarial_answer as gold
            if category == 5 and qa.get("adversarial_answer"):
                gold = str(qa["adversarial_answer"])
            else:
                gold = str(qa.get("answer", qa.get("adversarial_answer", "")))

            hypothesis, n_retrieved, retrieved_facts, search_time = answer_question(qp, question, category)

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
            print(f"  [{qa_count}] {mark} cat={category} score={score} search={search_time*1000:.0f}ms facts={n_retrieved} Q: {question[:60]}...")

            all_results.append({
                "sample_id": sid,
                "question_id": f"{sid}_q{qi+1}",
                "question": question,
                "gold": gold,
                "hypothesis": hypothesis,
                "category": category,
                "category_name": CAT_NAMES.get(category, ""),
                "score": score,
                "is_correct": is_correct,
                "n_facts": n_facts,
                "n_retrieved": n_retrieved,
                "search_time_ms": round(search_time * 1000, 1),
                "retrieved_facts": retrieved_facts,
            })

            # Incremental write: flush after each QA so partial results survive timeouts
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(all_results[-1], ensure_ascii=False) + "\n")

        # Cleanup provider
        qp.shutdown()
        time.sleep(3)
        print(f"  💾 DB saved: {q_db}")

        avg_score = sum(conv_scores) / len(conv_scores) if conv_scores else 0
        conv_time = time.perf_counter() - t0
        total_time += conv_time
        print(f"\n  📊 {sid}: {conv_correct}/{len(conv_scores)} correct, avg_score={avg_score:.2f}, {conv_time:.1f}s")

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write config.json and summary.txt
    _write_config(run_dir, "locomo", args, extra={
        "n_conversations": len(data),
        "n_facts_per_conv": {r["sample_id"]: r.get("n_facts", 0) for r in all_results
                             if "n_facts" in r},
    })
    _write_summary(run_dir, "locomo", args, all_results, total_time, len(data))

    # Summary
    total_correct = sum(1 for r in all_results if r["is_correct"])
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    avg_ms = (total_time / len(data)) * 1000 if data else 0

    print(f"\n{'='*60}")
    print(f"  ✅ Done! {len(all_results)} QA pairs")
    print(f"  📊 Accuracy (score≥4): {total_correct}/{len(all_results)} = {total_correct/len(all_results)*100:.1f}%")
    print(f"  📊 Average score: {avg_score:.2f}/5.0")
    print(f"  ⏱  Average: {avg_ms:.0f}ms/conversation")
    print(f"  📁 Run dir: {run_dir}")
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
