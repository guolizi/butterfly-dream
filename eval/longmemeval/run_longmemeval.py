#!/usr/bin/env python3
"""Butterfly Dream adapter for LongMemEval benchmark (ICLR 2025).

流程：
  1. 加载 LongMemEval 数据集
  2. 对每个问题，把历史会话喂入 Butterfly Dream 提取事实
  3. 用自然语言问题检索相关事实（带 entity/date 标签）
  4. 基于检索到的事实生成回答（CoT step-by-step）
  5. 输出 JSONL 供 evaluate_qa.py 评分

用法：
    python run_longmemeval.py --subset oracle
    python run_longmemeval.py --subset oracle --limit 50
    python run_longmemeval.py --subset oracle --limit 50 --model owl-alpha
    python run_longmemeval.py --db-dir eval/runs/2026-06-08_1234_locomo_tag/  # 复用已有 DB
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
_extraction_logger = logging.getLogger("longmemeval_extraction")
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


# ── Dataset loading ──────────────────────────────────────────────────

QUESTION_TYPES = [
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]


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


# ── Extraction ───────────────────────────────────────────────────────

def process_sessions(provider: ButterflyDreamMemoryProvider, sessions: list,
                     haystack_dates: list | None = None):
    """Feed haystack sessions into Butterfly Dream for extraction.

    Processes each session separately so the extraction LLM handles
    manageable chunks and session boundaries are preserved.
    Retries on 429 / empty extraction with exponential backoff.
    Logs success/failure per session to extraction.log.

    Args:
        haystack_dates: Optional list of session dates (parallel to sessions).
                        Injected into the first message of each session so
                        the extraction LLM can anchor relative time references.
    """
    total = len(sessions)
    succeeded = 0
    failed_sessions = []

    _log(f"Processing {total} sessions ...")

    for si, session in enumerate(sessions):
        sess_date = haystack_dates[si] if haystack_dates and si < len(haystack_dates) else ""

        # Set session date so extraction LLM can resolve relative dates
        if sess_date:
            provider._session_date = sess_date

        session_msgs = []
        for turn in session:
            content = turn["content"]
            # Inject date into the first message (same pattern as LoCoMo adapter)
            if not session_msgs and sess_date:
                content = f"[Date: {sess_date}] {content}"
            session_msgs.append({"role": turn["role"], "content": content})
        if not session_msgs:
            _log(f"  [{si+1}/{total}] session[{si}]: empty, skipped", "warning")
            continue

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
                _log(f"[{si+1}/{total}] session[{si}]: ✅ extracted {new_facts} facts ({n_turns} turns, attempt {attempt+1})")
                session_ok = True
                break  # extraction succeeded
            # Back off before retry
            wait = 5 * (attempt + 1)
            _log(f"[{si+1}/{total}] session[{si}]: ⏳ no new facts, retry in {wait}s (attempt {attempt+1}/4)", "warning")
            time.sleep(wait)

        if session_ok:
            succeeded += 1
        else:
            failed_sessions.append(si)
            _log(f"[{si+1}/{total}] session[{si}]: ❌ FAILED after 4 attempts", "error")

        time.sleep(3.0)  # rate limit between sessions

    _log(f"Extraction done: {succeeded}/{total} sessions succeeded")
    if failed_sessions:
        _log(f"Failed sessions: {', '.join(str(s) for s in failed_sessions)}", "error")


# ── Answer generation ───────────────────────────────────────────────

def answer_question(provider: ButterflyDreamMemoryProvider, question: str) -> tuple:
    """Search memory and generate an answer.

    Returns (answer, n_facts_retrieved, retrieved_facts, search_time_ms).

    The context is built with entity labels [Name] and date tags [YYYY-MM-DD]
    so the answer LLM knows who each fact belongs to and when it occurred.
    """
    from butterfly_dream.retrieval import ThreeDimRetriever

    retriever = ThreeDimRetriever(provider._store)
    t0 = time.perf_counter()
    results = retriever.search(query=question, scenario="chat", limit=20)
    search_time = time.perf_counter() - t0

    if not results:
        return ("No information available.", 0, [], search_time)

    # Use top 15 for LLM context (avoid excluding valid facts at ranks 11-15)
    # Prepend entity labels [Name] + date tags so the model knows context
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

    # Log all top 20 retrieved facts
    retrieved_facts = [{
        "rank": i + 1, "score": round(r["score"], 4),
        "content": r["content"], "content_date": r.get("content_date", ""),
        "entities": fact_entities.get(r.get("fact_id"), ""),
    } for i, r in enumerate(results)]

    _log(f"Retrieved {len(results)} facts in {search_time*1000:.0f}ms for Q: {question[:80]}...")
    for rf in retrieved_facts[:5]:
        ents = rf.get("entities", "")
        tag = f"[{ents}] " if ents else ""
        date = rf.get("content_date", "")
        dtag = f"[{date}] " if date else ""
        _log(f"  [{rf['rank']}] score={rf['score']:.4f} | {tag}{dtag}{rf['content'][:90]}")
    if len(retrieved_facts) > 5:
        _log(f"  ... {len(retrieved_facts)-5} more facts")

    return (_generate_answer(question, context), len(context_parts), retrieved_facts, search_time)


def _generate_answer(question: str, context: str) -> str:
    """Generate a direct answer based on retrieved context.

    Uses a simple, direct prompt (no CoT wrapper) so the output is
    just the answer text — compatible with LongMemEval's external scoring
    which expects clean answer strings without reasoning or prefixes.

    Context already contains entity labels [Name] and date tags [YYYY-MM-DD].
    """
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
    return result if result else "No information available."


# ── Output helpers ───────────────────────────────────────────────────

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
        "",
        "--- Per Type ---",
    ]
    # Per-type breakdown
    by_type = defaultdict(lambda: {"total": 0, "n_facts": 0})
    for r in all_results:
        t = r.get("question_type", "unknown")
        by_type[t]["total"] += 1
        by_type[t]["n_facts"] += r.get("n_facts", 0)
    for t in sorted(by_type):
        s = by_type[t]
        avg_f = s["n_facts"] / s["total"] if s["total"] else 0
        lines.append(f"  {t:35s} {s['total']:4d} questions, avg facts={avg_f:.0f}")

    (run_dir / "summary.txt").write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────

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
    parser.add_argument("--db-dir", default="",
                        help="Reuse existing DBs from this directory (skip extraction)")
    args = parser.parse_args()

    # Create run directory
    run_dir = _create_run_dir("longmemeval", args.tag)
    set_run_dir(run_dir)
    _setup_extraction_log(run_dir)
    _log(f"Run started: {run_dir.name}")
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

    # ── Per-question loop ────────────────────────────────────────────
    results = []
    total_time = 0
    extraction_errors = 0
    total_facts = 0

    output_path = run_dir / "results.jsonl"

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        question = entry["question"]
        answer = entry["answer"]
        question_type = entry["question_type"]
        sessions = entry["haystack_sessions"]
        dates = entry.get("haystack_dates")

        t0 = time.perf_counter()

        # Decide DB path and whether to extract
        skip_extract = False
        if args.db_dir:
            q_db = str(Path(args.db_dir) / f"{qid}.db")
            if not Path(q_db).exists():
                print(f"  ⚠️  DB not found at {q_db}, will create & extract")
                q_db = str(get_db_path("longmemeval", qid))
            else:
                skip_extract = True
                _log(f"  ♻️  Reusing existing DB: {q_db}")
        else:
            q_db = str(get_db_path("longmemeval", qid))

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
        if skip_extract:
            n_facts = q_provider._store.count_facts() if q_provider._store else 0
            extract_time = 0
            _log(f"  ✅ Skipped extraction, {n_facts} existing facts")
        else:
            total_sessions = len(sessions)
            _log(f"  📝 {total_sessions} sessions to extract")
            try:
                process_sessions(q_provider, sessions, haystack_dates=dates)
            except Exception as e:
                extraction_errors += 1
                _log(f"  ⚠️  Extraction error for {qid}: {e}", "error")
            n_facts = q_provider._store.count_facts() if q_provider._store else 0
            extract_time = time.perf_counter() - t0
            _log(f"  ✅ Extracted {n_facts} facts in {extract_time:.1f}s")
        total_facts += n_facts

        # Step 2: Answer the question
        try:
            hypothesis, n_retrieved, retrieved_facts, search_time = answer_question(q_provider, question)
        except Exception as e:
            hypothesis = f"Error: {e}"
            n_retrieved = 0
            retrieved_facts = []
            search_time = 0
            _log(f"  ⚠️  Answer error for {qid}: {e}", "error")

        # Cleanup
        q_provider.shutdown()
        time.sleep(3)
        _log(f"  💾 DB saved: {q_db}")

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Save result
        result = {
            "question_id": qid,
            "question_type": question_type,
            "question": question,
            "hypothesis": hypothesis,
            "n_facts": n_facts,
            "n_retrieved": n_retrieved,
            "search_time_ms": round(search_time * 1000, 1),
            "retrieved_facts": retrieved_facts,
        }
        results.append(result)

        # Incremental write: flush after each question so partial results survive timeouts
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Progress
        avg_ms = (total_time / (i + 1)) * 1000
        _log(f"[{i+1}/{len(data)}] ⏱{avg_ms:.0f}ms/q  📝{n_facts} facts  🔍{n_retrieved} ret  ❌{extraction_errors} errors  [{question_type}] {question[:50]}...")

    # ── Final output ─────────────────────────────────────────────────
    # Re-write full results (overwrite incremental file to ensure completeness)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write config.json and summary.txt
    _write_config(run_dir, "longmemeval", args, extra={
        "n_questions": len(data),
        "total_facts_extracted": total_facts,
        "extraction_errors": extraction_errors,
    })
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
