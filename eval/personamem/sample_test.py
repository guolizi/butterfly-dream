#!/usr/bin/env python3
"""PersonaMem 3-size sample test: 10 random questions from each of 32K/128K/1M."""
import csv, random, json, os, tempfile, time, sys
from pathlib import Path

random.seed(42)

def _load_hermes_env():
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file(): return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k not in os.environ:
                os.environ[k] = v.strip().strip("\"'").strip()

_load_hermes_env()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.chdir(str(Path(__file__).resolve().parent.parent.parent))

from run_personamem import _parse_options, process_context, answer_question
from butterfly_dream import ButterflyDreamMemoryProvider

def load_contexts(size):
    path = Path(f"eval/personamem/data/shared_contexts_{size}.jsonl")
    if not path.exists(): return {}
    ctx = {}
    with open(path) as f:
        for line in f:
            for k, v in json.loads(line).items():
                ctx[k] = v
    return ctx

results_all = {}
for size in ["32k", "128k", "1M"]:
    print(f"\n{'='*60}")
    print(f"📊 {size.upper()} — 随机抽 10 题")
    print(f"{'='*60}")

    with open(f"eval/personamem/data/questions_{size}.csv") as f:
        questions = list(csv.DictReader(f))

    sample = random.sample(questions, 10)
    contexts = load_contexts(size)
    print(f"  Loaded {len(contexts)} contexts")

    correct = 0
    for i, q in enumerate(sample):
        qid = q["question_id"]
        question_text = q["user_question_or_message"]
        gold_letter = q["correct_answer"].strip()
        options = _parse_options(q["all_options"])
        ctx_id = q["shared_context_id"]
        end_idx = int(q["end_index_in_shared_context"])
        qtype = q["question_type"]

        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            q_db = tmp.name
        qp = ButterflyDreamMemoryProvider({
            "db_path": q_db, "llm_extract": True,
            "extraction_model": {"provider": "openrouter", "model": "owl-alpha"},
            "trivial_filter": True,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
            "reflection": False,
        })
        qp.initialize(session_id=f"pm-{qid}")

        try:
            if ctx_id in contexts:
                process_context(qp, contexts[ctx_id], end_idx)
        except Exception as e:
            print(f"  ⚠️  Extract error: {e}")

        n_facts = qp._store.count_facts() if qp._store else 0

        try:
            hypothesis = answer_question(qp, question_text, options)
        except Exception as e:
            hypothesis = ""
            print(f"  ⚠️  Answer error: {e}")

        qp.shutdown()
        try: os.unlink(q_db)
        except: pass

        elapsed = time.perf_counter() - t0
        is_correct = hypothesis.lower() == gold_letter.lower()
        if is_correct: correct += 1
        mark = "✅" if is_correct else "❌"
        print(f"  [{i+1:2d}/10] {mark} gold={gold_letter} pred={hypothesis:5s} ⏱{elapsed:.0f}s 📝{n_facts}f  {qtype[:35]}")

    acc = correct / 10 * 100
    results_all[size] = correct
    print(f"\n  📊 {size.upper()}: {correct}/10 = {acc:.0f}%")

print(f"\n{'='*60}")
print(f"📊 汇总")
for s, c in results_all.items():
    print(f"  {s.upper():5s}: {c}/10 = {c*10:.0f}%")
print(f"{'='*60}")
