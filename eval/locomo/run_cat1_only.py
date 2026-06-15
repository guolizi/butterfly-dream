#!/usr/bin/env python3
"""Run only Cat1 questions for conv-26 using existing DB."""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_utils import get_model_config, set_run_dir, _RUNS_DIR, get_db_path
from run_locomo import (
    answer_question, judge_answer, _write_config, _write_summary,
    CAT_NAMES, _setup_extraction_log, _log
)

from butterfly_dream import ButterflyDreamMemoryProvider

# ── Load data ──
data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
data = json.loads(data_path.read_bytes())
conv26 = [d for d in data if d["sample_id"] == "conv-26"]
assert conv26, "conv-26 not found!"
conv26 = conv26[0]

# Filter Cat1
qa_list = [qa for qa in conv26["qa"] if qa["category"] == 1]
print(f"🎯 Running {len(qa_list)} Cat1 questions for conv-26")

# ── Init provider with existing DB ──
db_path = "/home/xx/butterfly-dream/eval/dbs/locomo/conv-26.db"
assert Path(db_path).exists(), f"DB not found: {db_path}"

qp = ButterflyDreamMemoryProvider({
    "db_path": db_path, "llm_extract": False,
    "trivial_filter": True,
    "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
    "reflection": False,
})
qp.initialize(session_id="locomo-conv-26")
n_facts = qp._store.count_facts() if qp._store else 0
print(f"💾 Loaded DB: {n_facts} facts")

# ── Create run dir ──
from datetime import datetime
ts = datetime.now().strftime("%Y-%m-%d_%H%M")
run_dir = _RUNS_DIR / f"{ts}_locomo_conv-26_cat1"
run_dir.mkdir(parents=True, exist_ok=True)
set_run_dir(run_dir)
_setup_extraction_log(run_dir)
print(f"📁 Run dir: {run_dir}")

# ── Answer & Judge ──
all_results = []
t0 = time.perf_counter()

for qi, qa in enumerate(qa_list):
    question = qa["question"]
    gold = str(qa.get("answer", ""))
    
    hypothesis, n_retrieved, retrieved_facts, search_time = answer_question(qp, question, category=1)
    score = judge_answer(question, gold, hypothesis)
    is_correct = score >= 4
    
    mark = "✅" if is_correct else "❌"
    print(f"  [{qi+1}/{len(qa_list)}] {mark} score={score} search={search_time*1000:.0f}ms Q: {question[:70]}...")
    
    all_results.append({
        "sample_id": "conv-26",
        "question_id": f"conv-26_cat1_q{qi+1}",
        "question": question,
        "gold": gold,
        "hypothesis": hypothesis,
        "category": 1,
        "category_name": CAT_NAMES.get(1, ""),
        "score": score,
        "is_correct": is_correct,
        "n_facts": n_facts,
        "n_retrieved": n_retrieved,
        "search_time_ms": round(search_time * 1000, 1),
        "retrieved_facts": retrieved_facts,
    })
    
    # Incremental write
    output_path = run_dir / "results.jsonl"
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(all_results[-1], ensure_ascii=False) + "\n")

qp.shutdown()
total_time = time.perf_counter() - t0

# ── Summary ──
total_correct = sum(1 for r in all_results if r["is_correct"])
avg_score = sum(r["score"] for r in all_results) / len(all_results)
print(f"\n{'='*50}")
print(f"✅ Cat1 done! {total_correct}/{len(all_results)} = {total_correct/len(all_results)*100:.1f}%")
print(f"📊 Avg score: {avg_score:.2f}/5.0")
print(f"⏱  Total: {total_time:.1f}s")

# Write config & summary
_write_config(run_dir, "locomo_cat1",
    type('Args', (), {
        "start": 0, "limit": 0, "sample": "", "conv": "conv-26",
        "tag": "cat1", "model": "", "no_judge": False, "db_dir": str(Path(db_path).parent),
    })(),
    extra={"n_cat1_questions": len(qa_list), "n_facts": n_facts})
_write_summary(run_dir, "locomo_cat1",
    type('Args', (), {
        "start": 0, "limit": 0, "sample": "", "conv": "conv-26",
        "tag": "cat1", "model": "", "no_judge": False, "db_dir": str(Path(db_path).parent),
    })(),
    all_results, total_time, 1)
