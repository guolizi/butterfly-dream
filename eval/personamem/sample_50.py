#!/usr/bin/env python3
"""Sample 3 PersonaMem (32K) questions per dimension and evaluate."""

import csv, json, random, sys, os, tempfile, time, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'eval'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from butterfly_dream import ButterflyDreamMemoryProvider
from run_personamem import process_context, answer_question, _parse_options
from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env
_load_hermes_env()

random.seed(42)

# Load data
data_dir = Path(__file__).resolve().parent / 'data'
with open(data_dir / 'shared_contexts_32k.jsonl') as f:
    ctx_line = f.readline()
ctx_data = json.loads(ctx_line)
ctx_key = list(ctx_data.keys())[0]
shared_messages = ctx_data[ctx_key]

questions = []
with open(data_dir / 'questions_32k.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append(row)

# Group by question_type and sample 3 per type
by_type = {}
for q in questions:
    qt = q.get('question_type', 'unknown')
    by_type.setdefault(qt, []).append(q)

sampled = []
for qt, pool in by_type.items():
    picked = random.sample(pool, min(3, len(pool)))
    sampled.extend(picked)

print(f'PersonaMem 32K: sampled {len(sampled)} questions across {len(by_type)} types')
for qt in sorted(by_type):
    print(f'  {qt}: {sum(1 for s in sampled if s.get("question_type") == qt)}')

def judge_answer(question, options, correct_answer, hypothesis):
    prompt = (
        f'Multiple choice question. Score 1-5.\n'
        f'5=correct option selected, 4=mostly right, 3=unclear, 2=wrong, 1=completely wrong\n\n'
        f'Q: {question}\nOptions: {options}\nCorrect: {correct_answer}\n'
        f'Generated: {hypothesis}\nOutput ONLY: {{"score": <1-5>}}'
    )
    messages = [{'role': 'user', 'content': prompt}]
    text = call_llm('judge', messages=messages, temperature=0)
    if not text:
        return 3
    m = re.search(r'"score"\s*:\s*(\d)', text)
    return int(m.group(1)) if m else 3


correct = 0
total_time = 0

for i, q in enumerate(sampled):
    context_id = q.get('shared_context_id', '')
    question = q['user_question_or_message']
    options = q.get('all_options', '')
    answer_ref = q.get('correct_answer', '')
    end_idx = int(q.get('end_index_in_shared_context', 500))

    t0 = time.perf_counter()

    db = tempfile.mktemp(suffix='.db')
    config = {'db_path': db, 'llm_extract': True,
              'extraction_model': {'provider': 'openrouter', 'model': 'owl-alpha'},
              'trivial_filter': True,
              'circuit_breaker': {'max_failures': 5, 'cooldown_seconds': 120},
              'reflection': False}
    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id=f'pm-{i}')

    try:
        process_context(provider, shared_messages, end_index=min(len(shared_messages), end_idx))
    except Exception as e:
        print(f'  [{i+1}] Extraction error: {e}')

    n_facts = provider._store.count_facts() if provider._store else 0

    try:
        parsed_opts = _parse_options(options)
        hypothesis = answer_question(provider, question, parsed_opts)
    except Exception as e:
        hypothesis = f'Error: {e}'

    score = judge_answer(question, options, answer_ref, hypothesis)
    elapsed = time.perf_counter() - t0
    total_time += elapsed

    if score >= 4:
        correct += 1

    mark = 'V' if score >= 4 else 'X'
    print(f'  [{i+1}/{len(sampled)}] [{mark}] score={score} facts={n_facts} {question[:50]}')

    provider.shutdown()
    try:
        os.unlink(db)
    except OSError:
        pass

    # Rate limit: 1.5s delay between questions
    time.sleep(1.5)

# Summary
print()
print('=' * 65)
acc = correct / len(sampled) * 100 if sampled else 0
avg_t = total_time / len(sampled) if sampled else 0
print(f'  PersonaMem 32K: {correct}/{len(sampled)} = {acc:.0f}%  avg={avg_t:.1f}s/q')
print('=' * 65)
