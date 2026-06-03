#!/usr/bin/env python3
"""Sample 3 LongMemEval questions per dimension and evaluate."""

import json, random, sys, os, tempfile, time, re
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'longmemeval'))

from butterfly_dream import ButterflyDreamMemoryProvider
from run_longmemeval import process_sessions, answer_question

from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env
_load_hermes_env()

random.seed(42)

# Load data
with open(PROJECT / 'eval' / 'longmemeval' / 'data' / 'longmemeval_oracle.json') as f:
    data = json.load(f)

# Group by question_type
by_type = defaultdict(list)
for e in data:
    by_type[e['question_type']].append(e)

# Sample 3 per type
sampled = []
types = sorted(by_type.keys())
for t in types:
    picked = random.sample(by_type[t], min(3, len(by_type[t])))
    sampled.extend(picked)

print(f'LongMemEval: sampled {len(sampled)} questions across {len(types)} types')
for t in types:
    count = sum(1 for s in sampled if s['question_type'] == t)
    print(f'  {t}: {count}')

def judge_answer(question, answer, hypothesis):
    prompt = (
        f'Score 1-5: 5=correct+complete, 4=correct+incomplete, '
        f'3=partially correct, 2=mostly wrong, 1=completely wrong\n\n'
        f'Q: {question}\nRef: {answer}\nGen: {hypothesis}\n'
        f'Output ONLY: {{"score": <1-5>}}'
    )
    messages = [{'role': 'user', 'content': prompt}]
    text = call_llm('judge', messages=messages, temperature=0)
    if not text:
        return 3
    m = re.search(r'"score"\s*:\s*(\d)', text)
    return int(m.group(1)) if m else 3


# Run
type_scores = defaultdict(list)
type_correct = defaultdict(int)
total_time = 0

for i, entry in enumerate(sampled):
    qid = entry['question_id']
    qtype = entry['question_type']
    question = entry['question']
    answer = entry['answer']

    t0 = time.perf_counter()

    # Fresh provider per question
    db = tempfile.mktemp(suffix='.db')
    config = {'db_path': db, 'llm_extract': True,
              'extraction_model': {'provider': 'openrouter', 'model': 'owl-alpha'},
              'trivial_filter': True,
              'circuit_breaker': {'max_failures': 5, 'cooldown_seconds': 120},
              'reflection': False}
    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id=f'lme-{qid}')

    try:
        process_sessions(provider, entry['haystack_sessions'])
    except Exception as e:
        print(f'  [{i+1}] Extraction error {qid}: {e}')

    n_facts = provider._store.count_facts() if provider._store else 0

    try:
        hypothesis = answer_question(provider, question)
    except Exception as e:
        hypothesis = f'Error: {e}'

    score = judge_answer(question, answer, hypothesis)
    elapsed = time.perf_counter() - t0
    total_time += elapsed

    type_scores[qtype].append(score)
    if score >= 4:
        type_correct[qtype] += 1

    mark = 'V' if score >= 4 else 'X'
    print(f'  [{i+1}/{len(sampled)}] [{mark}] {qtype:30s} score={score} facts={n_facts} {question[:45]}')

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
total_correct = sum(type_correct.values())
total_count = sum(len(v) for v in type_scores.values())
for t in types:
    scores = type_scores[t]
    if scores:
        n = len(scores)
        c = type_correct[t]
        avg = sum(scores) / len(scores)
        print(f'  {t:35s} {c}/{n} = {c/n*100:.0f}%  avg={avg:.1f}')
acc = total_correct / total_count * 100 if total_count else 0
avg_t = total_time / total_count if total_count else 0
print(f'  {"Total":35s} {total_correct}/{total_count} = {acc:.0f}%  avg={avg_t:.1f}s/q')
print('=' * 65)
