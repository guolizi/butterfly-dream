#!/usr/bin/env python3
"""Sample 50 PersonaMem (32K) questions and evaluate."""

import csv, json, random, sys, os, tempfile, time, urllib.request, re
from pathlib import Path

# Load env
env_path = Path.home() / '.hermes' / '.env'
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        key, _, val = line.partition('=')
        if key.strip() not in os.environ:
            os.environ[key.strip()] = val.strip().strip('"')

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from butterfly_dream import ButterflyDreamMemoryProvider
from run_personamem import process_context, answer_question, _parse_options
from butterfly_dream.__init__ import _resolve_provider_credentials

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

# Sample 50 random questions
sampled = random.sample(questions, min(50, len(questions)))
print(f'PersonaMem 32K: sampled {len(sampled)} questions from {len(questions)} total')

# Load judge credentials
judge_base, judge_key = _resolve_provider_credentials('openrouter')


def judge_answer(question, options, correct_answer, hypothesis):
    prompt = (
        f'Multiple choice question. Score 1-5.\n'
        f'5=correct option selected, 4=mostly right, 3=unclear, 2=wrong, 1=completely wrong\n\n'
        f'Q: {question}\nOptions: {options}\nCorrect: {correct_answer}\n'
        f'Generated: {hypothesis}\nOutput ONLY: {{"score": <1-5>}}'
    )
    payload = {'model': 'owl-alpha',
               'messages': [{'role': 'user', 'content': prompt}],
               'temperature': 0, 'max_tokens': 500}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{judge_base}/chat/completions', data=body,
        headers={'Authorization': f'Bearer {judge_key}', 'Content-Type': 'application/json'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rdata = json.loads(resp.read())
        text = rdata['choices'][0]['message']['content'].strip()
        m = re.search(r'"score"\s*:\s*(\d)', text)
        return int(m.group(1)) if m else 3
    except Exception:
        return 3


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
