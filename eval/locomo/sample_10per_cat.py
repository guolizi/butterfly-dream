#!/usr/bin/env python3
"""Sample 3 questions per LoCoMo dimension and evaluate."""

import json, random, sys, os, tempfile, time, re
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from butterfly_dream import ButterflyDreamMemoryProvider
from run_locomo import flatten_conversation, flatten_session, get_session_names, load_dataset, CAT_NAMES, process_conversation

from eval_utils import get_model_config, resolve_credentials, call_llm, _load_hermes_env
_load_hermes_env()

random.seed(42)

data = load_dataset(str(Path(__file__).resolve().parent / 'data' / 'locomo10.json'))

all_qa = []
for conv in data:
    conv_id = conv['sample_id']
    conv_dict = conv['conversation']
    for qa in conv['qa']:
        all_qa.append({'conv_id': conv_id, 'conv_dict': conv_dict, 'qa': qa})

# Sample 3 per category
sampled = []
for cat in [1, 2, 3, 4, 5]:
    pool = [q for q in all_qa if q['qa']['category'] == cat and 'answer' in q['qa']]
    picked = random.sample(pool, min(3, len(pool)))
    sampled.extend(picked)

print(f'每维度抽 3 题, 共 {len(sampled)} 题')

by_conv = defaultdict(list)
for s in sampled:
    by_conv[s['conv_id']].append(s)

def judge_answer(question, answer, hypothesis):
    prompt = (
        f'Given a question, reference answer, and generated answer, score 1-5.\n'
        f'5=Correct+complete, 4=Correct+incomplete, 3=Partially correct, '
        f'2=Mostly wrong, 1=Completely wrong\n\n'
        f'Question: {question}\nReference: {answer}\nGenerated: {hypothesis}\n\n'
        f'Output ONLY: {{"score": <1-5>}}'
    )
    messages = [{'role': 'user', 'content': prompt}]
    text = call_llm('judge', messages=messages, temperature=0)
    if not text:
        return 3
    m = re.search(r'"score"\s*:\s*(\d)', text)
    return int(m.group(1)) if m else 3


def answer_question(provider, question):
    from butterfly_dream.retrieval import ThreeDimRetriever
    retriever = ThreeDimRetriever(provider._store)
    results = retriever.search(query=question, scenario='chat', limit=15)
    context = '\n'.join(r.get('content', '') for r in results)
    prompt = (
        f'You are answering questions about a long conversation between two people. '
        f'Use ALL relevant facts from the context below. Be specific — include names, '
        f'dates, locations, and details. If multiple facts relate to the question, '
        f'combine them into a complete answer.\n\n'
        f'Context:\n{context}\n\n'
        f'Question: {question}\n\n'
        f'Answer (be specific and complete):'
    )
    messages = [
        {'role': 'system', 'content': 'Answer based on the memory context. Be specific with names, dates, and details. Combine multiple relevant facts into a complete answer.'},
        {'role': 'user', 'content': prompt}]
    result = call_llm('answer', messages=messages, max_tokens=1024)
    return result if result else 'Error: no API key configured'

cat_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
cat_correct = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_time = 0

for conv_id, qa_list in by_conv.items():
    conv_dict = [c for c in data if c['sample_id'] == conv_id][0]['conversation']
    session_names = get_session_names(conv_dict)
    total_turns = sum(len(conv_dict[s]) for s in session_names)
    print(f'\n--- {conv_id}: {len(session_names)} sessions, {total_turns} turns, {len(qa_list)} QA ---')

    db = tempfile.mktemp(suffix='.db')
    config = {'db_path': db, 'llm_extract': True,
              'extraction_model': {'provider': 'openrouter', 'model': 'owl-alpha'},
              'trivial_filter': True}
    provider = ButterflyDreamMemoryProvider(config)
    provider.initialize(session_id=f'locomo-{conv_id}')
    t0 = time.perf_counter()
    process_conversation(provider, conv_dict)
    n_facts = provider._store.count_facts() if provider._store else 0
    print(f'  facts={n_facts}, {time.perf_counter()-t0:.1f}s')

    for item in qa_list:
        qa = item['qa']
        cat = qa['category']
        answer_ref = qa.get('answer') or qa.get('adversarial_answer') or ''
        if isinstance(answer_ref, list):
            answer_ref = answer_ref[0] if answer_ref else ''
        if not answer_ref:
            continue
        t1 = time.perf_counter()
        hyp = answer_question(provider, qa['question'])
        score = judge_answer(qa['question'], answer_ref, hyp)
        elapsed = time.perf_counter() - t1
        total_time += elapsed
        cat_scores[cat].append(score)
        if score >= 4:
            cat_correct[cat] += 1
        mark = 'V' if score >= 4 else 'X'
        print(f'  [{mark}] cat={cat} score={score} {qa["question"][:55]}')

    provider.shutdown()
    try:
        os.unlink(db)
    except OSError:
        pass

    # Rate limit: 1.5s delay between questions
    time.sleep(1.5)

# Summary
print()
print('=' * 60)
total_correct = sum(cat_correct.values())
total_count = sum(len(v) for v in cat_scores.values())
for cat in [1, 2, 3, 4, 5]:
    scores = cat_scores[cat]
    if scores:
        n = len(scores)
        c = cat_correct[cat]
        avg = sum(scores) / len(scores)
        name = CAT_NAMES.get(cat, '?')
        print(f'  [{cat}] {name:30s} {c}/{n} = {c/n*100:.0f}%  avg={avg:.1f}')
acc = total_correct / total_count * 100 if total_count else 0
avg_t = total_time / total_count if total_count else 0
print(f'  {"Total":31s} {total_correct}/{total_count} = {acc:.0f}%  avg={avg_t:.1f}s/q')
print('=' * 60)
