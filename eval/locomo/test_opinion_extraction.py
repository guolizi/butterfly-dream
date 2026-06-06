#!/usr/bin/env python3
"""Test opinion category extraction on conv-30 session_3."""

import sys, json, os, tempfile, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from butterfly_dream import ButterflyDreamMemoryProvider
from run_locomo import flatten_session, _load_hermes_env, get_model_config

logging.basicConfig(level=logging.INFO)
_load_hermes_env()

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'locomo10.json')
    with open(data_path) as f:
        data = json.load(f)

    conv30 = data[1]
    conv = conv30["conversation"]
    print(f"Conv-30, session_3 共 14 条消息\n")

    # Show conversation
    msgs = flatten_session(conv, "session_3")
    for msg in msgs:
        icon = "👤" if msg["role"] == "user" else "🤖"
        print(f"  {icon} {msg['content'][:200]}")
        print()
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="opinion_test_")
    db_path = os.path.join(tmpdir, "test.db")

    model_cfg = get_model_config("extraction")
    print(f"Model: {model_cfg.get('model')}\n")
    sys.stdout.flush()

    provider = ButterflyDreamMemoryProvider({
        "db_path": db_path,
        "llm_extract": True,
        "extraction_model": model_cfg,
        "trivial_filter": True,
        "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
        "reflection": False,
    })
    provider.initialize(session_id="opinion-test-s3")

    print("⏳ 开始提取...")
    sys.stdout.flush()
    t0 = time.time()

    before = provider._store.count_facts() if provider._store else 0
    provider._last_extracted_idx = 0
    provider.on_session_end(msgs)

    # Wait up to 90s for extraction to complete
    last_n = 0
    stable_count = 0
    for i in range(180):  # 90 seconds max
        time.sleep(0.5)
        if not provider._store:
            continue
        n = provider._store.count_facts()
        if n != last_n:
            print(f"  [{i*0.5:.0f}s] facts: {n}", end="\r")
            sys.stdout.flush()
            last_n = n
            stable_count = 0
        elif n > 0:
            stable_count += 1
            if stable_count > 10:  # stable for 5s
                print(f"\n  [{i*0.5:.0f}s] stable at {n} facts ✓")
                break
        if i == 170:
            print(f"\n  ⏰ timeout ({i*0.5:.0f}s)")

    elapsed = time.time() - t0
    n_facts = provider._store.count_facts() if provider._store else 0
    print(f"\n⏱ {elapsed:.0f}s — 共提取 {n_facts} 条事实\n")

    # Read all facts from DB
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT f.fact_id, f.content, f.category, f.importance, f.content_date, "
                       "GROUP_CONCAT(fe.entity_id) as ents "
                       "FROM facts f LEFT JOIN fact_entities fe ON f.fact_id=fe.fact_id "
                       "GROUP BY f.fact_id ORDER BY f.fact_id")
    db_facts = cur.fetchall()
    conn.close()

    if not db_facts:
        print("❌ 无任何事实被提取")
        # 检查线程状态
        print(f"  线程数: {len(provider._extract_threads)}")
        for t in provider._extract_threads:
            print(f"  线程 {t.name}: alive={t.is_alive()}")
    else:
        for f in db_facts:
            fid, content, cat, imp, cdate, ents = f
            date_str = f" [{cdate}]" if cdate else ""
            ents_str = ents or "—"
            print(f"  fact[{fid:2d}] {cat:15s} imp={int(imp):2d}{date_str} ent={ents_str}")
            print(f"    {content[:180]}")
            print()

        opinion_facts = [f for f in db_facts if f[2] == 'opinion']
        if opinion_facts:
            print("=" * 60)
            print(f"📌 opinion 类事实 ({len(opinion_facts)} 条):")
            for f in opinion_facts:
                print(f"  fact[{f[0]}] imp={int(f[3])}: {f[1][:150]}")
        else:
            print("❌ 没有 opinion 类事实")

    provider.shutdown()
    import shutil
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
