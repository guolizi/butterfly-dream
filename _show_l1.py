#!/usr/bin/env python3
"""Show L1 stats."""
import sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row

total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
by_type = conn.execute("SELECT type, COUNT(*) as cnt FROM facts GROUP BY type").fetchall()
by_cat = conn.execute("SELECT category, COUNT(*) as cnt FROM facts GROUP BY category ORDER BY cnt DESC").fetchall()
entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
provenance = conn.execute("SELECT source_session_id, COUNT(*) as cnt FROM provenance GROUP BY source_session_id ORDER BY source_session_id").fetchall()

print(f"\n🦋 L1 事实池统计")
print(f"{'─'*50}")
print(f"\n 📊 总览")
print(f"   事实总数: {total}")
print(f"   实体数:   {entities}")
print(f"\n 📋 按类型")
for r in by_type:
    print(f"   {r['type']}: {r['cnt']}")
print(f"\n 📋 按分类 Top 10")
for r in by_cat:
    print(f"   {r['category']}: {r['cnt']}")
print(f"\n 📋 按来源会话")
for r in provenance:
    print(f"   {r['source_session_id']}: {r['cnt']} 条")

print(f"\n 📝 前 10 条事实")
rows = conn.execute("SELECT fact_id, content, type, category, importance FROM facts ORDER BY fact_id LIMIT 10").fetchall()
for r in rows:
    print(f"   #{r['fact_id']} [{r['type']}/{r['category']}] imp={r['importance']:.1f}")
    print(f"     {r['content'][:80]}")

conn.close()
