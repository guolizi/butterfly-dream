#!/usr/bin/env python3
"""Check dance-related keywords in the micro_facts index."""
import sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row

# Check 'dance' in conversation
c = conn.execute("SELECT turn_id, content FROM conversation_turns WHERE LOWER(content) LIKE '%dance%'")
print("'dance' in conversation:")
for r in c.fetchall():
    print(f"  turn {r['turn_id']}: {r['content'][:120]}")

# Check what keywords we have around dance
c = conn.execute("SELECT DISTINCT keyword FROM micro_facts WHERE keyword LIKE '%dan%'")
print("\nmicro_facts with 'dan':")
for r in c.fetchall():
    print(f"  {r['keyword']}")

# Check FTS5 for dance
c = conn.execute("""
    SELECT rowid, snippet(conversation_turns_fts, 0, '<b>', '</b>', '...', 40)
    FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'danc*'
    LIMIT 5
""")
print("\nFTS5 'danc*':")
for r in c.fetchall():
    print(f"  turn {r[0]}: {r[1]}")

# Show some sample jieba POS tags for dance sentences
import jieba.posseg as pseg
sample = "I'm starting a dance studio"
print("\njieba POS for 'I'm starting a dance studio':")
for w, f in pseg.cut(sample):
    print(f"  {w} ({f})", end="")
print()

sample2 = "The dance group is performing"
print("\njieba POS for 'The dance group is performing':")
for w, f in pseg.cut(sample2):
    print(f"  {w} ({f})", end="")
print()

conn.close()
