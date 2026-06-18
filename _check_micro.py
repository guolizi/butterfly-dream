#!/usr/bin/env python3
"""Check for missing keywords and fix stop words."""
import sqlite3
import re

DB_PATH = "eval/dbs/locomo/conv26_v2.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Check if "studio" appears in conversation
cursor = conn.execute("""
    SELECT turn_id, content FROM conversation_turns
    WHERE content LIKE '%studio%' OR content LIKE '%Studio%'
""")
print("'studio' in conversation:")
for r in cursor.fetchall():
    print(f"  turn {r['turn_id']}: {r['content'][:100]}")

# Check "paris"
cursor = conn.execute("""
    SELECT turn_id, content FROM conversation_turns
    WHERE content LIKE '%paris%' OR content LIKE '%Paris%'
""")
print("\n'paris' in conversation:")
for r in cursor.fetchall():
    print(f"  turn {r['turn_id']}: {r['content'][:100]}")

# Check "dance" (without ing)
cursor = conn.execute("""
    SELECT turn_id, content FROM conversation_turns
    WHERE content LIKE '%dance%' OR content LIKE '%Dance%'
""")
print("\n'dance' in conversation:")
for r in cursor.fetchall():
    print(f"  turn {r['turn_id']}: {r['content'][:120]}")

# Check how many micro_facts have junk keywords
cursor = conn.execute("""
    SELECT keyword, COUNT(*) as cnt FROM micro_facts
    WHERE keyword IN ('what', 're', 've', 'll', 'm', 's', 't', 'don', 'didn', 'doesn', 'isn', 'aren', 'wasn', 'weren', 'haven', 'hasn', 'hadn', 'won', 'wouldn', 'couldn', 'shouldn', 'mightn', 'mustn', 'needn', 'daren')
    GROUP BY keyword ORDER BY cnt DESC
""")
print("\nJunk keywords:")
total_junk = 0
for r in cursor.fetchall():
    total_junk += r['cnt']
    print(f"  {r['keyword']}: {r['cnt']}")
print(f"Total junk: {total_junk}")

conn.close()
