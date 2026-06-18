#!/usr/bin/env python3
"""Verify L0 DB and test FTS5 properly."""
import sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row

# FTS5 snippet with correct column index (0 = content)
cursor = conn.execute("""
    SELECT rowid, snippet(conversation_turns_fts, 0, '<b>', '</b>', '...', 40)
    FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'dancing'
    LIMIT 5
""")
print("FTS5 search 'dancing':")
for row in cursor.fetchall():
    print(f"  turn {row[0]}: {row[1]}")

# Search for 'studio'
cursor = conn.execute("""
    SELECT rowid, snippet(conversation_turns_fts, 0, '<b>', '</b>', '...', 40)
    FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'studio'
    LIMIT 5
""")
print("\nFTS5 search 'studio':")
for row in cursor.fetchall():
    print(f"  turn {row[0]}: {row[1]}")

# Search for 'adoption'
cursor = conn.execute("""
    SELECT rowid, snippet(conversation_turns_fts, 0, '<b>', '</b>', '...', 40)
    FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'adoption'
    LIMIT 5
""")
print("\nFTS5 search 'adoption':")
for row in cursor.fetchall():
    print(f"  turn {row[0]}: {row[1]}")

# Show session summary
cursor = conn.execute("""
    SELECT session_id, 
           COUNT(*) as turns,
           MIN(turn_order) as first,
           MAX(turn_order) as last,
           COUNT(CASE WHEN role='user' THEN 1 END) as user_turns,
           COUNT(CASE WHEN role='assistant' THEN 1 END) as asst_turns
    FROM conversation_turns 
    GROUP BY session_id 
    ORDER BY session_id
""")
print("\nSession summary:")
total = 0
for row in cursor.fetchall():
    total += row['turns']
    print(f"  {row['session_id']}: {row['turns']} turns (U:{row['user_turns']} A:{row['asst_turns']})")
print(f"Total: {total} turns")

# Verify schema
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print("\nTables:")
for row in cursor.fetchall():
    print(f"  {row[0]}")

conn.close()
