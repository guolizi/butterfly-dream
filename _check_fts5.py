#!/usr/bin/env python3
"""Check FTS5 table schema."""
import sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")

# Check FTS5 schema
cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name='conversation_turns_fts'")
print("FTS5 schema:", cursor.fetchone()[0])

# Check columns
cursor = conn.execute("PRAGMA table_info(conversation_turns_fts)")
print("\nFTS5 columns:")
for r in cursor.fetchall():
    print(f"  {r}")

# Try a simple FTS5 query
cursor = conn.execute("""
    SELECT rowid, content FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'adoption'
    LIMIT 3
""")
print("\nFTS5 MATCH 'adoption':")
for r in cursor.fetchall():
    print(f"  rowid={r[0]}: {r[1][:80]}")

conn.close()
