#!/usr/bin/env python3
"""Print session 2-3 for LLM extraction."""
import sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row

for sid in ["session_2", "session_3"]:
    turns = conn.execute("SELECT turn_order, role, content FROM conversation_turns WHERE session_id=? ORDER BY turn_order", (sid,)).fetchall()
    print(f"\n=== {sid} ({len(turns)} turns) ===")
    for t in turns:
        speaker = "Caroline" if t["role"] == "user" else "Melanie"
        print(f"  [{t['turn_order']}] {speaker}: {t['content']}")
