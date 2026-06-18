#!/usr/bin/env python3
"""
Step 1: Create v2 L0 database and import conv-26 conversation data.
FTS5 triggers use simple identity function (no jieba dependency for now).
"""
import json
import sqlite3
import os

DB_PATH = "eval/dbs/locomo/conv26_v2.db"
DATA_PATH = "eval/locomo/data/locomo10.json"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.row_factory = sqlite3.Row

# Register a simple jieba_segment stub (just returns the text as-is)
conn.create_function("jieba_segment", 1, lambda x: x)

print("=== Creating L0 tables ===")

conn.executescript("""
CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    turn_order  INTEGER NOT NULL,
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_turns_person_session
    ON conversation_turns(person, session_id, turn_order);

CREATE VIRTUAL TABLE IF NOT EXISTS conversation_turns_fts
    USING fts5(content, person, content=conversation_turns, content_rowid=turn_id);

CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(rowid, content, person)
        VALUES (new.turn_id, jieba_segment(new.content), new.person);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(conversation_turns_fts, rowid, content, person)
        VALUES ('delete', old.turn_id, old.content, old.person);
END;

CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(conversation_turns_fts, rowid, content, person)
        VALUES ('delete', old.turn_id, old.content, old.person);
    INSERT INTO conversation_turns_fts(rowid, content, person)
        VALUES (new.turn_id, jieba_segment(new.content), new.person);
END;

CREATE TABLE IF NOT EXISTS micro_facts (
    micro_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    turn_id     INTEGER NOT NULL REFERENCES conversation_turns(turn_id) ON DELETE CASCADE,
    snippet     TEXT NOT NULL,
    promoted    INTEGER DEFAULT 0,
    miss_count  INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_micro_person_keyword
    ON micro_facts(person, keyword);
CREATE INDEX IF NOT EXISTS idx_micro_promoted
    ON micro_facts(promoted) WHERE promoted = 0;

CREATE TABLE IF NOT EXISTS promotion_queue (
    queue_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    keyword     TEXT NOT NULL,
    miss_count  INTEGER DEFAULT 1 CHECK(miss_count >= 1),
    turn_ids    TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    updated_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(person, keyword)
);
""")

print("✅ L0 tables created")

# === Import conv-26 data ===
print("\n=== Importing conv-26 conversation ===")

with open(DATA_PATH, 'r') as f:
    data = json.load(f)

conv26 = [d for d in data if d["sample_id"] == "conv-26"][0]
conv = conv26["conversation"]
person = "Caroline"

session_keys = sorted(
    [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")],
    key=lambda k: int(k.split("_")[1])
)

total_turns = 0
for sk in session_keys:
    session_id = sk
    turns = conv[sk]
    for turn_order, turn in enumerate(turns, 1):
        speaker = turn["speaker"]
        text = turn["text"]
        role = "user" if speaker == person else "assistant"
        conn.execute(
            "INSERT INTO conversation_turns (person, session_id, role, content, turn_order) VALUES (?, ?, ?, ?, ?)",
            (person, session_id, role, text, turn_order)
        )
        total_turns += 1

conn.commit()
print(f"✅ Imported {total_turns} turns for person='{person}'")
print(f"   Sessions: {len(session_keys)}")

# Verify
cursor = conn.execute("SELECT COUNT(*) FROM conversation_turns")
print(f"   DB count: {cursor.fetchone()[0]}")

cursor = conn.execute("""
    SELECT role, COUNT(*) as cnt FROM conversation_turns GROUP BY role
""")
print("\nRole distribution:")
for row in cursor.fetchall():
    print(f"  {row['role']}: {row['cnt']}")

# Verify FTS5
cursor = conn.execute("SELECT COUNT(*) FROM conversation_turns_fts")
print(f"\nFTS5 index: {cursor.fetchone()[0]} rows")

# Show a sample FTS5 search
cursor = conn.execute("""
    SELECT rowid, snippet(conversation_turns_fts, 1, '<b>', '</b>', '...', 30)
    FROM conversation_turns_fts
    WHERE conversation_turns_fts MATCH 'dance'
    LIMIT 5
""")
print("\nFTS5 search 'dance':")
for row in cursor.fetchall():
    print(f"  turn {row[0]}: {row[1][:120]}")

conn.close()
print("\n✅ L0 database ready!")
