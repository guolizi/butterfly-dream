#!/usr/bin/env python3
"""
Step 2: Create L1 tables in conv26_v2.db.
"""
import sqlite3
import os

DB_PATH = "eval/dbs/locomo/conv26_v2.db"

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA foreign_keys=ON")
conn.row_factory = sqlite3.Row

# Register jieba_segment stub for FTS5 triggers
conn.create_function("jieba_segment", 1, lambda x: x)

print("=== Creating L1 tables ===")

conn.executescript("""
-- 3.1 统一事实表
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    content         TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'event'
                    CHECK(type IN ('event', 'knowledge', 'behavior')),
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    importance      REAL DEFAULT 0.5,
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    is_persistent   INTEGER DEFAULT 0,
    content_date    TEXT,
    created_at      TEXT DEFAULT (datetime('now','localtime')),
    updated_at      TEXT DEFAULT (datetime('now','localtime')),
    heat_zone       TEXT DEFAULT 'hot'
                    CHECK(heat_zone IN ('hot', 'warm', 'cold', 'ice')),
    cooling_factor  REAL DEFAULT 1.0,
    emotion_tag     TEXT,
    abstract_level  INTEGER DEFAULT 0,
    is_abstract     INTEGER DEFAULT 0,
    embedding       BLOB,
    structured_data TEXT,
    UNIQUE(person, content)
);

CREATE INDEX IF NOT EXISTS idx_facts_person_type
    ON facts(person, type);
CREATE INDEX IF NOT EXISTS idx_facts_importance
    ON facts(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created
    ON facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_content_date
    ON facts(content_date);
CREATE INDEX IF NOT EXISTS idx_facts_heat_zone
    ON facts(heat_zone) WHERE heat_zone IN ('hot', 'warm');
CREATE INDEX IF NOT EXISTS idx_facts_person_abstract
    ON facts(person, is_abstract) WHERE is_abstract = 1;

-- FTS5 for facts
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, category, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags, category)
        VALUES (new.fact_id, jieba_segment(new.content), new.tags, new.category);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags, category)
        VALUES ('delete', old.fact_id, old.content, old.tags, old.category);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags, category)
        VALUES ('delete', old.fact_id, old.content, old.tags, old.category);
    INSERT INTO facts_fts(rowid, content, tags, category)
        VALUES (new.fact_id, jieba_segment(new.content), new.tags, new.category);
END;

-- 3.2 行为模式
CREATE TABLE IF NOT EXISTS behavior_patterns (
    pattern_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id             INTEGER NOT NULL UNIQUE REFERENCES facts(fact_id) ON DELETE CASCADE,
    person              TEXT NOT NULL,
    status              TEXT DEFAULT 'tentative'
                        CHECK(status IN ('tentative', 'confirming', 'confirmed', 'superseded', 'evolved')),
    confidence          REAL DEFAULT 0.0,
    valid_from          TEXT,
    valid_until         TEXT,
    pattern_type        TEXT DEFAULT 'routine'
                        CHECK(pattern_type IN ('routine', 'emotion-driven', 'value-driven')),
    evolved_from_pattern_id INTEGER REFERENCES behavior_patterns(pattern_id) ON DELETE SET NULL,
    source_fact_ids     TEXT,
    created_at          TEXT DEFAULT (datetime('now','localtime')),
    updated_at          TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_bp_person_status
    ON behavior_patterns(person, status);
CREATE INDEX IF NOT EXISTS idx_bp_confidence
    ON behavior_patterns(confidence DESC);

-- 3.3 实体
CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    embedding   BLOB,
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(person, name)
);

CREATE INDEX IF NOT EXISTS idx_entities_person_name
    ON entities(person, name);

-- 事实-实体关联
CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

-- 3.4 事实间关系
CREATE TABLE IF NOT EXISTS fact_relations (
    relation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person        TEXT NOT NULL,
    source_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    target_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL
                  CHECK(relation_type IN ('abstracts_from', 'contradicted_by', 'supports', 'evolved_from')),
    context       TEXT,
    created_at    TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(source_fact_id, target_fact_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_fr_target
    ON fact_relations(target_fact_id);
CREATE INDEX IF NOT EXISTS idx_fr_source
    ON fact_relations(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fr_type
    ON fact_relations(relation_type);

-- 3.5 合并日志
CREATE TABLE IF NOT EXISTS merge_log (
    merge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person           TEXT NOT NULL,
    kept_fact_id     INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    absorbed_fact_id INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
    merged_content   TEXT,
    merge_reason     TEXT DEFAULT 'auto',
    created_at       TEXT DEFAULT (datetime('now','localtime'))
);

-- 3.6 媒体附件
CREATE TABLE IF NOT EXISTS media_attachments (
    media_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    storage_type  TEXT NOT NULL DEFAULT 'file' CHECK(storage_type IN ('file', 'url')),
    file_path     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    file_size     INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    caption       TEXT DEFAULT '',
    transcript    TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now','localtime'))
);

-- 4.4 溯源
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person        TEXT NOT NULL,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    source_type   TEXT NOT NULL
                  CHECK(source_type IN ('llm_extraction', 'l0_promotion', 'l3_abstraction',
                                        'l4_narrative', 'user_input', 'historical_import')),
    source_session_id TEXT,
    source_turn_id    INTEGER REFERENCES conversation_turns(turn_id) ON DELETE SET NULL,
    confidence        REAL DEFAULT 0.7,
    created_at        TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_prov_fact
    ON provenance(fact_id);
""")

conn.commit()

# Verify
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cursor.fetchall()]
print("Tables after L1 creation:")
for t in tables:
    print(f"  {t}")

conn.close()
print("\n✅ L1 tables created!")
