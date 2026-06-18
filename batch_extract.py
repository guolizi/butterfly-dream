"""
L1 事实批量提取入库脚本
用法: python3 batch_extract.py <session_ids>
"""
import sqlite3
import json
import sys
import re

DB = '/home/xx/butterfly-dream/memory_store.db'
CONV_FILE = '/home/xx/butterfly-dream/eval/locomo/data/conv-26_bilingual.md'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def parse_sessions():
    """解析 conv-26 文件，返回 {session_id: {"time": str, "text": str}}"""
    with open(CONV_FILE, 'r') as f:
        content = f.read()
    
    sessions = {}
    # 匹配 session 标题行
    pattern = r'## session_(\d+) \(([^)]+)\)\n(.*?)(?=\n## session_\d+|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for sid, time_str, text in matches:
        sessions[f"session_{sid}"] = {
            "time": time_str.strip(),
            "text": text.strip()
        }
    
    return sessions

def create_tables_if_not_exist(conn):
    """确保所有 L1 表存在"""
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
    CREATE INDEX IF NOT EXISTS idx_facts_person_type ON facts(person, type);
    CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance DESC);
    CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_facts_content_date ON facts(content_date);
    CREATE INDEX IF NOT EXISTS idx_facts_heat_zone ON facts(heat_zone) WHERE heat_zone IN ('hot', 'warm');
    CREATE INDEX IF NOT EXISTS idx_facts_person_abstract ON facts(person, is_abstract) WHERE is_abstract = 1;

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
    CREATE INDEX IF NOT EXISTS idx_entities_person_name ON entities(person, name);

    CREATE TABLE IF NOT EXISTS fact_entities (
        fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        PRIMARY KEY (fact_id, entity_id)
    );

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
    CREATE INDEX IF NOT EXISTS idx_fr_target ON fact_relations(target_fact_id);
    CREATE INDEX IF NOT EXISTS idx_fr_source ON fact_relations(source_fact_id);
    CREATE INDEX IF NOT EXISTS idx_fr_type ON fact_relations(relation_type);

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
    CREATE INDEX IF NOT EXISTS idx_prov_fact ON provenance(fact_id);

    CREATE TABLE IF NOT EXISTS emotion_events (
        event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        person          TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        emotion_model   TEXT DEFAULT 'vad-3d'
                        CHECK(emotion_model IN ('vad-3d', 'emotion-21d')),
        emotion_vector  TEXT NOT NULL,
        valence   REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(emotion_vector, '$[0]')
            END
        ),
        arousal   REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(emotion_vector, '$[1]')
            END
        ),
        dominance REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(emotion_vector, '$[2]')
            END
        ),
        emotion_label   TEXT,
        emotion_target  TEXT,
        intensity       REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN sqrt(
                (json_extract(emotion_vector, '$[0]') * json_extract(emotion_vector, '$[0]') +
                 json_extract(emotion_vector, '$[1]') * json_extract(emotion_vector, '$[1]') +
                 json_extract(emotion_vector, '$[2]') * json_extract(emotion_vector, '$[2]')) / 3.0
            )
            END
        ),
        primary_fact_id     INTEGER REFERENCES facts(fact_id) ON DELETE SET NULL,
        related_fact_ids    TEXT,
        source          TEXT NOT NULL DEFAULT 'user'
                        CHECK(source IN ('user', 'assistant', 'l0_promotion', 'inferred')),
        initial_importance  REAL DEFAULT 0.5,
        significance_reason TEXT,
        trigger_topics      TEXT,
        appraisal_dimensions TEXT,
        notes           TEXT,
        created_at      TEXT DEFAULT (datetime('now','localtime')),
        updated_at      TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_ee_person_time ON emotion_events(person, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_ee_person_source ON emotion_events(person, source);
    CREATE INDEX IF NOT EXISTS idx_ee_primary_fact ON emotion_events(primary_fact_id);

    CREATE TABLE IF NOT EXISTS emotion_triggers (
        trigger_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        person          TEXT NOT NULL,
        trigger_type    TEXT NOT NULL
                        CHECK(trigger_type IN ('topic', 'entity', 'event_type', 'location')),
        trigger_value   TEXT NOT NULL,
        typical_valence   REAL,
        typical_arousal   REAL,
        typical_dominance REAL,
        trigger_count     INTEGER DEFAULT 1,
        confidence        REAL DEFAULT 0.5,
        notes             TEXT,
        created_at        TEXT DEFAULT (datetime('now','localtime')),
        updated_at        TEXT DEFAULT (datetime('now','localtime')),
        UNIQUE(person, trigger_type, trigger_value)
    );
    CREATE INDEX IF NOT EXISTS idx_et_person_type ON emotion_triggers(person, trigger_type);

    CREATE TABLE IF NOT EXISTS emotion_states (
        state_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        person          TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        session_id      TEXT,
        emotion_model   TEXT DEFAULT 'vad-3d'
                        CHECK(emotion_model IN ('vad-3d', 'emotion-21d')),
        current_vector  TEXT NOT NULL,
        valence   REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(current_vector, '$[0]')
            END
        ),
        arousal   REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(current_vector, '$[1]')
            END
        ),
        dominance REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN json_extract(current_vector, '$[2]')
            END
        ),
        intensity       REAL GENERATED ALWAYS AS (
            CASE WHEN emotion_model = 'vad-3d'
            THEN sqrt(
                (json_extract(current_vector, '$[0]') * json_extract(current_vector, '$[0]') +
                 json_extract(current_vector, '$[1]') * json_extract(current_vector, '$[1]') +
                 json_extract(current_vector, '$[2]') * json_extract(current_vector, '$[2]')) / 3.0
            )
            END
        ),
        source_event_ids  TEXT,
        trigger_topics    TEXT,
        source          TEXT NOT NULL DEFAULT 'user'
                        CHECK(source IN ('user', 'assistant', 'l0_promotion', 'inferred')),
        created_at      TEXT DEFAULT (datetime('now','localtime'))
    );
    CREATE INDEX IF NOT EXISTS idx_es_person_time ON emotion_states(person, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_es_person_source ON emotion_states(person, source);
    """)
    conn.commit()
    print("✅ 表已就绪")

def insert_extraction_result(conn, person, session_id, items):
    """将提取结果写入数据库"""
    facts_inserted = 0
    emotions_inserted = 0
    relations_inserted = 0
    
    for item in items:
        dim = item.get("dimension", item.get("type", ""))
        
        if dim == "event":
            try:
                sd = json.dumps(item.get("structured_data"), ensure_ascii=False) if item.get("structured_data") else None
            except:
                sd = None
            conn.execute("""
                INSERT OR IGNORE INTO facts(person, content, type, category, tags, importance, content_date, emotion_tag, structured_data)
                VALUES (?, ?, 'event', ?, ?, ?, ?, ?, ?)
            """, (person, item["content"], item.get("category", "general"), 
                  item.get("tags", ""), item.get("importance", 0.5),
                  item.get("content_date"), item.get("emotion_tag"), sd))
            if conn.total_changes > 0:
                facts_inserted += 1
                fid = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content=?", (person, item["content"])).fetchone()
                if fid:
                    # provenance
                    conn.execute("""
                        INSERT INTO provenance(person, fact_id, source_type, source_session_id, confidence)
                        VALUES (?, ?, 'llm_extraction', ?, 0.8)
                    """, (person, fid[0], session_id))
                    # entities
                    for ent_name in item.get("entities", []):
                        conn.execute("INSERT OR IGNORE INTO entities(person, name) VALUES (?, ?)", (person, ent_name))
                        eid = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", (person, ent_name)).fetchone()
                        if eid:
                            conn.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?, ?)", (fid[0], eid[0]))
        
        elif dim == "knowledge":
            conn.execute("""
                INSERT OR IGNORE INTO facts(person, content, type, category, tags, importance, emotion_tag)
                VALUES (?, ?, 'knowledge', ?, ?, ?, ?)
            """, (person, item["content"], item.get("category", "general"),
                  item.get("tags", ""), item.get("importance", 0.5),
                  item.get("emotion_tag")))
            if conn.total_changes > 0:
                facts_inserted += 1
                fid = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content=?", (person, item["content"])).fetchone()
                if fid:
                    conn.execute("""
                        INSERT INTO provenance(person, fact_id, source_type, source_session_id, confidence)
                        VALUES (?, ?, 'llm_extraction', ?, 0.8)
                    """, (person, fid[0], session_id))
                    for ent_name in item.get("entities", []):
                        conn.execute("INSERT OR IGNORE INTO entities(person, name) VALUES (?, ?)", (person, ent_name))
                        eid = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", (person, ent_name)).fetchone()
                        if eid:
                            conn.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?, ?)", (fid[0], eid[0]))
        
        elif dim == "behavior":
            conn.execute("""
                INSERT OR IGNORE INTO facts(person, content, type, category, tags, importance, emotion_tag)
                VALUES (?, ?, 'behavior', ?, ?, ?, ?)
            """, (person, item["content"], item.get("category", "general"),
                  item.get("tags", ""), item.get("importance", 0.5),
                  item.get("emotion_tag")))
            if conn.total_changes > 0:
                facts_inserted += 1
                fid = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content=?", (person, item["content"])).fetchone()
                if fid:
                    conn.execute("""
                        INSERT INTO provenance(person, fact_id, source_type, source_session_id, confidence)
                        VALUES (?, ?, 'llm_extraction', ?, 0.8)
                    """, (person, fid[0], session_id))
                    for ent_name in item.get("entities", []):
                        conn.execute("INSERT OR IGNORE INTO entities(person, name) VALUES (?, ?)", (person, ent_name))
                        eid = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", (person, ent_name)).fetchone()
                        if eid:
                            conn.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?, ?)", (fid[0], eid[0]))
        
        elif dim == "emotion":
            # Find primary fact
            pfid = None
            if item.get("emotion_target"):
                target = item["emotion_target"]
                if target.startswith("event:"):
                    target_content = target.replace("event:", "")
                    pf = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content LIKE ? LIMIT 1", 
                                     (person, f"%{target_content}%")).fetchone()
                    if pf:
                        pfid = pf[0]
            
            conn.execute("""
                INSERT INTO emotion_events(person, timestamp, emotion_vector, emotion_label,
                    emotion_target, primary_fact_id, source, initial_importance, significance_reason,
                    trigger_topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person, item.get("timestamp", ""), json.dumps(item.get("emotion_vector", [0,0.5,0.5])),
                  item.get("emotion_label"), item.get("emotion_target"), pfid,
                  item.get("source", "user"), item.get("importance", 0.5),
                  item.get("significance_reason"), json.dumps(item.get("trigger_topics", []), ensure_ascii=False)))
            emotions_inserted += 1
        
        elif dim == "relation":
            # Store relation info for later processing
            relations_inserted += 1
    
    conn.commit()
    return facts_inserted, emotions_inserted, relations_inserted

def show_stats(conn):
    """显示统计信息"""
    cur = conn.execute("SELECT type, COUNT(*) as cnt FROM facts GROUP BY type ORDER BY type")
    rows = cur.fetchall()
    print("\n📊 事实统计:")
    for r in rows:
        print(f"  {r['type']:10s}: {r['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM emotion_events")
    print(f"  情感事件 : {cur.fetchone()['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM entities")
    print(f"  实体     : {cur.fetchone()['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM provenance")
    print(f"  溯源记录 : {cur.fetchone()['cnt']}")

if __name__ == '__main__':
    conn = get_conn()
    create_tables_if_not_exist(conn)
    
    sessions = parse_sessions()
    print(f"📖 解析到 {len(sessions)} 个会话")
    
    if len(sys.argv) > 1:
        # 只处理指定会话
        target_sessions = sys.argv[1:]
        sessions = {k: v for k, v in sessions.items() if k in target_sessions}
        print(f"🎯 目标会话: {list(sessions.keys())}")
    
    # 清空旧数据（重新提取时）
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        conn.executescript("""
            DELETE FROM fact_entities;
            DELETE FROM fact_relations;
            DELETE FROM provenance;
            DELETE FROM emotion_triggers;
            DELETE FROM emotion_states;
            DELETE FROM emotion_events;
            DELETE FROM entities;
            DELETE FROM facts;
            DELETE FROM conversation_turns;
        """)
        conn.commit()
        print("🔄 已清空旧数据")
    
    print("\n📋 会话列表:")
    for sid, info in sessions.items():
        time_str = info['time']
        first_line = info['text'].split('\n')[0][:60] if info['text'] else '(空)'
        print(f"  {sid}: {time_str} — {first_line}...")
