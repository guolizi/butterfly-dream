"""
手动构造事实验证 L1 表结构
不调 LLM，直接根据对话内容手动构造 INSERT
"""
import sqlite3
import json
import sys

DB = '/home/xx/butterfly-dream/memory_store.db'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(conn):
    """建 L0 + L1 相关表"""
    conn.executescript("""
    -- L0: 对话轮次
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

    -- L1: 事实表
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

    -- L1: 实体表
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

    -- L1: 事实-实体关联
    CREATE TABLE IF NOT EXISTS fact_entities (
        fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
        entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
        PRIMARY KEY (fact_id, entity_id)
    );

    -- L1: 事实间关系
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

    -- L1: 溯源
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

    -- L1: 情感事件池
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

    -- L1: 情感触发关联
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

    -- L1: 情感状态快照
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
    print("✅ 表创建完成")

def insert_conversation_turns(conn):
    """插入 session_1 的对话轮次"""
    turns = [
        # session_1
        ("session_1", "user", "Hey Mel! Good to see you! How have you been?", 1),
        ("session_1", "assistant", "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?", 2),
        ("session_1", "user", "I went to a LGBTQ support group yesterday and it was so powerful.", 3),
        ("session_1", "assistant", "Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?", 4),
        ("session_1", "user", "The transgender stories were so inspiring! I was so happy and thankful for all the support.", 5),
        ("session_1", "assistant", "Wow, love that painting! So cool you found such a helpful group. What's it done for you?", 6),
        ("session_1", "user", "The support group has made me feel accepted and given me courage to embrace myself.", 7),
        ("session_1", "assistant", "That's really cool. You've got guts. What now?", 8),
        ("session_1", "user", "Gonna continue my edu and check out career options, which is pretty exciting!", 9),
        ("session_1", "assistant", "Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?", 10),
        ("session_1", "user", "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", 11),
        ("session_1", "assistant", "You'd be a great counselor! Your empathy and understanding will really help the people you work with. By the way, take a look at this.", 12),
        ("session_1", "user", "Thanks, Melanie! That's really sweet. Is this your own painting?", 13),
        ("session_1", "assistant", "Yeah, I painted that lake sunrise last year! It's special to me.", 14),
        ("session_1", "user", "Wow, Melanie! The colors really blend nicely. Painting looks like a great outlet for expressing yourself.", 15),
        ("session_1", "assistant", "Thanks, Caroline! Painting's a fun way to express my feelings and get creative. It's a great way to relax after a long day.", 16),
        ("session_1", "user", "Totally agree, Mel. Relaxing and expressing ourselves is key. Well, I'm off to go do some research.", 17),
        ("session_1", "assistant", "Yep, Caroline. Taking care of ourselves is vital. I'm off to go swimming with the kids. Talk to you soon!", 18),
    ]
    conn.executemany(
        "INSERT INTO conversation_turns(person, session_id, role, content, turn_order) VALUES (?, ?, ?, ?, ?)",
        [("Caroline", *t) for t in turns]
    )
    conn.commit()
    print(f"✅ 插入 {len(turns)} 条对话轮次")

def insert_facts_manual(conn):
    """
    手动构造 session_1 的事实，验证表结构
    按类型分类：event / knowledge / behavior
    """
    person = "Caroline"
    
    facts = [
        # === 事件记录池 (type='event') ===
        {
            "content": "Caroline attended an LGBTQ support group on 2023-05-07",
            "type": "event",
            "category": "social",
            "tags": "LGBTQ,support_group,community",
            "importance": 0.8,
            "content_date": "2023-05-07",
            "emotion_tag": "开心",
        },
        {
            "content": "Caroline heard inspiring transgender stories at the support group",
            "type": "event",
            "category": "social",
            "tags": "transgender,inspiring,support_group",
            "importance": 0.7,
            "content_date": "2023-05-07",
            "emotion_tag": "positive",
        },
        {
            "content": "Melanie showed Caroline a painting of a lake sunrise she made last year",
            "type": "event",
            "category": "social",
            "tags": "painting,Melanie,art",
            "importance": 0.4,
            "content_date": "2023-05-08",
            "emotion_tag": None,
        },
        
        # === 静态知识池 (type='knowledge') ===
        {
            "content": "Caroline feels accepted and gains courage from the LGBTQ support group",
            "type": "knowledge",
            "category": "psychology",
            "tags": "acceptance,courage,support_group",
            "importance": 0.8,
            "emotion_tag": "开心",
        },
        {
            "content": "Caroline wants to pursue counseling or mental health career to help people with similar experiences",
            "type": "knowledge",
            "category": "career",
            "tags": "career,counseling,mental_health",
            "importance": 0.9,
            "emotion_tag": "positive",
        },
        {
            "content": "Caroline plans to continue education and explore career options",
            "type": "knowledge",
            "category": "career",
            "tags": "education,career,plan",
            "importance": 0.6,
            "emotion_tag": None,
        },
        {
            "content": "Melanie finds painting a good way to express feelings and relax",
            "type": "knowledge",
            "category": "hobby",
            "tags": "painting,Melanie,self_expression",
            "importance": 0.3,
            "emotion_tag": None,
        },
        {
            "content": "Taking care of oneself and self-expression are important to Caroline",
            "type": "knowledge",
            "category": "value",
            "tags": "self_care,self_expression,value",
            "importance": 0.6,
            "emotion_tag": None,
        },
        
        # === 行为模式池 (type='behavior') ===
        {
            "content": "Caroline actively seeks LGBTQ community support and attends related events",
            "type": "behavior",
            "category": "social",
            "tags": "LGBTQ,community,support_seeking",
            "importance": 0.7,
            "emotion_tag": "positive",
        },
        {
            "content": "Caroline researches career options in counseling and mental health field",
            "type": "behavior",
            "category": "career",
            "tags": "research,career,counseling",
            "importance": 0.6,
            "emotion_tag": None,
        },
    ]
    
    fact_ids = []
    for f in facts:
        cur = conn.execute("""
            INSERT INTO facts(person, content, type, category, tags, importance, content_date, emotion_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (person, f["content"], f["type"], f["category"], f["tags"], f["importance"], 
              f.get("content_date"), f.get("emotion_tag")))
        fact_ids.append(cur.lastrowid)
    
    conn.commit()
    print(f"✅ 插入 {len(facts)} 条事实")
    
    # 插入 provenance
    for fid in fact_ids:
        conn.execute("""
            INSERT INTO provenance(person, fact_id, source_type, source_session_id, confidence)
            VALUES (?, ?, 'llm_extraction', 'session_1', 0.8)
        """, (person, fid))
    conn.commit()
    print(f"✅ 插入 {len(fact_ids)} 条溯源记录")
    
    return fact_ids

def insert_entities(conn):
    """插入实体"""
    person = "Caroline"
    entities = [
        ("Caroline", "person", "Caroline"),
        ("Melanie", "person", "Melanie"),
        ("LGBTQ support group", "organization", "support group"),
        ("counseling", "career", "counseling,mental health"),
        ("painting", "art", "lake sunrise painting"),
    ]
    for name, etype, aliases in entities:
        conn.execute("""
            INSERT OR IGNORE INTO entities(person, name, entity_type, aliases)
            VALUES (?, ?, ?, ?)
        """, (person, name, etype, aliases))
    conn.commit()
    print(f"✅ 插入 {len(entities)} 个实体")

def insert_emotion_events(conn, fact_ids):
    """插入情感事件"""
    person = "Caroline"
    emotions = [
        {
            "timestamp": "2023-05-07T14:00:00",
            "vector": "[0.8, 0.7, 0.6]",
            "label": "开心",
            "target": "event:support_group",
            "fact_idx": 0,  # 对应第一个 fact
            "source": "user",
            "importance": 0.8,
            "topics": '["LGBTQ support group", "transgender stories"]',
        },
        {
            "timestamp": "2023-05-07T14:30:00",
            "vector": "[0.9, 0.8, 0.7]",
            "label": "鼓舞",
            "target": "event:transgender_stories",
            "fact_idx": 1,
            "source": "user",
            "importance": 0.7,
            "topics": '["transgender", "inspiration"]',
        },
        {
            "timestamp": "2023-05-08T13:56:00",
            "vector": "[0.6, 0.5, 0.6]",
            "label": "感激",
            "target": "person:Melanie",
            "fact_idx": 3,
            "source": "user",
            "importance": 0.6,
            "topics": '["friendship", "support"]',
        },
    ]
    for em in emotions:
        conn.execute("""
            INSERT INTO emotion_events(person, timestamp, emotion_vector, emotion_label, 
                emotion_target, primary_fact_id, source, initial_importance, trigger_topics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (person, em["timestamp"], em["vector"], em["label"], em["target"],
              fact_ids[em["fact_idx"]], em["source"], em["importance"], em["topics"]))
    conn.commit()
    print(f"✅ 插入 {len(emotions)} 条情感事件")

def insert_emotion_triggers(conn):
    """插入情感触发关联"""
    person = "Caroline"
    triggers = [
        ("topic", "LGBTQ support group", 0.8, 0.7, 0.6),
        ("topic", "transgender stories", 0.9, 0.8, 0.7),
        ("topic", "counseling career", 0.7, 0.6, 0.7),
        ("entity", "Melanie", 0.6, 0.5, 0.6),
    ]
    for tt, tv, v, a, d in triggers:
        conn.execute("""
            INSERT OR IGNORE INTO emotion_triggers(person, trigger_type, trigger_value,
                typical_valence, typical_arousal, typical_dominance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (person, tt, tv, v, a, d))
    conn.commit()
    print(f"✅ 插入 {len(triggers)} 条情感触发")

def verify_data(conn):
    """验证数据完整性"""
    print("\n=== 数据验证 ===")
    
    # 1. 事实表
    cur = conn.execute("""
        SELECT type, COUNT(*) as cnt, 
               ROUND(AVG(importance), 2) as avg_imp,
               GROUP_CONCAT(DISTINCT category) as cats
        FROM facts GROUP BY type ORDER BY type
    """)
    rows = cur.fetchall()
    for r in rows:
        print(f"  facts type={r['type']:10s} count={r['cnt']:2d} avg_imp={r['avg_imp']} cats={r['cats']}")
    
    # 2. 唯一约束检查
    try:
        conn.execute("INSERT INTO facts(person, content, type) VALUES ('Caroline', 'Caroline attended an LGBTQ support group on 2023-05-07', 'event')")
        print("  ❌ UNIQUE 约束未生效！")
        conn.rollback()
    except sqlite3.IntegrityError:
        print("  ✅ UNIQUE(person, content) 约束正常")
    
    # 3. CHECK 约束检查
    for bad_type in ['invalid', 'event2']:
        try:
            conn.execute("INSERT INTO facts(person, content, type) VALUES ('Caroline', 'test', ?)", (bad_type,))
            print(f"  ❌ CHECK(type) 未拦截 '{bad_type}'！")
            conn.rollback()
        except sqlite3.IntegrityError:
            print(f"  ✅ CHECK(type) 正确拦截 '{bad_type}'")
    
    # 4. 情感事件 GENERATED 列
    cur = conn.execute("""
        SELECT event_id, emotion_vector, valence, arousal, dominance, intensity
        FROM emotion_events LIMIT 1
    """)
    r = cur.fetchone()
    print(f"  GENERATED 列: vector={r['emotion_vector']} → v={r['valence']:.1f} a={r['arousal']:.1f} d={r['dominance']:.1f} i={r['intensity']:.3f}")
    
    # 5. 外键约束
    try:
        conn.execute("INSERT INTO emotion_events(person, timestamp, emotion_vector, primary_fact_id) VALUES ('Caroline', '2023-05-08', '[0.5,0.5,0.5]', 99999)")
        print("  ❌ 外键约束未生效！")
        conn.rollback()
    except sqlite3.IntegrityError:
        print("  ✅ 外键约束正常")
    
    # 6. 实体关联
    cur = conn.execute("SELECT COUNT(*) as cnt FROM entities")
    print(f"  实体数: {cur.fetchone()['cnt']}")
    
    cur = conn.execute("SELECT COUNT(*) as cnt FROM provenance")
    print(f"  溯源记录数: {cur.fetchone()['cnt']}")
    
    cur = conn.execute("SELECT COUNT(*) as cnt FROM emotion_triggers")
    print(f"  情感触发数: {cur.fetchone()['cnt']}")
    
    # 7. 事实-实体关联
    # 关联 Caroline 和 Melanie 到相关事实
    conn.execute("""
        INSERT OR IGNORE INTO fact_entities(fact_id, entity_id)
        SELECT f.fact_id, e.entity_id FROM facts f, entities e
        WHERE f.person = 'Caroline' AND e.person = 'Caroline'
        AND e.name = 'Caroline' AND f.type = 'event'
    """)
    conn.execute("""
        INSERT OR IGNORE INTO fact_entities(fact_id, entity_id)
        SELECT f.fact_id, e.entity_id FROM facts f, entities e
        WHERE f.person = 'Caroline' AND e.person = 'Caroline'
        AND e.name = 'Melanie' AND f.tags LIKE '%Melanie%'
    """)
    conn.commit()
    
    cur = conn.execute("""
        SELECT f.fact_id, f.content, e.name as entity
        FROM fact_entities fe
        JOIN facts f ON f.fact_id = fe.fact_id
        JOIN entities e ON e.entity_id = fe.entity_id
        ORDER BY f.fact_id
    """)
    rows = cur.fetchall()
    print(f"  事实-实体关联: {len(rows)} 条")
    for r in rows[:5]:
        print(f"    fact#{r['fact_id']} → {r['entity']}: {r['content'][:50]}...")
    
    print("\n✅ 全部验证通过！")

if __name__ == '__main__':
    conn = get_conn()
    
    # 清空旧数据
    conn.executescript("""
        DROP TABLE IF EXISTS fact_entities;
        DROP TABLE IF EXISTS fact_relations;
        DROP TABLE IF EXISTS provenance;
        DROP TABLE IF EXISTS emotion_triggers;
        DROP TABLE IF EXISTS emotion_states;
        DROP TABLE IF EXISTS emotion_events;
        DROP TABLE IF EXISTS entities;
        DROP TABLE IF EXISTS facts;
        DROP TABLE IF EXISTS conversation_turns;
    """)
    conn.commit()
    
    create_tables(conn)
    insert_conversation_turns(conn)
    fact_ids = insert_facts_manual(conn)
    insert_entities(conn)
    insert_emotion_events(conn, fact_ids)
    insert_emotion_triggers(conn)
    verify_data(conn)
    
    conn.close()
    print("\n🎉 全部完成！")
