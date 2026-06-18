"""
修复实体关系：按 v2 schema 建 entity_relations 表 + 同步实体关系
"""
import sqlite3, json

DB = '/home/xx/butterfly-dream/memory_store.db'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn

def main():
    conn = get_conn()
    
    # 1. 建 entity_relations 表（v2 schema）
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entity_relations (
            relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            person      TEXT NOT NULL,
            source_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            target_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
            relation    TEXT DEFAULT 'related_to',
            weight      REAL DEFAULT 1.0,
            created_at  TEXT DEFAULT (datetime('now','localtime')),
            UNIQUE(source_id, target_id, relation)
        );
        CREATE INDEX IF NOT EXISTS idx_er_source_target
            ON entity_relations(source_id, target_id);
    """)
    print("✅ entity_relations 表已建")
    
    # 2. 更新 entities 类型
    conn.execute("UPDATE entities SET entity_type='person' WHERE name IN ('Caroline', 'Melanie')")
    conn.execute("UPDATE entities SET entity_type='organization' WHERE name='LGBTQ互助小组'")
    conn.execute("UPDATE entities SET entity_type='organization' WHERE name LIKE '%小组%' OR name LIKE '%机构%' OR name LIKE '%社区%'")
    print("✅ entities 类型已更新")
    
    # 3. 插入 Caroline→Melanie 的朋友关系
    caroline = conn.execute("SELECT entity_id FROM entities WHERE person='Caroline' AND name='Caroline'").fetchone()
    melanie = conn.execute("SELECT entity_id FROM entities WHERE person='Caroline' AND name='Melanie'").fetchone()
    
    if caroline and melanie:
        cid, mid = caroline[0], melanie[0]
        try:
            conn.execute("""
                INSERT INTO entity_relations (person, source_id, target_id, relation, weight)
                VALUES ('Caroline', ?, ?, 'friend_of', 0.85)
            """, (cid, mid))
            print(f"✅ Caroline({cid}) → Melanie({mid}) friend_of 已插入")
        except Exception as e:
            print(f"⏭️  Caroline→Melanie 关系已存在: {e}")
    
    # 4. 从 facts.entities JSON 同步到 fact_entities 表
    facts = conn.execute("SELECT fact_id, entities FROM facts WHERE entities IS NOT NULL AND entities != '[]'").fetchall()
    synced = 0
    for fid, entities_json in facts:
        try:
            names = json.loads(entities_json)
            for name in names:
                # 查找或创建实体
                ent = conn.execute(
                    "SELECT entity_id FROM entities WHERE person='Caroline' AND name=?",
                    (name,)
                ).fetchone()
                if ent:
                    eid = ent[0]
                else:
                    cur = conn.execute(
                        "INSERT INTO entities (person, name, entity_type) VALUES ('Caroline', ?, 'unknown')",
                        (name,)
                    )
                    eid = cur.lastrowid
                
                # 插入 fact_entities 关联
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                        (fid, eid)
                    )
                    synced += 1
                except Exception:
                    pass
        except json.JSONDecodeError:
            pass
    
    conn.commit()
    print(f"✅ fact_entities 同步完成: {synced} 条关联")
    
    # 5. 验证
    print("\n=== 验证 ===")
    print(f"entities 表: {conn.execute('SELECT COUNT(*) FROM entities').fetchone()[0]} 条")
    print(f"entity_relations 表: {conn.execute('SELECT COUNT(*) FROM entity_relations').fetchone()[0]} 条")
    print(f"fact_entities 表: {conn.execute('SELECT COUNT(*) FROM fact_entities').fetchone()[0]} 条")
    
    print("\n=== entities 详情 ===")
    for r in conn.execute("SELECT entity_id, name, entity_type FROM entities ORDER BY entity_id"):
        print(f"  {r[0]}: {r[1]} ({r[2]})")
    
    print("\n=== entity_relations 详情 ===")
    for r in conn.execute("""
        SELECT er.relation_id, e1.name, e2.name, er.relation, er.weight
        FROM entity_relations er
        JOIN entities e1 ON er.source_id = e1.entity_id
        JOIN entities e2 ON er.target_id = e2.entity_id
    """):
        print(f"  {r[0]}: {r[1]} --[{r[3]}]--> {r[2]} (weight={r[4]})")
    
    conn.close()

if __name__ == "__main__":
    main()
