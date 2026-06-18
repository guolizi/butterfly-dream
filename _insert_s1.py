#!/usr/bin/env python3
"""Insert session_1 extracted facts into DB."""
import json, sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA foreign_keys=ON")
conn.create_function("jieba_segment", 1, lambda x: x)  # stub for FTS5 trigger

facts = json.loads("""[
  {"content": "Caroline attended an LGBTQ support group on 7 May 2023.", "type": "event", "category": "event", "tags": "lgbtq,support group,attendance", "importance": 0.9, "content_date": "2023-05-07", "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline found the LGBTQ support group experience powerful.", "type": "knowledge", "category": "opinion", "tags": "lgbtq,support group,opinion", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline found the transgender stories at the support group inspiring.", "type": "knowledge", "category": "opinion", "tags": "transgender,inspiring,stories,support group", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline felt happy and thankful for all the support at the group.", "type": "knowledge", "category": "emotion", "tags": "happiness,gratitude,support group", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "The support group made Caroline feel accepted.", "type": "knowledge", "category": "emotion", "tags": "acceptance,support group,emotion", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "The support group gave Caroline courage to embrace herself.", "type": "knowledge", "category": "state", "tags": "courage,self-acceptance,support group", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline plans to continue her education.", "type": "knowledge", "category": "goal", "tags": "education,future plans", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline plans to check out career options.", "type": "knowledge", "category": "goal", "tags": "career,future plans", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline is keen on counseling or working in mental health.", "type": "knowledge", "category": "career", "tags": "counseling,mental health,career interest", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline would love to support those with similar issues through counseling or mental health work.", "type": "knowledge", "category": "goal", "tags": "helping others,counseling,mental health", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline thinks painting is a great outlet for self-expression.", "type": "knowledge", "category": "opinion", "tags": "painting,self-expression,opinion", "importance": 0.5, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null},
  {"content": "Caroline believes relaxing and expressing ourselves is key.", "type": "knowledge", "category": "opinion", "tags": "relaxation,self-expression,opinion", "importance": 0.5, "content_date": null, "entities": ["Caroline"], "session_id": "session_1", "turn_id": null}
]""")

new_count = 0
dup_count = 0
for f in facts:
    try:
        cur = conn.execute("""
            INSERT INTO facts (person, content, type, category, tags, importance, content_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("Caroline", f["content"], f["type"], f["category"], f["tags"], f["importance"], f["content_date"]))
        fid = cur.lastrowid
        
        conn.execute("""
            INSERT INTO provenance (person, fact_id, source_type, source_session_id)
            VALUES (?, ?, 'llm_extraction', ?)
        """, ("Caroline", fid, f["session_id"]))
        
        for ename in f.get("entities", []):
            if ename == "Caroline":
                continue
            e = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", ("Caroline", ename)).fetchone()
            if not e:
                cur2 = conn.execute("INSERT INTO entities (person, name) VALUES (?, ?)", ("Caroline", ename))
                eid = cur2.lastrowid
            else:
                eid = e["entity_id"]
            conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fid, eid))
        
        new_count += 1
        print(f"  ✅ #{fid}: {f['content'][:70]}")
    except sqlite3.IntegrityError:
        dup_count += 1
        print(f"  ⏭️  Duplicate: {f['content'][:60]}")

conn.commit()
conn.close()
print(f"\n📊 session_1: {new_count} new, {dup_count} duplicates")
