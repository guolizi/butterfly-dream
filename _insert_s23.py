#!/usr/bin/env python3
"""Insert session_2 and session_3 facts into DB."""
import json, sqlite3

conn = sqlite3.connect("eval/dbs/locomo/conv26_v2.db")
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA foreign_keys=ON")
conn.create_function("jieba_segment", 1, lambda x: x)

facts = json.loads("""[
  {"content": "Caroline is researching adoption agencies to pursue her dream of having a family and giving a loving home to children who need it.", "type": "behavior", "category": "activity", "tags": "adoption,research,family", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline has a dream to have a family and give a loving home to children who need it.", "type": "knowledge", "category": "goal", "tags": "adoption,family,dream", "importance": 1.0, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline's goal is to give children a loving home through adoption.", "type": "knowledge", "category": "goal", "tags": "adoption,family,goal", "importance": 1.0, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline is grateful for support from friends and mentors regarding her adoption journey.", "type": "knowledge", "category": "emotion", "tags": "gratitude,support,adoption", "importance": 0.6, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline is feeling hopeful and optimistic about her adoption journey.", "type": "knowledge", "category": "emotion", "tags": "hopeful,optimistic,adoption", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline chose an adoption agency because they help LGBTQ+ folks with adoption.", "type": "behavior", "category": "preference", "tags": "adoption,agency,LGBTQ+,inclusivity", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "The adoption agency's inclusivity and support spoke to Caroline.", "type": "knowledge", "category": "opinion", "tags": "adoption,inclusivity,support", "importance": 0.6, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline is thrilled to make a family for children who need one through adoption.", "type": "knowledge", "category": "emotion", "tags": "thrilled,family,adoption", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline acknowledges it will be tough being a single parent through adoption.", "type": "knowledge", "category": "opinion", "tags": "single parent,adoption,challenge", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline is up for the challenge of being a single parent through adoption.", "type": "knowledge", "category": "state", "tags": "determined,adoption,single parent", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline will do her best to make sure adopted children have a safe and loving home.", "type": "knowledge", "category": "goal", "tags": "adoption,safe home,loving home", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_2", "turn_id": null},
  {"content": "Caroline spoke at a school event about her transgender journey and encouraged students to get involved in the LGBTQ community.", "type": "event", "category": "event", "tags": "LGBTQ,transgender,school event,public speaking,advocacy", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline started transitioning three years ago (approximately 2020).", "type": "knowledge", "category": "health", "tags": "transition,transgender,timeline", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline shared her personal journey, struggles, and development since coming out during her school talk.", "type": "event", "category": "event", "tags": "public speaking,coming out,personal journey,struggles", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline feels conversations about gender identity and inclusion are necessary.", "type": "knowledge", "category": "opinion", "tags": "gender identity,inclusion,opinion", "importance": 0.6, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline is thankful for being able to give a voice to the trans community.", "type": "behavior", "category": "emotion", "tags": "gratitude,trans community,advocacy", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline has been blessed with lots of love and support throughout her transition journey.", "type": "knowledge", "category": "state", "tags": "support,love,transition", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline wants to pass on the love and support she received to others.", "type": "behavior", "category": "goal", "tags": "giving back,support,community", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline believes sharing stories can build a strong, supportive community of hope.", "type": "knowledge", "category": "opinion", "tags": "storytelling,community,hope,opinion", "importance": 0.6, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline intends to keep using her voice to make a change and lift others up.", "type": "behavior", "category": "goal", "tags": "advocacy,goal,change,community", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline's friends, family and mentors are her rocks who motivate her and give her strength.", "type": "knowledge", "category": "person", "tags": "support system,friends,family,mentors,motivation", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline has known her close friends for 4 years.", "type": "knowledge", "category": "person", "tags": "friends,duration,relationship", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline moved from her home country approximately 4 years ago.", "type": "event", "category": "event", "tags": "relocation,home country,move", "importance": 0.9, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline went through a tough breakup.", "type": "event", "category": "event", "tags": "breakup,relationship,past", "importance": 0.8, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline is super thankful for her friends' love and help, especially after her tough breakup.", "type": "behavior", "category": "emotion", "tags": "gratitude,friends,support,breakup", "importance": 0.7, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null},
  {"content": "Caroline believes family is everything.", "type": "knowledge", "category": "opinion", "tags": "family,values,opinion", "importance": 0.6, "content_date": null, "entities": ["Caroline"], "session_id": "session_3", "turn_id": null}
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
print(f"\n📊 session_2+3: {new_count} new, {dup_count} duplicates")
