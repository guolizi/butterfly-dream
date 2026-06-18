"""
插入 session_4-6 提取结果
"""
import sqlite3, json

DB = '/home/xx/butterfly-dream/memory_store.db'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def insert_items(conn, person, session_id, items):
    facts_inserted = 0
    emotions_inserted = 0
    for item in items:
        dim = item.get("dimension", "")
        if dim in ("event", "knowledge", "behavior"):
            ftype = dim
            sd = None
            if dim == "event" and item.get("structured_data"):
                sd = json.dumps(item["structured_data"], ensure_ascii=False)
            conn.execute("""
                INSERT OR IGNORE INTO facts(person, content, type, category, tags, importance, content_date, emotion_tag, structured_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person, item["content"], ftype, item.get("category", "general"),
                  item.get("tags", ""), item.get("importance", 0.5),
                  item.get("content_date"), item.get("emotion_tag"), sd))
            if conn.total_changes > 0:
                facts_inserted += 1
                fid = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content=?", (person, item["content"])).fetchone()
                if fid:
                    conn.execute("INSERT INTO provenance(person, fact_id, source_type, source_session_id, confidence) VALUES (?, ?, 'llm_extraction', ?, 0.8)", (person, fid[0], session_id))
                    for ent_name in item.get("entities", []):
                        conn.execute("INSERT OR IGNORE INTO entities(person, name) VALUES (?, ?)", (person, ent_name))
                        eid = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", (person, ent_name)).fetchone()
                        if eid:
                            conn.execute("INSERT OR IGNORE INTO fact_entities(fact_id, entity_id) VALUES (?, ?)", (fid[0], eid[0]))
        elif dim == "emotion":
            pfid = None
            target = item.get("emotion_target", "")
            if target and target.startswith("event:"):
                tc = target.replace("event:", "")
                pf = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content LIKE ? LIMIT 1", (person, f"%{tc}%")).fetchone()
                if pf: pfid = pf[0]
            conn.execute("""
                INSERT INTO emotion_events(person, timestamp, emotion_vector, emotion_label, emotion_target, primary_fact_id, source, initial_importance, significance_reason, trigger_topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (person, item.get("timestamp", ""), json.dumps(item.get("emotion_vector", [0,0.5,0.5])),
                  item.get("emotion_label"), item.get("emotion_target"), pfid,
                  item.get("source", "user"), item.get("importance", 0.5),
                  item.get("significance_reason"), json.dumps(item.get("trigger_topics", []), ensure_ascii=False)))
            emotions_inserted += 1
    conn.commit()
    return facts_inserted, emotions_inserted

# session_4
s4 = [
    {"dimension": "event", "content": "Caroline 参加了LGBTQ+心理咨询工作坊", "type": "event", "category": "career", "tags": "counseling,workshop,LGBTQ", "importance": 0.8, "content_date": "2023-06-23", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ+心理咨询工作坊", "time": "2023-06-23", "location": None}},
    {"dimension": "event", "content": "Caroline 在18岁生日时收到朋友制作的手绘碗", "type": "event", "category": "personal", "tags": "birthday,gift,art", "importance": 0.6, "content_date": "2013", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "收到", "object": "手绘碗", "time": "2013", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 来自瑞典", "type": "knowledge", "category": "personal", "tags": "Sweden,origin", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 的项链是奶奶送的，代表爱、信念和力量", "type": "knowledge", "category": "personal", "tags": "necklace,family,symbol", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 想从事心理咨询工作，帮助跨性别者", "type": "knowledge", "category": "career", "tags": "counseling,transgender,career", "importance": 0.8, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 通过心理咨询和支持小组改善了自己的生活", "type": "knowledge", "category": "personal", "tags": "counseling,support,self-improvement", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 对某个领域产生兴趣时，她会主动参加相关活动来学习", "type": "behavior", "category": "psychology", "tags": "learning,initiative", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "Caroline 热衷于为他人创造安全、温馨的成长空间", "type": "behavior", "category": "psychology", "tags": "empathy,support", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 在工作坊中感到深受启发", "emotion_vector": [0.8, 0.7, 0.6], "emotion_label": "深受启发", "emotion_target": "event:workshop", "source": "user", "importance": 0.7, "significance_reason": "用户表达工作坊带来的启发", "trigger_topics": ["心理咨询工作坊", "跨性别"], "timestamp": "2023-06-27T10:37:00"},
    {"dimension": "emotion", "content": "Caroline 对奶奶送的项链感到珍视和感恩", "emotion_vector": [0.7, 0.8, 0.5], "emotion_label": "珍视", "emotion_target": "object:necklace", "source": "user", "importance": 0.6, "significance_reason": "用户描述项链的特殊意义", "trigger_topics": ["项链", "奶奶", "瑞典"], "timestamp": "2023-06-27T10:37:00"},
    {"dimension": "emotion", "content": "Caroline 对Melanie的鼓励感到感激", "emotion_vector": [0.6, 0.7, 0.8], "emotion_label": "感激", "emotion_target": "person:Melanie", "source": "user", "importance": 0.5, "significance_reason": "用户回应Melanie的鼓励", "trigger_topics": ["鼓励", "支持"], "timestamp": "2023-06-27T10:37:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8},
]

# session_5
s5 = [
    {"dimension": "event", "content": "Caroline 上周参加了LGBTQ+骄傲游行", "type": "event", "category": "social", "tags": "pride,LGBTQ,parade", "importance": 0.8, "content_date": "2023-06-26", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ+骄傲游行", "time": "2023-06-26", "location": None}},
    {"dimension": "event", "content": "Caroline 这个月要去参加跨性别会议", "type": "event", "category": "social", "tags": "transgender,conference,advocacy", "importance": 0.7, "content_date": "2023-07", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "跨性别会议", "time": "2023-07", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 正在学钢琴", "type": "knowledge", "category": "hobby", "tags": "piano,learning", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 想从事心理咨询和心理健康领域", "type": "knowledge", "category": "career", "tags": "counseling,mental_health,career", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 参加LGBTQ+社区活动时，她会感到归属感并产生用自己故事帮助他人、回馈社区的愿望", "type": "behavior", "category": "social", "tags": "community,belonging,giving_back,helping_others", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 在骄傲游行中感到归属感和快乐", "emotion_vector": [0.9, 0.8, 0.7], "emotion_label": "归属快乐", "emotion_target": "event:pride_parade", "source": "user", "importance": 0.8, "significance_reason": "用户主动表达强烈的归属感和快乐", "trigger_topics": ["骄傲游行", "LGBTQ社群"], "timestamp": "2023-07-03T13:36:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8},
]

# session_6
s6 = [
    {"dimension": "event", "content": "Caroline 上周和朋友家人去野餐了", "type": "event", "category": "social", "tags": "picnic,friends,family", "importance": 0.6, "content_date": "2023-06-29", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "野餐", "object": None, "time": "2023-06-29", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 正在为未来的孩子建一个图书馆", "type": "knowledge", "category": "family", "tags": "library,children,future", "importance": 0.7, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 正在研究心理咨询和心理健康工作", "type": "knowledge", "category": "career", "tags": "counseling,mental_health,career", "importance": 0.6, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 经历转变时，她会依赖并感激她的支持网络", "type": "behavior", "category": "psychology", "tags": "support,transition,reliance", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 对朋友和家人的支持感到感激", "emotion_vector": [0.8, 0.5, 0.7], "emotion_label": "感激", "emotion_target": "entity:support_network", "source": "user", "importance": 0.8, "significance_reason": "用户多次表达对支持网络的感激", "trigger_topics": ["朋友", "家人", "支持"], "timestamp": "2023-07-06T20:18:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8},
]

if __name__ == '__main__':
    conn = get_conn()
    total_f, total_e = 0, 0
    for sid, items in [("session_4", s4), ("session_5", s5), ("session_6", s6)]:
        f, e = insert_items(conn, "Caroline", sid, items)
        total_f += f; total_e += e
        print(f"  {sid}: {f} facts, {e} emotions")
    print(f"\n✅ 总计: {total_f} 事实, {total_e} 情感事件")
    
    cur = conn.execute("SELECT type, COUNT(*) as cnt FROM facts GROUP BY type ORDER BY type")
    rows = cur.fetchall()
    print("\n📊 事实分布:")
    for r in rows:
        print(f"  {r['type']:10s}: {r['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM emotion_events")
    print(f"  情感事件 : {cur.fetchone()['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM entities")
    print(f"  实体     : {cur.fetchone()['cnt']}")
    conn.close()
