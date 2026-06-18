"""
将提取结果写入数据库
"""
import sqlite3
import json

DB = '/home/xx/butterfly-dream/memory_store.db'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

# session_1 结果
s1 = [
    {"dimension": "event", "content": "Caroline 昨天参加了LGBTQ互助小组", "type": "event", "category": "social", "tags": "LGBTQ,support_group", "importance": 0.8, "content_date": "2023-05-07", "emotion_tag": "positive", "entities": ["Caroline", "LGBTQ互助小组"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ互助小组", "time": "2023-05-07", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 对心理咨询或心理健康领域感兴趣", "type": "knowledge", "category": "career", "tags": "counseling,mental_health", "importance": 0.7, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 感到被接纳时，她会去探索职业方向", "type": "behavior", "category": "psychology", "tags": "acceptance,career_exploration", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 在互助小组中感到开心和感激", "emotion_vector": [0.9, 0.8, 0.6], "emotion_label": "开心感激", "emotion_target": "event:support_group", "source": "user", "importance": 0.7, "significance_reason": "用户主动表达开心和感激", "trigger_topics": ["LGBTQ互助小组", "跨性别故事"], "timestamp": "2023-05-08T13:56:00"},
    {"dimension": "emotion", "content": "Caroline 感到被接纳和勇气", "emotion_vector": [0.8, 0.7, 0.9], "emotion_label": "被接纳有勇气", "emotion_target": "event:support_group", "source": "user", "importance": 0.7, "significance_reason": "用户描述小组带来的感受", "trigger_topics": ["LGBTQ互助小组", "自我接纳"], "timestamp": "2023-05-08T13:56:00"},
    {"dimension": "emotion", "content": "Caroline 对继续学业和探索职业感到兴奋", "emotion_vector": [0.7, 0.8, 0.5], "emotion_label": "兴奋期待", "emotion_target": "future:career_education", "source": "user", "importance": 0.6, "significance_reason": "用户表达对未来的兴奋", "trigger_topics": ["学业", "职业方向"], "timestamp": "2023-05-08T13:56:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
]

# session_2 结果
s2 = [
    {"dimension": "event", "content": "Caroline 正在研究领养机构", "type": "event", "category": "family", "tags": "adoption,research", "importance": 0.8, "content_date": "2023-05-25", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "研究", "object": "领养机构", "time": "2023-05-25", "location": None}},
    {"dimension": "event", "content": "Caroline 选择了一家帮助 LGBTQ+ 群体的领养机构", "type": "event", "category": "family", "tags": "adoption,LGBTQ+,inclusivity", "importance": 0.85, "content_date": "2023-05-25", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "选择", "object": "领养机构", "time": "2023-05-25", "location": None}},
    {"dimension": "event", "content": "Caroline 作为单亲家长准备迎接领养挑战", "type": "event", "category": "family", "tags": "adoption,single parent,challenge", "importance": 0.8, "content_date": "2023-05-25", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "准备迎接", "object": "领养挑战", "time": "2023-05-25", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 梦想组建家庭，给需要爱的孩子一个温暖的家", "type": "knowledge", "category": "family", "tags": "family,dream,adoption", "importance": 0.9, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 认为自我关怀很重要", "type": "knowledge", "category": "psychology", "tags": "self-care,importance", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 感激朋友和导师的支持", "type": "knowledge", "category": "social", "tags": "gratitude,support", "importance": 0.7, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 的目标是给孩子们一个充满爱的家", "type": "knowledge", "category": "family", "tags": "goal,love,home", "importance": 0.85, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 选择领养机构因为其包容性和对 LGBTQ+ 的支持", "type": "knowledge", "category": "family", "tags": "adoption,inclusivity,LGBTQ+", "importance": 0.8, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 作为单亲家长，决心迎接挑战", "type": "knowledge", "category": "family", "tags": "single parent,determination,challenge", "importance": 0.75, "emotion_tag": "positive", "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 感到有支持时，她会更坚定地追求梦想", "type": "behavior", "category": "psychology", "tags": "support,motivation", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "当 Caroline 受到鼓励时，她会表达感激并决心努力", "type": "behavior", "category": "psychology", "tags": "encouragement,gratitude,determination", "importance": 0.65, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 对 Melanie 参加慈善跑感到骄傲", "emotion_vector": [0.8, 0.6, 0.7], "emotion_label": "骄傲", "emotion_target": "event:charity_race", "source": "user", "importance": 0.6, "significance_reason": "用户主动表达对朋友成就的骄傲", "trigger_topics": ["慈善跑", "心理健康"], "timestamp": "2023-05-25T13:14:00"},
    {"dimension": "emotion", "content": "Caroline 对领养感到充满希望和乐观", "emotion_vector": [0.85, 0.7, 0.6], "emotion_label": "希望乐观", "emotion_target": "event:adoption", "source": "user", "importance": 0.85, "significance_reason": "用户主动表达对领养的积极情感", "trigger_topics": ["领养", "组建家庭"], "timestamp": "2023-05-25T13:14:00"},
    {"dimension": "emotion", "content": "Caroline 对领养感到激动", "emotion_vector": [0.9, 0.8, 0.7], "emotion_label": "激动", "emotion_target": "event:adoption", "source": "user", "importance": 0.8, "significance_reason": "用户表达对组建家庭的激动", "trigger_topics": ["领养", "家庭", "挑战"], "timestamp": "2023-05-25T13:14:00"},
    {"dimension": "emotion", "content": "Caroline 对 Melanie 的支持感到感激", "emotion_vector": [0.8, 0.3, 0.6], "emotion_label": "感激", "emotion_target": "person:Melanie", "source": "user", "importance": 0.7, "significance_reason": "用户对朋友鼓励的感激回应", "trigger_topics": ["支持", "鼓励"], "timestamp": "2023-05-25T13:14:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.9}
]

# session_3 结果
s3 = [
    {"dimension": "event", "content": "Caroline 上周在学校做了关于跨性别经历的演讲，鼓励学生参与LGBTQ社群", "type": "event", "category": "social", "tags": "school,speech,transgender,LGBTQ", "importance": 0.9, "content_date": "2023-06-02", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "演讲", "object": "跨性别经历", "time": "2023-06-02", "location": "学校"}},
    {"dimension": "event", "content": "Caroline 上周与Melanie见面", "type": "event", "category": "social", "tags": "meeting,friend", "importance": 0.6, "content_date": "2023-06-02", "emotion_tag": "positive", "entities": ["Caroline", "Melanie"], "structured_data": {"subject": "Caroline", "action": "见面", "object": "Melanie", "time": "2023-06-02", "location": None}},
    {"dimension": "knowledge", "content": "Caroline 三年前开始跨性别转变", "type": "knowledge", "category": "psychology", "tags": "transgender,transition", "importance": 0.9, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 四年前从祖国搬来，认识了现在的朋友", "type": "knowledge", "category": "personal", "tags": "relocation,friends", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 经历过一次艰难的分手", "type": "knowledge", "category": "personal", "tags": "breakup", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 的朋友、家人和导师是她的支柱", "type": "knowledge", "category": "psychology", "tags": "support_system", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 认为分享经历有助于促进理解和接纳", "type": "knowledge", "category": "belief", "tags": "sharing,understanding,acceptance", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "knowledge", "content": "Caroline 认为通过分享故事可以建立强大的互助社群", "type": "knowledge", "category": "belief", "tags": "community,hope", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "Caroline 会通过分享自己的故事来帮助和激励他人", "type": "behavior", "category": "social", "tags": "storytelling,helping,sharing", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "Caroline 会鼓励学生参与LGBTQ社群", "type": "behavior", "category": "social", "tags": "encouragement,LGBTQ", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "behavior", "content": "Caroline 会继续用她的声音去改变和鼓舞他人", "type": "behavior", "category": "social", "tags": "advocacy,voice", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
    {"dimension": "emotion", "content": "Caroline 做演讲时感到充满力量", "emotion_vector": [0.8, 0.8, 0.7], "emotion_label": "充满力量", "emotion_target": "event:school_speech", "source": "user", "importance": 0.8, "significance_reason": "用户主动表达演讲时的强烈正面情感", "trigger_topics": ["学校演讲", "跨性别经历"], "timestamp": "2023-06-09T19:55:00"},
    {"dimension": "emotion", "content": "Caroline 对Melanie的支持感到感激", "emotion_vector": [0.7, 0.7, 0.6], "emotion_label": "感激", "emotion_target": "person:Melanie", "source": "user", "importance": 0.7, "significance_reason": "用户多次表达对支持的感激", "trigger_topics": ["支持", "鼓励"], "timestamp": "2023-06-09T19:55:00"},
    {"dimension": "emotion", "content": "Caroline 看到观众共鸣时感到开心", "emotion_vector": [0.7, 0.7, 0.6], "emotion_label": "开心", "emotion_target": "event:audience_reaction", "source": "user", "importance": 0.6, "significance_reason": "用户表达看到观众共鸣时的正面情感", "trigger_topics": ["观众反应", "共鸣"], "timestamp": "2023-06-09T19:55:00"},
    {"dimension": "emotion", "content": "Caroline 对拥有支持系统感到幸运和感激", "emotion_vector": [0.7, 0.7, 0.7], "emotion_label": "感激", "emotion_target": "entity:support_system", "source": "user", "importance": 0.7, "significance_reason": "用户表达对朋友家人支持的感激", "trigger_topics": ["支持系统", "朋友", "家人"], "timestamp": "2023-06-09T19:55:00"},
    {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.9}
]

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
            pfid = None
            target = item.get("emotion_target", "")
            if target and target.startswith("event:"):
                tc = target.replace("event:", "")
                pf = conn.execute("SELECT fact_id FROM facts WHERE person=? AND content LIKE ? LIMIT 1", 
                                 (person, f"%{tc}%")).fetchone()
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
            pass  # 关系信息暂存，后续处理
    
    conn.commit()
    return facts_inserted, emotions_inserted

if __name__ == '__main__':
    conn = get_conn()
    
    total_facts = 0
    total_emotions = 0
    
    for session_id, items in [("session_1", s1), ("session_2", s2), ("session_3", s3)]:
        f, e = insert_items(conn, "Caroline", session_id, items)
        total_facts += f
        total_emotions += e
        print(f"  {session_id}: {f} facts, {e} emotions")
    
    print(f"\n✅ 总计: {total_facts} 事实, {total_emotions} 情感事件")
    
    # 统计
    cur = conn.execute("SELECT type, COUNT(*) as cnt FROM facts GROUP BY type ORDER BY type")
    rows = cur.fetchall()
    print("\n📊 事实分布:")
    for r in rows:
        print(f"  {r['type']:10s}: {r['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM emotion_events")
    print(f"  情感事件 : {cur.fetchone()['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM entities")
    print(f"  实体     : {cur.fetchone()['cnt']}")
    cur = conn.execute("SELECT COUNT(*) as cnt FROM provenance")
    print(f"  溯源记录 : {cur.fetchone()['cnt']}")
    
    conn.close()
