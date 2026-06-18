#!/usr/bin/env python3
"""提取 session_1 的 L1 事实，不调 LLM，直接手工编码提取。"""

import sqlite3
import sys

DB = "eval/dbs/locomo/conv26_v2.db"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA foreign_keys = ON")

# ── 定义所有事实 ──────────────────────────────────────────────
# (person, content, type, category, importance, content_date, turn_ids, entities, is_abstract)

facts = []

def add(person, content, typ, cat, imp, date, turns, ents, abstract=False):
    facts.append({
        "person": person, "content": content, "type": typ,
        "category": cat, "importance": imp, "content_date": date,
        "turn_ids": turns, "entities": ents, "is_abstract": abstract,
    })

# ===== Caroline 的事实 =====

# Turn 1: 打招呼
add("Caroline", "Caroline greeted Melanie warmly.", "event", "event", 0.3,
    "2023-05-08", [1], ["Caroline", "Melanie"])

# Turn 3: 互助小组
add("Caroline", "Caroline attended an LGBTQ support group.", "event", "event", 0.8,
    "2023-05-07", [3], ["Caroline", "LGBTQ support group"])
add("Caroline", "Caroline found the LGBTQ support group powerful.", "knowledge", "opinion", 0.6,
    None, [3], ["Caroline", "LGBTQ support group"])

# Turn 5: 跨性别故事
add("Caroline", "Caroline heard transgender stories at the support group.", "event", "event", 0.7,
    "2023-05-07", [5], ["Caroline", "LGBTQ support group"])
add("Caroline", "Caroline found the transgender stories inspiring.", "knowledge", "opinion", 0.6,
    None, [5], ["Caroline"])
add("Caroline", "Caroline felt happy and thankful for the support at the LGBTQ group.", "knowledge", "state", 0.6,
    None, [5], ["Caroline", "LGBTQ support group"])

# Turn 7: 互助小组的影响
add("Caroline", "The support group made Caroline feel accepted.", "knowledge", "state", 0.7,
    None, [7], ["Caroline", "LGBTQ support group"])
add("Caroline", "The support group gave Caroline courage to embrace herself.", "knowledge", "state", 0.7,
    None, [7], ["Caroline", "LGBTQ support group"])

# Turn 9: 教育和职业规划
add("Caroline", "Caroline plans to continue her education.", "knowledge", "goal", 0.7,
    None, [9], ["Caroline"])
add("Caroline", "Caroline intends to explore career options.", "knowledge", "goal", 0.6,
    None, [9], ["Caroline"])

# Turn 11: 职业兴趣
add("Caroline", "Caroline is interested in becoming a counselor or working in mental health.", "knowledge", "preference", 0.7,
    None, [11], ["Caroline"])
add("Caroline", "Caroline wants to support people with similar issues through counseling.", "knowledge", "goal", 0.7,
    None, [11], ["Caroline"])

# Turn 13: 对画的评论
add("Caroline", "Caroline complimented Melanie's lake sunrise painting.", "event", "event", 0.4,
    "2023-05-08", [13], ["Caroline", "Melanie"])

# Turn 15: 画的意义
add("Caroline", "Caroline thinks painting is a great outlet for expressing oneself.", "knowledge", "opinion", 0.5,
    None, [15], ["Caroline"])

# Turn 17: 要做研究
add("Caroline", "Caroline is going to do some research.", "event", "event", 0.4,
    "2023-05-08", [17], ["Caroline"])

# ===== Melanie 的事实 =====

# Turn 2: 忙碌
add("Melanie", "Melanie is busy with kids and work.", "knowledge", "state", 0.5,
    None, [2], ["Melanie"])

# Turn 4: 对互助小组感兴趣
add("Melanie", "Melanie expressed interest in Caroline's support group experience.", "event", "event", 0.5,
    "2023-05-08", [4], ["Melanie", "Caroline"])

# Turn 6: 鼓励 Caroline
add("Melanie", "Melanie encouraged Caroline to share more about her support group experience.", "event", "event", 0.4,
    "2023-05-08", [6], ["Melanie", "Caroline"])

# Turn 8: 鼓励 Caroline 追求目标
add("Melanie", "Melanie encouraged Caroline to pursue her goals.", "event", "event", 0.5,
    "2023-05-08", [8], ["Melanie", "Caroline"])

# Turn 10: 问职业兴趣
add("Melanie", "Melanie asked about Caroline's career interests.", "event", "event", 0.4,
    "2023-05-08", [10], ["Melanie", "Caroline"])

# Turn 12: 鼓励做咨询师
add("Melanie", "Melanie told Caroline she would be a great counselor.", "event", "event", 0.5,
    "2023-05-08", [12], ["Melanie", "Caroline"])
add("Melanie", "Melanie believes Caroline's empathy and understanding will help others.", "knowledge", "opinion", 0.5,
    None, [12], ["Melanie", "Caroline"])

# Turn 14: 画了湖景日出
add("Melanie", "Melanie painted a lake sunrise last year.", "event", "event", 0.5,
    "2022", [14], ["Melanie"])
add("Melanie", "The lake sunrise painting is special to Melanie.", "knowledge", "opinion", 0.4,
    None, [14], ["Melanie"])

# Turn 16: 画画的放松作用
add("Melanie", "Melanie thinks painting is a fun way to express feelings and relax.", "knowledge", "opinion", 0.5,
    None, [16], ["Melanie"])
add("Melanie", "Melanie uses painting to express her feelings, get creative, and relax.", "knowledge", "activity", 0.5,
    None, [16], ["Melanie"])

# Turn 18: 带孩子游泳
add("Melanie", "Melanie is going swimming with her kids.", "event", "event", 0.4,
    "2023-05-08", [18], ["Melanie"])

# ===== 行为模式 =====
add("Caroline", "Caroline actively seeks emotional support from community groups.", "behavior", "behavior", 0.6,
    None, [3, 5, 7], ["Caroline"])
add("Caroline", "Caroline shares personal experiences and emotions openly with friends.", "behavior", "behavior", 0.5,
    None, [1, 3, 5, 7, 9, 11, 13, 15, 17], ["Caroline"])
add("Caroline", "Caroline values creative expression and compliments others' creative work.", "behavior", "behavior", 0.4,
    None, [13, 15], ["Caroline"])

add("Melanie", "Melanie encourages and supports friends in their personal growth.", "behavior", "behavior", 0.6,
    None, [4, 6, 8, 10, 12], ["Melanie"])
add("Melanie", "Melanie balances parenting responsibilities with personal interests like painting.", "behavior", "behavior", 0.5,
    None, [2, 14, 16, 18], ["Melanie"])
add("Melanie", "Melanie uses creative activities like painting to relax and express herself.", "behavior", "behavior", 0.5,
    None, [14, 16], ["Melanie"])

print(f"Total facts to insert: {len(facts)}")

# ── 插入 ──────────────────────────────────────────────────────
inserted = 0
for f in facts:
    try:
        # 插入 fact
        cur = conn.execute("""
            INSERT INTO facts (person, content, type, category, importance, content_date, is_abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (f["person"], f["content"], f["type"], f["category"],
              f["importance"], f["content_date"], 1 if f["is_abstract"] else 0))
        fact_id = cur.lastrowid

        # 插入 provenance
        for tid in f["turn_ids"]:
            conn.execute("""
                INSERT INTO provenance (fact_id, source_type, source_session_id, source_turn_id, person, confidence)
                VALUES (?, 'historical_import', 'session_1', ?, ?, 1.0)
            """, (fact_id, tid, f["person"]))

        # 插入 entities
        for ent_name in f["entities"]:
            conn.execute("INSERT OR IGNORE INTO entities (name) VALUES (?)", (ent_name,))
            row = conn.execute("SELECT entity_id FROM entities WHERE name = ?", (ent_name,)).fetchone()
            if row:
                conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                             (fact_id, row["entity_id"]))

        conn.commit()
        inserted += 1

    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint" in str(e):
            print(f"  ⚠ 跳过重复: {f['content'][:60]}")
            conn.rollback()
        else:
            print(f"  ✗ 错误: {e}")
            conn.rollback()
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        conn.rollback()

print(f"\n成功插入: {inserted}/{len(facts)}")

# ── 统计 ──────────────────────────────────────────────────────
print("\n=== 统计 ===")
for person in ["Caroline", "Melanie"]:
    rows = conn.execute("""
        SELECT type, COUNT(*) as cnt FROM facts
        WHERE fact_id IN (
            SELECT fact_id FROM provenance WHERE source_session_id = 'session_1'
        ) AND person = ?
        GROUP BY type ORDER BY type
    """, (person,)).fetchall()
    total = sum(r["cnt"] for r in rows)
    types = ", ".join(f"{r['type']}:{r['cnt']}" for r in rows)
    print(f"  {person}: {total} ({types})")

# content_date 统计
dated = conn.execute("""
    SELECT COUNT(*) FROM facts
    WHERE fact_id IN (
        SELECT fact_id FROM provenance WHERE source_session_id = 'session_1'
    ) AND content_date IS NOT NULL
""").fetchone()[0]
total_facts = conn.execute("""
    SELECT COUNT(*) FROM facts
    WHERE fact_id IN (
        SELECT fact_id FROM provenance WHERE source_session_id = 'session_1'
    )
""").fetchone()[0]
print(f"\n  content_date 已填: {dated}/{total_facts}")

conn.close()
