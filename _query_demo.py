#!/usr/bin/env python3
"""
L0 数据库查询演示 — conv-26 (Caroline × Melanie)
"""
import sqlite3
from collections import defaultdict

DB_PATH = "eval/dbs/locomo/conv26_v2.db"

def print_sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def search_fts5(conn, query, limit=5):
    """方式 1: FTS5 全文搜索"""
    sql = """
        SELECT t.turn_id, t.session_id, t.turn_order, t.role,
               snippet(conversation_turns_fts, 0, '【', '】', '…', 40) as highlighted
        FROM conversation_turns_fts
        JOIN conversation_turns t ON conversation_turns_fts.rowid = t.turn_id
        WHERE conversation_turns_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    cursor = conn.execute(sql, (query, limit))
    return cursor.fetchall()

def search_keyword(conn, keyword, limit=10):
    """方式 2: 微事实关键词检索"""
    sql = """
        SELECT m.keyword, m.snippet, t.session_id, t.turn_order, t.role, t.content
        FROM micro_facts m
        JOIN conversation_turns t ON m.turn_id = t.turn_id
        WHERE m.keyword = ?
        LIMIT ?
    """
    cursor = conn.execute(sql, (keyword, limit))
    return cursor.fetchall()

def search_multi_keyword(conn, keywords, limit=10):
    """方式 3: 多关键词交集检索"""
    placeholders = ','.join('?' * len(keywords))
    sql = f"""
        SELECT m.keyword, t.turn_id, t.session_id, t.turn_order, t.role, t.content
        FROM micro_facts m
        JOIN conversation_turns t ON m.turn_id = t.turn_id
        WHERE m.keyword IN ({placeholders})
        ORDER BY t.turn_id
    """
    cursor = conn.execute(sql, keywords)
    
    turns = defaultdict(list)
    for row in cursor.fetchall():
        turns[row['turn_id']].append(row)
    
    multi_hit = {tid: rows for tid, rows in turns.items() if len(rows) >= 2}
    sorted_turns = sorted(multi_hit.items(), key=lambda x: -len(x[1]))
    return sorted_turns[:limit]

def search_session(conn, session_id):
    """方式 4: 查看某次会话的全部内容"""
    sql = """
        SELECT turn_order, role, content FROM conversation_turns
        WHERE session_id = ? ORDER BY turn_order
    """
    cursor = conn.execute(sql, (session_id,))
    return cursor.fetchall()

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    print("🐱 L0 数据库查询演示 — conv-26 (Caroline × Melanie)")
    
    # === 方式 1: FTS5 全文搜索 ===
    print_sep("方式 1: FTS5 全文搜索 — 搜关键词（支持模糊/短语/布尔）")
    
    for q in ["adoption", "dancing", "lgbtq", '"support group"', "adoption AND family"]:
        results = search_fts5(conn, q)
        print(f"\n🔍 MATCH '{q}': {len(results)} 条")
        for r in results:
            print(f"  [{r['session_id']}:{r['turn_order']}] {r['role']}")
            print(f"    {r['highlighted'][:100]}")
    
    # === 方式 2: 微事实关键词检索 ===
    print_sep("方式 2: 微事实关键词检索 — 精确匹配（速度快）")
    
    for kw in ["adoption", "dancing", "painting", "lgbtq"]:
        results = search_keyword(conn, kw)
        print(f"\n🔑 '{kw}': {len(results)} 条")
        for r in results[:3]:
            print(f"  [{r['session_id']}:{r['turn_order']}] {r['content'][:80]}")
    
    # === 方式 3: 多关键词交集 ===
    print_sep("方式 3: 多关键词交集 — 找同时聊到多个话题的轮次")
    
    keywords = ["adoption", "lgbtq", "family", "support"]
    results = search_multi_keyword(conn, keywords)
    print(f"\n🔗 同时包含 {keywords} 中 ≥2 个的轮次:")
    for turn_id, rows in results:
        kws = [r['keyword'] for r in rows]
        r = rows[0]
        print(f"  [{r['session_id']}:{r['turn_order']}] 命中: {kws}")
        print(f"    {r['content'][:80]}")
    
    # === 方式 4: 按会话浏览 ===
    print_sep("方式 4: 按会话浏览 — 看某次对话的全部内容")
    
    results = search_session(conn, "session_1")
    print(f"\n📋 session_1 全部 {len(results)} 轮:")
    for r in results:
        print(f"  [{r['turn_order']}] {r['role']}: {r['content'][:80]}")
    
    # === 实用技巧 ===
    print_sep("💡 FTS5 搜索技巧")
    
    print("① 前缀搜索: MATCH 'adop*' → 匹配 adoption, adopt, adopting...")
    results = search_fts5(conn, "adop*", 3)
    for r in results:
        print(f"   [{r['session_id']}:{r['turn_order']}] {r['highlighted'][:100]}")
    
    print("\n② 短语搜索: MATCH '\"support group\"' → 精确短语匹配")
    results = search_fts5(conn, '"support group"', 3)
    for r in results:
        print(f"   [{r['session_id']}:{r['turn_order']}] {r['highlighted'][:100]}")
    
    print("\n③ 布尔 AND: MATCH 'adoption AND family' → 同时包含")
    results = search_fts5(conn, "adoption AND family", 3)
    for r in results:
        print(f"   [{r['session_id']}:{r['turn_order']}] {r['highlighted'][:100]}")
    
    print("\n④ 布尔 OR: MATCH 'piano OR violin' → 任一包含")
    results = search_fts5(conn, "piano OR violin", 3)
    for r in results:
        print(f"   [{r['session_id']}:{r['turn_order']}] {r['highlighted'][:100]}")
    
    print("\n⑤ 排除: MATCH 'adoption NOT agency' → 包含 adoption 但不含 agency")
    results = search_fts5(conn, "adoption NOT agency", 3)
    for r in results:
        print(f"   [{r['session_id']}:{r['turn_order']}] {r['highlighted'][:100]}")
    
    # === 数据统计 ===
    print_sep("📊 数据统计")
    
    cursor = conn.execute("SELECT COUNT(*) FROM conversation_turns")
    print(f"总对话轮次: {cursor.fetchone()[0]}")
    
    cursor = conn.execute("SELECT COUNT(DISTINCT keyword) FROM micro_facts")
    print(f"唯一关键词数: {cursor.fetchone()[0]}")
    
    cursor = conn.execute("SELECT COUNT(*) FROM micro_facts")
    print(f"微事实索引总数: {cursor.fetchone()[0]}")
    
    cursor = conn.execute("""
        SELECT keyword, COUNT(*) as cnt FROM micro_facts
        GROUP BY keyword ORDER BY cnt DESC LIMIT 10
    """)
    print("最热关键词 Top 10:")
    for r in cursor.fetchall():
        print(f"  {r['keyword']}: {r['cnt']} 次")
    
    conn.close()

if __name__ == "__main__":
    main()
