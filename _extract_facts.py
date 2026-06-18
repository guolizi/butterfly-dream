#!/usr/bin/env python3
"""
Extract L1 facts from conv-26 using LLM, session by session.
"""
import json
import sqlite3
import os
import sys
from hermes_tools import delegate_task

DB_PATH = "eval/dbs/locomo/conv26_v2.db"

def get_session_data(conn, session_id):
    rows = conn.execute("""
        SELECT turn_order, role, content FROM conversation_turns
        WHERE session_id = ? ORDER BY turn_order
    """, (session_id,)).fetchall()
    return [dict(r) for r in rows]

def insert_fact(conn, person, fact):
    """Insert a fact and its provenance. Returns fact_id or None if duplicate."""
    try:
        cur = conn.execute("""
            INSERT INTO facts (person, content, type, category, tags, importance, content_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            person,
            fact["content"],
            fact.get("type", "event"),
            fact.get("category", "general"),
            fact.get("tags", ""),
            float(fact.get("importance", 0.5)),
            fact.get("content_date")
        ))
        fact_id = cur.lastrowid
        
        # Provenance
        conn.execute("""
            INSERT INTO provenance (person, fact_id, source_type, source_session_id, source_turn_id)
            VALUES (?, ?, 'llm_extraction', ?, ?)
        """, (person, fact_id, fact.get("session_id"), fact.get("turn_id")))
        
        # Entities
        for entity_name in fact.get("entities", []):
            # Get or create entity
            e = conn.execute("SELECT entity_id FROM entities WHERE person=? AND name=?", 
                           (person, entity_name)).fetchone()
            if e:
                entity_id = e["entity_id"]
            else:
                cur2 = conn.execute("INSERT INTO entities (person, name) VALUES (?, ?)",
                                  (person, entity_name))
                entity_id = cur2.lastrowid
            
            conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                       (fact_id, entity_id))
        
        return fact_id
    except sqlite3.IntegrityError:
        return None

def extract_session_facts(session_id, turns):
    """Use LLM to extract facts from a session."""
    # Build conversation text
    conv_text = ""
    for t in turns:
        speaker = "Caroline" if t["role"] == "user" else "Melanie"
        conv_text += f"[{t['turn_order']}] {speaker}: {t['content']}\n"
    
    prompt = f"""You are a fact extraction system for a memory database. Extract ALL facts about Caroline from this conversation session.

RULES:
1. Extract facts ONLY about Caroline (the person whose memory this is)
2. Each fact must be a complete, standalone statement
3. Include temporal information when available (dates, relative time)
4. Categorize each fact:
   - type: "event" (something that happened, has time anchor), "knowledge" (stable fact), "behavior" (regular pattern)
   - category: "person", "activity", "goal", "preference", "event", "place", "possession", "state", "opinion", "health", "family", "career", "emotion"
5. importance: 0.0-1.0 (how significant is this fact for understanding Caroline)
6. tags: comma-separated keywords
7. entities: list of people/places/things mentioned in the fact
8. content_date: ISO date if mentioned, otherwise null

Conversation session {session_id}:
{conv_text}

Return a JSON array of facts. Each fact object:
{{
  "content": "Caroline attended an LGBTQ support group on 8 May 2023",
  "type": "event",
  "category": "event",
  "tags": "lgbtq,support group,mental health",
  "importance": 0.7,
  "content_date": "2023-05-08",
  "entities": ["Caroline", "LGBTQ support group"],
  "session_id": "{session_id}",
  "turn_id": null
}}

Extract ALL meaningful facts. Be thorough but don't make up information. If a date is relative (e.g. "yesterday", "last week"), try to calculate the absolute date from the session context."""
    
    result = delegate_task(
        goal=prompt,
        context=f"Extract facts from {session_id} of conv-26 (Caroline & Melanie conversation)",
        toolsets=["terminal"]
    )
    
    return result

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    
    # Get sessions
    sessions = conn.execute("""
        SELECT DISTINCT session_id FROM conversation_turns
        ORDER BY session_id
    """).fetchall()
    session_ids = [r["session_id"] for r in sessions]
    
    print(f"🦋 Processing {len(session_ids)} sessions for fact extraction")
    
    total_facts = 0
    total_new = 0
    
    for sid in session_ids:
        turns = get_session_data(conn, sid)
        print(f"\n{'='*60}")
        print(f"📋 {sid} — {len(turns)} turns")
        
        # Extract via LLM
        result = extract_session_facts(sid, turns)
        print(f"   LLM result type: {type(result).__name__}")
        
        # The result is a list of task results from delegate_task
        # Each result has a 'summary' field
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("summary", "") if isinstance(result[0], dict) else str(result[0])
            print(f"   Summary: {summary[:500]}")
            
            # Try to parse JSON from the summary
            try:
                # Find JSON array in the summary
                import re
                json_match = re.search(r'\[.*?\]', summary, re.DOTALL)
                if json_match:
                    facts = json.loads(json_match.group())
                    print(f"   Extracted {len(facts)} facts")
                    
                    for fact in facts:
                        fid = insert_fact(conn, "Caroline", fact)
                        if fid:
                            total_new += 1
                            print(f"   ✅ #{fid}: {fact['content'][:80]}")
                        else:
                            print(f"   ⏭️  Duplicate: {fact['content'][:60]}")
                    
                    total_facts += len(facts)
                    conn.commit()
            except Exception as e:
                print(f"   ⚠️  Parse error: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Total: {total_facts} facts extracted, {total_new} new")
    
    conn.close()

if __name__ == "__main__":
    main()
