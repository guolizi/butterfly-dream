#!/usr/bin/env python3
"""
Rebuild micro_facts index with better keyword filtering.
"""
import sqlite3
import jieba
import jieba.posseg as pseg
import re

DB_PATH = "eval/dbs/locomo/conv26_v2.db"

# 停用词
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "because", "so", "than", "as", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "very", "too", "just", "also",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "can", "could", "shall", "should", "may", "might", "must",
    "get", "got", "getting", "gotten",
    "go", "goes", "going", "went", "gone",
    "say", "says", "said", "saying",
    "make", "makes", "made", "making",
    "know", "knows", "knew", "known",
    "think", "thinks", "thought", "thinking",
    "take", "takes", "took", "taken",
    "see", "sees", "saw", "seen",
    "come", "comes", "came", "come",
    "want", "wants", "wanted", "wanting",
    "look", "looks", "looked", "looking",
    "use", "uses", "used", "using",
    "find", "finds", "found", "finding",
    "give", "gives", "gave", "given",
    "tell", "tells", "told", "telling",
    "like", "likes", "liked", "liking",
    "also", "well", "really", "very", "quite",
    "yeah", "yes", "no", "ok", "okay", "hey", "oh", "ah", "wow",
    "one", "two", "thing", "things", "way", "lot", "lots",
    "much", "many", "some", "any", "something", "someone", "somebody",
    "everyone", "everybody", "everything", "nothing", "nobody",
    "always", "never", "sometimes", "often", "usually",
    "still", "already", "yet", "even", "just", "actually",
    "maybe", "perhaps", "probably", "definitely",
    "sure", "right", "good", "great", "nice", "awesome", "cool",
    "thanks", "thank", "welcome",
    "day", "days", "time", "times", "year", "years", "month", "months", "week", "weeks",
    "now", "today", "tomorrow", "yesterday",
    "first", "last", "next", "previous",
    "new", "old", "different", "same",
    "little", "big", "long", "short", "high", "low",
    "need", "needs", "needed", "needing",
    "try", "tries", "tried", "trying",
    "start", "starts", "started", "starting",
    "stop", "stops", "stopped", "stopping",
    "keep", "keeps", "kept", "keeping",
    "let", "lets", "let", "letting",
    "help", "helps", "helped", "helping",
    "talk", "talks", "talked", "talking",
    "mean", "means", "meant", "meaning",
    "feel", "feels", "felt", "feeling",
    "hope", "hopes", "hoped", "hoping",
    "love", "loves", "loved", "loving",
    "hate", "hates", "hated", "hating",
    "put", "puts", "put", "putting",
    "set", "sets", "set", "setting",
    "bring", "brings", "brought", "bringing",
    "call", "calls", "called", "calling",
    "work", "works", "worked", "working",
    "play", "plays", "played", "playing",
    "run", "runs", "ran", "running",
    "move", "moves", "moved", "moving",
    "live", "lives", "lived", "living",
    "leave", "leaves", "left", "leaving",
    "turn", "turns", "turned", "turning",
    "ask", "asks", "asked", "asking",
    "answer", "answers", "answered", "answering",
    "show", "shows", "showed", "shown", "showing",
    "seem", "seems", "seemed", "seeming",
    "back", "away", "here", "there",
    "though", "although", "however", "therefore",
    "since", "until", "while", "after", "before",
    "about", "around", "between", "through",
    "without", "within", "along", "among",
    "because", "cause",
    # English contractions (jieba mis-tags these as nouns)
    "what", "re", "ve", "ll", "m", "s", "t", "don", "didn", "doesn",
    "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
    "won", "wouldn", "couldn", "shouldn", "mightn", "mustn",
    "needn", "daren", "ain",
    # Chinese
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "为什么",
    "因为", "所以", "但是", "如果", "虽然", "然后", "而且", "或者", "还是",
    "可以", "能够", "应该", "必须", "可能", "已经", "正在", "将要",
    "把", "被", "让", "给", "对", "从", "向", "在", "于", "与", "以",
    "啊", "吧", "吗", "呢", "哦", "嗯", "哈", "呀", "啦",
    "这个", "那个", "这些", "那些", "这里", "那里",
    "这样", "那样", "这么", "那么", "怎么", "怎样",
    "做", "当", "让", "使", "用", "拿",
    "来", "去", "出", "进", "回", "过", "起",
    "大", "小", "多", "少", "高", "低", "长", "短",
    "能", "能够", "可以", "可能", "会", "要", "想",
    "知道", "觉得", "认为", "以为", "相信",
    "告诉", "问", "回答", "说", "讲", "谈",
    "看", "听", "闻", "尝", "摸",
    "吃", "喝", "穿", "住", "行",
    "买", "卖", "开", "关", "放",
    "等", "等等", "比如", "例如",
    "还", "更", "最", "越", "太",
    "只", "才", "就", "便", "也",
    "又", "再", "也", "还", "仍",
    "已", "曾", "刚", "正", "将",
    "必", "须", "得", "应", "该",
    "真", "很", "挺", "蛮", "怪",
    "别", "不要", "不用", "不必",
    "每", "各", "某", "另",
    "整", "全", "全部", "所有",
    "半", "几", "多", "少", "许多",
    "来", "去", "上", "下", "进", "出",
    "回", "过", "起", "开", "到",
    "成", "完", "好", "掉", "住",
}

# 额外保留的关键词
EXTRA_KEEP = {
    "dancing", "painting", "running", "swimming", "hiking", "camping",
    "cooking", "baking", "reading", "writing", "drawing", "singing",
    "traveling", "shopping", "cleaning", "gardening", "knitting",
    "yoga", "meditation", "jogging", "walking",
    "happy", "sad", "angry", "excited", "nervous", "anxious",
    "stressed", "relaxed", "tired", "bored", "lonely", "grateful",
    "proud", "confident", "inspired", "motivated", "determined",
    "support", "help", "care", "love", "trust", "respect",
    "adoption", "counseling", "therapy", "mental health",
    "lgbtq", "transgender", "support group",
    "piano", "violin", "guitar", "music",
    "art", "photography", "design",
    "career", "job", "business", "startup",
    "studio", "gallery", "exhibition", "festival",
    "family", "friend", "relationship",
    "travel", "trip", "vacation", "holiday",
    "party", "celebration", "wedding", "birthday",
    "school", "college", "university", "class", "course",
    "volunteer", "charity", "donation", "fundraising",
    "health", "fitness", "exercise", "diet",
    "nature", "beach", "mountain", "park", "garden",
    "book", "movie", "song", "show", "concert",
    "dog", "cat", "pet", "animal",
    "food", "restaurant", "coffee", "tea",
    "home", "house", "apartment", "room",
    "city", "paris", "rome", "london", "new york",
    "summer", "winter", "spring", "fall", "autumn",
    "morning", "afternoon", "evening", "night",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

def is_valid_keyword(word: str) -> bool:
    """Check if a word is a valid keyword."""
    word = word.strip()
    if len(word) < 2:
        return False
    if word.lower() in STOP_WORDS:
        return False
    if re.match(r'^[\d\s\.\,\!\?\;\:\'\"\-\(\)\[\]\{\}]+$', word):
        return False
    if not re.search(r'[a-zA-Z\u4e00-\u9fff]', word):
        return False
    # Skip pure English contractions (single letter + optional letters)
    if re.match(r'^[a-z]{1,3}$', word) and word not in EXTRA_KEEP:
        return False
    # Skip pure numbers
    if re.match(r'^\d+$', word):
        return False
    return True

def extract_keywords(text: str) -> set[str]:
    """Extract keywords from text using jieba POS tagging."""
    keywords = set()
    words = pseg.cut(text)
    
    for word, flag in words:
        word = word.strip().lower()
        if len(word) < 2:
            continue
        if not is_valid_keyword(word):
            continue
        
        # Keep explicitly listed keywords
        if word in EXTRA_KEEP:
            keywords.add(word)
            continue
        
        # Keep nouns and proper nouns (jieba tags)
        if flag in ("n", "nr", "ns", "nt", "nz", "vn", "an", "eng"):
            keywords.add(word)
        # Keep English NN/NNS/NNP/NNPS
        elif flag in ("NN", "NNS", "NNP", "NNPS"):
            keywords.add(word)
        # Keep VBG (gerunds) that are activity-like
        elif flag == "VBG" and len(word) >= 4:
            keywords.add(word)
    
    return keywords

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Clear existing micro_facts
    conn.execute("DELETE FROM micro_facts")
    conn.commit()
    
    # Get all turns
    cursor = conn.execute("""
        SELECT turn_id, person, content
        FROM conversation_turns
        ORDER BY turn_id
    """)
    turns = cursor.fetchall()
    
    print(f"Processing {len(turns)} turns...")
    
    inserted = 0
    for turn in turns:
        turn_id = turn["turn_id"]
        person = turn["person"]
        content = turn["content"]
        
        keywords = extract_keywords(content)
        
        for kw in keywords:
            snippet = content[:80].replace("\n", " ")
            conn.execute(
                "INSERT INTO micro_facts (person, keyword, turn_id, snippet) VALUES (?, ?, ?, ?)",
                (person, kw, turn_id, snippet)
            )
            inserted += 1
    
    conn.commit()
    
    print(f"Inserted {inserted} micro_facts")
    
    # Stats
    cursor = conn.execute("""
        SELECT keyword, COUNT(*) as cnt
        FROM micro_facts
        GROUP BY keyword
        ORDER BY cnt DESC
        LIMIT 30
    """)
    print("\nTop keywords:")
    for row in cursor.fetchall():
        print(f"  {row['keyword']}: {row['cnt']} turns")
    
    cursor = conn.execute("SELECT COUNT(DISTINCT keyword) FROM micro_facts")
    print(f"\nUnique keywords: {cursor.fetchone()[0]}")
    
    # Test searches
    print("\n=== Search tests ===")
    for test_word in ["dancing", "adoption", "studio", "paris", "support", "painting", "dance"]:
        cursor = conn.execute("""
            SELECT m.keyword, m.snippet, t.session_id, t.turn_order
            FROM micro_facts m
            JOIN conversation_turns t ON m.turn_id = t.turn_id
            WHERE m.keyword = ?
            LIMIT 3
        """, (test_word,))
        results = cursor.fetchall()
        print(f"\n'{test_word}': {len(results)} matches")
        for r in results:
            print(f"  [{r['session_id']}:{r['turn_order']}] {r['snippet'][:60]}")
    
    conn.close()

if __name__ == "__main__":
    main()
