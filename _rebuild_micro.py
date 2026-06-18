"""从 conversation_turns 重建全部 micro_facts 索引"""
import sqlite3
import re

db = 'eval/dbs/locomo/conv26_v2.db'
conn = sqlite3.connect(db)

# 中文+英文停用词
STOPWORDS = {
    # 中文
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些',
    '吗', '吧', '啊', '呢', '哦', '嗯', '哈', '呀', '啦', '哇', '哟',
    '什么', '怎么', '为什么', '因为', '所以', '但是', '可是', '然后',
    '那个', '这个', '这里', '那里', '这样', '那样', '可以', '应该',
    '已经', '还是', '还是', '或者', '如果', '虽然', '而且', '不过',
    '真的', '觉得', '知道', '看到', '听到', '想到', '告诉', '谢谢',
    '请', '让', '把', '被', '从', '对', '比', '向', '跟', '与',
    # 英文
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'and', 'but', 'or', 'if', 'because',
    'so', 'than', 'then', 'also', 'very', 'just', 'like', 'about',
    'not', 'no', 'yes', 'ok', 'okay', 'oh', 'ah', 'well', 'hey',
    'get', 'got', 'go', 'went', 'come', 'came', 'take', 'took',
    'make', 'made', 'say', 'said', 'tell', 'told', 'think', 'thought',
    'know', 'knew', 'see', 'saw', 'want', 'need', 'let', 'thing',
    'things', 'way', 'really', 'actually', 'even', 'still', 'much',
    'many', 'some', 'any', 'every', 'all', 'both', 'each',
    'what', 'when', 'where', 'who', 'which', 'how',
    'too', 'quite', 'pretty', 'kind', 'sort', 'little', 'bit',
    'right', 'sure', 'yeah', 'wow', 'gosh', 'hmm', 'um', 'uh',
    'thanks', 'thank', 'please', 'sorry',
}

# 专有名词（保持完整）
PROPER_NOUNS = {'caroline', 'melanie', 'mel', 'lgbtq', 'lgbt'}

# 情感/状态关键词（提取为单字词时的补充）
EMOTION_WORDS = {
    'happy', 'sad', 'angry', 'excited', 'worried', 'nervous', 'proud',
    'grateful', 'hopeful', 'inspired', 'motivated', 'scared', 'lonely',
    'blessed', 'thankful', 'confident', 'curious', 'amazed', 'surprised',
    '开心', '高兴', '难过', '伤心', '生气', '兴奋', '担心', '紧张',
    '自豪', '感激', '希望', '感动', '害怕', '孤独', '自信', '好奇',
}

def extract_keywords(text):
    """从文本中提取有意义的短语（1-3个词）"""
    text = text.lower().strip()
    if not text:
        return []
    
    # 拆成句子
    sentences = re.split(r'[。！？.!?\n]', text)
    
    keywords = set()
    
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 3:
            continue
        
        # 提取中文关键词（2-6字短语）
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,6}', sent)
        for w in cn_words:
            if w not in STOPWORDS:
                keywords.add(w)
        
        # 提取英文关键词（保留专有名词和重要词汇）
        en_words = re.findall(r"[a-zA-Z']{3,}", sent)
        for w in en_words:
            wl = w.lower().strip("'")
            if wl in PROPER_NOUNS:
                keywords.add(wl)
            elif wl in EMOTION_WORDS:
                keywords.add(wl)
            elif len(wl) >= 4 and wl not in STOPWORDS:
                keywords.add(wl)
        
        # 提取英文双词短语（重要概念）
        en_bigrams = re.findall(r"([a-zA-Z']{3,})\s+([a-zA-Z']{3,})", sent)
        for w1, w2 in en_bigrams:
            w1l, w2l = w1.lower().strip("'"), w2.lower().strip("'")
            if w1l not in STOPWORDS and w2l not in STOPWORDS:
                bigram = f"{w1l}_{w2l}"
                # 只保留有意义的搭配
                if len(bigram) >= 8:
                    keywords.add(bigram)
    
    return list(keywords)


# 清空旧数据
conn.execute('DELETE FROM micro_facts')
conn.commit()

# 获取所有轮次
turns = conn.execute('''
    SELECT turn_id, person, session_id, content, turn_order
    FROM conversation_turns
    ORDER BY session_id, turn_order
''').fetchall()

print(f'总轮次数: {len(turns)}')

inserted = 0
current_session = None
for t in turns:
    turn_id, person, session_id, content, turn_order = t
    
    if session_id != current_session:
        print(f'\n  session_id={session_id}...', end=' ', flush=True)
        current_session = session_id
    
    keywords = extract_keywords(content)
    snippet = content[:80] if content else ''
    
    for kw in keywords:
        conn.execute(
            'INSERT INTO micro_facts (person, keyword, turn_id, snippet, promoted, miss_count) '
            'VALUES (?, ?, ?, ?, 0, 0)',
            (person, kw, turn_id, snippet)
        )
        inserted += 1

conn.commit()

# 统计
stats = conn.execute('''
    SELECT m.session_id, COUNT(*) as cnt, COUNT(DISTINCT m.person) as persons
    FROM (
        SELECT ct.session_id, mf.micro_id, mf.person
        FROM micro_facts mf
        JOIN conversation_turns ct ON mf.turn_id = ct.turn_id
    ) m
    GROUP BY m.session_id
    ORDER BY m.session_id
''').fetchall()

print(f'\n\n=== 重建完成 ===')
print(f'总 micro_facts: {inserted}')
print(f'\n按 session 分布:')
for s in stats:
    print(f'  {s[0]}: {s[1]}条 (参与者{s[2]}人)')

# 按 person 统计
person_stats = conn.execute('''
    SELECT person, COUNT(*) as cnt 
    FROM micro_facts 
    GROUP BY person 
    ORDER BY cnt DESC
''').fetchall()
print(f'\n按 person 分布:')
for p in person_stats:
    print(f'  {p[0]}: {p[1]}条')

conn.close()
