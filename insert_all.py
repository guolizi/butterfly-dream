"""
批量插入所有提取结果到数据库
"""
import sqlite3, json, sys

DB = '/home/xx/butterfly-dream/memory_store.db'

def get_conn():
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn

# 所有会话的提取结果
results = {
    # session_1 (already inserted via earlier batch)
    # session_2 (already inserted via earlier batch)
    # session_3 (already inserted via earlier batch)
    
    # session_4-6 (already inserted via insert_s4_6.py)
    
    # session_7-9
    "session_7": [
        {"dimension": "event", "content": "Caroline 两天前参加了LGBTQ会议", "type": "event", "category": "professional", "tags": "LGBTQ,conference,networking", "importance": 0.7, "content_date": "2023-07-10", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ会议", "time": "2023-07-10", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为支持小组是安全的分享空间", "type": "knowledge", "category": "belief", "tags": "support_group,safe_space,sharing", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为在支持小组中分享经历有助于自我成长", "type": "knowledge", "category": "belief", "tags": "support_group,sharing,growth", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 参加LGBTQ相关的支持小组和会议", "type": "behavior", "category": "social", "tags": "LGBTQ,support_group,conference", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对参加LGBTQ会议感到兴奋", "emotion_vector": [0.8, 0.8, 0.7], "emotion_label": "兴奋", "emotion_target": "event:conference", "source": "user", "importance": 0.7, "significance_reason": "用户表达对会议的积极感受", "trigger_topics": ["LGBTQ会议", "社交"], "timestamp": "2023-07-12T16:33:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_8": [
        {"dimension": "event", "content": "Caroline 上周五去参加了领养咨询会", "type": "event", "category": "family", "tags": "adoption,consultation", "importance": 0.7, "content_date": "2023-07-07", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "领养咨询会", "time": "2023-07-07", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为领养咨询会信息量很大", "type": "knowledge", "category": "belief", "tags": "adoption,consultation,informative", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为领养咨询会让她对领养过程有了更清晰的了解", "type": "knowledge", "category": "belief", "tags": "adoption,clarity,process", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 主动了解领养过程", "type": "behavior", "category": "family", "tags": "adoption,research,proactive", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对领养咨询会感到兴奋", "emotion_vector": [0.8, 0.7, 0.7], "emotion_label": "兴奋", "emotion_target": "event:adoption_consultation", "source": "user", "importance": 0.7, "significance_reason": "用户表达对领养咨询会的积极感受", "trigger_topics": ["领养", "咨询"], "timestamp": "2023-07-15T13:51:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_9": [
        {"dimension": "event", "content": "Caroline 最近在深入探索心理咨询职业", "type": "event", "category": "career", "tags": "counseling,career_exploration", "importance": 0.6, "content_date": "2023-07-17", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "探索", "object": "心理咨询职业", "time": "2023-07-17", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为心理咨询师可以真正帮助到人", "type": "knowledge", "category": "belief", "tags": "counseling,helping,career", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为通过自己的经历帮助他人很有意义", "type": "knowledge", "category": "belief", "tags": "helping,experience,meaningful", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 探索心理咨询作为职业方向", "type": "behavior", "category": "career", "tags": "counseling,career,exploration", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对探索心理咨询职业感到充满希望", "emotion_vector": [0.7, 0.7, 0.6], "emotion_label": "充满希望", "emotion_target": "career:counseling", "source": "user", "importance": 0.6, "significance_reason": "用户表达对职业探索的积极感受", "trigger_topics": ["心理咨询", "职业"], "timestamp": "2023-07-17T14:31:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_10-12
    "session_10": [
        {"dimension": "event", "content": "Caroline 上周二加入了一个新的LGBTQ+支持小组", "type": "event", "category": "social", "tags": "LGBTQ+,support_group,new", "importance": 0.7, "content_date": "2023-07-18", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "加入", "object": "LGBTQ+支持小组", "time": "2023-07-18", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为支持小组提供了一个安全的分享空间", "type": "knowledge", "category": "belief", "tags": "support_group,safe_space,sharing", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 通过加入支持小组寻求社群连接", "type": "behavior", "category": "social", "tags": "support_group,community,connection", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对加入新的支持小组感到兴奋和感激", "emotion_vector": [0.8, 0.8, 0.7], "emotion_label": "兴奋感激", "emotion_target": "event:new_support_group", "source": "user", "importance": 0.7, "significance_reason": "用户表达对加入新小组的积极感受", "trigger_topics": ["LGBTQ+", "支持小组"], "timestamp": "2023-07-20T20:56:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_11": [
        {"dimension": "event", "content": "Caroline 最近参加了一个LGBTQ活动", "type": "event", "category": "social", "tags": "LGBTQ,event,community", "importance": 0.6, "content_date": "2023-08-14", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ活动", "time": "2023-08-14", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为在LGBTQ+社区中找到归属感很重要", "type": "knowledge", "category": "belief", "tags": "LGBTQ+,community,belonging", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 积极参加LGBTQ+社区活动", "type": "behavior", "category": "social", "tags": "LGBTQ+,community,participation", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 在LGBTQ+社区中找到归属感感到开心", "emotion_vector": [0.8, 0.8, 0.7], "emotion_label": "开心", "emotion_target": "community:belonging", "source": "user", "importance": 0.6, "significance_reason": "用户表达在社区中的归属感", "trigger_topics": ["LGBTQ+", "归属感"], "timestamp": "2023-08-14T14:24:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_12": [
        {"dimension": "event", "content": "Caroline 最近有一个重要的里程碑", "type": "event", "category": "milestone", "tags": "milestone,progress", "importance": 0.7, "content_date": "2023-08-17", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "达成", "object": "重要里程碑", "time": "2023-08-17", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为实现重要里程碑让人充满动力", "type": "knowledge", "category": "belief", "tags": "milestone,motivation,progress", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 庆祝自己的进步和里程碑", "type": "behavior", "category": "psychology", "tags": "celebration,milestone,progress", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对最近的里程碑感到自豪和兴奋", "emotion_vector": [0.9, 0.8, 0.8], "emotion_label": "自豪兴奋", "emotion_target": "event:milestone", "source": "user", "importance": 0.7, "significance_reason": "用户表达对里程碑的自豪", "trigger_topics": ["里程碑", "进步"], "timestamp": "2023-08-17T13:50:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_13-15
    "session_13": [
        {"dimension": "event", "content": "Caroline 向领养机构提交了申请", "type": "event", "category": "family", "tags": "adoption,application,submitted", "importance": 0.8, "content_date": "2023-08-23", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "提交", "object": "领养申请", "time": "2023-08-23", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为领养是组建家庭的好方式", "type": "knowledge", "category": "value", "tags": "adoption,family,value", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 积极推动领养进程", "type": "behavior", "category": "family", "tags": "adoption,proactive,progress", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对提交领养申请感到兴奋和紧张", "emotion_vector": [0.8, 0.6, 0.7], "emotion_label": "兴奋紧张", "emotion_target": "event:adoption_application", "source": "user", "importance": 0.8, "significance_reason": "用户表达对领养申请的复杂情绪", "trigger_topics": ["领养", "申请"], "timestamp": "2023-08-23T15:31:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_14": [
        {"dimension": "event", "content": "Caroline 上周徒步时遇到了一些不愉快", "type": "event", "category": "outdoor", "tags": "hiking,unpleasant_experience", "importance": 0.5, "content_date": "2023-08-18", "emotion_tag": "negative", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "徒步", "object": None, "time": "2023-08-18", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 喜欢徒步", "type": "knowledge", "category": "hobby", "tags": "hiking,hobby", "importance": 0.4, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 通过徒步放松心情", "type": "behavior", "category": "psychology", "tags": "hiking,relaxation", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对徒步时遇到的不愉快感到沮丧", "emotion_vector": [-0.5, 0.6, 0.3], "emotion_label": "沮丧", "emotion_target": "event:hiking_incident", "source": "user", "importance": 0.5, "significance_reason": "用户表达徒步中的负面经历", "trigger_topics": ["徒步", "不愉快"], "timestamp": "2023-08-25T13:33:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    "session_15": [
        {"dimension": "event", "content": "Caroline 最近在推进领养申请", "type": "event", "category": "family", "tags": "adoption,application,progress", "importance": 0.7, "content_date": "2023-08-28", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "推进", "object": "领养申请", "time": "2023-08-28", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为领养是一个漫长的过程但值得", "type": "knowledge", "category": "belief", "tags": "adoption,process,worthwhile", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 耐心推进领养申请", "type": "behavior", "category": "family", "tags": "adoption,patience,progress", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对领养申请的进展感到乐观", "emotion_vector": [0.7, 0.7, 0.6], "emotion_label": "乐观", "emotion_target": "event:adoption_progress", "source": "user", "importance": 0.6, "significance_reason": "用户表达对领养进展的积极态度", "trigger_topics": ["领养", "进展"], "timestamp": "2023-08-28T15:19:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_16 (from the last batch - session_16)
    "session_16": [
        {"dimension": "event", "content": "Caroline 上周末和朋友骑自行车出游", "type": "event", "category": "social", "tags": "biking,friends", "importance": 0.6, "content_date": "2023-09-09", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "骑自行车", "object": None, "time": "2023-09-09", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 从17岁左右开始创作艺术", "type": "knowledge", "category": "art", "tags": "art,started,age_17", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 是跨性别女性", "type": "knowledge", "category": "identity", "tags": "transgender,woman", "importance": 0.8, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 为LGBTQ+社区做志愿服务", "type": "knowledge", "category": "volunteering", "tags": "LGBTQ+,volunteering", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 没做过陶艺但愿意尝试新艺术形式", "type": "knowledge", "category": "art", "tags": "pottery,new_experience", "importance": 0.3, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 通过绘画探索性别认同和表达情感", "type": "behavior", "category": "psychology", "tags": "painting,gender_identity,expression", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 在转变期间通过创作艺术理解和接纳自己", "type": "behavior", "category": "psychology", "tags": "art,self_acceptance,transition", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 从事志愿服务帮助LGBTQ+社区", "type": "behavior", "category": "social", "tags": "volunteering,LGBTQ+", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对自己的跨性别身份感到骄傲", "emotion_vector": [0.9, 0.8, 0.9], "emotion_label": "骄傲", "emotion_target": "self", "source": "user", "importance": 0.8, "significance_reason": "用户表达对真实自我的骄傲", "trigger_topics": ["跨性别", "自我接纳"], "timestamp": "2023-09-13T00:09:00"},
        {"dimension": "emotion", "content": "Caroline 对朋友、家人和导师的支持感到感激", "emotion_vector": [0.8, 0.7, 0.8], "emotion_label": "感激", "emotion_target": "others", "source": "user", "importance": 0.6, "significance_reason": "用户表达对支持的感激", "trigger_topics": ["支持", "感恩"], "timestamp": "2023-09-13T00:09:00"},
        {"dimension": "emotion", "content": "Caroline 与接纳她的人在一起感到快乐", "emotion_vector": [0.8, 0.9, 0.8], "emotion_label": "快乐", "emotion_target": "relationships", "source": "user", "importance": 0.7, "significance_reason": "用户表达在接纳关系中的快乐", "trigger_topics": ["接纳", "关系"], "timestamp": "2023-09-13T00:09:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_17 (from the last batch)
    "session_17": [
        {"dimension": "event", "content": "Caroline 联系了导师咨询领养建议", "type": "event", "category": "family", "tags": "adoption,mentor", "importance": 0.8, "content_date": "2023-10-13", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "联系", "object": "导师", "time": "2023-10-13", "location": None}},
        {"dimension": "event", "content": "Caroline 参加了跨性别诗歌朗诵会", "type": "event", "category": "social", "tags": "poetry,transgender,empowerment", "importance": 0.7, "content_date": "2023-10-06", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "跨性别诗歌朗诵会", "time": "2023-10-06", "location": None}},
        {"dimension": "event", "content": "Caroline 创作了代表自由和真实的画", "type": "event", "category": "art", "tags": "drawing,freedom,self-expression", "importance": 0.6, "content_date": "2023-10-13", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "创作", "object": "代表自由和真实的画", "time": "2023-10-13", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 的梦想是领养孩子并为需要帮助的孩子提供有爱的家", "type": "knowledge", "category": "family", "tags": "adoption,dream", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 最近在尝试抽象画作为自我表达方式", "type": "knowledge", "category": "hobby", "tags": "abstract_painting,self-expression", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 画了一幅代表自由和真实的画，提醒自己忠于自我和拥抱女性身份", "type": "knowledge", "category": "art", "tags": "drawing,freedom,womanhood", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 通过抽象画释放情感和表达自我", "type": "behavior", "category": "psychology", "tags": "abstract_painting,emotion,expression", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 通过艺术表达自我认同和女性身份", "type": "behavior", "category": "psychology", "tags": "art,self-identity,womanhood", "importance": 0.5, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对开始领养新篇章感到兴奋", "emotion_vector": [0.9, 0.8, 0.7], "emotion_label": "兴奋", "emotion_target": "event:adoption_journey", "source": "user", "importance": 0.8, "significance_reason": "用户表达对领养的兴奋", "trigger_topics": ["领养", "新篇章"], "timestamp": "2023-10-13T10:31:00"},
        {"dimension": "emotion", "content": "Caroline 对参加跨性别诗歌朗诵会感到有力量和鼓舞", "emotion_vector": [0.8, 0.9, 0.7], "emotion_label": "鼓舞", "emotion_target": "event:poetry_reading", "source": "user", "importance": 0.7, "significance_reason": "用户描述朗诵会充满力量和启发", "trigger_topics": ["诗歌朗诵会", "跨性别", "力量"], "timestamp": "2023-10-13T10:31:00"},
        {"dimension": "emotion", "content": "Caroline 对自己的画作感到自豪和满足", "emotion_vector": [0.8, 0.7, 0.9], "emotion_label": "自豪", "emotion_target": "artwork:freedom_painting", "source": "user", "importance": 0.6, "significance_reason": "用户解释画作的意义并展示", "trigger_topics": ["画", "自由", "真实"], "timestamp": "2023-10-13T10:31:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_18 (from the last batch)
    "session_18": [
        {"dimension": "event", "content": "Caroline 和Melanie聊天", "type": "event", "category": "social", "tags": "chat,friendship", "importance": 0.3, "content_date": "2023-10-20", "emotion_tag": None, "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "聊天", "object": "Melanie", "time": "2023-10-20", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为家人就是一切", "type": "knowledge", "category": "value", "tags": "family,value", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为孩子有很强的韧性", "type": "knowledge", "category": "belief", "tags": "children,resilience", "importance": 0.4, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 在朋友遇到困难时给予安慰和支持", "type": "behavior", "category": "social", "tags": "support,friendship,comfort", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 对Melanie儿子的事故感到担心", "emotion_vector": [-0.3, 0.7, 0.4], "emotion_label": "担心", "emotion_target": "event:accident", "source": "user", "importance": 0.5, "significance_reason": "用户对朋友家庭的关心", "trigger_topics": ["事故", "安全"], "timestamp": "2023-10-20T18:55:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
    ],
    # session_19 (just extracted)
    "session_19": [
        {"dimension": "event", "content": "Caroline 通过了领养机构的面试", "type": "event", "category": "family", "tags": "adoption,interview,passed", "importance": 0.9, "content_date": "2023-10-20", "emotion_tag": "positive", "entities": ["Caroline"], "structured_data": {"subject": "Caroline", "action": "通过", "object": "领养机构面试", "time": "2023-10-20", "location": None}},
        {"dimension": "knowledge", "content": "Caroline 认为领养是一种回馈和表达爱的方式", "type": "knowledge", "category": "value", "tags": "adoption,giving_back,love", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为爱和接纳应该是每个人的权利", "type": "knowledge", "category": "value", "tags": "love,acceptance,right", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为做真实的自己、诚实地生活是自由的", "type": "knowledge", "category": "value", "tags": "authenticity,freedom,honesty", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为自我接纳是一个漫长的过程", "type": "knowledge", "category": "psychology", "tags": "self-acceptance,journey", "importance": 0.6, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "knowledge", "content": "Caroline 认为帮助他人成长和带来安慰能带来快乐", "type": "knowledge", "category": "value", "tags": "helping,joy,growth", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 想把自己得到的支持传递给需要帮助的人", "type": "behavior", "category": "psychology", "tags": "support,giving_back,helping", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 希望建立自己的家庭，给孩子们一个家", "type": "behavior", "category": "family", "tags": "family,adoption,home", "importance": 0.8, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 准备好向需要的人提供爱和支持", "type": "behavior", "category": "psychology", "tags": "love,support,readiness", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "behavior", "content": "Caroline 感激身边人的支持，认为他们的鼓励造就了她", "type": "behavior", "category": "psychology", "tags": "gratitude,support,encouragement", "importance": 0.7, "emotion_tag": None, "entities": ["Caroline"]},
        {"dimension": "emotion", "content": "Caroline 通过领养面试后感到非常兴奋和感激", "emotion_vector": [0.9, 0.9, 0.8], "emotion_label": "兴奋感激", "emotion_target": "event:adoption_interview", "source": "user", "importance": 0.9, "significance_reason": "用户表达强烈的兴奋和感激", "trigger_topics": ["领养面试", "家庭"], "timestamp": "2023-10-22T09:55:00"},
        {"dimension": "emotion", "content": "Caroline 对帮助他人感到快乐", "emotion_vector": [0.8, 0.7, 0.9], "emotion_label": "快乐满足", "emotion_target": "behavior:helping_others", "source": "user", "importance": 0.7, "significance_reason": "用户表达帮助他人带来的喜悦", "trigger_topics": ["帮助他人", "成长"], "timestamp": "2023-10-22T09:55:00"},
        {"dimension": "emotion", "content": "Caroline 对自我接纳的旅程感到感激", "emotion_vector": [0.7, 0.8, 0.6], "emotion_label": "感激", "emotion_target": "experience:self_acceptance_journey", "source": "user", "importance": 0.6, "significance_reason": "用户表达对过去支持者的感激", "trigger_topics": ["自我接纳", "支持"], "timestamp": "2023-10-22T09:55:00"},
        {"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.9}
    ]
}

def insert_facts(conn, session_id, facts):
    """插入事实到数据库"""
    cursor = conn.cursor()
    count = 0
    
    for fact in facts:
        dim = fact["dimension"]
        
        if dim == "emotion":
            # 插入情感事件
            cursor.execute("""
                INSERT INTO emotion_events 
                (person, content, emotion_vector, emotion_label, emotion_target, 
                 source, initial_importance, significance_reason, trigger_topics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "Caroline",
                fact["content"],
                json.dumps(fact["emotion_vector"]),
                fact.get("emotion_label"),
                fact.get("emotion_target"),
                fact.get("source", "user"),
                fact["importance"],
                fact.get("significance_reason"),
                json.dumps(fact.get("trigger_topics", [])),
                fact["timestamp"]
            ))
            count += 1
            
        elif dim == "relation":
            # 跳过关系插入 - 表结构是事实级关联(fact_id→fact_id)，不是实体级
            # 后面再单独处理实体关系
            pass
            count += 1
            
        else:
            # 插入事实（event/knowledge/behavior）
            cursor.execute("""
                INSERT INTO facts 
                (person, content, type, category, tags, importance, 
                 content_date, emotion_tag, entities, structured_data, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "Caroline",
                fact["content"],
                fact["type"],
                fact.get("category"),
                fact.get("tags"),
                fact["importance"],
                fact.get("content_date"),
                fact.get("emotion_tag"),
                json.dumps(fact.get("entities", [])),
                json.dumps(fact.get("structured_data")) if fact.get("structured_data") else None,
                session_id
            ))
            count += 1
    
    conn.commit()
    return count

def main():
    conn = get_conn()
    total = 0
    
    for session_id, facts in results.items():
        n = insert_facts(conn, session_id, facts)
        print(f"  {session_id}: {n} 条")
        total += n
    
    conn.close()
    print(f"\n✅ 总计插入 {total} 条事实")

if __name__ == "__main__":
    main()
