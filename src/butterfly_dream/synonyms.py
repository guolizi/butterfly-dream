"""Synonym dictionary for FTS5 query expansion in retrieval.

This file provides a maintainable, centralized mapping of words to their
synonyms. The retrieval pipeline uses these to expand query tokens as
OR groups in FTS5, bridging vocabulary gaps between user queries and
stored fact texts (e.g. query "children" → also search "students", "youth").

Design principles:
  - Cover *common* vocabulary gaps, not every LoCoMo case.
  - ~2-6 synonyms per entry.  Too many dilute precision.
  - Keys should be lemma/base forms (the query sanitizer lemmatizes before lookup).
  - Include both English and Chinese (jieba-segmented) entries.
  - Organize by semantic category for easy maintenance.
"""

from __future__ import annotations

import json
import os
from typing import Final

# ---------------------------------------------------------------------------
# Built-in synonym map
# ---------------------------------------------------------------------------
# Keys are lowercase, ideally lemmatized forms.
# Values are lowercase synonym lists (the expander lowercases everything).
_CORE: dict[str, list[str]] = {

    # ====== English ======

    # --- People & Relationships ---
    "family":     ["kid", "children", "parent"],
    "child":      ["student", "youth", "teen", "kid"],
    "children":   ["student", "youth", "teen", "kid"],
    "friend":     ["buddy", "companion", "pal"],
    "parent":     ["mother", "father", "mom", "dad"],

    # --- Participation verbs ---
    "participate": ["join", "attend", "go"],
    "join":        ["attend", "participate", "sign"],
    "attend":      ["join", "participate", "go"],

    # --- Activity / hobby nouns ---
    "activity":   ["hobby", "sport", "pastime", "interest"],
    "activities": ["hobby", "sport", "pastime", "interest"],
    "play":       ["game", "sport", "activity"],
    "travel":     ["trip", "vacation", "journey", "tour"],

    # --- Art / creative ---
    # Q44: query "what kind of art" → fact text "abstract painting", "paintings"
    # FTS5 prefix "art*" doesn't match "painting" (starts with "paint").
    "art":        ["painting", "drawing", "sculpture"],

    # --- Music / performance ---
    # Q62: query "musical artists/bands" → fact text "concert", "perform"
    "music":      ["song", "concert", "band", "artist", "musical"],
    "musical":    ["music", "concert", "show", "performance"],
    "artist":     ["singer", "musician", "performer", "band"],
    "band":       ["group", "performer", "orchestra", "ensemble"],
    "concert":    ["show", "performance", "gig", "live"],

    # --- change / transition ---
    # Q66: query "changes" → fact text "transition", "body changes"
    "change":     ["transition", "shift", "transformation", "difference"],

    # --- Help / support ---
    "help":       ["support", "assist", "aid"],
    "support":    ["help", "assist", "aid"],

    # --- Ally / advocacy ---
    # Q47: query "ally" → fact text "support", "friend", "advocate"
    "ally":       ["support", "friend", "advocate"],
    "advocate":   ["support", "ally", "champion"],

    # --- Preference / emotion ---
    "like":       ["enjoy", "love", "appreciate", "prefer"],
    "hate":       ["dislike", "despise", "detest"],
    "happy":      ["glad", "joyful", "delighted", "pleased"],
    "sad":        ["unhappy", "depressed", "down", "upset"],
    "feel":       ["experience", "sense", "go through"],

    # --- Possession & transactions ---
    "have":       ["own", "possess"],
    "buy":        ["purchase", "acquire", "get"],
    "get":        ["receive", "obtain", "acquire"],

    # --- Communication ---
    "talk":       ["speak", "discuss", "chat", "converse"],
    "say":        ["tell", "mention", "state", "remark"],
    "meet":       ["gather", "get together", "meet up"],

    # --- Work / study ---
    "work":       ["job", "career", "employment", "occupation"],
    "study":      ["learn", "research", "education"],
    "pursue":     ["chase", "strive", "go after"],

    # --- Time / planning ---
    "plan":       ["intend", "hope", "goal", "aspire", "want"],
    "recent":     ["last", "past", "recently"],

    # --- General verbs ---
    "make":       ["create", "build", "produce", "craft"],
    "fix":        ["repair", "mend", "restore"],
    "use":        ["utilize", "employ"],
    "give":       ["offer", "provide", "present"],

    # ====== Chinese (jieba-segmented) ======

    # --- 人物/关系 ---
    "家庭":   ["家人", "孩子", "子女", "亲戚"],
    "家人":   ["家庭", "孩子", "子女"],
    "孩子":   ["学生", "青年", "少年", "小孩", "子女"],
    "朋友":   ["好友", "哥们", "伙伴", "闺蜜"],

    # --- 动作/参与 ---
    "参加":   ["参与", "加入", "出席", "报名"],
    "参与":   ["参加", "加入", "出席"],
    "加入":   ["参加", "参与", "报名"],

    # --- 活动/爱好 ---
    "活动":   ["爱好", "运动", "娱乐", "消遣"],
    "爱好":   ["活动", "兴趣", "嗜好", "喜欢"],
    "喜欢":   ["爱好", "喜爱", "热爱", "钟情"],
    "玩":     ["游戏", "运动", "娱乐"],

    # --- 帮助/支持 ---
    "帮助":   ["支持", "协助", "帮忙", "援助"],
    "支持":   ["帮助", "协助", "支援"],

    # --- 时间/计划 ---
    "时间":   ["日期", "时候", "何时"],
    "计划":   ["打算", "目标", "希望", "准备"],
    "最近":   ["近期", "前几天", "上次"],
    "经常":   ["常常", "总是", "频繁"],

    # --- 地点/出行 ---
    "地方":   ["地点", "位置", "场所", "哪里"],
    "去":     ["到", "前往", "出发"],
    "旅行":   ["旅游", "度假", "出行", "游玩"],

    # --- 沟通/交流 ---
    "说":     ["告诉", "讲", "提到", "聊"],
    "聊天":   ["讨论", "交流", "谈话", "聊"],
    "讨论":   ["聊天", "交流", "商量", "谈"],

    # --- 情感/状态 ---
    "开心":   ["快乐", "高兴", "愉快", "幸福"],
    "难过":   ["伤心", "不开心", "沮丧", "低落"],
    "感觉":   ["觉得", "感受", "感到"],
    "想":     ["希望", "打算", "想要", "愿望"],

    # --- 工作/学习 ---
    "工作":   ["上班", "职业", "事业", "打工"],
    "学习":   ["学", "读书", "研究", "上学"],

    # --- 日常动词 ---
    "买":     ["购买", "购置", "入手"],
    "看":     ["阅读", "观看", "浏览", "读"],
    "做":     ["干", "搞", "弄", "进行"],
    "吃":     ["喝", "品尝", "用餐"],
    "给":     ["送", "提供", "递"],
}

# ---------------------------------------------------------------------------
# Optional user-custom synonym file
# ---------------------------------------------------------------------------
_CUSTOM_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "synonyms_custom.json",
)


def _load_custom() -> dict[str, list[str]]:
    """Load user-custom synonyms from JSON, merged on top of built-in."""
    if not os.path.exists(_CUSTOM_PATH):
        return {}
    try:
        with open(_CUSTOM_PATH) as f:
            custom: dict[str, list[str]] = json.load(f)
        return custom
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Resolved synonym map: built-in core + any user overrides.
SYNONYM_MAP: Final[dict[str, list[str]]] = {**_CORE, **_load_custom()}


def get_synonyms(token: str) -> list[str]:
    """Return the synonym list for *token*, or an empty list.

    Lookup is case-insensitive.  Returns a copy so callers can safely
    mutate the result list.
    """
    return list(SYNONYM_MAP.get(token.lower(), []))


def add_synonym(word: str, synonyms: list[str]) -> None:
    """Add or overwrite a synonym entry at runtime (testing convenience)."""
    SYNONYM_MAP[word.lower()] = [s.lower() for s in synonyms]
