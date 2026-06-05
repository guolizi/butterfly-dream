"""Three-dimensional fact retriever for Butterfly Dream.

Combines three scoring dimensions:
  - Relevance:    How semantically related is this fact to the query?
  - Recency:      How recently was this fact created/updated?
  - Importance:   How intrinsically important is this fact?

Final score = (α × relevance + β × recency + γ × importance) × trust
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

import jieba

logger = logging.getLogger(__name__)

# Default scenario weights
SCENARIO_WEIGHTS = {
    "chat":      {"relevance": 0.5, "recency": 0.3, "importance": 0.2},
    "technical": {"relevance": 0.5, "recency": 0.2, "importance": 0.3},
    "longterm":  {"relevance": 0.3, "recency": 0.1, "importance": 0.6},
    "qa":        {"relevance": 0.6, "recency": 0.3, "importance": 0.1},
    "balanced":  {"relevance": 0.4, "recency": 0.3, "importance": 0.3},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse SQLite datetime string to datetime object."""
    if not dt_str:
        return None
    try:
        # Handle 'YYYY-MM-DD HH:MM:SS' format (SQLite default, UTC)
        if "+" in dt_str or "Z" in dt_str or "T" in dt_str:
            # ISO 8601 with timezone
            from datetime import timezone as tz
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        else:
            # SQLite default: no timezone → assume UTC
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _recency_score(dt: Optional[datetime], half_life_days: float = 30.0) -> float:
    """Exponential decay: 0.5^(age / half_life). Returns 1.0 for now, 0.5 at half_life, ~0 at ∞."""
    if dt is None:
        return 0.5  # neutral for unknown timestamps
    age = (_now() - dt).total_seconds() / 86400.0  # age in days
    if age < 0:
        age = 0.0
    if half_life_days <= 0:
        return 1.0  # no decay
    return 0.5 ** (age / half_life_days)


def _importance_score(importance: Optional[float]) -> float:
    """Normalize importance (1-10) to [0, 1]."""
    if importance is None:
        return 0.5
    return max(0.0, min(1.0, (importance - 1.0) / 9.0))


class ThreeDimRetriever:
    """Multi-strategy fact retrieval with three-dimensional scoring.

    Pipeline:
    1. FTS5 search to get candidate pool (limit × 3)
    2. Score each candidate on three dimensions
    3. Weighted combine → sort → return top N
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        half_life_days: float = 30.0,
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        custom_weights: dict | None = None,
    ):
        self.store = store
        self.half_life_days = half_life_days
        self.hrr_dim = hrr_dim
        self._custom_weights = custom_weights

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 10,
        scenario: str = "balanced",
        recency_weight: Optional[float] = None,
        relevance_weight: Optional[float] = None,
        importance_weight: Optional[float] = None,
        persistent_only: bool = False,
        fts_mode: str = "or",
        mmr_lambda: Optional[float] = None,
    ) -> list[dict]:
        """Three-dimensional search: relevance × recency × importance × trust.

        Args:
            query: Search query string.
            category: Optional category filter.
            min_trust: Minimum trust score threshold.
            limit: Max results to return.
            scenario: Weight preset ("chat", "technical", "longterm", "qa", "balanced").
            persistent_only: If True, only return facts marked as persistent.
            recency_weight: Override recency weight for this call.
            relevance_weight: Override relevance weight for this call.
            importance_weight: Override importance weight for this call.
            mmr_lambda: MMR diversity parameter (0.0–1.0). Higher = more relevance,
                lower = more diversity. Default 0.7 when MMR is enabled. Set to 1.0
                to disable MMR (pure relevance ranking). MMR is automatically applied
                when the candidate pool contains near-duplicate facts (Jaccard > 0.6).

        Returns:
            List of fact dicts with 'score' field, sorted descending.
        """
        # Resolve weights — merge scenario presets with instance custom weights
        base = SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["balanced"])
        if scenario == "custom" and self._custom_weights:
            base = self._custom_weights
        weights = base.copy()
        if relevance_weight is not None:
            weights["relevance"] = relevance_weight
        if recency_weight is not None:
            weights["recency"] = recency_weight
        if importance_weight is not None:
            weights["importance"] = importance_weight

        # Stage 1: Get FTS5 candidates
        candidates = self._fts_candidates(query, category, min_trust, limit * 3, persistent_only, fts_mode=fts_mode)

        # Stage 1.5: Also fetch candidates by semantic category if query matches
        semantic_cats = self._query_to_semantic_categories(query)
        if semantic_cats:
            cat_candidates = self._semantic_category_candidates(
                semantic_cats, category, min_trust, limit * 3, persistent_only
            )
            # Merge: add category candidates not already in FTS5 results
            seen_ids = {c.get("fact_id") for c in candidates}
            for c in cat_candidates:
                if c.get("fact_id") not in seen_ids:
                    c["fts_rank"] = 0.0  # no FTS rank, will rely on other dimensions
                    candidates.append(c)
                    seen_ids.add(c["fact_id"])

        if not candidates:
            return []

        # Stage 2: Score on all three dimensions
        query_tokens = self._tokenize(query)
        scored = []

        # Semantic category boost: facts matching detected categories get a relevance bump
        _CAT_BOOST = 0.25  # boost for matching semantic category

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            # --- Relevance ---
            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                try:
                    fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                    hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
                except Exception:
                    hrr_sim = 0.5
            else:
                hrr_sim = 0.5

            relevance = (
                self.fts_weight * fts_score
                + self.jaccard_weight * jaccard
                + self.hrr_weight * hrr_sim
            )

            # Boost relevance if fact's category matches query intent
            if semantic_cats and fact.get("category") in semantic_cats:
                relevance = min(1.0, relevance + _CAT_BOOST)

            # --- Recency ---
            created = _parse_datetime(fact.get("created_at"))
            recency = _recency_score(created, self.half_life_days)

            # --- Importance ---
            importance_raw = fact.get("importance")
            importance = _importance_score(importance_raw)

            # --- Combine: three-dimensional score × trust ---
            score = (
                weights["relevance"] * relevance
                + weights["recency"] * recency
                + weights["importance"] * importance
            ) * fact["trust_score"]

            scored.append({
                **fact,
                "_relevance": round(relevance, 4),
                "_recency": round(recency, 4),
                "_importance": round(importance, 4),
                "score": round(score, 4),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Stage 3: Dedup near-duplicate facts via clustering
        scored = self._dedup(scored, limit)

        return scored[:limit]

    # -- Internal pipeline helpers --------------------------------------------

    def _dedup(self, scored: list[dict], limit: int) -> list[dict]:
        """Remove near-duplicate facts to improve diversity.

        Groups facts whose Jaccard similarity exceeds 0.4, keeps only the
        highest-scoring fact per group, then backfills from remaining
        candidates until `limit` is reached. This prevents 6 near-identical
        "counseling career" facts from crowding out a more specific
        "work with trans people" fact.
        """
        if len(scored) <= limit:
            return scored

        _DEDUP_THRESHOLD = 0.4

        # Pre-tokenize once
        token_cache = [self._tokenize(s["content"]) for s in scored]

        selected: list[dict] = []
        selected_indices: set[int] = set()

        for i, candidate in enumerate(scored):
            if i in selected_indices:
                continue

            # Check if this candidate is near-duplicate of any already-selected
            cand_tokens = token_cache[i]
            is_dup = False
            for sel_idx in selected_indices:
                if self._jaccard_similarity(cand_tokens, token_cache[sel_idx]) > _DEDUP_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                selected.append(candidate)
                selected_indices.add(i)

            if len(selected) >= limit:
                break

        return selected

    def _fact_similarity(self, a: dict, b: dict) -> float:
        """Similarity between two facts (Jaccard on content tokens)."""
        tokens_a = self._tokenize(a["content"])
        tokens_b = self._tokenize(b["content"])
        return self._jaccard_similarity(tokens_a, tokens_b)

    def _fts_candidates(
        self,
        query: str,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 30,
        persistent_only: bool = False,
        fts_mode: str = "or",
    ) -> list[dict]:
        """Stage 1: Fetch candidates from FTS5 full-text search.

        Searches both facts_fts and media_attachments_fts in parallel,
        then merges results by fact_id. Media matches bring in their
        parent fact and include a 'media' list and '_media_match' flag.
        """
        # Sanitize query for FTS5 special characters
        safe_query = self._sanitize_fts_query(query, fts_mode)
        if not safe_query:
            return []

        # Query facts_fts — use separate parameterized queries for safety
        try:
            if category:
                if persistent_only:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.category = ? AND f.trust_score >= ? AND f.is_persistent = 1
                           ORDER BY rank LIMIT ?""",
                        (safe_query, category, min_trust, limit),
                    )
                else:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.category = ? AND f.trust_score >= ?
                           ORDER BY rank LIMIT ?""",
                        (safe_query, category, min_trust, limit),
                    )
            else:
                if persistent_only:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.trust_score >= ? AND f.is_persistent = 1
                           ORDER BY rank LIMIT ?""",
                        (safe_query, min_trust, limit),
                    )
                else:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.trust_score >= ?
                           ORDER BY rank LIMIT ?""",
                        (safe_query, min_trust, limit),
                    )
        except Exception:
            rows = []

        results = []
        seen_fact_ids = {}
        for row in rows:
            d = {key: row[key] for key in row.keys()}
            d["fts_rank"] = 1.0 / (1.0 + math.exp(d.get("rank", 0) or 0))
            d["media"] = []
            d["_media_match"] = False
            results.append(d)
            seen_fact_ids[d["fact_id"]] = d

        # Also search media_attachments_fts
        try:
            media_rows = self.store.execute_query(
                """SELECT m.rowid AS media_id, m.rank, ma.*
                   FROM media_attachments_fts m
                   JOIN media_attachments ma ON m.rowid = ma.media_id
                   WHERE media_attachments_fts MATCH ?
                   ORDER BY m.rank LIMIT ?""",
                (safe_query, limit),
            )
        except Exception:
            media_rows = []  # table might not exist in old DBs

        for row in media_rows:
            media = {key: row[key] for key in row.keys()}
            media_rank = media.pop("rank", 0)  # pop rank before ma.* columns shadow it
            media_fts_score = 1.0 / (1.0 + math.exp(float(media_rank or 0)))
            fid = media["fact_id"]

            if fid in seen_fact_ids:
                # Append media to existing fact result
                existing = seen_fact_ids[fid]
                existing["media"].append(media)
                existing["_media_match"] = True
                # Boost relevance from media match
                existing["fts_rank"] = max(existing["fts_rank"], media_fts_score)
            else:
                # Fetch the parent fact and add it with media
                if persistent_only:
                    fact_rows = self.store.execute_query(
                        "SELECT * FROM facts WHERE fact_id=? AND trust_score>=? AND is_persistent = 1",
                        (fid, min_trust),
                    )
                else:
                    fact_rows = self.store.execute_query(
                        "SELECT * FROM facts WHERE fact_id=? AND trust_score>=?",
                        (fid, min_trust),
                    )
                if fact_rows:
                    fact_row = fact_rows[0]
                    fact_dict = {key: fact_row[key] for key in fact_row.keys()}
                    fact_dict["fts_rank"] = media_fts_score
                    fact_dict["media"] = [media]
                    fact_dict["_media_match"] = True
                    results.append(fact_dict)
                    seen_fact_ids[fid] = fact_dict

        # Sort by fts_rank descending (best match first)
        results.sort(key=lambda x: x["fts_rank"], reverse=True)
        return results[:limit]

    # -- Semantic category helpers --------------------------------------------

    @staticmethod
    def _query_to_semantic_categories(query: str) -> list[str]:
        """Map query keywords to semantic categories. Supports English and Chinese."""
        q = query.lower()
        categories = []

        # English mappings
        EN_MAP = [
            (["where", "location", "place", "which city", "which country", "which town"], "place"),
            (["when", "what time", "what date", "how long", "how many years", "how many days",
              "how old", "since when", "which year", "which month", "which day"], "time"),
            (["who", "whom", "whose", "which person"], "person"),
            (["what happened", "what did", "what event", "what events", "what was the"], "event"),
            (["what activities", "what activity", "what hobbies", "what hobby",
              "what sports", "what sport", "what pastime", "what pastimes",
              "what does", "what do", "how do you", "how often"], "activity"),
            (["what is", "what are", "how would", "describe"], "identity"),
            (["like", "favorite", "prefer", "enjoy", "love", "hate", "dislike",
              "taste", "interest"], "preference"),
            (["want", "plan", "goal", "wish", "hope", "aspire", "intend",
              "going to", "will do"], "goal"),
            (["what project", "what projects", "working on", "building",
              "what tech", "what stack", "what framework"], "project"),
            (["what tool", "what tools", "what software", "what app", "what apps",
              "what program", "what programs", "what do you use", "how do you use"], "tool"),
            (["do you have", "what do you own", "what do you have", "any pets",
              "any cars", "any property"], "possession"),
            (["how are you", "how do you feel", "what is your status",
              "what is your state", "how is it going"], "state"),
        ]
        # Chinese mappings
        ZH_MAP = [
            (["在哪", "哪里", "什么地方", "何处", "哪个城市", "哪个国家", "来源", "来自",
              "住在", "家在", "搬去", "搬到", "去哪"], "place"),
            (["什么时候", "几月", "哪天", "多久", "多长时间", "几年", "何时", "哪一年", "多大",
              "几号", "哪天", "几点", "多晚"], "time"),
            (["谁", "哪个人", "什么人", "认识", "朋友", "家人", "同事", "邻居", "亲戚"], "person"),
            (["发生了什么", "什么事", "什么事件", "什么情况", "怎么了", "出什么事"], "event"),
            (["做什么", "干什么", "什么活动", "什么爱好", "什么运动", "怎么锻炼", "平时做什么",
              "经常", "每天", "每周", "总是", "习惯", "一般"], "activity"),
            (["是什么", "什么样的", "什么身份", "什么职业", "做什么工作", "工作是"], "identity"),
            (["喜欢", "爱好", "偏好", "爱", "讨厌", "不喜欢", "兴趣", "最爱", "最讨厌",
              "宁愿", "倾向", "更喜欢"], "preference"),
            (["想", "计划", "目标", "打算", "希望", "愿望", "想要", "准备", "将来"], "goal"),
            (["什么项目", "在做什么", "什么技术", "什么框架", "用什么开发", "开发什么",
              "做什么项目", "技术栈"], "project"),
            (["什么工具", "什么软件", "用什么", "什么程序", "什么app", "什么应用",
              "用什么软件", "用什么工具"], "tool"),
            (["有没有", "拥有", "有什么", "养了什么", "名下", "养了", "有只", "有只猫",
              "有只狗", "有辆车", "有套房"], "possession"),
            (["最近怎么样", "状态如何", "什么状态", "什么情况", "还好吗", "怎么样"], "state"),
        ]

        for keywords, cat in EN_MAP + ZH_MAP:
            for kw in keywords:
                if kw in q:
                    categories.append(cat)
                    break

        # Deduplicate while preserving order
        return list(dict.fromkeys(categories))

    def _semantic_category_candidates(
        self,
        semantic_cats: list[str],
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 30,
        persistent_only: bool = False,
    ) -> list[dict]:
        """Fetch facts by category (semantic classification)."""
        placeholders = ",".join("?" for _ in semantic_cats)
        conditions = f"f.category IN ({placeholders}) AND f.trust_score >= ?"
        params: list = list(semantic_cats) + [min_trust]

        if persistent_only:
            conditions += " AND f.is_persistent = 1"

        params.append(limit)
        try:
            rows = self.store.execute_query(
                f"SELECT * FROM facts f WHERE {conditions} ORDER BY f.importance DESC, f.created_at DESC LIMIT ?",
                tuple(params),
            )
        except Exception:
            return []

        results = []
        for row in rows:
            d = {key: row[key] for key in row.keys()}
            d["fts_rank"] = 0.0
            d["media"] = []
            d["_media_match"] = False
            results.append(d)
        return results

    @staticmethod
    def _sanitize_fts_query(query: str, fts_mode: str = "or") -> str:
        """Remove FTS5 special characters, lemmatize English, filter stop words, and collapse whitespace."""
        import re
        # Remove characters that could break FTS5 syntax
        # Keep # (common in C#/F#) — it's safe in FTS5; + is kept for C++/F++
        safe = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef#+]', ' ', query)
        # Jieba-segment CJK text to match FTS5 indexing
        tokens = []
        for word in safe.split():
            if re.search(r'[\u4e00-\u9fff]', word):
                tokens.extend(jieba.cut(word))
            else:
                tokens.append(word)
        # Lemmatize English tokens (reduce verb/noun forms to base form)
        try:
            from nltk.stem import WordNetLemmatizer
            wnl = WordNetLemmatizer()
            tokens = [
                wnl.lemmatize(t, 'v') if t.isascii() and t.isalpha() and len(t) > 2 else t
                for t in tokens
            ]
        except ImportError:
            pass  # NLTK not installed, skip lemmatization
        safe = ' '.join(tokens)
        # Collapse whitespace
        safe = ' '.join(safe.split())
        if len(safe) < 2:
            return ""
        # Filter stop words — these match almost everything in OR mode
        # and add noise without improving relevance
        STOP_WORDS = {
            # English
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must',
            'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'theirs',
            'this', 'that', 'these', 'those', 'here', 'there',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'into', 'about', 'between', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'not', 'so', 'if', 'then', 'than', 'too',
            'very', 'also', 'some', 'any', 'all',
            'no', 'only', 'own', 'same', 'other', 'such',
            'further', 'once', 'again', 'further', 'even', 'still',
            # Chinese
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '被',
            '从', '把', '些', '所', '过', '对', '里', '为', '与', '及', '等',
        }
        tokens_clean = [t for t in safe.split() if t.lower() not in STOP_WORDS]
        if not tokens_clean:
            # Fallback: if all tokens were stop words, keep original
            tokens_clean = safe.split()
        # Use '*' prefix matching to bridge jieba segmentation gaps.
        # jieba may segment the same text differently in queries vs indexed content
        # (e.g. "橘猫" is one token in index but "橘" + "猫叫" in query "橘猫叫什么名字").
        # Prefix matching ensures partial jieba tokens still produce candidates.
        op = ' AND ' if fts_mode == 'and' else ' OR '
        if len(tokens_clean) > 1:
            return op.join(t + '*' for t in tokens_clean)
        # Single token: use prefix match only if CJK (jieba may over-split)
        if re.search(r'[\u4e00-\u9fff]', safe):
            return tokens_clean[0] + '*' if tokens_clean else ''
        return tokens_clean[0] if tokens_clean else ''

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of normalized tokens."""
        return tokenize(text)

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        return jaccard_similarity(a, b)


# Module-level helpers (also usable by store.py)
def tokenize(text: str) -> set[str]:
    """Tokenize text into a set of normalized tokens.

    Uses jieba for CJK word segmentation and regex for English tokens,
    producing semantically meaningful tokens for both languages.
    This powers Jaccard similarity in dedup, merge, and retrieval.
    """
    import re
    tokens = set()
    # English / Latin words
    for token in re.findall(r'[a-zA-Z][a-zA-Z0-9_\-+#]{1,}', text):
        tokens.add(token.lower())
    # CJK text — jieba word segmentation (more accurate than bigrams)
    cjk_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in cjk_parts:
        tokens.update(jieba.cut(part))
    return tokens


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)
