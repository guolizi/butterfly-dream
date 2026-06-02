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

logger = logging.getLogger(__name__)

# Default scenario weights
SCENARIO_WEIGHTS = {
    "chat":      {"relevance": 0.4, "recency": 0.4, "importance": 0.2},
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
        candidates = self._fts_candidates(query, category, min_trust, limit * 3, persistent_only)
        if not candidates:
            return []

        # Stage 2: Score on all three dimensions
        query_tokens = self._tokenize(query)
        scored = []

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
        return scored[:limit]

    # -- Internal pipeline helpers --------------------------------------------

    def _fts_candidates(
        self,
        query: str,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 30,
        persistent_only: bool = False,
    ) -> list[dict]:
        """Stage 1: Fetch candidates from FTS5 full-text search.

        Searches both facts_fts and media_attachments_fts in parallel,
        then merges results by fact_id. Media matches bring in their
        parent fact and include a 'media' list and '_media_match' flag.
        """
        # Sanitize query for FTS5 special characters
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        # Base WHERE clause
        trust_clause = "f.trust_score >= ?"
        persistent_clause = "AND f.is_persistent = 1" if persistent_only else ""

        # Query facts_fts
        if category:
            rows = self.store.execute_query(
                f"""SELECT f.*, rank FROM facts_fts
                   JOIN facts f ON facts_fts.rowid = f.fact_id
                   WHERE facts_fts MATCH ? AND f.category = ? AND {trust_clause} {persistent_clause}
                   ORDER BY rank LIMIT ?""",
                (safe_query, category, min_trust, limit),
            )
        else:
            rows = self.store.execute_query(
                f"""SELECT f.*, rank FROM facts_fts
                   JOIN facts f ON facts_fts.rowid = f.fact_id
                   WHERE facts_fts MATCH ? AND {trust_clause} {persistent_clause}
                   ORDER BY rank LIMIT ?""",
                (safe_query, min_trust, limit),
            )

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
                persistent_filter = "AND is_persistent = 1" if persistent_only else ""
                fact_rows = self.store.execute_query(
                    f"SELECT * FROM facts WHERE fact_id=? AND trust_score>=? {persistent_filter}",
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

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Remove FTS5 special characters and collapse whitespace."""
        import re
        # Remove characters that could break FTS5 syntax
        # Keep # (common in C#/F#) — it's safe in FTS5; + is kept for C++/F++
        safe = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef#+]', ' ', query)
        # Collapse whitespace
        safe = ' '.join(safe.split())
        if len(safe) < 2:
            return ""
        # Convert to AND query for multi-word
        return safe

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
    """Tokenize text into a set of normalized tokens."""
    import re
    tokens = set()
    for token in re.findall(r'[a-zA-Z][a-zA-Z0-9_\-+#]{1,}', text):
        tokens.add(token.lower())
    cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i] + cjk_chars[i + 1])
    for char in cjk_chars:
        tokens.add(char)
    return tokens


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)
