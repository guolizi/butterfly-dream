"""SQLite-backed fact store with three-dimensional scoring metadata.

Extends the Holographic store with:
- Importance scoring (1-10, LLM-assigned + user-adjustable)
- Recency tracking (automatic timestamps with configurable decay)
- Entity relationship graph for multi-hop reasoning
- Fact merging / conflict resolution for same-entity facts
"""

import logging
import re
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    importance      REAL DEFAULT 5.0,          -- 1.0 ~ 10.0, LLM-assigned
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation    TEXT DEFAULT 'related_to',
    weight      REAL DEFAULT 1.0,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created   ON facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category  ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name   ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""

# Trust adjustment constants
_HELPFUL_DELTA   = 0.05
_UNHELPFUL_DELTA = -0.10
_IMPORTANCE_DELTA = 0.5  # Per feedback on importance
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0
_IMPORTANCE_MIN = 1.0
_IMPORTANCE_MAX = 10.0

# Entity extraction patterns (from Holographic, enhanced for CJK)
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)
_RE_CJK_BRACKETS = re.compile(
    r"[「『]([^」』]+)[」』]|《([^》]+)》|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019"
)
_RE_QUOTED_CN = re.compile(r'["\u201c\u2018]([^"\u201d\u2019]{2,})["\u201d\u2019]')


class MemoryStore:
    """Thread-safe SQLite-backed fact store with importance + trust tracking."""

    def __init__(self, db_path: str, default_trust: float = 0.5, hrr_dim: int = 1024):
        self._db_path = db_path
        self._default_trust = default_trust
        self._hrr_dim = hrr_dim
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- CRUD ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        importance: float = 5.0,
        entities: Optional[list[str]] = None,
    ) -> dict:
        """Store a fact with importance scoring. Merges with existing if duplicate content."""
        with self._lock:
            # Check for duplicate by content
            existing = self._conn.execute(
                "SELECT fact_id, importance, trust_score FROM facts WHERE content = ?",
                (content,),
            ).fetchone()

            if existing:
                # Merge: keep higher importance, keep higher trust
                fact_id, old_imp, old_trust = existing
                new_importance = max(old_imp, importance)
                new_trust = max(old_trust, self._default_trust)
                self._conn.execute(
                    "UPDATE facts SET importance=?, trust_score=?, updated_at=datetime('now') WHERE fact_id=?",
                    (new_importance, new_trust, fact_id),
                )
                logger.debug("Merged duplicate fact #%d (importance %.1f)", fact_id, new_importance)
            else:
                # Encode HRR vector
                hrr_vector = self._encode_hrr(content, entities or [])
                hrr_blob = hrr.phases_to_bytes(hrr_vector) if hrr_vector is not None else None

                self._conn.execute(
                    """INSERT INTO facts (content, category, tags, importance, trust_score, hrr_vector)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (content, category, tags, importance, self._default_trust, hrr_blob),
                )
                fact_id = self._conn.lastrowid

            # Process entities
            extracted = self._extract_entities(content)
            if entities:
                extracted.extend(entities)
            if extracted:
                self._link_entities(fact_id, extracted)

            self._conn.commit()
            return {"fact_id": fact_id, "content": content, "importance": importance}

    def get_fact(self, fact_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def update_fact(self, fact_id: int, **kwargs) -> bool:
        allowed = {"content", "category", "tags", "importance", "trust_score"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [fact_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE facts SET {set_clause} WHERE fact_id=?", values
            )
            self._conn.commit()
            return True

    def remove_fact(self, fact_id: int) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            return True

    def list_facts(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_facts(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # -- Feedback --------------------------------------------------------------

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT trust_score, importance, helpful_count, retrieval_count FROM facts WHERE fact_id=?",
                (fact_id,),
            ).fetchone()
            if not row:
                return {"error": "fact not found"}
            trust, importance, helpful_count, retrieval_count = row
            if helpful:
                trust = min(_TRUST_MAX, trust + _HELPFUL_DELTA)
                importance = min(_IMPORTANCE_MAX, importance + _IMPORTANCE_DELTA)
                helpful_count += 1
            else:
                trust = max(_TRUST_MIN, trust + _UNHELPFUL_DELTA)
                importance = max(_IMPORTANCE_MIN, importance - _IMPORTANCE_DELTA)
            self._conn.execute(
                """UPDATE facts SET trust_score=?, importance=?, helpful_count=?,
                   retrieval_count=?, updated_at=datetime('now') WHERE fact_id=?""",
                (trust, importance, helpful_count, retrieval_count, fact_id),
            )
            self._conn.commit()
            return {"fact_id": fact_id, "trust_score": trust, "importance": importance}

    # -- Entity management -----------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text."""
        entities = set()
        for pattern in (_RE_CAPITALIZED, _RE_DOUBLE_QUOTE, _RE_SINGLE_QUOTE,
                        _RE_CJK_BRACKETS, _RE_QUOTED_CN):
            for match in pattern.finditer(text):
                for group in match.groups():
                    if group and len(group) > 1:
                        entities.add(group.strip())
        # AKA patterns
        for match in _RE_AKA.finditer(text):
            entities.add(match.group(1).strip())
            entities.add(match.group(2).strip())
        return [e for e in entities if len(e) >= 2]

    def _link_entities(self, fact_id: int, entity_names: list[str]) -> None:
        """Associate entities with a fact, creating them if needed."""
        for name in entity_names:
            # Upsert entity
            self._conn.execute(
                "INSERT INTO entities (name) VALUES (?) ON CONFLICT(name) DO NOTHING",
                (name,),
            )
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if row:
                # Link fact ↔ entity
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, row[0]),
                )

    def get_entity_facts(self, entity_name: str, limit: int = 20) -> list[dict]:
        """Get all facts linked to an entity."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT f.* FROM facts f
                   JOIN fact_entities fe ON f.fact_id = fe.fact_id
                   JOIN entities e ON fe.entity_id = e.entity_id
                   WHERE e.name = ? OR e.aliases LIKE ?
                   ORDER BY f.importance DESC, f.trust_score DESC
                   LIMIT ?""",
                (entity_name, f"%{entity_name}%", limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        """BFS traversal for related entities up to `depth` hops."""
        with self._lock:
            visited = set()
            queue = [(entity_name, 0)]
            results = []
            while queue:
                name, d = queue.pop(0)
                if name in visited or d > depth:
                    continue
                visited.add(name)
                rows = self._conn.execute(
                    """SELECT DISTINCT e2.name FROM entities e1
                       JOIN fact_entities fe1 ON e1.entity_id = fe1.entity_id
                       JOIN fact_entities fe2 ON fe1.fact_id = fe2.fact_id AND fe2.entity_id != fe1.entity_id
                       JOIN entities e2 ON fe2.entity_id = e2.entity_id
                       WHERE e1.name = ?""",
                    (name,),
                ).fetchall()
                for (related_name,) in rows:
                    if related_name not in visited:
                        results.append({"source": name, "target": related_name, "depth": d + 1})
                        queue.append((related_name, d + 1))
            return results

    # -- HRR encoding ----------------------------------------------------------

    def _encode_hrr(self, content: str, entities: list[str] = None) -> Optional["np.ndarray"]:
        """Encode fact content + entities into HRR phase vector."""
        if not hrr._HAS_NUMPY:
            return None
        try:
            if entities:
                return hrr.encode_fact(content, entities, self._hrr_dim)
            return hrr.encode_text(content, self._hrr_dim)
        except Exception:
            return None

    def compute_hrr_similarity(self, fact_id: int, query: str) -> float:
        """HRR similarity between a stored fact and a query string."""
        if not hrr._HAS_NUMPY:
            return 0.5
        with self._lock:
            row = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE fact_id = ? AND hrr_vector IS NOT NULL",
                (fact_id,),
            ).fetchone()
            if not row:
                return 0.5
            try:
                fact_vec = hrr.bytes_to_phases(row[0])
                query_vec = hrr.encode_text(query, self._hrr_dim)
                return (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
            except Exception:
                return 0.5

    # -- Close ----------------------------------------------------------------

    def close(self):
        self._conn.close()
