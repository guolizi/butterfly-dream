"""🦋 Butterfly Dream — 3-Dimensional Memory Plugin for Hermes Agent.

A MemoryProvider plugin that scores facts across three dimensions:
  Relevance (semantic), Recency (temporal decay), Importance (LLM-assigned).

庄周梦蝶 — 记忆如蝶，翩跹于时间、意义与关联的三维空间。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from hermes_cli.config import cfg_get

from .store import MemoryStore
from .retrieval import ThreeDimRetriever, SCENARIO_WEIGHTS

logger = logging.getLogger(__name__)

# Known provider base URLs (can be overridden via {PROVIDER}_BASE_URL env)
_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1",
    "minimax": "https://api.minimax.chat/v1",
    "ollama": "http://localhost:11434/v1",
}


# ---------------------------------------------------------------------------
# LLM extraction prompt (enhanced with importance scoring)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant for an AI agent. Analyze conversation turns and extract facts worth remembering — with importance scores.

Extract facts about:
1. User preferences, habits, and personal information
2. Project decisions, architecture choices, and technical rationale
3. Tool configurations, setup steps, and environment details
4. Key conventions and agreements made during the conversation
5. Any other information that would be useful to remember across sessions

Rules:
- Only extract concrete, specific facts. Skip small talk and greetings.
- Prefer concise, self-contained statements.
- If nothing worth extracting, return an empty array.
- Deduplicate: don't extract the same fact multiple times.

**Importance scoring (1-10):**
- 9-10: Critical identity/security info, core project architecture, irreversible decisions
- 7-8: Important preferences, key technical choices, significant constraints
- 5-6: Useful context, typical settings, common patterns
- 3-4: Minor preferences, temporary states, easily rediscoverable info
- 1-2: Trivial details, likely to change, not worth remembering long-term

Return a JSON array of objects, each with:
- "content": the fact statement (plain text, max 400 chars)
- "category": one of "user_pref", "project", "tool", "general"
- "tags": optional comma-separated tags
- "importance": integer 1-10 (how important is this fact to remember?)

Example:
[
  {"content": "User prefers VS Code for Python development with black formatter", "category": "user_pref", "tags": "editor,python", "importance": 6},
  {"content": "Project uses FastAPI with SQLAlchemy async session pattern", "category": "project", "tags": "backend,stack", "importance": 8},
  {"content": "User mentioned they like matcha lattes", "category": "general", "tags": "preference", "importance": 3}
]"""


# ---------------------------------------------------------------------------
# Tool schemas (extended with importance & scenario support)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Three-dimensional memory with algebraic reasoning. "
        "Use for deep recall across relevance, recency, and importance.\\n\\n"
        "ACTIONS:\\n"
        "• add — Store a fact the user would expect you to remember.\\n"
        "• search — 3D keyword/semantic search ('editor config', 'deploy process').\\n"
        "  Pass scenario='chat'|'technical'|'longterm'|'qa' to tune weights.\\n"
        "• probe — Entity recall: ALL facts about a person/thing.\\n"
        "• related — What connects to an entity? Structural adjacency.\\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\\n"
        "• update/remove/list — CRUD operations.\\n\\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {
                "type": "array", "items": {"type": "string"},
                "description": "Entity names for 'reason'.",
            },
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "importance": {
                "type": "integer", "description": "Importance 1-10 (used for 'add').",
            },
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
            "scenario": {
                "type": "string",
                "enum": ["chat", "technical", "longterm", "qa", "balanced"],
                "description": "Retrieval weight scenario (default: 'balanced').",
            },
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains both trust and importance — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Load butterfly-dream config from config.yaml."""
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "butterfly-dream", default={}) or {}
    except Exception:
        return {}


def _resolve_provider_credentials(provider: str) -> tuple[str, str]:
    """Resolve (base_url, api_key) for a given provider name."""
    prefix = provider.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    base_url = os.environ.get(f"{prefix}_BASE_URL", _DEFAULT_BASE_URLS.get(provider, ""))
    return base_url.rstrip("/"), api_key


def _call_extraction_llm(
    messages_text: str,
    provider: str,
    model: str,
    timeout: int = 30,
) -> list[dict]:
    """Call the extraction LLM and return parsed fact objects with importance.

    Returns list of {"content", "category", "tags", "importance"}.
    Returns empty list on any error (fail-safe).
    """
    base_url, api_key = _resolve_provider_credentials(provider)
    if not api_key:
        logger.warning("ButterflyDream LLM extract: no API key for '%s'", provider)
        return []
    if not base_url:
        logger.warning("ButterflyDream LLM extract: no base URL for '%s'", provider)
        return []
    if not model:
        logger.warning("ButterflyDream LLM extract: no model specified")
        return []

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract facts from these conversation turns:\n\n{messages_text}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        logger.warning("ButterflyDream LLM extract request failed: %s", e)
        return []

    try:
        content = response_data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("ButterflyDream LLM extract: parse failed: %s", e)
        return []

    if isinstance(parsed, dict):
        for key in ("facts", "memories", "extractions", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        logger.warning("ButterflyDream LLM extract: unexpected format: %s", type(parsed).__name__)
        return []

    # Validate and normalize
    facts = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if len(content) < 10:
            continue
        content = content[:400]
        category = str(item.get("category", "general")).strip()
        if category not in ("user_pref", "project", "tool", "general"):
            category = "general"
        tags = str(item.get("tags", "")).strip()
        importance = int(item.get("importance", 5))
        importance = max(1, min(10, importance))
        facts.append({
            "content": content,
            "category": category,
            "tags": tags,
            "importance": importance,
        })

    return facts


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ButterflyDreamMemoryProvider(MemoryProvider):
    """Three-dimensional memory: Relevance × Recency × Importance.

    Builds on Holographic's fact store and HRR vector encoding, adding:
    - LLM-assigned importance scoring during extraction
    - Exponential recency decay with configurable half-life
    - Scenario-aware weight presets for retrieval
    - Entity relationship graph for multi-hop reasoning
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: Optional[MemoryStore] = None
        self._retriever: Optional[ThreeDimRetriever] = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

        # LLM extraction config
        llm_cfg = self._config.get("extraction_model", {})
        self._extraction_provider = str(llm_cfg.get("provider", "deepseek"))
        self._extraction_model = str(llm_cfg.get("model", "deepseek-v4-flash"))

        # Extraction state
        self._llm_extract_enabled = self._config.get("llm_extract", False)
        self._last_extracted_idx = 0

    @property
    def name(self) -> str:
        return "butterfly-dream"

    def is_available(self) -> bool:
        return True

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.butterfly-dream."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["butterfly-dream"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/butterfly_memory.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "llm_extract", "description": "LLM-based fact extraction with importance scoring", "default": "true", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "min_trust_threshold", "description": "Minimum trust threshold for retrieval", "default": "0.3"},
            {"key": "recency_half_life_days", "description": "Days for recency score to decay by half", "default": "30"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/butterfly_memory.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)

        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        half_life = float(self._config.get("recency_half_life_days", 30.0))

        # Retrieval weights from config
        ret_cfg = self._config.get("retrieval", {})
        rel_w = float(ret_cfg.get("relevance_weight", 0.4))
        rec_w = float(ret_cfg.get("recency_weight", 0.3))
        imp_w = float(ret_cfg.get("importance_weight", 0.3))

        # Normalize weights to sum to 1.0
        total = rel_w + rec_w + imp_w
        if total > 0:
            rel_w /= total
            rec_w /= total
            imp_w /= total
        SCENARIO_WEIGHTS["custom"] = {
            "relevance": rel_w,
            "recency": rec_w,
            "importance": imp_w,
        }

        self._store = MemoryStore(
            db_path=db_path,
            default_trust=default_trust,
            hrr_dim=hrr_dim,
        )
        self._retriever = ThreeDimRetriever(
            store=self._store,
            half_life_days=half_life,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id
        self._last_extracted_idx = 0

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store.count_facts()
        except Exception:
            total = 0
        if total == 0:
            return (
                "# 🦋 Butterfly Dream Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable facts with three-dimensional scoring.\n"
                "Use fact_feedback to rate facts after using them (trains trust + importance)."
            )
        return (
            f"# 🦋 Butterfly Dream Memory\n"
            f"Active. {total} facts stored with 3D scoring (Relevance × Recency × Importance).\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Prefetch relevant facts before agent processes a user message."""
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(
                query, min_trust=self._min_trust, limit=5, scenario="balanced",
            )
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                imp = r.get("importance", 5)
                lines.append(f"- [{trust:.1f} trust | {imp:.0f} imp] {r.get('content', '')}")
            return "## 🦋 Butterfly Dream Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("ButterflyDream prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: list | None = None) -> None:
        """No-op: extraction is async (pre-compress and session-end only)."""
        pass

    def on_pre_compress(self, messages: list) -> str:
        """Extract facts before context compression discards messages — includes importance scoring."""
        if not self._llm_extract_enabled or not self._store or not messages:
            return ""
        msgs_copy = list(messages)

        def _extract_async():
            try:
                facts = self._run_llm_extraction(msgs_copy)
                if facts:
                    logger.info("ButterflyDream pre-compress extracted %d facts", len(facts))
            except Exception as e:
                logger.debug("ButterflyDream pre-compress extraction failed: %s", e)

        t = threading.Thread(target=_extract_async, daemon=True, name="butterfly-compress")
        t.start()
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Final extraction at session end with importance scoring."""
        if not self._store or not messages:
            return
        if self._llm_extract_enabled:
            new_msgs = messages[self._last_extracted_idx:]
            if new_msgs:
                facts = self._run_llm_extraction(new_msgs)
                if facts:
                    logger.info("ButterflyDream session-end extracted %d facts", len(facts))

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts with default importance."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                # Importance: user profile writes get higher default (7), general gets 5
                importance = 7 if target == "user" else 5
                self._store.add_fact(content, category=category, importance=importance)
            except Exception as e:
                logger.debug("ButterflyDream memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
        self._store = None
        self._retriever = None

    # -- LLM extraction (enhanced with importance) -----------------------------

    def _run_llm_extraction(self, messages: list) -> list[dict]:
        """Extract facts with importance scoring via LLM.

        Returns list of stored facts (with fact_id).
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 10:
                continue
            if role in ("user", "assistant"):
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {content[:1000]}")

        if len(lines) < 2:
            return []

        text = "\n\n".join(lines)
        if len(text) > 24000:
            head = text[:12000]
            tail = text[-10000:]
            text = head + "\n\n... [truncated] ...\n\n" + tail

        facts = _call_extraction_llm(
            messages_text=text,
            provider=self._extraction_provider,
            model=self._extraction_model,
        )

        stored = []
        for fact in facts:
            try:
                result = self._store.add_fact(
                    content=fact["content"],
                    category=fact.get("category", "general"),
                    tags=fact.get("tags", ""),
                    importance=fact.get("importance", 5),
                )
                stored.append(result)
            except Exception as e:
                logger.debug("ButterflyDream store fact failed: %s", e)

        return stored

    # -- Tool handlers ---------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        action = args.get("action", "")
        try:
            if action == "add":
                return self._handle_add(args)
            elif action == "search":
                return self._handle_search(args)
            elif action == "probe":
                return self._handle_probe(args)
            elif action == "related":
                return self._handle_related(args)
            elif action == "reason":
                return self._handle_reason(args)
            elif action == "contradict":
                return self._handle_contradict(args)
            elif action == "update":
                return self._handle_update(args)
            elif action == "remove":
                return self._handle_remove(args)
            elif action == "list":
                return self._handle_list(args)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            logger.error("ButterflyDream fact_store error: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

    def _handle_add(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content is required"})
        category = args.get("category", "general")
        tags = args.get("tags", "")
        importance = args.get("importance", 5)
        if not isinstance(importance, int):
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5
        importance = max(1, min(10, importance))
        result = self._store.add_fact(content, category=category, tags=tags, importance=importance)
        return json.dumps(result)

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})
        min_trust = float(args.get("min_trust", self._min_trust))
        limit = int(args.get("limit", 10))
        scenario = args.get("scenario", "balanced")
        results = self._retriever.search(query, min_trust=min_trust, limit=limit, scenario=scenario)
        return json.dumps(results, default=str)

    def _handle_probe(self, args: dict) -> str:
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required"})
        limit = int(args.get("limit", 20))
        facts = self._store.get_entity_facts(entity, limit=limit)
        return json.dumps(facts, default=str)

    def _handle_related(self, args: dict) -> str:
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required"})
        depth = int(args.get("depth", 2))
        relations = self._store.get_related_entities(entity, depth=depth)
        return json.dumps(relations, default=str)

    def _handle_reason(self, args: dict) -> str:
        entities = args.get("entities", [])
        if not entities:
            return json.dumps({"error": "entities is required"})
        limit = int(args.get("limit", 10))
        # Gather facts shared by all specified entities
        shared = None
        for entity_name in entities[:5]:  # cap at 5 entities
            facts = self._store.get_entity_facts(entity_name, limit=50)
            fact_ids = {f.get("fact_id") for f in facts if f.get("fact_id")}
            if shared is None:
                shared = fact_ids
            else:
                shared &= fact_ids
        if not shared:
            return json.dumps([])
        # Fetch full facts
        results = []
        for fid in sorted(shared)[:limit]:
            fact = self._store.get_fact(fid)
            if fact:
                results.append(dict(fact))
        return json.dumps(results, default=str)

    def _handle_contradict(self, args: dict) -> str:
        """Find facts with conflicting claims (same entity, opposing content)."""
        # Simple heuristic: find facts sharing entities with opposite sentiment words
        contradict_pairs = []
        entities = self._store._conn.execute(
            "SELECT name FROM entities LIMIT 50"
        ).fetchall()
        for (name,) in entities:
            facts = self._store.get_entity_facts(name, limit=20)
            content_list = [(f["fact_id"], f["content"]) for f in facts]
            for i, (id1, c1) in enumerate(content_list):
                for j, (id2, c2) in enumerate(content_list):
                    if j <= i:
                        continue
                    if self._is_contradictory(c1, c2):
                        contradict_pairs.append({
                            "entity": name,
                            "fact_id_a": id1,
                            "content_a": c1,
                            "fact_id_b": id2,
                            "content_b": c2,
                        })
        return json.dumps(contradict_pairs[:20], default=str)

    @staticmethod
    def _is_contradictory(a: str, b: str) -> bool:
        """Rough heuristic: check for negation markers between similar statements."""
        a_lower = a.lower()
        b_lower = b.lower()
        # Check if they share key terms but one uses negation
        negation_words = {"not", "don't", "doesn't", "didn't", "won't", "can't",
                         "isn't", "aren't", "wasn't", "weren't", "never", "no",
                         "不喜欢", "不要", "不是", "没有", "不行"}
        a_tokens = set(a_lower.split())
        b_tokens = set(b_lower.split())
        common = a_tokens & b_tokens
        if len(common) < 3:
            return False
        has_negation_a = any(n in a_tokens for n in negation_words)
        has_negation_b = any(n in b_tokens for n in negation_words)
        return has_negation_a != has_negation_b

    def _handle_update(self, args: dict) -> str:
        fact_id = args.get("fact_id")
        if fact_id is None:
            return json.dumps({"error": "fact_id is required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        kwargs = {}
        for key in ("content", "category", "tags", "importance", "trust_delta"):
            if key in args:
                kwargs[key] = args[key]
        if self._store.update_fact(fact_id, **kwargs):
            return json.dumps({"success": True, "fact_id": fact_id})
        return json.dumps({"error": "fact not found", "fact_id": fact_id})

    def _handle_remove(self, args: dict) -> str:
        fact_id = args.get("fact_id")
        if fact_id is None:
            return json.dumps({"error": "fact_id is required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        if self._store.remove_fact(fact_id):
            return json.dumps({"success": True, "fact_id": fact_id})
        return json.dumps({"error": "fact not found", "fact_id": fact_id})

    def _handle_list(self, args: dict) -> str:
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        facts = self._store.list_facts(limit=limit, offset=offset)
        return json.dumps(facts, default=str)

    def _handle_fact_feedback(self, args: dict) -> str:
        action = args.get("action", "")
        fact_id = args.get("fact_id")
        if fact_id is None or action not in ("helpful", "unhelpful"):
            return json.dumps({"error": "fact_id and action (helpful/unhelpful) are required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        result = self._store.record_feedback(fact_id, helpful=(action == "helpful"))
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

# The Hermes plugin loader discovers this via __init__.py containing
# "MemoryProvider" or "register_memory_provider" in its source.
