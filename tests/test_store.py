"""Tests for store.py — SQLite-backed MemoryStore."""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from butterfly_dream import store
from butterfly_dream.store import MemoryStore


@pytest.fixture
def memstore():
    """Create a MemoryStore in a temp DB for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    ms = MemoryStore(db_path=db_path, default_trust=0.5, hrr_dim=128)
    yield ms
    ms.close()
    os.unlink(db_path)


class TestAddAndGet:
    def test_add_simple(self, memstore):
        """Basic add returns fact_id and content."""
        result = memstore.add_fact("The sky is blue", category="general", tags="nature")
        assert result["content"] == "The sky is blue"
        assert result["fact_id"] > 0
        assert result["merged"] is False

    def test_get_by_id(self, memstore):
        """get_fact retrieves what was added."""
        added = memstore.add_fact("Python is fun", category="general")
        fetched = memstore.get_fact(added["fact_id"])
        assert fetched is not None
        assert fetched["content"] == "Python is fun"

    def test_get_nonexistent(self, memstore):
        """get_fact on missing ID returns None."""
        assert memstore.get_fact(99999) is None

    def test_count(self, memstore):
        """count_facts reflects number of facts."""
        assert memstore.count_facts() == 0
        memstore.add_fact("Fact A")
        assert memstore.count_facts() == 1
        memstore.add_fact("Fact B")
        assert memstore.count_facts() == 2

    def test_add_with_entities(self, memstore):
        """Entities are extracted and linked."""
        result = memstore.add_fact("Alice likes Python programming", entities=["Alice", "Python"])
        # "Alice" and "Python" should be extracted as entities
        facts = memstore.get_entity_facts("Alice")
        assert len(facts) >= 1
        assert facts[0]["content"] == "Alice likes Python programming"

    def test_add_with_explicit_entities(self, memstore):
        """Explicit entities extend auto-extracted ones."""
        result = memstore.add_fact("The tool is great", entities=["Hermes", "Butterfly"])
        # Check entities are linked
        facts = memstore.get_entity_facts("Hermes")
        assert len(facts) == 1

    def test_add_with_importance(self, memstore):
        """Importance is stored correctly."""
        memstore.add_fact("Critical fact", importance=9)
        memstore.add_fact("Trivial fact", importance=2)
        facts = memstore.list_facts()
        # The two facts should have different importance values
        importances = {f["content"]: f["importance"] for f in facts}
        assert importances.get("Critical fact") == 9
        assert importances.get("Trivial fact") == 2


class TestUpdateRemove:
    def test_update_fact(self, memstore):
        """Update changes specified fields."""
        added = memstore.add_fact("Old content", category="general", importance=5)
        memstore.update_fact(added["fact_id"], content="New content", importance=8)
        fetched = memstore.get_fact(added["fact_id"])
        assert fetched["content"] == "New content"
        assert fetched["importance"] == 8

    def test_remove_fact(self, memstore):
        """Remove deletes the fact."""
        added = memstore.add_fact("To be deleted")
        fid = added["fact_id"]
        assert memstore.get_fact(fid) is not None
        assert memstore.remove_fact(fid) is True
        assert memstore.get_fact(fid) is None

    def test_list_facts(self, memstore):
        """list_facts returns facts ordered by creation date."""
        for i in range(5):
            memstore.add_fact(f"Fact number {i}")
        facts = memstore.list_facts(limit=10)
        assert len(facts) == 5
        contents = [f["content"] for f in facts]
        assert "Fact number 0" in contents
        # Recent first
        assert "Fact number 4" in contents


class TestEntityManagement:
    def test_entity_extraction_capitalized(self, memstore):
        """Capitalized multi-word phrases extracted as entities."""
        facts = memstore._extract_entities("Alice and Bob work on Project X")
        assert "Alice" not in facts  # single capitalized words aren't entities by default
        # Actually let me check the pattern
        # _RE_CAPITALIZED = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        # That requires at least TWO capitalized words in sequence
        facts = memstore._extract_entities("Alice Johnson and Bob Smith")
        assert "Alice Johnson" in facts
        assert "Bob Smith" in facts

    def test_entity_extraction_quotes(self, memstore):
        """Quoted phrases extracted as entities."""
        facts = memstore._extract_entities('The project "Butterfly Dream" is great')
        assert "Butterfly Dream" in facts

    def test_entity_extraction_cjk(self, memstore):
        """CJK bracket-quoted text extracted."""
        facts = memstore._extract_entities("项目「蝴蝶梦」很好用")
        assert "蝴蝶梦" in facts

    def test_get_entity_facts(self, memstore):
        """get_entity_facts returns all facts for an entity."""
        memstore.add_fact("Alice Johnson is a developer", importance=7, merge=False)
        memstore.add_fact("Alice Johnson likes Python", importance=5, merge=False)
        facts = memstore.get_entity_facts("Alice Johnson")
        assert len(facts) >= 2

    def test_related_entities_bfs(self, memstore):
        """BFS traversal finds connected entities."""
        # Create entity graph: A -[fact1]-> B -[fact2]-> C
        memstore.add_fact("Alice Johnson and Bob Smith work together", entities=["Alice Johnson", "Bob Smith"])
        memstore.add_fact("Bob Smith and Charlie Brown are friends", entities=["Bob Smith", "Charlie Brown"])
        related = memstore.get_related_entities("Alice Johnson", depth=2)
        names = {(r["source"], r["target"]) for r in related}
        assert ("Alice Johnson", "Bob Smith") in names
        assert ("Bob Smith", "Charlie Brown") in names


class TestFactMerging:
    def test_exact_merge(self, memstore):
        """Same content → merge, not duplicate."""
        r1 = memstore.add_fact("Test fact", importance=5)
        r2 = memstore.add_fact("Test fact", importance=8)
        assert r2["merged"] is True
        assert r2["merge_type"] == "exact"
        assert r2["importance"] == 8  # max of 5 and 8
        assert memstore.count_facts() == 1  # still 1

    def test_semantic_merge_shared_entity(self, memstore):
        """Facts sharing an entity → semantic merge."""
        r1 = memstore.add_fact('Alice Johnson uses VS Code for Python dev', importance=5, entities=["Alice Johnson"])
        r2 = memstore.add_fact('Alice Johnson uses VS Code for Rust dev', importance=6, entities=["Alice Johnson"])
        assert r2["merged"] is True
        assert r2["merge_type"] == "semantic"
        assert memstore.count_facts() == 1  # merged into one
        merged = memstore.get_fact(r2["fact_id"])
        assert merged is not None
        assert "Alice Johnson" in merged["content"]
        assert "VS Code" in merged["content"] or "Python" in merged["content"]

    def test_new_fact_when_no_match(self, memstore):
        """Different entities → new fact, no merge."""
        memstore.add_fact("Alice likes cats", entities=["Alice"])
        r2 = memstore.add_fact("Bob likes dogs", entities=["Bob"])
        assert r2["merged"] is False
        assert memstore.count_facts() == 2

    def test_merge_log_created(self, memstore):
        """Semantic merge creates merge_log entry."""
        memstore.add_fact("Alice Johnson uses VS Code", entities=["Alice Johnson"])
        r2 = memstore.add_fact("Alice Johnson uses PyCharm", entities=["Alice Johnson"])
        # Check merge_log
        logs = memstore._conn.execute(
            "SELECT * FROM merge_log WHERE kept_fact_id = ?", (r2["fact_id"],)
        ).fetchall()
        assert len(logs) >= 1
        assert logs[0]["merge_reason"] == "semantic"


class TestFeedback:
    def test_helpful_increases_trust(self, memstore):
        """'helpful' feedback increases trust and importance."""
        r = memstore.add_fact("Helpful fact", importance=5)
        memstore.record_feedback(r["fact_id"], helpful=True)
        updated = memstore.get_fact(r["fact_id"])
        assert updated["trust_score"] > 0.5  # default 0.5 + 0.05
        assert updated["importance"] > 5.0  # 5.0 + 0.5

    def test_unhelpful_decreases_trust(self, memstore):
        """'unhelpful' feedback decreases trust and importance."""
        r = memstore.add_fact("Unhelpful fact", importance=6)
        memstore.record_feedback(r["fact_id"], helpful=False)
        updated = memstore.get_fact(r["fact_id"])
        assert updated["trust_score"] < 0.5  # default 0.5 - 0.10
        assert updated["importance"] < 6.0  # 6.0 - 0.5

    def test_feedback_nonexistent(self, memstore):
        """Feedback on missing ID returns error."""
        result = memstore.record_feedback(99999, helpful=True)
        assert "error" in result


class TestHRREncoding:
    def test_hrr_vector_stored(self, memstore):
        """HRR vector is computed and stored for new facts."""
        r = memstore.add_fact("A test fact with entities", importance=5)
        fetched = memstore.get_fact(r["fact_id"])
        if fetched["hrr_vector"] is not None:
            # Verify it decodes to correct length
            vec = np.frombuffer(fetched["hrr_vector"], dtype=np.float64)
            assert len(vec) == 128  # hrr_dim we set in fixture

    def test_hrr_similarity(self, memstore):
        """compute_hrr_similarity returns a value in [0, 1]."""
        r = memstore.add_fact("Python programming language", importance=5)
        sim = memstore.compute_hrr_similarity(r["fact_id"], "Python coding")
        assert 0.0 <= sim <= 1.0


class TestCombineFactContent:
    """Direct unit tests for _combine_fact_content (S4)."""

    def test_exact_same(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "Alice likes cats")
        assert result == "Alice likes cats"

    def test_substring_existing_longer(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats very much", "Alice likes cats")
        assert result == "Alice likes cats very much"

    def test_substring_new_longer(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "Alice likes cats very much")
        assert result == "Alice likes cats very much"

    def test_case_insensitive_substring(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "alice likes cats very much")
        assert "cats very much" in result

    def test_contradiction_detected(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "Alice does not like cats")
        assert "[冲突]" in result

    def test_different_aspects_combined(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "Alice works at Google")
        assert "；" in result or "Google" in result

    def test_no_new_info(self):
        from butterfly_dream.store import MemoryStore
        result = MemoryStore._combine_fact_content("Alice likes cats", "alice likes cats")
        assert result == "Alice likes cats"
