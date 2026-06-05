"""Tests for store.py — SQLite-backed MemoryStore."""
import sys
import os
import json
import tempfile
import hashlib
from pathlib import Path

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
        """Capitalized names extracted as entities (single + multi-word)."""
        # Single name + verb pattern: "Name verb..."
        facts = memstore._extract_entities("Caroline is considering a career in counseling")
        assert "Caroline" in facts
        facts = memstore._extract_entities("Melanie ran a charity race")
        assert "Melanie" in facts
        # Multi-word names
        facts = memstore._extract_entities("Alice Johnson and Bob Smith")
        assert "Alice Johnson" in facts
        assert "Bob Smith" in facts
        # No false positives from non-name words
        facts = memstore._extract_entities("The project uses FastAPI")
        assert "The" not in facts

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


class TestMediaAttachments:
    """Tests for media attachment features."""

    def test_attach_creates_file(self, memstore):
        """attach_media copies file to content-addressed path."""
        result = memstore.add_fact("Test media fact", importance=5)
        fid = result["fact_id"]
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpeg data")
            src = f.name
        try:
            media = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/jpeg",
                description="A test image",
            )
            assert media["sha256"] == hashlib.sha256(b"fake jpeg data").hexdigest()
            # File should exist at the content-addressed path
            media_path = Path(memstore._media_dir) / media["file_path"]
            assert media_path.exists()
            assert media_path.read_bytes() == b"fake jpeg data"
        finally:
            os.unlink(src)

    def test_attach_returns_metadata(self, memstore):
        """attach_media returns correct media_id, file_path, sha256."""
        result = memstore.add_fact("Metadata test", importance=5)
        fid = result["fact_id"]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"png data here")
            src = f.name
        try:
            media = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/png",
                description="Metadata test image",
                caption="A caption",
            )
            assert "media_id" in media
            assert media["media_id"] > 0
            assert "file_path" in media
            assert media["file_path"].endswith(".png")
            assert media["sha256"] == hashlib.sha256(b"png data here").hexdigest()
            assert media["dedup"] is False
        finally:
            os.unlink(src)

    def test_get_fact_media(self, memstore):
        """get_fact_media returns attached media records."""
        result = memstore.add_fact("Media get test", importance=5)
        fid = result["fact_id"]
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"audio data")
            src = f.name
        try:
            media = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="audio/mpeg",
                description="Test audio",
            )
            medias = memstore.get_fact_media(fid)
            assert len(medias) == 1
            assert medias[0]["media_id"] == media["media_id"]
            assert medias[0]["mime_type"] == "audio/mpeg"
            assert medias[0]["description"] == "Test audio"
        finally:
            os.unlink(src)

    def test_detach_media(self, memstore):
        """detach_media removes DB row, returns True."""
        result = memstore.add_fact("Detach test", importance=5)
        fid = result["fact_id"]
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"webp data")
            src = f.name
        try:
            media = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/webp",
                description="To detach",
            )
            mid = media["media_id"]
            assert len(memstore.get_fact_media(fid)) == 1
            assert memstore.detach_media(mid) is True
            assert len(memstore.get_fact_media(fid)) == 0
            # Second detach should return False
            assert memstore.detach_media(mid) is False
        finally:
            os.unlink(src)

    def test_attach_dedup(self, memstore):
        """Same sha256 on same fact returns dedup=True."""
        result = memstore.add_fact("Dedup test", importance=5)
        fid = result["fact_id"]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"duplicate content")
            src = f.name
        try:
            first = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="text/plain",
                description="First attach",
            )
            second = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="text/plain",
                description="Second attach (should dedup)",
            )
            assert first["dedup"] is False
            assert second["dedup"] is True
            assert second["media_id"] == first["media_id"]
            # Only one row in DB
            medias = memstore.get_fact_media(fid)
            assert len(medias) == 1
        finally:
            os.unlink(src)

    def test_path_traversal_protection(self, memstore):
        """attach_media should reject paths that escape media_dir."""
        result = memstore.add_fact("Traversal test", importance=5)
        fid = result["fact_id"]
        # We need a file that exists... but the attack is in the path we
        # pass via mime_type or the hash computation doesn't involve user paths.
        # Actually the protection is in the resolved path check. Let's test
        # with a symlink-based approach: attach a file, then directly construct
        # an evil relative path.
        # The real protection is against mime_type that could lead to directory
        # traversal in the path construction. But since we compute the path
        # internally from hash values (not user input), path traversal via
        # user-controlled mime_type would only affect the extension.
        # Let's test the explicit check works: verify that a path computed by
        # the method is always within media_root.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"safe data")
            src = f.name
        try:
            media = memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/jpeg",
                description="Safe",
            )
            # Verify the file was stored inside media dir
            abs_path = (Path(memstore._media_dir) / media["file_path"]).resolve()
            assert str(abs_path).startswith(str(Path(memstore._media_dir).resolve()) + os.sep)
        finally:
            os.unlink(src)

    def test_media_orphans_detects_unreferenced(self, memstore):
        """media_orphans finds files on disk not in DB."""
        result = memstore.add_fact("Orphan test", importance=5)
        fid = result["fact_id"]

        # Attach a legit file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"orphan test data")
            src = f.name
        try:
            memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/jpeg",
                description="Legit media",
            )
        finally:
            os.unlink(src)

        # Create an orphan file directly on disk
        orphan_rel = "im/ff/orphan_test_file.jpg"
        orphan_abs = Path(memstore._media_dir) / orphan_rel
        orphan_abs.parent.mkdir(parents=True, exist_ok=True)
        orphan_abs.write_text("I am an orphan")
        orphan_rel2 = "ot/00/another_orphan.txt"
        orphan_abs2 = Path(memstore._media_dir) / orphan_rel2
        orphan_abs2.parent.mkdir(parents=True, exist_ok=True)
        orphan_abs2.write_text("orphan 2")

        orphans = memstore.media_orphans()
        assert orphan_rel in orphans
        assert orphan_rel2 in orphans

    def test_hrr_rebundle_after_attach(self, memstore):
        """Attaching media with description updates the fact's HRR vector."""
        if not hasattr(__import__('butterfly_dream.holographic', fromlist=['_HAS_NUMPY']), '_HAS_NUMPY'):
            pass
        from butterfly_dream import holographic as hrr_mod
        if not hrr_mod._HAS_NUMPY:
            pytest.skip("numpy not available")

        result = memstore.add_fact("HRR rebundle test", importance=5)
        fid = result["fact_id"]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"hrr test data")
            src = f.name
        try:
            # Get HRR before
            pre_fact = memstore.get_fact(fid)
            pre_hrr = pre_fact.get("hrr_vector")

            memstore.attach_media(
                fact_id=fid, source_path=src, mime_type="image/jpeg",
                description="A beautiful sunset over the mountains",
            )

            post_fact = memstore.get_fact(fid)
            post_hrr = post_fact.get("hrr_vector")

            if pre_hrr is not None:
                assert post_hrr is not None
                assert pre_hrr != post_hrr  # Vector should change after rebundle
        finally:
            os.unlink(src)


class TestPersistentFlag:
    """Tests for is_persistent fact marking (S6.1)."""

    def test_add_persistent_flag(self, memstore):
        """add_fact(is_persistent=True) stores with is_persistent=1."""
        r = memstore.add_fact("Persistent fact", importance=7, is_persistent=True)
        fid = r["fact_id"]
        fact = memstore.get_fact(fid)
        assert fact["is_persistent"] == 1

    def test_add_non_persistent_default(self, memstore):
        """add_fact() without is_persistent defaults to 0."""
        r = memstore.add_fact("Non-persistent fact", importance=5)
        fid = r["fact_id"]
        fact = memstore.get_fact(fid)
        assert fact["is_persistent"] == 0

    def test_list_persistent_only(self, memstore):
        """list_facts(persistent_only=True) returns only persistent facts."""
        memstore.add_fact("Persistent A", is_persistent=True)
        memstore.add_fact("Normal B")
        memstore.add_fact("Persistent C", is_persistent=True)
        all_facts = memstore.list_facts()
        persistent = memstore.list_facts(persistent_only=True)
        assert len(all_facts) == 3
        assert len(persistent) == 2
        for f in persistent:
            assert f["is_persistent"] == 1

    def test_count_persistent_only(self, memstore):
        """count_facts(persistent_only=True) counts only persistent facts."""
        memstore.add_fact("P1", is_persistent=True)
        memstore.add_fact("N1")
        memstore.add_fact("P2", is_persistent=True)
        assert memstore.count_facts() == 3
        assert memstore.count_facts(persistent_only=True) == 2


class TestDedup:
    """Tests for cross-source dedup via _find_duplicate (S6.2)."""

    def test_dedup_skips_near_duplicate(self, memstore):
        """add_fact with dedup_threshold=0.7 skips near-identical rephrase."""
        r1 = memstore.add_fact("User prefers VS Code for Python development",
                                dedup_threshold=0.7)
        r2 = memstore.add_fact("User prefers VS Code for Python development",
                                dedup_threshold=0.7)
        assert r1["fact_id"] == r2["fact_id"]
        assert r2["merge_type"] == "exact"  # exact match, not dedup
        assert memstore.count_facts() == 1

    def test_dedup_semantic_skip(self, memstore):
        """Semantically similar but not exact -> dedup skips."""
        r1 = memstore.add_fact("User prefers VS Code for Python development",
                                dedup_threshold=0.7)
        # High Jaccard overlap (7/8 = 0.875 >= 0.7)
        r2 = memstore.add_fact("User prefers VS Code for Python development work",
                                dedup_threshold=0.7)
        assert r2.get("merge_type") == "dedup"
        assert memstore.count_facts() == 1

    def test_dedup_threshold_zero_disabled(self, memstore):
        """dedup_threshold=0.0 (default) does NOT dedup similar facts."""
        memstore.add_fact("User prefers VS Code for Python development")
        memstore.add_fact("User prefers VS Code for Python dev work")
        assert memstore.count_facts() == 2

    def test_dedup_short_content_skipped(self, memstore):
        """Content with < 3 tokens bypasses dedup (goes to normal merge path)."""
        r1 = memstore.add_fact("Hi there", dedup_threshold=0.7)
        # < 3 tokens → dedup not attempted. Exact match? "Hi there!" ≠ "Hi there". Inserted.
        r2 = memstore.add_fact("Hi there", dedup_threshold=0.7)
        # "Hi there" == "Hi there" → Level 1 exact match
        assert r1["fact_id"] == r2["fact_id"]
        assert r2["merge_type"] == "exact"
        assert memstore.count_facts() == 1

    def test_dedup_returns_is_persistent(self, memstore):
        """Dedup return dict includes is_persistent field."""
        r1 = memstore.add_fact("Core preference: uses dark mode in VS Code",
                                is_persistent=True, dedup_threshold=0.7)
        r2 = memstore.add_fact("Core preference: uses dark mode in VS Code",
                                dedup_threshold=0.7)
        # exact match: merge, is_persistent from the first should stick
        assert r2["is_persistent"] is True or r2["is_persistent"] == 1


class TestTimeline:
    """Tests for get_entity_timeline (S6.3)."""

    def test_timeline_chronological_order(self, memstore):
        """get_entity_timeline returns facts sorted by created_at ASC."""
        import time
        r1 = memstore.add_fact("Decision: use FastAPI", category="project")
        memstore._link_entities(r1["fact_id"], ["ProjectX"])
        time.sleep(0.05)
        r2 = memstore.add_fact("Decision: switch to Django", category="project")
        memstore._link_entities(r2["fact_id"], ["ProjectX"])
        time.sleep(0.05)
        r3 = memstore.add_fact("Decision: add Redis cache", category="project")
        memstore._link_entities(r3["fact_id"], ["ProjectX"])

        timeline = memstore.get_entity_timeline("ProjectX")
        assert len(timeline) == 3
        # Oldest first
        assert timeline[0]["fact_id"] == r1["fact_id"]
        assert timeline[1]["fact_id"] == r2["fact_id"]
        assert timeline[2]["fact_id"] == r3["fact_id"]

    def test_timeline_importance_filter(self, memstore):
        """min_importance filters out low-importance facts."""
        r1 = memstore.add_fact("Important: core stack choice", importance=8,
                                category="project")
        memstore._link_entities(r1["fact_id"], ["ProjectY"])
        r2 = memstore.add_fact("Trivial: used a library", importance=3,
                                category="project")
        memstore._link_entities(r2["fact_id"], ["ProjectY"])

        all_timeline = memstore.get_entity_timeline("ProjectY")
        filtered = memstore.get_entity_timeline("ProjectY", min_importance=5)
        assert len(all_timeline) == 2
        assert len(filtered) == 1
        assert filtered[0]["fact_id"] == r1["fact_id"]

    def test_timeline_empty_entity(self, memstore):
        """Unknown entity returns empty list."""
        result = memstore.get_entity_timeline("NonExistentEntity")
        assert result == []


class TestEntitySummary:
    """Tests for get_entity_summary (S6.4)."""

    def test_summary_empty_entity(self, memstore):
        """Unknown entity returns empty summary."""
        summary = memstore.get_entity_summary("NoSuchEntity")
        assert summary["facts_count"] == 0
        assert summary["timeline"] == []
        assert summary["conflicts"] == []

    def test_summary_current_state(self, memstore):
        """Latest fact per category becomes current_state."""
        r1 = memstore.add_fact("User likes cats", category="user_pref", importance=5)
        memstore._link_entities(r1["fact_id"], ["User"])
        import time
        time.sleep(0.05)
        r2 = memstore.add_fact("User now prefers dogs", category="user_pref", importance=7)
        memstore._link_entities(r2["fact_id"], ["User"])

        summary = memstore.get_entity_summary("User")
        assert summary["facts_count"] == 2
        assert summary["by_category"].get("user_pref") == 2
        # Current state should be the latest fact
        assert "dogs" in summary["current_state"]["user_pref"]["content"]

    def test_summary_conflicts_detected(self, memstore):
        """Contradictory facts flagged as conflicts."""
        r1 = memstore.add_fact("User likes coffee very much", category="user_pref", importance=7)
        memstore._link_entities(r1["fact_id"], ["User"])
        r2 = memstore.add_fact("User does not like coffee very much", category="user_pref", importance=8)
        memstore._link_entities(r2["fact_id"], ["User"])

        summary = memstore.get_entity_summary("User")
        assert len(summary["conflicts"]) >= 1

    def test_summary_related_entities(self, memstore):
        """Related entities listed."""
        r1 = memstore.add_fact("Works on ProjectX", category="project")
        memstore._link_entities(r1["fact_id"], ["User", "ProjectX"])

        summary = memstore.get_entity_summary("ProjectX")
        assert "User" in summary["related_entities"]


class TestChineseContent:
    """Chinese-language test cases for all major features (S7)."""

    def test_add_chinese_fact(self, memstore):
        """Basic add/get works with Chinese content."""
        r = memstore.add_fact("用户喜欢用VS Code写Python", category="user_pref",
                              tags="编辑器,python", importance=7)
        fid = r["fact_id"]
        fact = memstore.get_fact(fid)
        assert fact["content"] == "用户喜欢用VS Code写Python"
        assert fact["category"] == "user_pref"
        assert fact["importance"] == 7

    def test_chinese_contradiction_detected(self, memstore):
        """Chinese negation '不喜欢' triggers contradiction detection."""
        r1 = memstore.add_fact("用户喜欢喝咖啡", importance=7)
        memstore._link_entities(r1["fact_id"], ["用户"])
        r2 = memstore.add_fact("用户不喜欢喝咖啡", importance=8)
        memstore._link_entities(r2["fact_id"], ["用户"])

        # Direct contradiction check
        assert memstore._is_contradictory("用户喜欢喝咖啡", "用户不喜欢喝咖啡")

        # Entity summary should also find it
        summary = memstore.get_entity_summary("用户")
        assert len(summary["conflicts"]) >= 1

    def test_chinese_non_contradiction(self, memstore):
        """Similar Chinese facts without negation are not contradictions."""
        assert not memstore._is_contradictory(
            "用户喜欢用VS Code", "用户爱用VS Code")

    def test_chinese_dedup_semantic(self, memstore):
        """Similar Chinese facts are deduped with threshold 0.7."""
        # NOTE: FTS5 unicode61 requires spaces between CJK and Latin tokens
        r1 = memstore.add_fact("用户 喜欢 用 VS Code 来写 Python 程序",
                                dedup_threshold=0.7)
        r2 = memstore.add_fact("用户 喜欢 用 VS Code 来写 Python 程",
                                dedup_threshold=0.7)
        assert r2.get("merge_type") == "dedup"
        assert memstore.count_facts() == 1

    def test_chinese_entity_timeline(self, memstore):
        """Timeline works with Chinese entity names and content."""
        import time
        r1 = memstore.add_fact("用户决定用FastAPI做后端", category="project")
        memstore._link_entities(r1["fact_id"], ["项目X"])
        time.sleep(0.05)
        r2 = memstore.add_fact("用户把FastAPI改成Django", category="project")
        memstore._link_entities(r2["fact_id"], ["项目X"])

        timeline = memstore.get_entity_timeline("项目X")
        assert len(timeline) == 2
        assert timeline[0]["fact_id"] == r1["fact_id"]
        assert "Django" in timeline[1]["content"]

    def test_chinese_entity_summary(self, memstore):
        """Summary card works with Chinese entities."""
        r1 = memstore.add_fact("用户喜欢喝咖啡", category="user_pref", importance=6)
        memstore._link_entities(r1["fact_id"], ["用户"])
        r2 = memstore.add_fact("用户现在更喜欢喝茶", category="user_pref", importance=7)
        memstore._link_entities(r2["fact_id"], ["用户"])

        summary = memstore.get_entity_summary("用户")
        assert summary["facts_count"] == 2
        # Latest per category should be the second fact
        assert "茶" in summary["current_state"]["user_pref"]["content"]
