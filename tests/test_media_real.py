"""Real-world end-to-end media attachment tests.

Uses actual image and audio files from the filesystem to verify:
- File copy to content-addressed storage
- SHA-256 dedup
- DB insertion + retrieval
- FTS5 search on descriptions
- Detach
- Orphan detection
"""
import sys
import os
import tempfile
import shutil
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from butterfly_dream.store import MemoryStore

# ── Real media files available on this system ──────────────────────

IMAGE_PNG = "/home/xx/.hermes/hermes-agent/assets/banner.png"
IMAGE_PNG2 = "/home/xx/.hermes/cache/images/neko_sunset.png"
AUDIO_MP3 = "/home/xx/.hermes/voices/keli.mp3"


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    tmpdir = tempfile.mkdtemp(prefix="butterfly_media_real_")
    db_path = os.path.join(tmpdir, "test_memory.db")
    store = MemoryStore(db_path, hrr_dim=128)  # small HRR dim for speed
    yield store, tmpdir, db_path
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestRealMediaAttach:
    """End-to-end tests with real files."""

    def test_attach_image_png(self, tmp_db):
        """Attach a real PNG image and verify every step."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("A banner image for testing", category="test")
        fact_id = fact["fact_id"]

        result = store.attach_media(
            fact_id=fact_id,
            source_path=IMAGE_PNG,
            mime_type="image/png",
            description="Hermes agent banner logo",
            caption="banner.png",
        )

        # 1. Result shape
        assert "media_id" in result
        assert result["dedup"] is False
        assert result["sha256"] == hashlib.sha256(
            open(IMAGE_PNG, "rb").read()
        ).hexdigest()

        # 2. File was copied to CAS path
        media_root = os.path.join(tmpdir, "media")
        abs_path = os.path.join(media_root, result["file_path"])
        assert os.path.isfile(abs_path), f"File not found at {abs_path}"
        assert os.path.getsize(abs_path) == os.path.getsize(IMAGE_PNG)
        assert hashlib.sha256(open(abs_path, "rb").read()).hexdigest() == result["sha256"]

        # 3. Path structure: im/{sha[:2]}/{sha}.png
        sha = result["sha256"]
        expected_rel = f"im/{sha[:2]}/{sha}.png"
        assert result["file_path"] == expected_rel, f"{result['file_path']} != {expected_rel}"

        # 4. DB record exists
        media_list = store.get_fact_media(fact_id)
        assert len(media_list) == 1
        row = media_list[0]
        assert row["mime_type"] == "image/png"
        assert row["file_path"] == expected_rel
        assert row["sha256"] == sha
        assert row["description"] == "Hermes agent banner logo"
        assert row["caption"] == "banner.png"
        assert row["file_size"] > 0

    def test_attach_audio_mp3(self, tmp_db):
        """Attach a real MP3 audio file."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Genshin voice clip", category="test")
        fact_id = fact["fact_id"]

        result = store.attach_media(
            fact_id=fact_id,
            source_path=AUDIO_MP3,
            mime_type="audio/mpeg",
            description="Keli character voice line",
            transcript="爆炸就是艺术！",
        )

        sha = result["sha256"]
        expected_rel = f"au/{sha[:2]}/{sha}.mp3"
        assert result["file_path"] == expected_rel

        # File on disk
        media_root = os.path.join(tmpdir, "media")
        assert os.path.isfile(os.path.join(media_root, result["file_path"]))

        # DB record with transcript
        media_list = store.get_fact_media(fact_id)
        assert len(media_list) == 1
        row = media_list[0]
        assert row["mime_type"] == "audio/mpeg"
        assert row["transcript"] == "爆炸就是艺术！"

    def test_dedup_same_file_same_fact(self, tmp_db):
        """Attaching the same file twice to the same fact returns dedup=True."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Dedup test", category="test")
        fact_id = fact["fact_id"]

        r1 = store.attach_media(fact_id, IMAGE_PNG, "image/png", "first")
        assert r1["dedup"] is False

        r2 = store.attach_media(fact_id, IMAGE_PNG, "image/png", "second")
        assert r2["dedup"] is True
        assert r2["media_id"] == r1["media_id"]
        assert r2["sha256"] == r1["sha256"]

        # Only one record in DB
        assert len(store.get_fact_media(fact_id)) == 1

    def test_same_file_different_facts(self, tmp_db):
        """Same file attached to different facts makes separate DB records
        but shares the disk file (CAS dedup)."""
        store, tmpdir, db_path = tmp_db
        f1 = store.add_fact("Fact one", category="test")
        f2 = store.add_fact("Fact two", category="test")

        r1 = store.attach_media(f1["fact_id"], IMAGE_PNG, "image/png", "one")
        r2 = store.attach_media(f2["fact_id"], IMAGE_PNG, "image/png", "two")

        # Different media_ids
        assert r1["media_id"] != r2["media_id"]
        # Same sha256
        assert r1["sha256"] == r2["sha256"]
        # Same disk file (only one copy)
        assert r1["file_path"] == r2["file_path"]
        assert len(store.get_fact_media(f1["fact_id"])) == 1
        assert len(store.get_fact_media(f2["fact_id"])) == 1

    def test_attach_multiple_files(self, tmp_db):
        """Multiple different files on one fact."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Multi media test", category="test")
        fid = fact["fact_id"]

        r1 = store.attach_media(fid, IMAGE_PNG, "image/png", "banner")
        r2 = store.attach_media(fid, IMAGE_PNG2, "image/png", "neko sunset")
        r3 = store.attach_media(fid, AUDIO_MP3, "audio/mpeg", "keli voice",
                                transcript="哒哒哒")

        assert r1["media_id"] != r2["media_id"]
        assert r2["media_id"] != r3["media_id"]

        media_list = store.get_fact_media(fid)
        assert len(media_list) == 3  # all three attached

        # Check mime types
        mimes = {m["mime_type"] for m in media_list}
        assert "image/png" in mimes
        assert "audio/mpeg" in mimes

    def test_detach_media(self, tmp_db):
        """Detach removes DB record but keeps file on disk."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Detach me", category="test")
        fid = fact["fact_id"]

        result = store.attach_media(fid, IMAGE_PNG, "image/png", "will detach")
        media_id = result["media_id"]
        file_path = os.path.join(tmpdir, "media", result["file_path"])

        assert os.path.isfile(file_path)  # file exists before detach
        assert store.detach_media(media_id) is True
        assert len(store.get_fact_media(fid)) == 0  # DB record gone
        assert os.path.isfile(file_path)  # file still on disk

    def test_media_orphans(self, tmp_db):
        """media_orphans() finds files not referenced in DB."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Orphan test", category="test")
        fid = fact["fact_id"]

        # Attach then detach
        r = store.attach_media(fid, IMAGE_PNG, "image/png", "orphan me")
        store.detach_media(r["media_id"])

        orphans = store.media_orphans()
        assert r["file_path"] in orphans, f"{r['file_path']} not in orphans: {orphans}"

    def test_fts5_search_description(self, tmp_db):
        """FTS5 can search by description after attaching media."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Searchable media fact", category="test")
        fid = fact["fact_id"]

        store.attach_media(
            fid, IMAGE_PNG, "image/png",
            description="A beautiful sunset with cherry blossoms",
        )
        store.attach_media(
            fid, IMAGE_PNG2, "image/png",
            description="Neko-chan lounging in the sun",
        )

        # Search via raw FTS5
        from butterfly_dream.store import _SCHEMA  # noqa: just ensure init

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        results = conn.execute(
            """SELECT m.description, m.file_path
               FROM media_attachments_fts f
               JOIN media_attachments m ON f.rowid = m.media_id
               WHERE media_attachments_fts MATCH ?""",
            ("cherry blossoms",),
        ).fetchall()

        assert len(results) >= 1
        descriptions = [r["description"] for r in results]
        assert any("cherry blossoms" in d for d in descriptions)

        # Also search japanese/romaji
        results2 = conn.execute(
            """SELECT m.description FROM media_attachments_fts f
               JOIN media_attachments m ON f.rowid = m.media_id
               WHERE media_attachments_fts MATCH ?""",
            ('"Neko-chan lounging"',),
        ).fetchall()
        assert len(results2) >= 1

        conn.close()

    def test_invalid_fact_id_raises(self, tmp_db):
        """Attaching to non-existent fact raises ValueError."""
        store, tmpdir, db_path = tmp_db
        with pytest.raises(ValueError, match="not found"):
            store.attach_media(99999, IMAGE_PNG, "image/png")

    def test_missing_source_file_raises(self, tmp_db):
        """Attaching non-existent file raises FileNotFoundError."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Missing file", category="test")
        with pytest.raises(FileNotFoundError):
            store.attach_media(fact["fact_id"], "/nonexistent/file.png", "image/png")

    def test_cascade_delete_fact_removes_media_db_keeps_file(self, tmp_db):
        """Removing a fact CASCADE-deletes media DB rows but keeps disk files."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Transient", category="test")
        fid = fact["fact_id"]

        r = store.attach_media(fid, IMAGE_PNG, "image/png", "transient media")

        # Remove fact — CASCADE should delete media_attachments row
        store.remove_fact(fid)
        assert store.get_fact(fid) is None  # fact gone
        assert len(store.get_fact_media(fid)) == 0  # media DB row cascade-deleted

        # File still on disk
        media_root = os.path.join(tmpdir, "media")
        assert os.path.isfile(os.path.join(media_root, r["file_path"]))


class TestMergeAndMedia:
    """Media survives semantic merge."""

    def test_media_persists_after_semantic_merge(self, tmp_db):
        """Media on a fact remains accessible after new content merges into it."""
        store, tmpdir, db_path = tmp_db
        fa = store.add_fact(
            "Alice loves fluffy cats",
            category="test", entities=["Alice"],
        )
        r = store.attach_media(
            fa["fact_id"], IMAGE_PNG, "image/png",
            description="Alice's cat photo",
        )

        # New content with shared entity + high Jaccard similarity merges into fact A
        fb = store.add_fact(
            "Alice loves cats a lot", category="test", entities=["Alice"],
        )
        assert fb["fact_id"] == fa["fact_id"]  # merged, not new
        assert fb["merged"] is True

        # Media still on the fact
        media = store.get_fact_media(fa["fact_id"])
        assert len(media) == 1
        assert media[0]["sha256"] == r["sha256"]

    def test_media_on_absorbed_content_never_needs_reparent(self, tmp_db):
        """New content absorbed during merge was never a DB row, so no re-parent."""
        store, tmpdir, db_path = tmp_db
        fa = store.add_fact("Bob builds robots", category="test", entities=["Bob"])
        store.attach_media(fa["fact_id"], IMAGE_PNG, "image/png", description="Bob's robot")

        fb = store.add_fact("Bob builds amazing robots", category="test", entities=["Bob"])
        assert fb["merged"] is True  # merged into fact A

        # Only one media record (on fact A), the absorbed content had no fact_id
        media = store.get_fact_media(fa["fact_id"])
        assert len(media) == 1


class TestRetrieverPipeline:
    """End-to-end search via ThreeDimRetriever (not just raw SQL)."""

    def test_media_search_through_retriever(self, tmp_db):
        """Searching via retriever correctly finds media-attached facts."""
        store, tmpdir, db_path = tmp_db
        # Need to mock Hermes imports for retriever too
        from butterfly_dream.retrieval import ThreeDimRetriever

        retriever = ThreeDimRetriever(store)

        fa = store.add_fact("Photo gallery of cherry blossoms", category="test")
        store.attach_media(
            fa["fact_id"], IMAGE_PNG, "image/png",
            description="Beautiful cherry blossoms at sunset",
        )

        # Search — this would crash with ValueError before P0-1 fix!
        results = retriever.search("cherry blossoms", limit=5)
        assert len(results) >= 1
        assert any(r.get("_media_match") for r in results)

    def test_media_search_no_match_does_not_crash(self, tmp_db):
        """Search without media matches doesn't crash."""
        from butterfly_dream.retrieval import ThreeDimRetriever
        store, tmpdir, db_path = tmp_db
        retriever = ThreeDimRetriever(store)

        store.add_fact("Plain text fact without media", category="test")
        store.add_fact("Another fact", category="test")

        # No media to match, but should not crash
        results = retriever.search("something", limit=5)
        assert len(results) >= 0  # just don't crash


class TestChunkedHashing:
    """Chunked SHA-256 produces identical results to single-read."""

    def test_chunked_matches_single_read(self, tmp_db):
        """64KB chunked hashing produces same SHA-256 as reading entire file."""
        store, tmpdir, db_path = tmp_db

        # Reference: single-read hash
        with open(IMAGE_PNG, "rb") as f:
            ref_hash = hashlib.sha256(f.read()).hexdigest()

        # Store's chunked hash (used inside attach_media)
        sha256_hash = hashlib.sha256()
        with open(IMAGE_PNG, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        chunked_hash = sha256_hash.hexdigest()

        assert chunked_hash == ref_hash

        # Also verify the store produces the same hash
        fact = store.add_fact("Hash test", category="test")
        result = store.attach_media(
            fact["fact_id"], IMAGE_PNG, "image/png", "hash verification",
        )
        assert result["sha256"] == ref_hash
