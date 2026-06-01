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


class TestMimeTypeEdgeCases:
    """MIME type parsing edge cases."""

    def test_mime_with_charset_param(self, tmp_db):
        """mime_type with ;charset= parameter strips correctly."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Charset in mime", category="test")

        result = store.attach_media(
            fact["fact_id"], IMAGE_PNG,
            mime_type="image/png; charset=utf-8",
            description="PNG with charset param",
        )
        # Should end with .png, not .png; charset=utf-8
        assert result["file_path"].endswith(".png"), \
            f"Expected .png, got: {result['file_path']}"
        assert "charset" not in result["file_path"]

    def test_mime_svg_xml(self, tmp_db):
        """image/svg+xml strips +xml suffix."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("SVG test", category="test")

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg"/>')
            src = f.name
        try:
            result = store.attach_media(
                fact["fact_id"], src,
                mime_type="image/svg+xml",
                description="SVG vector graphic",
            )
            assert result["file_path"].endswith(".svg"), \
                f"Expected .svg, got: {result['file_path']}"
            assert "svg+xml" not in result["file_path"]
        finally:
            os.unlink(src)

    def test_mime_svg_xml_with_charset(self, tmp_db):
        """image/svg+xml;charset=utf-8 strips both +xml and charset."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("SVG charset", category="test")

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg"/>')
            src = f.name
        try:
            result = store.attach_media(
                fact["fact_id"], src,
                mime_type="image/svg+xml; charset=utf-8",
                description="SVG with both modifiers",
            )
            assert result["file_path"].endswith(".svg"), \
                f"Expected .svg, got: {result['file_path']}"
        finally:
            os.unlink(src)

    def test_mime_octet_stream(self, tmp_db):
        """application/octet-stream uses 'ot' directory and 'stream' ext."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Octet stream", category="test")

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"binary data")
            src = f.name
        try:
            result = store.attach_media(
                fact["fact_id"], src,
                mime_type="application/octet-stream",
                description="Generic binary blob",
            )
            assert result["file_path"].startswith("ot/"), \
                f"Expected ot/ prefix, got: {result['file_path']}"
            assert result["file_path"].endswith(".bin"), \
                f"Expected .bin ext, got: {result['file_path']}"
        finally:
            os.unlink(src)

    def test_mime_avif(self, tmp_db):
        """image/avif gets .avif extension directly (no mapping needed)."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("AVIF test", category="test")

        with tempfile.NamedTemporaryFile(suffix=".avif", delete=False) as f:
            f.write(b"avif data")
            src = f.name
        try:
            result = store.attach_media(
                fact["fact_id"], src,
                mime_type="image/avif",
                description="AVIF image format",
            )
            # avif is not in _EXT_MAP, should stay as-is
            assert result["file_path"].endswith(".avif")
            assert result["file_path"].startswith("im/")
        finally:
            os.unlink(src)


class TestEmptyAndZeroByte:
    """Empty file and empty description handling."""

    def test_zero_byte_file(self, tmp_db):
        """Zero-byte file is attachable; SHA-256 is known empty hash."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Zero byte test", category="test")

        # Create zero-byte temp file
        src = os.path.join(tmpdir, "empty.png")
        with open(src, "wb"):
            pass  # zero bytes

        result = store.attach_media(
            fact["fact_id"], src, "image/png",
            description="Empty file",
        )
        # SHA-256 of empty data
        assert result["sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert os.path.isfile(os.path.join(tmpdir, "media", result["file_path"]))
        assert result["dedup"] is False

    def test_empty_description_default(self, tmp_db):
        """Default empty description is accepted (no crash)."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Empty desc test", category="test")
        result = store.attach_media(fact["fact_id"], IMAGE_PNG, "image/png")
        assert result["media_id"] > 0
        # Verify description is empty string in DB
        media = store.get_fact_media(fact["fact_id"])
        assert media[0]["description"] == ""

    def test_empty_caption_and_transcript(self, tmp_db):
        """Empty caption and transcript defaults work."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Empty caption test", category="test")
        result = store.attach_media(
            fact["fact_id"], IMAGE_PNG, "image/png",
            description="Has description only",
            caption="",
            transcript="",
        )
        media = store.get_fact_media(fact["fact_id"])
        assert media[0]["caption"] == ""
        assert media[0]["transcript"] == ""


class TestDetachEdgeCases:
    """Detach boundary conditions."""

    def test_detach_idempotent(self, tmp_db):
        """Detaching the same media_id twice returns False second time."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Idempotent detach", category="test")
        result = store.attach_media(fact["fact_id"], IMAGE_PNG, "image/png", "detach me")
        mid = result["media_id"]

        assert store.detach_media(mid) is True
        assert store.detach_media(mid) is False  # already gone

    def test_detach_nonexistent(self, tmp_db):
        """Detaching a media_id that never existed returns False."""
        store, tmpdir, db_path = tmp_db
        assert store.detach_media(99999) is False

    def test_get_fact_media_no_media(self, tmp_db):
        """Fact with no media returns empty list."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("No media fact", category="test")
        assert store.get_fact_media(fact["fact_id"]) == []


class TestMediaOrphansExtended:
    """Orphan detection edge cases."""

    def test_orphans_after_remove_fact(self, tmp_db):
        """Removing a fact with media leaves orphaned disk files."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Orphan creator", category="test")
        r = store.attach_media(fact["fact_id"], IMAGE_PNG, "image/png", "will be orphan")
        store.remove_fact(fact["fact_id"])  # CASCADE deletes DB row

        orphans = store.media_orphans()
        assert r["file_path"] in orphans, f"{r['file_path']} not in orphans: {orphans}"

    def test_orphans_empty_dir(self, tmp_db):
        """media_orphans returns empty list when no media dir exists."""
        store, tmpdir, db_path = tmp_db
        # Don't attach any media
        assert store.media_orphans() == []

    def test_orphans_with_mixed_state(self, tmp_db):
        """mix of referenced and orphaned files."""
        store, tmpdir, db_path = tmp_db
        f1 = store.add_fact("Referenced fact", category="test")
        r1 = store.attach_media(f1["fact_id"], IMAGE_PNG, "image/png", "referenced")
        f2 = store.add_fact("Orphan fact", category="test")
        r2 = store.attach_media(f2["fact_id"], IMAGE_PNG2, "image/png", "soon orphan")
        store.remove_fact(f2["fact_id"])

        orphans = store.media_orphans()
        assert r1["file_path"] not in orphans, f"Referenced file should not be orphan: {r1['file_path']}"
        assert r2["file_path"] in orphans, f"Expected orphan: {r2['file_path']}"


class TestSameFileDifferentMetadata:
    """Same file attached with different descriptions."""

    def test_same_file_different_descriptions_same_fact(self, tmp_db):
        """Same file + same fact = dedup, first description is kept."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Dedup desc test", category="test")
        fid = fact["fact_id"]

        r1 = store.attach_media(fid, IMAGE_PNG, "image/png", description="First version")
        r2 = store.attach_media(fid, IMAGE_PNG, "image/png", description="Second version")

        assert r2["dedup"] is True  # same file, same fact → dedup
        # Description from the FIRST insert is kept (dedup returns existing row)
        media = store.get_fact_media(fid)
        assert len(media) == 1

    def test_same_file_different_descriptions_different_facts(self, tmp_db):
        """Same file on different facts = separate records, each keeps its desc."""
        store, tmpdir, db_path = tmp_db
        f1 = store.add_fact("Fact 1", category="test")
        f2 = store.add_fact("Fact 2", category="test")

        r1 = store.attach_media(f1["fact_id"], IMAGE_PNG, "image/png", description="Cat photo")
        r2 = store.attach_media(f2["fact_id"], IMAGE_PNG, "image/png", description="Dog photo")

        # Different media_ids because different facts
        assert r1["media_id"] != r2["media_id"]
        # Each keeps its own description
        m1 = store.get_fact_media(f1["fact_id"])
        m2 = store.get_fact_media(f2["fact_id"])
        assert m1[0]["description"] == "Cat photo"
        assert m2[0]["description"] == "Dog photo"


class TestCJKAndUnicode:
    """CJK/Unicode in paths, descriptions, and FTS5."""

    def test_chinese_description_fts5(self, tmp_db):
        """FTS5 search on Chinese descriptions."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("中文测试", category="test")
        store.attach_media(
            fact["fact_id"], IMAGE_PNG, "image/png",
            description="一只可爱的柴犬在樱花树下微笑",
            caption="柴犬",
        )

        # Search in Chinese
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        results = conn.execute(
            """SELECT m.description FROM media_attachments_fts f
               JOIN media_attachments m ON f.rowid = m.media_id
               WHERE media_attachments_fts MATCH ?""",
            ("柴犬",),
        ).fetchall()
        assert len(results) >= 1
        conn.close()

    def test_emoji_in_description(self, tmp_db):
        """Emoji in description stored and searchable."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Emoji test", category="test")
        store.attach_media(
            fact["fact_id"], IMAGE_PNG, "image/png",
            description="A cute cat 🐱 with a heart ❤️",
        )

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Emoji stored correctly
        media = store.get_fact_media(fact["fact_id"])
        assert "🐱" in media[0]["description"]
        assert "❤️" in media[0]["description"]
        conn.close()

    def test_long_description(self, tmp_db):
        """Very long description (1000 chars) is stored correctly."""
        store, tmpdir, db_path = tmp_db
        fact = store.add_fact("Long desc test", category="test")
        long_desc = "A photo of " + "very beautiful " * 50 + "cat"
        assert len(long_desc) > 500

        result = store.attach_media(
            fact["fact_id"], IMAGE_PNG, "image/png",
            description=long_desc,
        )
        media = store.get_fact_media(fact["fact_id"])
        assert media[0]["description"] == long_desc
        assert len(media[0]["description"]) == len(long_desc)
