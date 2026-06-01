"""Tests for media compression module.

Tests both unit-level compression logic and integration into
MemoryStore.attach_media with real files.
"""
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from PIL import Image

from butterfly_dream.store import MemoryStore
from butterfly_dream.media_compressor import (
    compress_media,
    DEFAULT_COMPRESSION_CONFIG,
)

# ── Real media files ────────────────────────────────────────────────

IMAGE_PNG = "/home/xx/.hermes/hermes-agent/assets/banner.png"
AUDIO_MP3 = "/home/xx/.hermes/voices/keli.mp3"
# A large noisy PNG that actually benefits from compression
TEST_NOISY_PNG = "/tmp/butterfly_test_images/test_noisy.png"
# Test video source (8.5MB H.264 + PCM, high quality → compresses well)
TEST_VIDEO = "/tmp/butterfly_test_images/test_video_source.mp4"

# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    tmpdir = tempfile.mkdtemp(prefix="butterfly_compression_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def store_compressed(tmp_dir):
    """A MemoryStore with compression enabled (default config)."""
    db_path = os.path.join(tmp_dir, "test_compressed.db")
    store = MemoryStore(
        db_path,
        hrr_dim=128,
        compression_config=DEFAULT_COMPRESSION_CONFIG.copy(),
    )
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def store_uncompressed(tmp_dir):
    """A MemoryStore with compression disabled."""
    db_path = os.path.join(tmp_dir, "test_uncompressed.db")
    store = MemoryStore(
        db_path,
        hrr_dim=128,
        compression_config={"enabled": False},
    )
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Unit tests: compress_media ──────────────────────────────────────

class TestCompressMediaUnit:
    """Direct tests on compress_media()."""

    def test_skip_when_disabled(self, tmp_dir):
        """Compression returns None when enabled=False."""
        result_path, result_mime = compress_media(
            IMAGE_PNG, "image/png", tmp_dir,
            config={"enabled": False},
        )
        assert result_path is None
        assert result_mime is None

    def test_compress_image_png(self, tmp_dir):
        """Compress a PNG image and verify resulting JPEG."""
        result_path, result_mime = compress_media(
            TEST_NOISY_PNG, "image/png", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )
        assert result_path is not None
        assert result_mime == "image/jpeg"
        assert os.path.exists(result_path)

        # Should be a valid JPEG
        img = Image.open(result_path)
        assert img.format == "JPEG"
        assert img.mode == "RGB"

        # Should be significantly smaller
        orig_size = os.path.getsize(TEST_NOISY_PNG)
        comp_size = os.path.getsize(result_path)
        assert comp_size < orig_size * 0.8, (
            f"Not meaningfully compressed: {orig_size} → {comp_size}"
        )

        # Clean up
        os.unlink(result_path)

    def test_compress_audio_mp3(self, tmp_dir):
        """Compress an MP3 audio file and verify resulting MP3."""
        result_path, result_mime = compress_media(
            AUDIO_MP3, "audio/mpeg", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )

        if result_path is None:
            pytest.skip("ffmpeg not available for audio compression")

        assert result_mime == "audio/mpeg"
        assert os.path.exists(result_path)
        assert result_path.endswith(".mp3")

        orig_size = os.path.getsize(AUDIO_MP3)
        comp_size = os.path.getsize(result_path)
        # The original is 320kbps; default is 128kbps so should be smaller
        assert comp_size < orig_size * 0.8, (
            f"Audio not meaningfully compressed: {orig_size} → {comp_size}"
        )

        os.unlink(result_path)

    def test_compress_video_mp4(self, tmp_dir):
        """Compress an MP4 video file and verify resulting MP4."""
        result_path, result_mime = compress_media(
            TEST_VIDEO, "video/mp4", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )

        if result_path is None:
            pytest.skip("ffmpeg not available for video compression")

        assert result_mime == "video/mp4"
        assert os.path.exists(result_path)
        assert result_path.endswith(".mp4")

        orig_size = os.path.getsize(TEST_VIDEO)
        comp_size = os.path.getsize(result_path)
        # 8.5M → should be well under 50%
        assert comp_size < orig_size * 0.5, (
            f"Video not meaningfully compressed: {orig_size} → {comp_size}"
        )

        os.unlink(result_path)

    def test_compress_unknown_type(self, tmp_dir):
        """Compression skips unknown MIME types."""
        # Write a minimal .bin file
        bin_path = os.path.join(tmp_dir, "test.bin")
        with open(bin_path, "wb") as f:
            f.write(b"\x00" * 100)
        result_path, result_mime = compress_media(
            bin_path, "application/octet-stream", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )
        assert result_path is None
        assert result_mime is None

    def test_compress_svg(self, tmp_dir):
        """SVG images are skipped (vector format)."""
        svg_path = os.path.join(tmp_dir, "test.svg")
        with open(svg_path, "w") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        result_path, result_mime = compress_media(
            svg_path, "image/svg+xml", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )
        assert result_path is None
        assert result_mime is None

    def test_compress_image_quality(self, tmp_dir):
        """Higher quality setting produces larger file."""
        from butterfly_dream.media_compressor import _compress_image

        # Low quality
        low_cfg = {"quality": 10, "max_dim": 1920, "convert_to_jpeg": True}
        low_path, _ = _compress_image(TEST_NOISY_PNG, low_cfg)
        assert low_path is not None
        low_size = os.path.getsize(low_path)
        os.unlink(low_path)

        # High quality
        high_cfg = {"quality": 95, "max_dim": 1920, "convert_to_jpeg": True}
        high_path, _ = _compress_image(TEST_NOISY_PNG, high_cfg)
        assert high_path is not None
        high_size = os.path.getsize(high_path)
        os.unlink(high_path)

        assert high_size > low_size, (
            f"Higher quality ({high_size}) should be larger than low ({low_size})"
        )

    def test_compress_image_resize(self, tmp_dir):
        """max_dim limits the compressed image dimensions."""
        from butterfly_dream.media_compressor import _compress_image

        cfg = {"quality": 85, "max_dim": 100, "convert_to_jpeg": True}
        result_path, _ = _compress_image(TEST_NOISY_PNG, cfg)
        if result_path is None:
            pytest.skip("Image too small for resize test")

        img = Image.open(result_path)
        assert img.width <= 100 and img.height <= 100, (
            f"Image not resized: {img.size}"
        )
        os.unlink(result_path)

    def test_compress_no_benefit(self, tmp_dir):
        """If compression doesn't save space, original is kept."""
        # Create a tiny 1x1 black pixel PNG — already tiny
        tiny_path = os.path.join(tmp_dir, "tiny.png")
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        img.save(tiny_path, "PNG")

        result_path, result_mime = compress_media(
            tiny_path, "image/png", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )
        # Should skip since compressed won't be smaller
        assert result_path is None
        assert result_mime is None

    def test_skip_large_file(self, tmp_dir):
        """Files exceeding max_size_mb are skipped."""
        # Create a 2MB dummy file
        big_path = os.path.join(tmp_dir, "big_test.mp4")
        with open(big_path, "wb") as f:
            f.write(b"\x00" * (2 * 1024 * 1024))

        result_path, result_mime = compress_media(
            big_path, "video/mp4", tmp_dir,
            config={"enabled": True, "max_size_mb": 1},  # 1MB limit → 2MB file skipped
        )
        assert result_path is None
        assert result_mime is None

    def test_skip_large_file_default_threshold(self, tmp_dir):
        """Default max_size_mb=100 skips files >100MB."""
        # Create a 150MB file (sparse, doesn't actually use disk)
        big_path = os.path.join(tmp_dir, "huge.mp4")
        with open(big_path, "wb") as f:
            f.seek(150 * 1024 * 1024 - 1)
            f.write(b"\x00")

        result_path, result_mime = compress_media(
            big_path, "video/mp4", tmp_dir,
            config=DEFAULT_COMPRESSION_CONFIG,
        )
        assert result_path is None  # skipped because >100MB

    def test_compress_below_threshold(self, tmp_dir):
        """Small files still get compressed when under max_size_mb."""
        result_path, result_mime = compress_media(
            TEST_NOISY_PNG, "image/png", tmp_dir,
            config={"enabled": True, "max_size_mb": 100},
        )
        assert result_path is not None
        assert result_mime == "image/jpeg"
        os.unlink(result_path)

    def test_max_size_mb_in_integration(self, tmp_dir):
        """MemoryStore respects max_size_mb from config."""
        cfg = {"enabled": True, "max_size_mb": 1}  # 1MB limit
        db_path = os.path.join(tmp_dir, "test_maxsize.db")
        store = MemoryStore(db_path, hrr_dim=128, compression_config=cfg)

        fact = store.add_fact("Big file test", category="test")
        fact_id = fact["fact_id"]

        # The noisy PNG is ~900KB, under 1MB → should compress
        result = store.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
        )
        assert result["media_id"] > 0
        media_root = Path(store._media_dir)
        stored_path = media_root / result["file_path"]
        assert stored_path.suffix == ".jpg"  # compressed

        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Integration tests: store with compression ───────────────────────

class TestCompressionIntegration:
    """Test that attach_media works end-to-end with compression."""

    def test_compressed_store_image(self, store_compressed):
        """Storing with compression produces a JPEG file on disk."""
        fact = store_compressed.add_fact("A test image", category="test")
        fact_id = fact["fact_id"]
        media_root = Path(store_compressed._media_dir)

        result = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
            description="Test image with compression",
        )

        assert "media_id" in result
        assert result["dedup"] is False

        # The stored file should be a JPEG
        stored_path = media_root / result["file_path"]
        assert stored_path.exists()
        assert stored_path.suffix == ".jpg"

        # Verify it's a valid JPEG
        img = Image.open(str(stored_path))
        assert img.format == "JPEG"

        # Verify SHA-256 matches (compressed content)
        stored_sha = hashlib.sha256(open(stored_path, "rb").read()).hexdigest()
        assert result["sha256"] == stored_sha

        # Verify DB record
        media_rows = store_compressed.get_fact_media(fact_id)
        assert len(media_rows) == 1
        # MIME should have been updated to image/jpeg
        assert media_rows[0]["mime_type"] == "image/jpeg"
        # file_size should be the compressed size
        assert media_rows[0]["file_size"] == stored_path.stat().st_size

    def test_no_compression_when_disabled(self, store_uncompressed):
        """With compression disabled, original PNG is stored as-is."""
        fact = store_uncompressed.add_fact("A test image no compress", category="test")
        fact_id = fact["fact_id"]
        media_root = Path(store_uncompressed._media_dir)

        result = store_uncompressed.attach_media(
            fact_id=fact_id,
            source_path=IMAGE_PNG,
            mime_type="image/png",
            description="No compression",
        )

        # Stored file should be PNG
        stored_path = media_root / result["file_path"]
        assert stored_path.exists()
        assert stored_path.suffix == ".png"

        img = Image.open(str(stored_path))
        assert img.format == "PNG"

        # SHA should match original PNG
        original_sha = hashlib.sha256(open(IMAGE_PNG, "rb").read()).hexdigest()
        assert result["sha256"] == original_sha

        # MIME should remain image/png
        media_rows = store_uncompressed.get_fact_media(fact_id)
        assert len(media_rows) == 1
        assert media_rows[0]["mime_type"] == "image/png"

    def test_dedup_with_compression(self, store_compressed):
        """Same source file attached twice dedup's on compressed content."""
        fact = store_compressed.add_fact("Dedup test", category="test")
        fact_id = fact["fact_id"]

        r1 = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
        )
        r2 = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
        )

        assert r1["sha256"] == r2["sha256"]
        assert r2["dedup"] is True
        assert r1["media_id"] == r2["media_id"]

    def test_compression_smaller_than_original(self, store_compressed, tmp_dir):
        """Compressed file should be measurably smaller for typical PNG."""
        fact = store_compressed.add_fact("Size comparison", category="test")
        fact_id = fact["fact_id"]

        result = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
        )

        # For the noisy PNG (900KB), JPEG compression should save ~75%
        media_root = Path(store_compressed._media_dir)
        stored_path = media_root / result["file_path"]
        orig_size = os.path.getsize(TEST_NOISY_PNG)
        stored_size = stored_path.stat().st_size

        assert stored_size < orig_size * 0.8, (
            f"Expected <80% of original, got {orig_size}→{stored_size}"
        )

    def test_compression_config_partial_override(self, tmp_dir):
        """Partial compression config overrides default."""
        # Only set enabled + image quality, rest should use defaults
        cfg = {
            "enabled": True,
            "image": {"quality": 50},
        }
        db_path = os.path.join(tmp_dir, "test_partial.db")
        store = MemoryStore(db_path, hrr_dim=128, compression_config=cfg)

        fact = store.add_fact("Partial config test", category="test")
        fact_id = fact["fact_id"]

        result = store.attach_media(
            fact_id=fact_id,
            source_path=TEST_NOISY_PNG,
            mime_type="image/png",
        )
        assert result["media_id"] > 0

        media_root = Path(store._media_dir)
        stored_path = media_root / result["file_path"]
        assert stored_path.exists()
        assert stored_path.suffix == ".jpg"

        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_compressed_store_audio(self, store_compressed):
        """Store an audio file with compression enabled."""
        fact = store_compressed.add_fact("A test audio", category="test")
        fact_id = fact["fact_id"]
        media_root = Path(store_compressed._media_dir)

        result = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=AUDIO_MP3,
            mime_type="audio/mpeg",
            description="Test audio with compression",
        )

        assert "media_id" in result
        stored_path = media_root / result["file_path"]
        assert stored_path.exists()

        # Should be an mp3 file
        assert stored_path.suffix == ".mp3"

        # Should be smaller than original (320kbps → 128kbps)
        orig_size = os.path.getsize(AUDIO_MP3)
        stored_size = stored_path.stat().st_size
        assert stored_size < orig_size * 0.8, (
            f"Audio compression not effective: {orig_size}→{stored_size}"
        )

        # DB record should have audio/mpeg
        media_rows = store_compressed.get_fact_media(fact_id)
        assert len(media_rows) == 1
        assert media_rows[0]["mime_type"] == "audio/mpeg"

    def test_compressed_store_video(self, store_compressed):
        """Store a video file with compression enabled."""
        fact = store_compressed.add_fact("A test video", category="test")
        fact_id = fact["fact_id"]
        media_root = Path(store_compressed._media_dir)

        result = store_compressed.attach_media(
            fact_id=fact_id,
            source_path=TEST_VIDEO,
            mime_type="video/mp4",
            description="Test video with compression",
        )

        assert "media_id" in result
        stored_path = media_root / result["file_path"]
        assert stored_path.exists()
        assert stored_path.suffix == ".mp4"

        # Should be significantly smaller
        orig_size = os.path.getsize(TEST_VIDEO)
        stored_size = stored_path.stat().st_size
        assert stored_size < orig_size * 0.5, (
            f"Video compression not effective: {orig_size}→{stored_size}"
        )

        # DB record
        media_rows = store_compressed.get_fact_media(fact_id)
        assert len(media_rows) == 1
        assert media_rows[0]["mime_type"] == "video/mp4"

    def test_no_config_defaults_to_no_compression(self, tmp_dir):
        """MemoryStore without compression_config should not compress."""
        db_path = os.path.join(tmp_dir, "test_no_cfg.db")
        store = MemoryStore(db_path, hrr_dim=128)  # no compression_config

        fact = store.add_fact("No config test", category="test")
        fact_id = fact["fact_id"]

        result = store.attach_media(
            fact_id=fact_id,
            source_path=IMAGE_PNG,
            mime_type="image/png",
        )

        original_sha = hashlib.sha256(open(IMAGE_PNG, "rb").read()).hexdigest()
        assert result["sha256"] == original_sha  # no compression

        shutil.rmtree(tmp_dir, ignore_errors=True)
