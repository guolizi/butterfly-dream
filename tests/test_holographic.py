"""Tests for holographic.py — HRR vector engine."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from butterfly_dream import holographic as hrr


class TestEncodeAtom:
    def test_deterministic(self):
        """Same input → same vector every time."""
        v1 = hrr.encode_atom("butterfly", 1024)
        v2 = hrr.encode_atom("butterfly", 1024)
        assert np.allclose(v1, v2)

    def test_different_words_differ(self):
        """Different words → different vectors (no collision)."""
        v1 = hrr.encode_atom("butterfly", 256)
        v2 = hrr.encode_atom("dream", 256)
        assert not np.allclose(v1, v2)

    def test_output_range(self):
        """Phase values in [0, 2π)."""
        v = hrr.encode_atom("test", 512)
        assert v.min() >= 0.0
        assert v.max() < 2.0 * math.pi

    def test_variable_dimensions(self):
        """Works for various dimensions."""
        for dim in [64, 128, 256, 1024]:
            v = hrr.encode_atom("test", dim)
            assert len(v) == dim


class TestBindUnbind:
    def test_bind_unbind_roundtrip(self):
        """unbind(bind(a, b), b) ≈ a (up to numerical precision)."""
        dim = 256
        a = hrr.encode_atom("alice", dim)
        b = hrr.encode_atom("bob", dim)
        memory = hrr.bind(a, b)
        retrieved = hrr.unbind(memory, b)
        assert np.allclose(a, retrieved)

    def test_bind_unbind_identity(self):
        """unbind(bind(a, b), a) ≈ b."""
        dim = 256
        a = hrr.encode_atom("x", dim)
        b = hrr.encode_atom("y", dim)
        retrieved = hrr.unbind(hrr.bind(a, b), a)
        assert np.allclose(b, retrieved)

    def test_bind_is_symmetric(self):
        """bind(a, b) == bind(b, a)."""
        dim = 256
        a = hrr.encode_atom("aaa", dim)
        b = hrr.encode_atom("bbb", dim)
        assert np.allclose(hrr.bind(a, b), hrr.bind(b, a))

    def test_different_key_fails(self):
        """unbind with wrong key gives unrelated vector."""
        dim = 256
        a = hrr.encode_atom("secret", dim)
        b = hrr.encode_atom("key", dim)
        wrong = hrr.encode_atom("wrong", dim)
        memory = hrr.bind(a, b)
        noise = hrr.unbind(memory, wrong)
        sim = hrr.similarity(a, noise)
        assert sim < 0.3  # barely similar


class TestBundle:
    def test_bundle_self_similarity(self):
        """A bundled vector should be similar to each input."""
        dim = 256
        v1 = hrr.encode_atom("cat", dim)
        v2 = hrr.encode_atom("dog", dim)
        v3 = hrr.encode_atom("bird", dim)
        bundled = hrr.bundle(v1, v2, v3)
        assert hrr.similarity(bundled, v1) > 0.5  # HRR bundle: O(sqrt(dim)) capacity
        assert hrr.similarity(bundled, v2) > 0.5
        assert hrr.similarity(bundled, v3) > 0.4

    def test_bundle_single(self):
        """Bundling one vector returns something similar to it."""
        dim = 256
        v = hrr.encode_atom("lonely", dim)
        bundled = hrr.bundle(v)
        assert hrr.similarity(v, bundled) > 0.95

    def test_bundle_identity(self):
        """Bundling identical vectors stays close."""
        dim = 128
        v = hrr.encode_atom("same", dim)
        bundled = hrr.bundle(v, v, v)
        assert hrr.similarity(v, bundled) > 0.9


class TestSimilarity:
    def test_self_similarity(self):
        """Similarity of a vector to itself = 1.0."""
        v = hrr.encode_atom("me", 256)
        assert hrr.similarity(v, v) == pytest.approx(1.0, abs=1e-10)

    def test_unrelated(self):
        """Unrelated vectors have near-zero similarity."""
        v1 = hrr.encode_atom("hello", 512)
        v2 = hrr.encode_atom("world", 512)
        sim = hrr.similarity(v1, v2)
        assert abs(sim) < 0.2

    def test_range(self):
        """Similarity is always in [-1, 1]."""
        for i in range(10):
            a = hrr.encode_atom(f"word{i}", 128)
            b = hrr.encode_atom(f"other{i}", 128)
            sim = hrr.similarity(a, b)
            assert -1.0 <= sim <= 1.0


class TestEncodeText:
    def test_empty_text(self):
        """Empty text returns the sentinel vector."""
        v = hrr.encode_text("", 256)
        sentinel = hrr.encode_atom("__hrr_empty__", 256)
        assert hrr.similarity(v, sentinel) > 0.9

    def test_same_meaning_similar(self):
        """Similar sentences produce similar vectors."""
        v1 = hrr.encode_text("the cat sat on the mat", 256)
        v2 = hrr.encode_text("a cat sits on a mat", 256)
        sim = hrr.similarity(v1, v2)
        assert sim > 0.1, f"Expected positive similarity, got {sim}"  # bag-of-words with shared tokens

    def test_different_topics_differ(self):
        """Different topic → low similarity."""
        v1 = hrr.encode_text("quantum physics equations", 256)
        v2 = hrr.encode_text("baking chocolate chip cookies", 256)
        sim = hrr.similarity(v1, v2)
        assert sim < 0.3, f"Expected low similarity, got {sim}"


class TestEncodeFact:
    def test_structured_encoding(self):
        """encode_fact produces a valid vector."""
        v = hrr.encode_fact("Alice likes cats", ["Alice"], 256)
        assert len(v) == 256
        assert v.min() >= 0.0
        assert v.max() < 2.0 * math.pi

    def test_fact_has_entity_info(self):
        """With vs without entity → different (but both encode same text core)."""
        v_with = hrr.encode_fact("Project uses FastAPI", ["Project"], 256)
        v_without = hrr.encode_fact("Project uses FastAPI", [], 256)
        # Should be similar but not identical (entity binding adds info)
        sim = hrr.similarity(v_with, v_without)
        assert 0.3 < sim < 0.95, f"Expected moderate-high similarity, got {sim}"

    def test_multiple_entities(self):
        """Multiple entities all get encoded."""
        v = hrr.encode_fact("Alice and Bob work on Project X", ["Alice", "Bob", "Project X"], 256)
        assert len(v) == 256


class TestSerialization:
    def test_roundtrip_bytes(self):
        """Phase vector survives serialize → deserialize."""
        v = hrr.encode_atom("persist", 1024)
        data = hrr.phases_to_bytes(v)
        assert isinstance(data, bytes)
        assert len(data) == 1024 * 8  # float64 = 8 bytes
        v2 = hrr.bytes_to_phases(data)
        assert np.allclose(v, v2)

    def test_zero_length(self):
        """Dimension-0 vector produces empty bytes (not a ValueError)."""
        data = hrr.phases_to_bytes(np.array([], dtype=np.float64))
        assert data == b""

    def test_wrong_dtype(self):
        """Convert back and verify dtype is float64."""
        v = hrr.encode_atom("dtype", 256)
        data = hrr.phases_to_bytes(v)
        v2 = hrr.bytes_to_phases(data)
        assert v2.dtype == np.float64


class TestSnr:
    def test_infinite_for_zero_items(self):
        """SNR is ∞ when there are 0 items."""
        assert hrr.snr_estimate(1024, 0) == float("inf")

    def test_positive_snr(self):
        """SNR > 0 for valid inputs."""
        snr = hrr.snr_estimate(1024, 10)
        assert snr > 0

    def test_warning_threshold(self):
        """SNR < 2 when capacity is exceeded."""
        snr = hrr.snr_estimate(64, 20)  # n_items > dim/4
        assert snr < 2.0
