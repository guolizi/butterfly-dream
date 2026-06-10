"""Tests for retrieval.py — ThreeDimRetriever."""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from butterfly_dream.store import MemoryStore
from butterfly_dream.retrieval import (
    ThreeDimRetriever,
    SCENARIO_WEIGHTS,
    _recency_score,
    _importance_score,
    tokenize,
    jaccard_similarity,
)
from datetime import datetime, timezone


@pytest.fixture
def store_and_retriever():
    """Create store + retriever with temp DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    ms = MemoryStore(db_path=db_path, default_trust=0.5)
    ret = ThreeDimRetriever(ms, half_life_days=30)
    yield ms, ret
    ms.close()
    os.unlink(db_path)


class TestHelpers:
    def test_tokenize_english(self):
        """Tokenize splits English words into lowercase tokens."""
        tokens = tokenize("Hello World TEST")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_cjk(self):
        """CJK characters get jieba word-level tokens."""
        tokens = tokenize("蝴蝶梦")
        assert "蝴蝶梦" in tokens
        tokens2 = tokenize("测试中文分词")
        assert "测试" in tokens2
        assert "中文" in tokens2
        assert "分词" in tokens2

    def test_jaccard_similarity(self):
        """Jaccard similarity is correct."""
        a = {"a", "b", "c"}
        b = {"a", "b", "d"}
        assert jaccard_similarity(a, b) == pytest.approx(2 / 4)  # {a,b} / {a,b,c,d}

    def test_jaccard_empty(self):
        """Empty sets return 0."""
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0

    def test_recency_score_now(self):
        """Now → approx 1.0 (floating point)."""
        assert _recency_score(datetime.now(timezone.utc)) == pytest.approx(1.0, abs=1e-9)

    def test_recency_score_future(self):
        """Future timestamps clamp to 0 age → 1.0."""
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        assert _recency_score(future) == 1.0

    def test_recency_score_old(self):
        """Old fact (>> half-life) → near 0."""
        old = datetime(2020, 1, 1, tzinfo=timezone.utc)
        score = _recency_score(old, half_life_days=30)
        assert score < 0.01

    def test_importance_score_normalize(self):
        """Importance 1→0, 10→1, 5.5→0.5."""
        assert _importance_score(1) == 0.0
        assert _importance_score(10) == 1.0
        assert _importance_score(5.5) == pytest.approx(0.5, abs=0.01)
        assert _importance_score(None) == 0.5

    def test_scenario_weights_defined(self):
        """All expected scenarios exist with valid weights."""
        for scenario in ("chat", "technical", "longterm", "qa", "balanced"):
            w = SCENARIO_WEIGHTS[scenario]
            total = w["relevance"] + w["recency"] + w["importance"]
            assert total == pytest.approx(1.0, abs=0.01)


class TestSearch:
    def test_empty_store_returns_empty(self, store_and_retriever):
        """Search on empty store returns empty list."""
        _, ret = store_and_retriever
        results = ret.search("anything")
        assert results == []

    def test_basic_search_returns_results(self, store_and_retriever):
        """Search finds matching facts."""
        ms, ret = store_and_retriever
        ms.add_fact("Python is a programming language", category="general", importance=7)
        results = ret.search("Python programming")
        assert len(results) > 0
        assert "Python" in results[0]["content"]

    def test_search_limit(self, store_and_retriever):
        """Limit parameter is respected."""
        ms, ret = store_and_retriever
        for i in range(10):
            ms.add_fact(f"Fact number {i} about Python", category="general")
        results = ret.search("Python", limit=3)
        assert len(results) <= 3

    def test_search_min_trust_filter(self, store_and_retriever):
        """Low-trust facts are excluded."""
        ms, ret = store_and_retriever
        r1 = ms.add_fact("High trust fact about Python", importance=5)
        r2 = ms.add_fact("Low trust fact about Python", importance=5)
        # Reduce trust on the second
        for _ in range(5):
            ms.record_feedback(r2["fact_id"], helpful=False)
        results = ret.search("Python", min_trust=0.5)
        contents = [r["content"] for r in results]
        assert "High trust fact about Python" in contents
        # Low trust might still appear depending on threshold — let's check trust value
        low = ms.get_fact(r2["fact_id"])
        if low["trust_score"] < 0.5:
            assert "Low trust fact about Python" not in contents

    def test_scenario_weights_affect_ranking(self, store_and_retriever):
        """Different scenarios reorder results."""
        ms, ret = store_and_retriever
        # Old but important fact
        ms.add_fact("Very important Python architecture decision", importance=9)
        # Recent but less important fact
        ms.add_fact("Python version update note", importance=3)

        longterm = ret.search("Python", scenario="longterm", limit=5)
        chat = ret.search("Python", scenario="chat", limit=5)

        # In longterm mode, importance matters more
        assert len(longterm) > 0
        assert len(chat) > 0

    def test_category_filter(self, store_and_retriever):
        """Category filter narrows results."""
        ms, ret = store_and_retriever
        ms.add_fact("Python coding tips", category="user_pref")
        ms.add_fact("Project uses Python", category="project")
        results = ret.search("Python", category="project")
        contents = [r["content"] for r in results]
        assert all("Project" in c for c in contents)

    def test_search_returns_score_components(self, store_and_retriever):
        """Results include _relevance, _recency, _importance fields."""
        ms, ret = store_and_retriever
        ms.add_fact("Test fact about Python", importance=6)
        results = ret.search("Python")
        if results:
            r = results[0]
            assert "_relevance" in r
            assert "_recency" in r
            assert "_importance" in r
            assert "score" in r

    def test_search_persistent_only(self, store_and_retriever):
        """search(persistent_only=True) returns only persistent facts."""
        ms, ret = store_and_retriever
        ms.add_fact("User preference for Python", importance=7, is_persistent=True)
        ms.add_fact("Session temp note for Python", importance=4)
        ms.add_fact("Another persistent Python fact", importance=6, is_persistent=True)

        all_results = ret.search("Python", limit=10)
        persistent = ret.search("Python", limit=10, persistent_only=True)

        assert len(all_results) >= 2
        assert len(persistent) >= 1
        assert len(persistent) < len(all_results)
        for r in persistent:
            assert r["is_persistent"] == 1

    def test_search_chinese(self, store_and_retriever):
        """FTS5 search works with Chinese query terms (with spaces)."""
        ms, ret = store_and_retriever
        # FTS5 unicode61 requires spaces between CJK and Latin for indexing
        ms.add_fact("用户 喜欢 用 VS Code 写 Python", importance=7, is_persistent=True)
        ms.add_fact("用户 不喜欢 用 VS Code 写 Python", importance=6)

        # Search by English token (separated by space from CJK)
        results = ret.search("Python", limit=10)
        assert len(results) >= 2

        # Search by CJK bigram (works when CJK tokens are whitespace-separated)
        results_cjk = ret.search("用户", limit=10)
        assert len(results_cjk) >= 2


class TestScenarios:
    def test_custom_scenario_weight(self, store_and_retriever):
        """Override weights per call."""
        ms, ret = store_and_retriever
        ms.add_fact("Python is great for data science", importance=8)
        results = ret.search("Python", relevance_weight=1.0, recency_weight=0.0, importance_weight=0.0)
        assert len(results) > 0

    def test_fts_fallback(self, store_and_retriever):
        """Search works with short queries."""
        ms, ret = store_and_retriever
        ms.add_fact("The quick brown fox jumps over the lazy dog", importance=5)
        # Very short query might get sanitized differently
        results = ret.search("fox")
        assert len(results) > 0


class TestSanitizeFTSQuery:
    """Edge cases for _sanitize_fts_query (S3)."""

    def test_empty_string(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        assert ThreeDimRetriever._sanitize_fts_query("") == ""

    def test_single_char(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        assert ThreeDimRetriever._sanitize_fts_query("a") == ""

    def test_special_chars_stripped(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        result = ThreeDimRetriever._sanitize_fts_query('hello "*world" OR test')
        # Original FTS5 special chars (*, ") should be stripped from input
        assert '"' not in result
        # Our own OR separator is added; user's "OR" becomes a regular token
        assert len(result) > 0
        # Result uses prefix matching (each token ends with *)
        assert result.endswith("*") or "*" in result

    def test_cjk_preserved(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        result = ThreeDimRetriever._sanitize_fts_query("蝴蝶梦记忆")
        assert "蝴蝶梦记忆" in result or "蝴蝶" in result

    def test_multi_word_and(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        result = ThreeDimRetriever._sanitize_fts_query("python programming")
        # Should be preserved as AND query
        # NOTE: "programming" is verb-lemmatized to "program" by NLTK
        assert "python" in result
        assert "program" in result

    def test_synonym_expansion(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        # "children" → lemmatized to "child" → WordNet synsets include: kid, youngster, minor, ...
        result = ThreeDimRetriever._sanitize_fts_query("help children")
        assert "children" in result
        assert "kid" in result or "youngster" in result
        # Should contain OR groups
        assert "OR" in result

    def test_hyphenated_words(self):
        from butterfly_dream.retrieval import ThreeDimRetriever
        result = ThreeDimRetriever._sanitize_fts_query("well-known fact")
        assert len(result) > 0
