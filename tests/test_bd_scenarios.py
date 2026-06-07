"""Tests for Butterfly Dream extraction + retrieval pipeline using bd_eval scenarios.

Two test levels:
- Unit tests (default): mock LLM extraction, test pipeline integration
- Integration tests (@pytest.mark.integration): call real LLM (needs API key)
"""
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.store import MemoryStore
from butterfly_dream.retrieval import ThreeDimRetriever
from tests.bd_scenarios import SINGLE_SESSION_SCENARIOS, MULTI_SESSION_SCENARIOS


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_db():
    """Create a temp DB path, yield, then clean up."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


# ==============================================================================
# Mock extraction: return canned facts based on conversation content
# ==============================================================================

def _mock_extract(conversation, *args, **kwargs):
    """Return canned facts based on conversation content."""
    full_text = " ".join(m["content"] for m in conversation).lower()

    facts = []
    
    # Determine if Chinese or English
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in full_text)
    
    # Chinese: 编程语言偏好
    if has_cjk and "rust" in full_text and "python" in full_text:
        facts.extend([
            {"content": "用户之前用Python写后端开发", "category": "preference", "importance": 8.0},
            {"content": "上个月从Python切换到Rust", "category": "preference", "importance": 9.0},
            {"content": "用户认为Rust比Python快", "category": "preference", "importance": 7.0},
        ])
    # Chinese: 工作与居住地
    elif has_cjk and "东京" in full_text and "google" in full_text:
        facts.extend([
            {"content": "用户搬到了日本东京生活", "category": "personal", "importance": 9.0},
            {"content": "用户在Google东京办公室工作", "category": "work", "importance": 9.0},
            {"content": "团队在涩谷做搜索相关开发", "category": "work", "importance": 8.0},
        ])
    # Chinese: 技术栈
    elif has_cjk and ("fastapi" in full_text or "postgresql" in full_text):
        facts.extend([
            {"content": "新项目后端使用FastAPI框架", "category": "project", "importance": 8.0},
            {"content": "新项目前端使用React框架", "category": "project", "importance": 8.0},
            {"content": "项目使用PostgreSQL数据库", "category": "project", "importance": 7.0},
            {"content": "项目部署在AWS ECS上使用Docker", "category": "project", "importance": 7.0},
        ])
    # Chinese: 宠物
    elif has_cjk and ("橘猫" in full_text or "金毛" in full_text):
        facts.extend([
            {"content": "用户养了一只三岁的橘猫叫小胖", "category": "personal", "importance": 8.0},
            {"content": "用户还养了一条金毛犬叫旺财", "category": "personal", "importance": 8.0},
            {"content": "小胖最喜欢吃鱼", "category": "personal", "importance": 7.0},
        ])
    # Chinese: 工作变更
    elif has_cjk and ("微软" in full_text or "字节跳动" in full_text):
        facts.extend([
            {"content": "用户在微软做Azure开发", "category": "work", "importance": 8.0},
            {"content": "用户主要负责Kubernetes相关服务", "category": "work", "importance": 8.0},
            {"content": "用户跳槽到字节跳动做TikTok推荐系统", "category": "work", "importance": 9.0},
            {"content": "用户的技术栈包含Go和Rust", "category": "work", "importance": 7.0},
        ])
    # English: language preference
    elif not has_cjk and "rust" in full_text and "python" in full_text:
        facts.extend([
            {"content": "User previously used Python for backend development", "category": "preference", "importance": 8.0},
            {"content": "User switched from Python to Rust last month", "category": "preference", "importance": 9.0},
            {"content": "User thinks Rust is faster than Python", "category": "preference", "importance": 7.0},
        ])
    # English: work/location
    elif not has_cjk and "tokyo" in full_text and "google" in full_text:
        facts.extend([
            {"content": "User moved to Tokyo, Japan", "category": "personal", "importance": 9.0},
            {"content": "User works at Google's Tokyo office", "category": "work", "importance": 9.0},
            {"content": "User's team works on search in Shibuya", "category": "work", "importance": 8.0},
        ])
    # English: tech stack
    elif not has_cjk and "fastapi" in full_text and "react" in full_text:
        facts.extend([
            {"content": "New project backend uses FastAPI framework", "category": "project", "importance": 8.0},
            {"content": "New project frontend uses React", "category": "project", "importance": 8.0},
            {"content": "Project uses PostgreSQL database", "category": "project", "importance": 7.0},
            {"content": "Project is deployed on AWS with Docker", "category": "project", "importance": 7.0},
        ])
    # English: pets
    elif not has_cjk and ("tabby" in full_text or ("cat" in full_text and "golden" in full_text)):
        facts.extend([
            {"content": "User has a three-year-old orange tabby cat named Mochi", "category": "personal", "importance": 8.0},
            {"content": "User also has a golden retriever named Biscuit", "category": "personal", "importance": 8.0},
            {"content": "Mochi loves eating fish", "category": "personal", "importance": 7.0},
        ])
    # English: health
    elif not has_cjk and "allergy" in full_text:
        facts.extend([
            {"content": "User has a severe nut allergy", "category": "health", "importance": 8.0},
            {"content": "User is lactose intolerant", "category": "health", "importance": 7.0},
            {"content": "User takes vitamin D supplements daily", "category": "health", "importance": 7.0},
        ])
    # English: schedule
    elif not has_cjk and "monday" in full_text and "friday" in full_text:
        facts.extend([
            {"content": "User works from home on Mondays and Fridays", "category": "work", "importance": 7.0},
            {"content": "User runs 5k every morning before work", "category": "habit", "importance": 7.0},
            {"content": "User drinks a protein smoothie after running", "category": "habit", "importance": 7.0},
            {"content": "User goes hiking in the mountains on weekends", "category": "habit", "importance": 7.0},
        ])
    # English: job change
    elif not has_cjk and "microsoft" in full_text and "meta" in full_text:
        facts.extend([
            {"content": "User worked at Microsoft on Azure", "category": "work", "importance": 8.0},
            {"content": "User mainly handled Kubernetes-related services", "category": "work", "importance": 8.0},
            {"content": "User switched to Meta working on Instagram recommendation", "category": "work", "importance": 9.0},
            {"content": "User's tech stack includes Go and Rust", "category": "work", "importance": 7.0},
        ])

    if not facts:
        facts.append({"content": f"User mentioned: {full_text[:80]}", "category": "general", "importance": 5.0})

    return facts


def _assert_fact_covers_golden(fact_contents, golden_fact):
    """Check if any stored fact contains the core meaning of a golden fact.
    
    For Chinese text: check substring overlap (Chinese has no word boundaries).
    For English: check that key content words from the golden fact appear.
    """
    gf_lower = golden_fact.lower()
    
    # Check each stored fact's content against golden fact
    for fc in fact_contents:
        fc_lower = fc.lower()
        
        # Chinese: check if the stored fact contains major parts of golden fact
        # by comparing n-gram overlap for Chinese, or direct substring for English
        if len(gf_lower) <= 15:
            # Short golden fact → direct substring match
            if gf_lower in fc_lower or fc_lower in gf_lower:
                return True
        else:
            # Long golden fact → check key content words
            # For Chinese: count shared characters
            if any(c in fc_lower for c in gf_lower if '\u4e00' <= c <= '\u9fff'):
                # Check substantial overlap for Chinese
                shared_chars = sum(1 for c in set(gf_lower) if '\u4e00' <= c <= '\u9fff' and c in fc_lower)
                total_cjk = sum(1 for c in set(gf_lower) if '\u4e00' <= c <= '\u9fff')
                if total_cjk > 0 and shared_chars / total_cjk >= 0.4:
                    return True
            
            # For English: check key words
            stopwords = {'the','a','an','in','on','at','to','for','of','and','or','is','was','are','has','have','had','be','been','being','with','from','by','as','but','not','no','user','his','her','its','their','my','your','our'}
            en_words = [w.strip(".,;:!?'\"()") for w in gf_lower.split()
                       if w.strip(".,;:!?'\"()") not in stopwords and len(w) > 2]
            if en_words:
                match_count = sum(1 for w in en_words if w in fc_lower)
                if match_count >= max(1, len(en_words) // 3):
                    return True
    
    return False


# ==============================================================================
# Unit tests: mock the LLM extraction, test pipeline integration
# ==============================================================================

class TestScenarioPipeline:
    """Test the full extract → store → retrieve pipeline with mocked LLM."""

    @pytest.mark.parametrize("scenario", SINGLE_SESSION_SCENARIOS,
                             ids=[s["name"][:30] for s in SINGLE_SESSION_SCENARIOS])
    def test_single_session_extract_and_query(self, scenario, temp_db):
        """Extract facts from a scenario, store them, then verify queries retrieve correct info."""
        config = {
            "db_path": temp_db,
            "llm_extract": True,
            "trivial_filter": False,
            "reflection": False,
        }
        provider = ButterflyDreamMemoryProvider(config)
        provider.initialize(session_id="test-bd-scenarios")
        assert provider._store is not None

        # Mock the LLM extraction
        with patch.object(provider, "_run_llm_extraction",
                          side_effect=_mock_extract):
            conv = scenario.get("conversation", [])
            extracted = provider._run_llm_extraction(conv)

            # Store the facts
            for fact in extracted:
                provider._store.add_fact(
                    content=fact["content"],
                    category=fact.get("category", "general"),
                    importance=float(fact.get("importance", 5.0)),
                )

        # Verify facts are stored
        all_facts = provider._store.list_facts(limit=100)
        fact_contents = [f["content"] for f in all_facts]
        assert len(fact_contents) > 0, f"No facts stored for {scenario['name']}"

        # Verify golden facts are covered
        for gf in scenario["golden_facts"]:
            assert _assert_fact_covers_golden(fact_contents, gf), \
                f"Golden fact not covered in stored facts for {scenario['name']}: {gf}"

        # Verify queries can retrieve relevant facts
        retriever = ThreeDimRetriever(provider._store)
        for query_str, expected_keywords in scenario.get("queries", []):
            results = retriever.search(query=query_str, scenario="chat", limit=5)
            assert len(results) > 0, f"No results for query: {query_str}"

            # Check if expected keywords appear in top results
            top_content = " ".join(r.get("content", "") for r in results[:3]).lower()
            found = any(kw.lower() in top_content for kw in expected_keywords)
            assert found, \
                f"Query '{query_str}' didn't find {expected_keywords} in top-3 for {scenario['name']}"

        provider.shutdown()

    def test_conversation_content_preserved(self, temp_db):
        """Verify fact extraction preserves conversation facts across storage."""
        config = {
            "db_path": temp_db,
            "llm_extract": True,
            "trivial_filter": False,
            "reflection": False,
        }
        provider = ButterflyDreamMemoryProvider(config)
        provider.initialize(session_id="test-content")
        assert provider._store is not None

        with patch.object(provider, "_run_llm_extraction") as mock_extract:
            mock_extract.return_value = [
                {"content": "User lives in Tokyo and works at Google", "category": "personal", "importance": 8.0},
                {"content": "User's team works on search technology", "category": "work", "importance": 7.0},
            ]
            extracted = provider._run_llm_extraction([
                {"role": "user", "content": "test conversation"}
            ])
            for fact in extracted:
                provider._store.add_fact(**fact)

        facts = provider._store.list_facts(limit=10)
        contents = {f["content"] for f in facts}
        assert "User lives in Tokyo and works at Google" in contents
        assert "User's team works on search technology" in contents

        provider.shutdown()


class TestMultiSession:
    """Cross-session extraction and retrieval."""

    def test_multi_session_facts_merged(self, temp_db):
        """Facts from multiple sessions coexist in the same store."""
        config = {
            "db_path": temp_db,
            "llm_extract": True,
            "trivial_filter": False,
            "reflection": False,
        }
        provider = ButterflyDreamMemoryProvider(config)
        provider.initialize(session_id="test-multi")
        assert provider._store is not None

        # Collect ALL conversations from ALL multi-session scenarios
        all_convs = []
        for scenario in MULTI_SESSION_SCENARIOS:
            for session in scenario["sessions"]:
                all_convs.append(session["conversation"])

        with patch.object(provider, "_run_llm_extraction",
                          side_effect=_mock_extract):
            for conv in all_convs:
                extracted = provider._run_llm_extraction(conv)
                for fact in extracted:
                    provider._store.add_fact(**fact)

        all_facts = provider._store.list_facts(limit=100)
        contents = " ".join(f["content"] for f in all_facts).lower()

        # Should have facts from all scenarios
        assert len(all_facts) >= 4, f"Expected >= 4 facts, got {len(all_facts)}"
        assert "microsoft" in contents or "azure" in contents or "微软" in contents
        assert "meta" in contents or "instagram" in contents or "字节跳动" in contents

        provider.shutdown()


# ==============================================================================
# Integration tests: real LLM extraction (requires API key)
# ==============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"),
                    reason="No OPENROUTER_API_KEY in environment")
class TestRealExtraction:
    """End-to-end tests that call the real LLM for extraction."""

    @pytest.mark.parametrize("scenario", SINGLE_SESSION_SCENARIOS[:2],
                             ids=lambda s: s["name"][:25])
    def test_real_llm_extraction(self, scenario, temp_db):
        """Call real LLM to extract facts from conversation, then verify."""
        from eval.eval_utils import get_model_config
        config = {
            "db_path": temp_db,
            "llm_extract": True,
            "extraction_model": get_model_config("extraction"),
            "trivial_filter": False,
            "reflection": False,
        }
        provider = ButterflyDreamMemoryProvider(config)
        provider.initialize(session_id="test-integration")
        assert provider._store is not None

        # Real extraction call
        conv = scenario.get("conversation", [])
        extracted = provider._run_llm_extraction(conv)

        # Store results
        for fact in extracted:
            provider._store.add_fact(
                content=fact["content"],
                category=fact.get("category", "general"),
                importance=float(fact.get("importance", 5.0)),
            )

        # Verify
        all_facts = provider._store.list_facts(limit=50)
        assert len(all_facts) > 0

        # Check query recall
        retriever = ThreeDimRetriever(provider._store)
        for query_str, expected in scenario.get("queries", []):
            results = retriever.search(query=query_str, scenario="chat", limit=5)
            if results:
                top = results[0].get("content", "").lower()
                found = any(kw.lower() in top for kw in expected)
                if not found:
                    print(f"  ⚠️  Query '{query_str}' expected {expected} but got: {results[0].get('content','')[:80]}")

        provider.shutdown()
