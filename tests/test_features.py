"""Tests for Butterfly Dream features: trivial filter, circuit breaker, reflection."""
import sys
import os
import json
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from butterfly_dream import ButterflyDreamMemoryProvider
from butterfly_dream.__init__ import (
    _TRIVIAL_PATTERNS,
    _CB_DEFAULT_MAX_FAILURES,
    _CB_DEFAULT_COOLDOWN,
    _REFLECTION_FREQUENCY,
    _call_extraction_llm,
)


# ==============================================================================
# Trivial message filter
# ==============================================================================

class TestTrivialFilter:
    """_is_trivial_content recognizes low-information messages."""

    @pytest.mark.parametrize("msg", [
        "ok", "okay", "OK", "Ok", "好的", "好", "嗯", "嗯嗯",
        "thanks", "Thank you", "thx", "谢谢", "多谢", "感谢",
        "got it", "gotcha", "understood", "理解", "懂了", "明白",
        "yes", "yep", "yeah", "yup", "no", "nope", "不是", "不对",
        "hello", "hi", "hey", "你好", "您好", "嗨",
        "nice", "great", "good", "确实", "不错", "好吧",
        "试一下", "试试", "先这样", "差不多了",
    ])
    def test_trivial_messages(self, msg):
        assert ButterflyDreamMemoryProvider._is_trivial_content(msg), f"'{msg}' should be trivial"

    @pytest.mark.parametrize("msg", [
        "I prefer VS Code for Python development",
        "The project uses FastAPI with SQLAlchemy",
        "我们今天讨论一下架构设计方案",
        "Can you help me debug this issue?",
        "The server is running on port 8080",
        "这个配置需要改一下才能用",
        "Let me check the database schema first",
        "Hey, can you look at this error?",
        "ok let me check that",
        "好的我来看一下这个问题",
    ])
    def test_non_trivial_messages(self, msg):
        assert not ButterflyDreamMemoryProvider._is_trivial_content(msg), f"'{msg}' should NOT be trivial"

    def test_trivial_with_whitespace(self):
        assert ButterflyDreamMemoryProvider._is_trivial_content("  ok  ")
        assert ButterflyDreamMemoryProvider._is_trivial_content("好的！")
        assert ButterflyDreamMemoryProvider._is_trivial_content("thanks~")

    def test_trivial_disabled(self):
        """When disabled, all content passes through."""
        provider = ButterflyDreamMemoryProvider({"trivial_filter": False, "llm_extract": False})
        assert provider._trivial_filter_enabled is False


# ==============================================================================
# Circuit breaker
# ==============================================================================

class TestCircuitBreaker:
    """_circuit_breaker_ok and _mark_extraction_result work correctly."""

    def test_initial_state(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        assert provider._circuit_breaker_ok() is True
        assert provider._extraction_failures == 0

    def test_success_resets(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        provider._extraction_failures = 2
        provider._mark_extraction_result(True)
        assert provider._extraction_failures == 0
        assert provider._circuit_breaker_ok() is True

    def test_failure_increments(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        provider._mark_extraction_result(False)
        assert provider._extraction_failures == 1
        assert provider._circuit_breaker_ok() is True

    def test_trips_after_max(self):
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 3, "cooldown_seconds": 60},
        })
        provider._mark_extraction_result(False)
        provider._mark_extraction_result(False)
        provider._mark_extraction_result(False)
        assert provider._extraction_failures == 3
        assert provider._circuit_breaker_ok() is False

    def test_recovers_after_cooldown(self):
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 2, "cooldown_seconds": 60},
        })
        provider._mark_extraction_result(False)
        provider._mark_extraction_result(False)
        assert provider._circuit_breaker_ok() is False

        # Set cooldown in the past
        provider._cooldown_until = time.time() - 1
        assert provider._circuit_breaker_ok() is True
        assert provider._extraction_failures == 0

    def test_configurable_threshold(self):
        """Custom max_failures from config."""
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 10},
        })
        assert provider._cb_max_failures == 5
        assert provider._cb_cooldown == 10

        for _ in range(3):
            provider._mark_extraction_result(False)
        assert provider._circuit_breaker_ok() is True

        for _ in range(2):
            provider._mark_extraction_result(False)
        assert provider._circuit_breaker_ok() is False

    def test_defaults_from_constants(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        assert provider._cb_max_failures == _CB_DEFAULT_MAX_FAILURES
        assert provider._cb_cooldown == _CB_DEFAULT_COOLDOWN


# ==============================================================================
# Reflection
# ==============================================================================

class TestReflection:
    """_run_reflection doesn't crash and respects guard conditions."""

    def test_no_store_returns_gracefully(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        provider._store = None
        provider._run_reflection()  # should not raise

    def test_reflection_config_defaults(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        assert provider._reflection_enabled is True
        assert _REFLECTION_FREQUENCY == 5

    def test_reflection_disabled(self):
        provider = ButterflyDreamMemoryProvider({"llm_extract": False, "reflection": False})
        assert provider._reflection_enabled is False

    def test_reflection_trigger_count(self):
        """on_session_end triggers reflection every _REFLECTION_FREQUENCY extractions."""
        provider = ButterflyDreamMemoryProvider({"llm_extract": False})
        provider._extraction_count = 3
        assert provider._extraction_count % _REFLECTION_FREQUENCY != 0

        provider._extraction_count = 5
        assert provider._extraction_count % _REFLECTION_FREQUENCY == 0

        provider._extraction_count = 10
        assert provider._extraction_count % _REFLECTION_FREQUENCY == 0


# ==============================================================================
# Integration: extraction flow with trivial filter + circuit breaker
# ==============================================================================

class TestExtractionFlow:
    """_run_llm_extraction respects trivial filter and circuit breaker."""

    def test_trivial_messages_skipped_in_extraction(self, monkeypatch):
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "trivial_filter": True,
        })

        called_lines = []

        def mock_llm(messages_text, **kwargs):
            called_lines.append(messages_text)
            return []

        monkeypatch.setattr("butterfly_dream._call_extraction_llm", mock_llm)

        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "thanks"},
        ]

        result = provider._run_llm_extraction(messages)
        assert result == []

        # All messages are trivial - LLM should not have been called
        # (because the guard at < 2 non-trivial lines triggers early return)
        # OR if called, the text shouldn't contain trivial phrases
        if called_lines:
            text = called_lines[0]
            for phrase in ("hi", "hello", "ok", "thanks"):
                assert phrase not in text, f"Trivial phrase '{phrase}' should not appear in LLM input"

    def test_circuit_breaker_blocks_llm_call(self, monkeypatch):
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 2, "cooldown_seconds": 60},
        })

        called = False

        def mock_llm(**kwargs):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr("butterfly_dream._call_extraction_llm", mock_llm)

        provider._extraction_failures = 2
        provider._cooldown_until = time.time() + 60

        messages = [
            {"role": "user", "content": "I prefer VS Code for development"},
            {"role": "assistant", "content": "Let me help you set that up"},
        ]

        result = provider._run_llm_extraction(messages)
        assert result == []
        assert called is False, "LLM should not be called when circuit breaker is active"

    def test_circuit_breaker_allows_after_cooldown(self, monkeypatch):
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 2, "cooldown_seconds": 60},
        })

        called = False

        def mock_llm(**kwargs):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr("butterfly_dream._call_extraction_llm", mock_llm)

        provider._extraction_failures = 2
        provider._cooldown_until = time.time() - 1

        messages = [
            {"role": "user", "content": "I prefer VS Code for development"},
            {"role": "assistant", "content": "Let me help you set that up"},
        ]

        result = provider._run_llm_extraction(messages)
        assert called is True, "LLM should be called after cooldown expires"

    def test_reflection_respects_circuit_breaker(self, monkeypatch):
        """Reflection should skip LLM call when circuit breaker is active."""
        provider = ButterflyDreamMemoryProvider({
            "llm_extract": False,
            "circuit_breaker": {"max_failures": 1, "cooldown_seconds": 60},
        })

        called = False

        def mock_llm(**kwargs):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr("butterfly_dream._call_extraction_llm", mock_llm)

        # Trip the circuit breaker
        provider._mark_extraction_result(False)
        assert provider._circuit_breaker_ok() is False

        # Reflection should not call LLM
        provider._run_reflection()
        assert called is False, "Reflection should not call LLM when circuit breaker is active"


# ==============================================================================
# LLM response parsing
# ==============================================================================

class TestLLMResponseParsing:
    """_call_extraction_llm correctly parses various LLM response formats."""

    @pytest.mark.parametrize("mock_response,expected_count,desc", [
        ({"facts": [{"content": "用户有一个有效事实用于测试", "category": "user_pref", "tags": "语言", "importance": 7}]}, 1, "标准 facts 键名"),
        ({"memories": [{"content": "用户住在北京朝阳区CBD", "category": "user_pref", "tags": "地址", "importance": 6}]}, 1, "memories 键名兼容"),
        ({"results": [{"content": "用户今天中午吃了火锅", "category": "general", "tags": "饮食", "importance": 3}]}, 1, "results 键名兼容"),
        ({"insights": [{"content": "用户最近开始每周去健身房", "category": "user_pref", "tags": "健康", "importance": 6}]}, 1, "insights 键名兼容"),
        ({"patterns": [{"content": "用户形成了一个固定工作模式", "category": "user_pref", "tags": "习惯", "importance": 5}]}, 1, "patterns 键名兼容"),
        ({"extractions": [{"content": "用户今天完成了项目里程碑", "category": "project", "tags": "项目", "importance": 7}]}, 1, "extractions 键名兼容"),
        ({"unknown_key": [{"content": "不应被解析的内容", "category": "general", "tags": "", "importance": 5}]}, 0, "未知键名应忽略"),
        ([{"content": "直接返回列表格式兼容测试", "category": "general", "tags": "", "importance": 5}], 1, "直接返回列表兼容"),
        ({"facts": [{"content": "ABC", "category": "general", "tags": "", "importance": 5}]}, 0, "过短内容应过滤 <10字符"),
        ({"facts": [{"content": "用户有一个无效分类测试", "category": "invalid", "tags": "", "importance": 5}]}, 1, "无效分类降级为general"),
        ({"facts": [{"content": "用户重要性上限裁剪测试内容", "category": "general", "tags": "", "importance": 15}]}, 1, "重要性上限裁剪为10"),
        ({"facts": [{"content": "用户重要性下限裁剪测试内容", "category": "general", "tags": "", "importance": 0}]}, 1, "重要性下限裁剪为1"),
        ({"facts": "not a list"}, 0, "facts 非列表安全返回"),
        ({}, 0, "空字典安全返回"),
    ])
    def test_parse_formats(self, mock_response, expected_count, desc, monkeypatch):
        from unittest.mock import patch, MagicMock
        import urllib.request

        inner = json.dumps(mock_response)
        response_body = json.dumps({"choices": [{"message": {"content": inner}}]})

        def mock_urlopen(*args, **kwargs):
            m = MagicMock()
            m.read.return_value = response_body.encode("utf-8")
            cm = MagicMock()
            cm.__enter__.return_value = m
            return cm

        with patch("urllib.request.urlopen", mock_urlopen):
            with patch("butterfly_dream.__init__._resolve_provider_credentials",
                       return_value=("https://test.api", "test-key")):
                facts = _call_extraction_llm("test", "test", "test", timeout=5)
                assert len(facts) == expected_count, f"{desc}: expected {expected_count}, got {len(facts)}"

    def test_parse_persistent_flag(self):
        """is_persistent flag is correctly parsed from various formats."""
        from unittest.mock import patch, MagicMock

        mock_data = {"facts": [
            {"content": "用户持久标记测试内容A通过", "category": "general", "tags": "", "importance": 5, "is_persistent": True},
            {"content": "用户非持久标记测试内容B通过", "category": "general", "tags": "", "importance": 5, "is_persistent": False},
            {"content": "用户字符串true标记测试通过", "category": "general", "tags": "", "importance": 5, "is_persistent": "true"},
            {"content": "用户字符串false标记测试通过", "category": "general", "tags": "", "importance": 5, "is_persistent": "false"},
        ]}
        inner = json.dumps(mock_data)
        body = json.dumps({"choices": [{"message": {"content": inner}}]})

        def mock_urlopen(*args, **kwargs):
            m = MagicMock()
            m.read.return_value = body.encode("utf-8")
            cm = MagicMock()
            cm.__enter__.return_value = m
            return cm

        with patch("urllib.request.urlopen", mock_urlopen):
            with patch("butterfly_dream.__init__._resolve_provider_credentials",
                       return_value=("https://test.api", "test-key")):
                facts = _call_extraction_llm("test", "test", "test", timeout=5)
                assert len(facts) == 4
                assert facts[0]["is_persistent"] is True
                assert facts[1]["is_persistent"] is False
                assert facts[2]["is_persistent"] is True
                assert facts[3]["is_persistent"] is False


# ==============================================================================
# Tokenize / entity extraction
# ==============================================================================

class TestTokenize:
    """tokenize() extracts entities from mixed Chinese/English text."""

    @pytest.mark.parametrize("text,expected_subset", [
        ("用户喜欢VS Code和Neovim", {"vs", "code", "neovim"}),
        ("用户在阿里巴巴和腾讯工作过", {"阿里巴巴", "腾讯"}),
        ("用户最喜欢的编程语言是Python和Rust", {"python", "rust"}),
        ("用户住在北京，工作在深圳", {"北京", "深圳"}),
        ("用户喜欢在macOS上开发", {"macos"}),
        ("C++和C#都是好语言", {"c++", "c#"}),
        ("普通句子没有特殊实体", set()),
        ("", set()),
    ])
    def test_entity_extraction(self, text, expected_subset):
        from butterfly_dream.retrieval import tokenize
        tokens = tokenize(text)
        for entity in expected_subset:
            assert entity in tokens, f"'{entity}' should be in tokens: {tokens}"
