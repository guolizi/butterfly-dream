"""conftest: Mock Hermes-specific imports for standalone testing."""
import sys
from unittest.mock import MagicMock
from abc import ABC, abstractmethod

# Define a proper MemoryProvider ABC so subclasses don't become MagicMocks
class _MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    def system_prompt_block(self) -> str: return ""
    def prefetch(self, query: str, **kwargs) -> str: return ""
    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None: pass
    def get_tool_schemas(self): return []
    def handle_tool_call(self, name, args, **kwargs): return ""
    def on_pre_compress(self, messages) -> str: return ""
    def on_session_end(self, messages) -> None: pass
    def on_memory_write(self, action, target, content, **kwargs) -> None: pass

# Mock Hermes modules before butterfly_dream imports them
hermes_mocks = {
    "agent": MagicMock(),
    "agent.memory_provider": MagicMock(),
    "tools": MagicMock(),
    "tools.registry": MagicMock(),
    "hermes_cli": MagicMock(),
    "hermes_cli.config": MagicMock(),
    "hermes_constants": MagicMock(),
}

for mod_name, mock in hermes_mocks.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock

# Replace MemoryProvider in the mock with our proper ABC
from agent.memory_provider import MemoryProvider
import agent.memory_provider as amp_mod
amp_mod.MemoryProvider = _MemoryProvider
# Also replace the global reference in sys.modules
sys.modules["agent.memory_provider"].MemoryProvider = _MemoryProvider
MemoryProvider.__name__ = "MemoryProvider"
