"""conftest: Mock Hermes-specific imports for standalone testing."""
import sys
from unittest.mock import MagicMock

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

# Ensure specific mocks have needed attributes
from agent.memory_provider import MemoryProvider  # type: ignore[import-untyped]
MemoryProvider.__name__ = "MemoryProvider"
