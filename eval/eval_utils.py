"""评测管线共享工具 — 统一模型配置加载。

所有 eval 脚本通过 get_model_config() 获取模型设置，
换模型只需改 eval/model_config.yaml，不用动代码。

用法：
    from eval_utils import get_model_config, call_llm

    cfg = get_model_config()          # 读 model_config.yaml
    cfg = get_model_config("judge")   # 只读 judge 部分
    answer = call_llm("answer", messages=[...])
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_EVAL_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _EVAL_DIR / "model_config.yaml"
_DB_DIR = _EVAL_DIR / "dbs"
_RUNS_DIR = _EVAL_DIR / "runs"
_config_cache = None
_active_run_dir: Path | None = None


def set_run_dir(run_dir: Path) -> None:
    """Set the active run directory. All subsequent get_db_path() calls
    will store DBs under ``run_dir/dbs/`` instead of the global dbs/ dir."""
    global _active_run_dir
    _active_run_dir = run_dir
    (run_dir / "dbs").mkdir(parents=True, exist_ok=True)


def get_run_dir() -> Path | None:
    """Return the active run directory, or None if not set."""
    return _active_run_dir


def get_db_path(benchmark: str, item_id: str) -> Path:
    """Get a persistent DB path for an eval run.

    Always stores to ``eval/dbs/{benchmark}/{item_id}.db`` so extraction
    results survive across runs and can be reused via --db-dir.
    """
    db_dir = _DB_DIR / benchmark
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{item_id}.db"


def _load_config() -> dict:
    """Load model_config.yaml (with fallback to hardcoded defaults)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults = {
        "extraction": {"provider": "openrouter", "model": "owl-alpha"},
        "answer":     {"provider": "openrouter", "model": "owl-alpha"},
        "judge":      {"provider": "openrouter", "model": "owl-alpha"},
    }

    if _CONFIG_PATH.exists():
        try:
            import yaml
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            # Merge with defaults (loaded overrides defaults)
            for key in defaults:
                if key in loaded:
                    defaults[key] = {**defaults[key], **loaded[key]}
        except Exception:
            pass

    _config_cache = defaults
    return _config_cache


def get_model_config(role: str = "all") -> dict:
    """Get model config for a given role.

    Args:
        role: "extraction", "answer", "judge", or "all" (returns entire dict).

    Returns:
        {"provider": "...", "model": "..."}
    """
    cfg = _load_config()
    if role == "all":
        return cfg
    return cfg.get(role, cfg.get("answer", {}))


# ---------------------------------------------------------------------------
# Credential resolution (same as butterfly_dream.__init__)
# ---------------------------------------------------------------------------

_DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek":   "https://api.deepseek.com",
    "openai":     "https://api.openai.com/v1",
    "zai":        "https://open.bigmodel.cn/api/paas/v4",
    "anthropic":  "https://api.anthropic.com/v1",
}


def resolve_credentials(provider: str) -> tuple[str, str]:
    """Resolve (base_url, api_key) for a provider."""
    prefix = provider.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    base_url = os.environ.get(
        f"{prefix}_BASE_URL",
        _DEFAULT_BASE_URLS.get(provider, ""),
    )
    return base_url.rstrip("/"), api_key


def _load_hermes_env():
    """Load ~/.hermes/.env into os.environ (idempotent)."""
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.is_file():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key not in os.environ:
                os.environ[key] = val.strip().strip("\"'")


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

import time as _time
_last_llm_call: float = 0.0
_MIN_CALL_INTERVAL: float = 3.0  # seconds between LLM calls (rate-limit safety)


def call_llm(
    role: str,
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: int = 60,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> str:
    """Call an LLM using the config for the given role.

    Rate-limited: enforces at least ``_MIN_CALL_INTERVAL`` seconds between calls.
    Retries up to 4 times on failure with exponential backoff (5/10/15/20s).

    Args:
        role: "extraction", "answer", or "judge".
        messages: OpenAI-format messages list.
        temperature, max_tokens: Generation params.
        provider_override, model_override: Bypass config for one-off tests.

    Returns:
        Assistant message content string. Empty string on error after all retries.
    """
    global _last_llm_call

    _load_hermes_env()
    cfg = get_model_config(role)
    provider = provider_override or cfg.get("provider", "openrouter")
    model = model_override or cfg.get("model", "owl-alpha")

    base_url, api_key = resolve_credentials(provider)
    if not api_key:
        return ""

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(4):
        # Rate limit: wait if needed
        elapsed = _time.monotonic() - _last_llm_call
        if elapsed < _MIN_CALL_INTERVAL:
            _time.sleep(_MIN_CALL_INTERVAL - elapsed)
        _last_llm_call = _time.monotonic()

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [LLM error: {e}]", file=sys.stderr)
            if attempt < 3:
                wait = 5 * (attempt + 1)
                print(f"  ⏳ Retrying in {wait}s (attempt {attempt+1}/4)", file=sys.stderr)
                _time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# CLI: show current config
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_model_config("all")
    print("Current eval model config:")
    for role, settings in cfg.items():
        print(f"  {role:12s} → {settings['provider']}/{settings['model']}")
    print(f"\nConfig file: {_CONFIG_PATH}")
