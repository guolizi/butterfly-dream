#!/usr/bin/env python3
"""Re-extract conv-26 with updated prompt, skipping QA evaluation."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from butterfly_dream import ButterflyDreamMemoryProvider
from eval_utils import get_db_path, get_model_config, _load_hermes_env
from run_locomo import (
    load_dataset, get_session_names, process_conversation,
    _run_clustering_after_extraction, _setup_extraction_log, _log,
)

_load_hermes_env()

# ── DB path ──
db_path = get_db_path("locomo", "conv-26")
print(f"🔧 Target DB: {db_path}")

# ── Delete old DB ──
for f in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  🗑️  Removed {f}")

# ── Load data ──
data_path = Path(__file__).resolve().parent / "data" / "locomo10.json"
data = load_dataset(str(data_path))
conv = [d for d in data if d["sample_id"] == "conv-26"]
if not conv:
    print("❌ conv-26 not found")
    sys.exit(1)
conv = conv[0]
print(f"📋 Loaded conv-26: {len(get_session_names(conv['conversation']))} sessions")

# ── Set up extraction log ──
os.makedirs("eval/runs/reextract", exist_ok=True)
handler = _setup_extraction_log(Path("eval/runs/reextract"))

# ── Create provider (fresh DB) ──
provider = ButterflyDreamMemoryProvider({
    "db_path": str(db_path),
    "llm_extract": True,
    "extraction_model": get_model_config("extraction"),
    "trivial_filter": True,
    "circuit_breaker": {"max_failures": 5, "cooldown_seconds": 120},
    "reflection": False,
})
provider.initialize(session_id="locomo-conv-26")

# ── Extract ──
t0 = time.perf_counter()
process_conversation(provider, conv["conversation"])
extract_time = time.perf_counter() - t0

n_facts = provider._store.count_facts() if provider._store else 0
print(f"\n✅ Extracted {n_facts} facts in {extract_time:.1f}s")

# ── Cluster ──
_run_clustering_after_extraction(provider._store)

# ── Summary ──
if provider._store:
    cur = provider._store._conn.cursor()
    cur.execute("SELECT COUNT(*) FROM entities")
    n_ents = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM entity_relations")
    n_rels = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM cluster_members")
    n_clustered = cur.fetchone()[0]
    print(f"\n📊 DB Summary:")
    print(f"   Entities: {n_ents}")
    print(f"   Relations: {n_rels}")
    print(f"   Clustered entities: {n_clustered}")

provider.shutdown()
print(f"\n💾 DB saved: {db_path}")
