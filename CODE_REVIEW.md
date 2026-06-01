# 🦋 Butterfly Dream — Comprehensive Code Review

**Reviewer:** Code Review Agent
**Date:** 2026-06-01
**Codebase:** `~/butterfly-dream/src/butterfly_dream/`
**Test suite:** 68/68 passing (3 test files, conftest.py with Hermes mocks)
**Environment:** Python 3.14.4, numpy 2.4.6

---

## EXECUTIVE SUMMARY

**Overall assessment: Sound foundation with serious bugs in retrieval math that undermine core functionality. Production readiness: NOT YET.**

The codebase demonstrates good engineering practices overall — type hints, docstrings, structured modules, thread safety, and meaningful decomposition. However, a critical bug in the FTS5 rank normalization actually **inverts** the relevance signal, meaning the search component actively favors bad matches over good ones. This single issue, combined with mutable global state, unchecked SQL f-strings, and no tests for the 720-line plugin entry point, makes the project unsuitable for production use without significant remediation.

**Issues found: 30 (1 CRITICAL, 3 HIGH, 10 MEDIUM, 11 LOW, 5 SUGGESTION)**

---

## 1. CRITICAL ISSUES

### C1. FTS5 Rank Normalization Is Inverted — Relevance Scores Are Backwards

**File:** `retrieval.py` line 245  
**Severity:** CRITICAL  
**Type:** Logic bug — retrieval quality

```python
d["fts_rank"] = 1.0 / (1.0 + abs(d.get("rank", 0) or 0))
```

**Problem:** SQLite FTS5's built-in BM25 ranking assigns **negative** scores where **more negative = better match** (values like -20 for perfect match, -1 for weak match, 0 for no match). The code applies `abs()` which makes the best match's rank `abs(-20) = 20` and the worst match's `abs(-1) = 1`. Then:

| Match quality | Raw rank | fts_rank | Interpretation |
|---|---|---|---|
| Best match | -20 | 0.048 | Gets LOWEST fts_rank |
| OK match | -5 | 0.167 | Medium |
| Barely match | -1 | 0.500 | Gets HIGHEST fts_rank |

The best match gets **10× lower** fts_rank than the worst match. This means the relevance signal actively **penalizes** the best FTS5 results.

Since `fts_weight=0.4` in the ThreeDimRetriever constructor, this corrupts ~40% of the final relevance score. The Jaccard and HRR components partially compensate, but this is a fundamental correctness bug.

**Fix:** Change to:
```python
# rank is ≤ 0, more negative = better match
d["fts_rank"] = -rank / (1.0 - rank) if rank < 0 else 1.0
```
This maps: -20→0.95, -5→0.83, -1→0.50, 0→1.0.

---

## 2. HIGH-SEVERITY ISSUES

### H1. Duplicate Import and Thread-Race in _find_merge_candidate

**File:** `store.py` line 233  
**Severity:** HIGH  
**Type:** Performance + maintainability

```python
from .retrieval import tokenize, jaccard_similarity
```

**Problem:** This import happens **inside the method body**, on every call to `_find_merge_candidate` (which is called on every `add_fact` with `merge=True`). This is both a performance issue (unnecessary Python import overhead on every fact addition) and a design smell. It's also not thread-safe — Python's `import` system has a global lock, but this could cause rare deadlocks in extreme concurrency scenarios.

**Fix:** Move to module-level imports.

### H2. Global SCENARIO_WEIGHTS Mutation Causes Cross-Session Contamination

**Files:** `__init__.py` lines 363-367, `retrieval.py` lines 29-35  
**Severity:** HIGH  
**Type:** Race condition / shared mutable state

```python
# In __init__.py initialize():
SCENARIO_WEIGHTS["custom"] = {
    "relevance": rel_w,
    "recency": rec_w,
    "importance": imp_w,
}
```

**Problem:** Every call to `initialize()` mutates the **module-level global** `SCENARIO_WEIGHTS` dict in `retrieval.py`. In a multi-session or multi-agent Hermes deployment, concurrent calls to `initialize()` will race on this shared dictionary. The "custom" key gets overwritten by whichever `initialize()` call finishes last, corrupting other sessions' weight configurations. Test isolation is also broken — test A setting custom weights can affect test B.

**Fix:** Store custom weights per-instance (on `self`) rather than mutating a shared module-level constant. The `search()` method should accept custom weights directly.

### H3. BFS Uses O(n) List Pop as Queue

**File:** `store.py` line 507  
**Severity:** HIGH  
**Type:** Performance — algorithmic complexity

```python
queue = [(entity_name, 0)]
while queue:
    name, d = queue.pop(0)  # O(n) for each pop!
```

**Problem:** `list.pop(0)` is O(n) because every element must be shifted. For deep BFS traversals (depth=2 is small, but depth could scale), this becomes O(n²). With hundreds of entities, this will be noticeably slow.

**Fix:** Use `collections.deque` with `popleft()` for O(1) operations.

---

## 3. MEDIUM-SEVERITY ISSUES

### M1. SQL Column Names Constructed via F-String (Whitelist-Guarded But Fragile)

**File:** `store.py` lines 396-400  
**Severity:** MEDIUM  
**Type:** Security / maintenance

```python
set_clause = ", ".join(f"{k}=?" for k in updates)
# ...
f"UPDATE facts SET {set_clause} WHERE fact_id=?"
```

**Context:** The `updates` dict is filtered through `allowed = {"content", "category", "tags", "importance", "trust_score"}` (line 391), so direct injection is blocked today. **However**, this pattern is a well-known SQL injection antipattern that becomes dangerous the moment someone extends `allowed` without realizing the implications. A `CASCADE` or subquery-containing column name would bypass the parameterization.

**Fix:** Use a mapping dictionary approach:
```python
_COLUMN_MAP = {"content": 1, "category": 1, "tags": 1, ...}
columns = [k for k in kwargs if k in _COLUMN_MAP]
```
Then iterate with positionals — no f-string interpolation of user-controlled keys at all.

### M2. `absorbed_fact_id` Set to 0 in Merge Log

**File:** `store.py` line 280-283  
**Severity:** MEDIUM  
**Type:** Data integrity

```python
self._conn.execute(
    """INSERT INTO merge_log (kept_fact_id, absorbed_fact_id, merged_content, merge_reason)
       VALUES (?, 0, ?, 'semantic')""",
    (fact_id, merged_content),
)
```

**Problem:** The merge log records `absorbed_fact_id=0` for semantic merges, but foreign key `fact_id=0` doesn't exist in the facts table. If foreign key enforcement (`PRAGMA foreign_keys = ON`) is ever enabled, this INSERT would fail. More importantly, the merge log is uninformative — you can't trace which fact was absorbed.

**Fix:** Either insert the new content as a separate fact first (then absorb its ID), or accept that semantic merges don't have an "absorbed" ID and use NULL.

### M3. Enormous Query Construction in _find_merge_candidate

**File:** `store.py` line 225  
**Severity:** MEDIUM  
**Type:** Denial of service / stability

```python
.format(",".join("?" * len(entities)))
```

**Problem:** If `entities` has thousands of entries (e.g., from a malformed LLM extraction), this generates a query with thousands of parameter placeholders and passes them all to SQLite. This is a DoS vector through the entity parameter.

**Fix:** Cap entity count before query construction, e.g., `entities = entities[:20]`.

### M4. `add_fact` Accepts Arbitrary Types Without Validation

**File:** `store.py` lines 142-183  
**Severity:** MEDIUM  
**Type:** Robustness

```python
def add_fact(self, content: str, category: str = "general", tags: str = "", importance: float = 5.0, ...)
```

**Problem:** While type hints say `str`, Python doesn't enforce them. A caller passing `content=None`, `importance="abc"`, or `tags=123` would cause SQLite errors or `float()` casting exceptions at runtime. The `importance` parameter is coerced in `_handle_add` (line 568) but not in `add_fact` itself.

**Fix:** Add explicit type guards at the top of `add_fact`:
```python
if not isinstance(content, str) or not content.strip():
    raise ValueError("content must be a non-empty string")
```

### M5. `_combine_fact_content` Contradiction Detection Is Fragile

**File:** `store.py` lines 327-346  
**Severity:** MEDIUM  
**Type:** Design correctness

```python
is_contradiction = len(common) >= 3 and e_has_neg != n_has_neg
```

**Problem:** The heuristic checks if two facts share ≥3 tokens but differ in negation. This fails on:
- Multi-sentence contradictions ("Alice likes cats. She also likes dogs." vs "Alice hates all pets") — different tokens, miss contradiction.
- Syntactic negation ("Alice is not unhappy about cats") — double negation, false positive.
- Short contradictions ("like" vs "don't like") — only 1 common token, miss contradiction.
- Contradiction via antonym rather than negation ("Alice loves cats" vs "Alice hates cats") — complete miss.

Combined content (line 344-345) also produces ungrammatical composites with `；` (CJK semicolon) and `⚡` which look bad in user-facing responses.

### M6. Pre-Compress Extraction Uses Correct Thread, but Session-End Does Not

**File:** `__init__.py` lines 427-443 (threaded) vs 455-463 (blocking)  
**Severity:** MEDIUM  
**Type:** Performance / blocking

`on_pre_compress` correctly runs extraction in a daemon thread. But `on_session_end` runs it **synchronously**, blocking the session teardown on an LLM API call (potentially 10+ seconds). This can delay the overall Hermes shutdown pipeline.

**Fix:** Use `threading.Thread` in `on_session_end` as well.

### M7. Hardcoded Magic Numbers for Message Truncation

**File:** `__init__.py` lines 504-507  
**Severity:** MEDIUM  
**Type:** Maintainability

```python
if len(text) > 24000:
    head = text[:12000]
    tail = text[-10000:]
```

**Problem:** Four magic numbers (`24000`, `12000`, `10000`, and the 2000-char gap in the middle). These are undocumented token-count heuristics that silently drop the middle of conversation context. A 4000-char gap between head and tail means potentially critical context is lost.

**Fix:** Make these configurable or use a more principled truncation (sliding window, key message retention, etc.).

### M8. Direct Access to Private `store._conn` in Retrieval

**File:** `retrieval.py` line 219  
**Severity:** MEDIUM  
**Type:** Encapsulation / coupling

```python
conn = self.store._conn
```

**Problem:** `ThreeDimRetriever` reaches into `MemoryStore._conn` (a private attribute with underscore prefix). This tightly couples the two classes. If `MemoryStore` ever changes its connection management (connection pooling, WAL mode reconnection, replication), `ThreeDimRetriever` silently breaks.

**Fix:** Expose a `store.get_connection()` or `store.execute()` public method.

### M9. Entity Aliases LIKE Pattern With User Data

**File:** `store.py` line 493-496  
**Severity:** MEDIUM  
**Type:** Query correctness / security

```python
WHERE e.name = ? OR e.aliases LIKE ?
```
with `(entity_name, f"%{entity_name}%", limit)`

**Problem:** The `LIKE '%{entity_name}%'` pattern has a `%` wildcard on both sides, passed via parameter. While not SQL injection (it's parameterized), if `entity_name` is `%`, this matches literally every entity. If it contains SQL wildcard characters (`_` matches any single character), the result set is silently wrong.

**Fix:** Escape `%` and `_` in the entity name: `entity_name.replace("%", r"\%").replace("_", r"\_")`.

### M10. In-Memory Fact Contradiction Detection Loads All Entities

**File:** `__init__.py` lines 629-648  
**Severity:** MEDIUM  
**Type:** Performance

```python
entities = self._store._conn.execute("SELECT name FROM entities LIMIT 50").fetchall()
for row in entities:
    facts = self._store.get_entity_facts(name, limit=20)
    # O(n²) comparison on each entity's facts
```

**Problem:** The `_handle_contradict` handler performs up to 50 entities × 20 facts × 19 comparisons = 19,000 comparisons in worst case, all **synchronously** in the tool call handler. Each `get_entity_facts` call makes a separate SQL query. This could time out the Hermes tool call handler.

---

## 4. LOW-SEVERITY ISSUES

### L1. No Dimension Validation in HRR Operations

**File:** `holographic.py` lines 55-86  
**Severity:** LOW

`bind`, `unbind`, `bundle`, and `similarity` all assume same-dimension inputs but never check. A dimension mismatch (e.g., `encode_atom("x", 256)` vs `encode_atom("y", 128)`) raises an opaque numpy broadcasting error instead of a clear ValueError.

### L2. Bundle Creates Temporary List of Complex Arrays

**File:** `holographic.py` line 77  
**Severity:** LOW (performance)

```python
complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
```

This allocates `len(vectors)` complex arrays before summing. For large bundles (100+ entities), this doubles peak memory. Use a generator or `np.add.reduce`.

### L3. `remove_fact` Always Returns True Even for Missing Facts

**File:** `store.py` lines 405-409  
**Severity:** LOW

```python
def remove_fact(self, fact_id: int) -> bool:
    with self._lock:
        self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._conn.commit()
        return True
```

Returns `True` even if `fact_id` didn't exist (0 rows affected). The tool handler assumes `False` means "not found" (line 692), which is inconsistent with the actual implementation.

### L4. `close()` Can Be Called Multiple Times Without Protection

**File:** `store.py` lines 559-560  
**Severity:** LOW

```python
def close(self):
    self._conn.close()
```

No guard against double-close. Calling `close()` twice raises `sqlite3.ProgrammingError`. The `shutdown()` method in `__init__.py` guards against this (lines 477-481), but direct callers don't.

### L5. Exact-Merge Doesn't Re-Encode HRR Vector

**File:** `store.py` lines 185-201  
**Severity:** LOW

When merging exact duplicates, the HRR vector is **not** updated even though importance/tags/trust change. The stale HRR vector doesn't reflect the updated metadata. Semantic merges correctly re-encode (line 267).

### L6. Legacy Build Backend in pyproject.toml

**File:** `pyproject.toml` line 3  
**Severity:** LOW

```toml
build-backend = "setuptools.backends._legacy:Backend"
```

The `_legacy` backend is a compatibility shim and may be removed in future setuptools versions. Should use `setuptools.build_meta`.

### L7. numpy Version Constraint May Break on Older Versions

**File:** `pyproject.toml` line 24  
**Severity:** LOW

```toml
"numpy>=1.26"
```

No upper bound. numpy 2.x has breaking changes (removed aliases, changed casting rules). Should pin `numpy>=1.26,<3` to prevent surprises.

### L8. No Import Guard for `numpy` on Fallback Path

**File:** `retrieval.py` lines 101-105  
**Severity:** LOW

```python
if hrr_weight > 0 and not hrr._HAS_NUMPY:
```

Accessing `hrr._HAS_NUMPY` is fine, but there's no guard if `hrr` itself fails to import. The try/except in store.py (line 19-21) and retrieval.py (line 22-24) catches ImportError, but both paths assume `numpy` availability at least at the top level.

### L9. Redundant `dict(fact)` Conversion in `_handle_reason`

**File:** `__init__.py` line 622  
**Severity:** LOW

```python
results.append(dict(fact))
```

`fact` is already a dict (from `get_fact`), so this is a no-op copy. Unnecessary allocation.

### L10. Hardcoded "trust" Fallback Key That Doesn't Exist

**File:** `__init__.py` line 415  
**Severity:** LOW

```python
trust = r.get("trust_score", r.get("trust", 0))
```

The fallback `"trust"` key doesn't exist in any data schema. If `trust_score` were missing, this returns 0 silently, which would incorrectly zero out all scoring (since `score *= trust`).

### L11. Plugin.yaml `requires` Overstates Dependency

**File:** `plugin.yaml` line 6  
**Severity:** LOW

```yaml
requires: [numpy]
```

The code handles numpy absence gracefully (returns 0.5 or None for HRR operations). Listing it as required prevents Hermes from loading the plugin on systems without numpy, even though it would degrade gracefully.

---

## 5. SUGGESTIONS

### S1. No Tests for `__init__.py` (The Plugin Entry Point)

**Files:** `tests/` — **no** `test_init.py` or equivalent  
**Severity:** SUGGESTION

The 720-line `__init__.py` file contains the entire Hermes plugin lifecycle — `initialize`, `handle_tool_call`, `prefetch`, `on_pre_compress`, `on_session_end`, `save_config`, `shutdown`, LLM extraction, and 9 tool action handlers. Zero lines of test code cover any of this.

**All 9 tool actions** (`add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`) and `fact_feedback` have no test coverage.

### S2. No Concurrency / Thread Safety Tests

The store uses `threading.Lock`, but no tests verify correctness under concurrent access. Common failure modes include: deadlock, double-insert, stale reads, and write skew.

### S3. No Tests for `_sanitize_fts_query` Edge Cases

FTS5 special characters (`*`, `"`, `-`, `OR`, `AND`, `NEAR`) are stripped, but the test suite only tests simple queries like `"Python"`. Testing would reveal whether CJK characters are handled correctly by the regex.

### S4. No Tests for `_combine_fact_content` Directly

The contradiction detection, substring detection, and CJK semicolon joining logic are tested only indirectly through end-to-end merge tests. Direct unit tests would catch edge cases like:
- Exact match with trailing whitespace
- Unicode normalization differences
- Contradiction with CJK negation words
- Boundary conditions (2 vs 3 common tokens)

### S5. No Benchmark / Performance Tests

The FTS5 → Jaccard → HRR pipeline does O(n × m × d) work per search (n=candidates, m=query tokens, d=HRR dim=1024). No benchmarks verify that search completes within acceptable latency for production use (sub-100ms for 30 candidates).

---

## 6. DESIGN OBSERVATIONS

### D1. Entity Extraction Is Heavily English-Biased

The entity extraction pipeline (`store.py` lines 110-120) has 6 regex patterns:
- 3 English-specific (`_RE_CAPITALIZED`, `_RE_DOUBLE_QUOTE`, `_RE_SINGLE_QUOTE`, `_RE_AKA`)
- 3 CJK-specific (`_RE_CJK_BRACKETS`, `_RE_QUOTED_CN`)

There's no support for many common languages (Arabic, Devanagari, Korean [has no spaces but uses hangul blocks], etc.). The CJK patterns also handle only a limited set of quotation conventions.

### D2. Merge-Heavy Architecture Creates Data Sink

The three-level merge strategy (exact → semantic → insert) means that semantically related facts **merge** rather than staying separate. Over time, this creates ever-larger composite facts that are hard to query precisely and degrade the Jaccard/HRR similarity signal. A production system might want both merge AND split capabilities, or a versioned fact approach.

### D3. No WAL Mode for SQLite

The SQLite connection is opened in default journal mode (DELETE). For a plugin that mixes reads (prefetch, search) and writes (add_fact, feedback), WAL mode would provide better concurrent performance.

### D4. LLM Extraction Prompt Is Hardcoded in Source

**File:** `__init__.py` lines 46-79

The 35-line LLM extraction prompt is embedded as a string constant. This makes it impossible to customize, A/B test, or update without deploying new code. Prompt engineering improvements require code changes.

---

## 7. REMEDIATION PRIORITY

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| P0 | **C1** — FTS5 rank inversion | 1 line | Fixes core retrieval correctness |
| P0 | **H2** — Global SCENARIO_WEIGHTS mutation | ~20 lines | Prevents multi-session corruption |
| P1 | **H1** — Deferred import | 1 line move | Performance + maintainability |
| P1 | **H3** — deque for BFS | 3 lines | Prevents O(n²) on large graphs |
| P1 | **M8** — Direct _conn access | ~5 lines | Design hygiene |
| P2 | **M1** — SQL f-string pattern | ~10 lines | Security hardening |
| P2 | **S1** — Plugin test coverage | New test file | Confidence in production |
| P3 | All MEDIUM and LOW items | Varies | Overall quality |

---

## 8. FILE-BY-FILE SUMMARY

### `holographic.py` (151 lines) — ⭐ EXCELLENT
Clean, well-documented, mathematically sound. No significant issues. The deterministic SHA-256 encoding approach is well-motivated. Minor: no dimension validation, memory-inefficient bundle.

### `store.py` (560 lines) — ⚠️ GOOD WITH CONCERNS
Solid SQLite schema design with FTS5 + triggers. Thread-safe lock pattern is correct. Issues: `list.pop(0)` O(n), SQL f-string pattern, deferred import, unguarded entity count, stale HRR on exact merge, missing type validation.

### `retrieval.py` (294 lines) — ❌ CRITICAL BUG
The FTS5 rank normalization is inverted, corrupting the core relevance signal. Everything else is well-structured. Good separation of pipeline stages. The scenario weight system is elegant.

### `__init__.py` (720 lines) — ⚠️ UNTESTED, GLOBAL STATE
Largest file with zero test coverage. Global SCENARIO_WEIGHTS mutation is the most serious design issue. LLM extraction is well-implemented with proper normalization. Tool handlers follow a consistent pattern.

### `plugin.yaml` (7 lines) — FINE
Minor: `requires: [numpy]` is stricter than necessary.

### `pyproject.toml` (32 lines) — MINOR
Legacy build backend, no upper bound on numpy.

### Test Suite (612 lines across 3 files) — GOOD COVERAGE FOR CORE, NONE FOR PLUGIN
holographic tests: excellent (atoms, bind/unbind, bundle, similarity, encode, serialization, SNR).
store tests: good (CRUD, entities, merging, feedback, HRR).
retrieval tests: good (helpers, search, scenarios).
**Missing: zero tests for __init__.py, no concurrency tests, no FTS5 sanitization edge cases.**

---

## 9. CONCLUSION

The Butterfly Dream project has a thoughtful architecture and well-structured core modules. The core issues to fix before production use are:

1. **Fix the FTS5 rank normalization** (1 line change, critical correctness)
2. **Eliminate global SCENARIO_WEIGHTS mutation** (eliminates session contamination)
3. **Add __init__.py test coverage** (currently 0%)
4. **Harden SQL patterns** (eliminate f-string SQL construction)

Estimated remediation effort: ~2-3 hours for all P0-P1 issues. The design is fundamentally sound and worth investing in.
