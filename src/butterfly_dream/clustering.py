"""Entity clustering engine for Butterfly Dream v2.

Discovers abstract entity groups from neural embedding similarity.
Pipeline:
  1. Load all entities with embeddings from store
  2. Compute all-pairs cosine similarity
  3. Build graph edges where similarity > threshold
  4. Find connected components → clusters
  5. Auto-name each cluster from its centroid member
  6. Write back: cluster table entries + is_member_of relations
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .store import MemoryStore

logger = logging.getLogger(__name__)

# Default clustering parameters
_DEFAULT_SIM_THRESHOLD = 0.65   # min cosine sim to be in same cluster (0.55 for CJK, 0.75+ for EN)
_DEFAULT_MIN_CLUSTER_SIZE = 2   # ignore singletons


def compute_clusters(
    store: MemoryStore,
    *,
    threshold: float = _DEFAULT_SIM_THRESHOLD,
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    relation_type: str = "includes",
) -> list[dict]:
    """Run entity embedding clustering and persist results (three-layer ontology).

    Creates an abstract entity in the entities table for each cluster, plus
    includes edges (L3) from abstract → concrete members in entity_relations.

    Args:
        store: MemoryStore instance (must have embedding service loaded).
        threshold: Cosine similarity threshold for co-cluster edges.
        min_cluster_size: Minimum entities per cluster (singletons skipped).
        relation_type: Relation to write in entity_relations (default 'includes').

    Returns:
        List of cluster dicts: {cluster_id, name, members, coherence, centroid}.
    """
    # ── 1. Load all entities with embeddings ──
    from .embedding import get_embedding_service
    embed_svc = get_embedding_service()

    rows = store.execute_query(
        "SELECT entity_id, name, embedding FROM entities WHERE embedding IS NOT NULL AND entity_type != 'abstract'"
    )
    if len(rows) < min_cluster_size:
        logger.info("clustering: only %d entities with embeddings, skipping", len(rows))
        return []

    entity_ids = []
    entity_names = []
    vectors = []
    for r in rows:
        blob = r["embedding"]
        if blob is None:
            continue
        vec = embed_svc.deserialize(bytes(blob))
        if vec is not None:
            entity_ids.append(r["entity_id"])
            entity_names.append(r["name"])
            vectors.append(vec)

    if len(vectors) < min_cluster_size:
        return []

    # ── 2. Compute all-pairs cosine similarity ──
    mat = np.stack(vectors)  # (N, D)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid div-by-zero
    sim_mat = (mat @ mat.T) / (norms @ norms.T)  # (N, N)

    # ── 3. Build adjacency graph (similarity > threshold) ──
    n = len(entity_ids)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i, j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # ── 4. Find connected components ──
    visited = [False] * n
    clusters_computed: list[list[int]] = []
    for i in range(n):
        if not visited[i]:
            # BFS
            component = []
            stack = [i]
            visited[i] = True
            while stack:
                v = stack.pop()
                component.append(v)
                for nb in adj[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            if len(component) >= min_cluster_size:
                clusters_computed.append(component)

    if not clusters_computed:
        logger.info("clustering: no clusters found at threshold=%.2f", threshold)
        return []

    # ── 5. For each cluster: name, centroid, persist ──
    results = []
    for component in clusters_computed:
        member_indices = component
        member_ids = [entity_ids[i] for i in member_indices]
        member_names = [entity_names[i] for i in member_indices]
        member_vecs = [vectors[i] for i in member_indices]

        # Centroid: mean of member vectors
        centroid = np.mean(member_vecs, axis=0)
        # Coherence: avg intra-cluster similarity
        n_comp = len(component)
        if n_comp >= 2:
            intra_sims = []
            for i_idx in member_indices:
                for j_idx in member_indices:
                    if i_idx < j_idx:
                        intra_sims.append(sim_mat[i_idx, j_idx])
            coherence = float(np.mean(intra_sims))
        else:
            coherence = 1.0

        # Auto-name: find the member with highest avg similarity to all others
        best_avg = -1.0
        best_idx = 0
        for idx, i in enumerate(member_indices):
            sims_to_others = []
            for j in member_indices:
                if i != j:
                    sims_to_others.append(sim_mat[i, j])
            if sims_to_others:
                avg = float(np.mean(sims_to_others))
                if avg > best_avg:
                    best_avg = avg
                    best_idx = idx
        cluster_name = _auto_cluster_name(member_names, best_idx)

        # Per-member centroid similarity (replaces old pairwise similarities)
        centroid_norm = float(np.linalg.norm(centroid))
        member_sims = []
        for vec in member_vecs:
            vec_norm = float(np.linalg.norm(vec))
            if centroid_norm > 0 and vec_norm > 0:
                sim = float(np.dot(centroid, vec) / (centroid_norm * vec_norm))
            else:
                sim = 0.0
            member_sims.append(round(sim, 4))

        # ── 6. Persist to DB ──
        try:
            centroid_blob = embed_svc.serialize(centroid)
        except Exception:
            centroid_blob = None

        cluster_id = store.create_cluster(
            name=cluster_name,
            cluster_type="auto",
            member_entity_ids=member_ids,
            similarities=member_sims,  # per-member centroid similarity
            centroid=centroid_blob,
            coherence=coherence,
            relation_type=relation_type,
        )

        results.append({
            "cluster_id": cluster_id,
            "name": cluster_name,
            "members": [{"id": eid, "name": ename}
                        for eid, ename in zip(member_ids, member_names)],
            "coherence": round(coherence, 3),
            "size": n_comp,
        })

        logger.info(
            "clustering: created cluster '%s' with %d members (coherence=%.3f)",
            cluster_name, n_comp, coherence,
        )

    return results


def _auto_cluster_name(member_names: list[str], centroid_idx: int) -> str:
    """Heuristic: use the centroid entity name + '类' suffix.

    Works well for Chinese entity names (跳绳→跳绳类, 游泳→游泳类).
    For English names, uses the longest common prefix heuristic.
    """
    if not member_names:
        return "未命名聚类"

    # Use centroid member name as base
    base = member_names[centroid_idx]

    # Check if all names share a common prefix (e.g., "运动_跳绳", "运动_游泳")
    if len(member_names) >= 2:
        # Find longest common prefix across all names
        prefix = _longest_common_prefix(member_names)
        if len(prefix) >= 2:
            base = prefix

    # If name already contains '类' or 'cluster' or 'category', don't append again
    if any(c in base for c in ("类", "cluster", "Cluster", "category", "Category")):
        return base

    # Chinese name → append 类
    if re.search(r'[\u4e00-\u9fff]', base):
        return f"{base}类"

    # English → "base category/group"
    return f"{base} category"


def _longest_common_prefix(strings: list[str]) -> str:
    """Longest common prefix among a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def cluster_stats(store: MemoryStore) -> dict:
    """Return summary statistics about current clusters."""
    clusters = store.get_all_clusters()
    if not clusters:
        return {"total_clusters": 0, "total_members": 0, "clusters": []}

    total_members = sum(c.get("member_count", 0) for c in clusters)
    avg_coherence = (
        sum(c.get("coherence", 0) for c in clusters) / len(clusters)
        if clusters else 0
    )

    return {
        "total_clusters": len(clusters),
        "total_members": total_members,
        "avg_coherence": round(avg_coherence, 3),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "name": c["name"],
                "type": c.get("cluster_type", "auto"),
                "member_count": c.get("member_count", 0),
                "coherence": c.get("coherence", 0),
            }
            for c in clusters
        ],
    }
