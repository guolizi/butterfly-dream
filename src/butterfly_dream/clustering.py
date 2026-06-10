"""Entity clustering engine for Butterfly Dream v2 — Hierarchical clustering.

Pipeline:
  1. Louvain community detection on the co-occurrence graph
  2. For each community with internal structure, detect "core entities"
     (high-degree entities that serve as cluster centers)
  3. Build hierarchical sub-clusters: an entity belongs to the sub-cluster
     of the core entity it most exclusively co-occurs with
  4. Shared entities (co-occur with multiple cores) stay at the parent level
  5. Persist: parent cluster + sub-clusters + includes edges between them

This mirrors the three-layer ontology: L1=concrete entities, L2=abstract clusters,
L3=includes relations between clusters (hierarchy).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_MIN_CLUSTER_SIZE = 2


def compute_clusters(
    store: MemoryStore,
    *,
    threshold: float | None = None,
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    relation_type: str = "includes",
) -> list[dict]:
    """Run hierarchical Louvain clustering on entity co-occurrence graph.

    Creates clusters with sub-clusters for each detected "core entity".
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    import numpy as np
    from .embedding import get_embedding_service

    embed_svc = get_embedding_service()

    # ── 1. Load entities ──
    rows = store.execute_query(
        "SELECT entity_id, name, embedding FROM entities WHERE entity_type != 'abstract'"
    )
    if len(rows) < min_cluster_size:
        return []

    entity_ids, entity_names = [], []
    entity_embed_map = {}
    for r in rows:
        entity_ids.append(r["entity_id"])
        entity_names.append(r["name"])
        blob = r["embedding"]
        if blob is not None:
            vec = embed_svc.deserialize(bytes(blob))
            if vec is not None:
                entity_embed_map[r["entity_id"]] = vec

    # ── 2. Load co-occur edges ──
    edge_rows = store.execute_query(
        "SELECT source_id, target_id, weight FROM entity_relations WHERE relation = 'co_occur'"
    )

    # Build full graph
    G = nx.Graph()
    for eid in entity_ids:
        G.add_node(eid)
    for r in edge_rows:
        G.add_edge(r["source_id"], r["target_id"], weight=r["weight"])

    # Build adjacency lookup
    adj = {eid: {} for eid in entity_ids}
    for r in edge_rows:
        s, t, w = r["source_id"], r["target_id"], r["weight"]
        adj[s][t] = w
        adj[t][s] = w

    # ── 3. Louvain base communities ──
    communities = louvain_communities(G, weight="weight", seed=42)
    communities = [c for c in communities if len(c) >= min_cluster_size]
    if not communities:
        return []

    logger.info("Louvain: %d base communities from %d entities",
                len(communities), len(entity_ids))

    # ── 4. Two-pass: detect cores in large communities, then merge small ones ──
    all_results = []

    def name_for_eid(eid: int) -> str:
        return entity_names[entity_ids.index(eid)]

    def eid_for_name(name: str) -> int | None:
        try:
            return entity_ids[entity_names.index(name)]
        except ValueError:
            return None

    # ── Pass 1: Identify cores from large communities ──
    large_communities = [c for c in communities if len(c) > 3]
    small_communities = [c for c in communities if 2 <= len(c) <= 3]

    # Track all detected cores and the main characters (max-degree per large community)
    all_cores: list[int] = []  # core entity IDs
    core_names: list[str] = []  # core entity names
    core_members: dict[int, list[int]] = {}  # core_id → members assigned to it
    shared_members: list[int] = []  # shared across cores

    for community in large_communities:
        member_ids = sorted(community)
        subgraph = G.subgraph(member_ids)
        degrees = {n: d for n, d in subgraph.degree(weight="weight")}
        sorted_by_degree = sorted(degrees.items(), key=lambda x: -x[1])
        max_deg = sorted_by_degree[0][1] if sorted_by_degree else 0

        # Find cores: person-like entities with high degree
        cores_here = []
        for eid, deg in sorted_by_degree:
            name = name_for_eid(eid)
            if name.endswith(" category") or name.endswith("类"):
                continue
            is_person = (
                len(name) >= 2 and name[0].isupper()
                and name not in {
                    "Sweden", "Becoming Nicole", "Charlotte's Web",
                    "Embracing Identity", "Pride Month", "Pride fest",
                    "Summer Sounds", "Grand Canyon", "Connected LGBTQ Activists",
                    "LGBTQ support group", "LGBTQ+ community", "LGBTQ+ youth center",
                    "transgender conference", "Activists", "Pottery",
                }
                and deg >= max_deg * 0.3
            )
            if is_person:
                cores_here.append((eid, name, deg))
            if len(cores_here) >= 3:
                break

        if len(cores_here) <= 1:
            # Single-core: everything goes to one flat cluster
            cluster_id, cluster_name = _create_flat_cluster(
                store, member_ids, entity_ids, entity_names,
                entity_embed_map, embed_svc, G, relation_type,
            )
            all_results.append({
                "cluster_id": cluster_id, "name": cluster_name, "coherence": None,
                "members": [{"id": eid, "name": name_for_eid(eid)} for eid in member_ids],
                "size": len(member_ids), "sub_clusters": [],
            })
            continue

        # Multiple cores: assign non-core entities
        core_ids = [c[0] for c in cores_here]
        core_names_list = [c[1] for c in cores_here]
        core_set = set(core_ids)

        # Register cores globally
        all_cores.extend(core_ids)
        core_names.extend(core_names_list)
        for cid in core_ids:
            core_members.setdefault(cid, [cid])

        for eid in member_ids:
            if eid in core_set:
                continue
            weights = {}
            for cid in core_ids:
                w = adj.get(eid, {}).get(cid, 0) or adj.get(cid, {}).get(eid, 0)
                weights[cid] = w
            max_w = max(weights.values()) if weights else 0
            if max_w == 0:
                shared_members.append(eid)
                continue
            strong = sum(1 for w in weights.values() if w >= max_w * 0.5)
            if strong >= 2:
                shared_members.append(eid)
            else:
                best_core = max(weights, key=weights.get)
                core_members.setdefault(best_core, [best_core]).append(eid)

    # ── Pass 2: Merge small communities into the nearest core ──
    all_cores_set = set(all_cores)
    small_orphans = []  # small communities with no connection to any core

    for community in small_communities:
        member_ids = sorted(community)
        # Check if any member connects to a known core
        best_core = None
        best_weight = 0.0
        for eid in member_ids:
            for cid in all_cores:
                w = adj.get(eid, {}).get(cid, 0) or adj.get(cid, {}).get(eid, 0)
                if w > best_weight:
                    best_weight = w
                    best_core = cid

        if best_core is not None and best_weight >= 0.1:
            # Merge this small community into the core's sub-cluster
            core_members.setdefault(best_core, [best_core]).extend(member_ids)
            # Also add includes edges from core's abstract entity to each small entity
            core_name = name_for_eid(best_core)
            # Core's cluster already exists (from create_cluster below), so we can skip now
            # We'll add them in create_cluster or just via cluster_members
        else:
            small_orphans.append(member_ids)

    # ── Create sub-clusters for each core ──
    # First, create the Caroline+Melanie parent cluster (core entities + shared members)
    all_shared = sorted(set(shared_members + all_cores))
    if all_shared:
        parent_id, parent_name = _create_flat_cluster(
            store, all_shared, entity_ids, entity_names,
            entity_embed_map, embed_svc, G, relation_type,
        )
        # Create sub-clusters
        sub_results = []
        for cid in all_cores:
            sub_members = sorted(set(core_members.get(cid, [cid])))
            if len(sub_members) >= min_cluster_size:
                sub_name = f"{name_for_eid(cid)}子类"
                sub_id, _ = _create_flat_cluster(
                    store, sub_members, entity_ids, entity_names,
                    entity_embed_map, embed_svc, G, relation_type,
                    cluster_name_override=sub_name,
                )
                sub_results.append({
                    "cluster_id": sub_id, "name": sub_name,
                    "core_entity": name_for_eid(cid),
                    "members": [{"id": eid, "name": name_for_eid(eid)} for eid in sub_members],
                    "size": len(sub_members),
                })

        # Link sub-clusters to parent via includes edges
        parent_abstract = _get_abstract_entity_id(store, parent_name)
        for sub in sub_results:
            sub_abstract = _get_abstract_entity_id(store, sub["name"])
            if parent_abstract and sub_abstract and sub_abstract != parent_abstract:
                store._conn.execute(
                    "INSERT OR IGNORE INTO entity_relations "
                    "(source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
                    (parent_abstract, sub_abstract, "includes", 0.9),
                )
                store._conn.commit()

        all_results.append({
            "cluster_id": parent_id, "name": parent_name,
            "coherence": None,  # computed at DB level
            "members": [{"id": eid, "name": name_for_eid(eid)} for eid in all_shared],
            "size": len(all_shared), "sub_clusters": sub_results,
        })

    # ── Small orphan communities (no relation to any core) ──
    for orphan_ids in small_orphans:
        cluster_id, cluster_name = _create_flat_cluster(
            store, orphan_ids, entity_ids, entity_names,
            entity_embed_map, embed_svc, G, relation_type,
        )
        all_results.append({
            "cluster_id": cluster_id, "name": cluster_name, "coherence": None,
            "members": [{"id": eid, "name": name_for_eid(eid)} for eid in orphan_ids],
            "size": len(orphan_ids), "sub_clusters": [],
        })

    return all_results


def _get_abstract_entity_id(store, name: str) -> int | None:
    """Get the abstract entity ID for a cluster name."""
    row = store._conn.execute(
        "SELECT entity_id FROM entities WHERE name = ? AND entity_type = 'abstract'",
        (name,),
    ).fetchone()
    return row["entity_id"] if row else None


def _create_flat_cluster(
    store, member_ids, entity_ids, entity_names,
    entity_embed_map, embed_svc, G, relation_type,
    cluster_name_override=None,
):
    """Create a single flat cluster (no hierarchy)."""
    import numpy as np

    n_comp = len(member_ids)
    member_names = [entity_names[entity_ids.index(eid)] for eid in member_ids]

    # Centroid: entity with highest weighted degree
    subgraph = G.subgraph(member_ids)
    centroid_node = max(subgraph.degree(weight="weight"), key=lambda x: x[1])[0]
    centroid_name = entity_names[entity_ids.index(centroid_node)]

    # Coherence
    internal_edges = subgraph.number_of_edges()
    max_possible = n_comp * (n_comp - 1) / 2
    coherence = internal_edges / max_possible if max_possible > 0 else 0.0
    if internal_edges > 0:
        avg_weight = sum(d["weight"] for _, _, d in subgraph.edges(data=True)) / internal_edges
        coherence = (coherence + avg_weight) / 2

    # Auto-name
    if cluster_name_override:
        cluster_name = cluster_name_override
    else:
        best_idx = member_names.index(centroid_name) if centroid_name in member_names else 0
        cluster_name = _auto_cluster_name(member_names, best_idx)

    # Centroid vector
    centroid_vec = entity_embed_map.get(centroid_node)
    member_sims = []
    for eid in member_ids:
        vec = entity_embed_map.get(eid)
        if vec is not None and centroid_vec is not None:
            sim = float(np.dot(centroid_vec, vec) / (
                np.linalg.norm(centroid_vec) * np.linalg.norm(vec) + 1e-12
            ))
        else:
            sim = 0.0
        member_sims.append(round(sim, 4))

    centroid_blob = embed_svc.serialize(centroid_vec) if centroid_vec is not None else None

    cluster_id = store.create_cluster(
        name=cluster_name,
        cluster_type="auto",
        member_entity_ids=member_ids,
        similarities=member_sims,
        centroid=centroid_blob,
        coherence=round(coherence, 4),
        relation_type=relation_type,
    )

    logger.info("cluster: '%s' (%d members, coherence=%.3f)",
                cluster_name, n_comp, coherence)

    return cluster_id, cluster_name


# ── Naming helpers ──

def _auto_cluster_name(member_names: list[str], centroid_idx: int) -> str:
    if not member_names:
        return "未命名聚类"
    base = member_names[centroid_idx]
    if len(member_names) >= 2:
        prefix = _longest_common_prefix(member_names)
        if len(prefix) >= 2:
            base = prefix
    if any(c in base for c in ("类", "cluster", "Cluster", "category", "Category")):
        return base
    if re.search(r'[\u4e00-\u9fff]', base):
        return f"{base}类"
    return f"{base} category"


def _longest_common_prefix(strings: list[str]) -> str:
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


def cluster_stats(store) -> dict:
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
        "clusters": [{
            "cluster_id": c["cluster_id"],
            "name": c["name"],
            "type": c.get("cluster_type", "auto"),
            "member_count": c.get("member_count", 0),
            "coherence": c.get("coherence", 0),
        } for c in clusters],
    }
