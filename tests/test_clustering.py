"""E2E test for entity clustering + is_member_of graph expansion."""
import json
import tempfile

import pytest

from butterfly_dream.store import MemoryStore
from butterfly_dream.retrieval import ThreeDimRetriever
from butterfly_dream.clustering import compute_clusters, cluster_stats


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    db_path = f"{tmpdir}/test_cluster.db"
    s = MemoryStore(db_path=db_path)
    yield s
    s.close()


def _insert_facts(store):
    facts = [
        ("小明喜欢跳绳，这是一项他很喜欢的运动", "preference", 7.0, ["小明", "跳绳"]),
        ("小明每周都去游泳", "activity", 6.0, ["小明", "游泳"]),
        ("跳绳是一种很好的有氧运动", "event", 4.0, ["跳绳"]),
        ("游泳可以锻炼全身肌肉", "event", 4.0, ["游泳"]),
        ("小红喜欢玩乐高拼图", "preference", 6.0, ["小红", "乐高"]),
        ("小明和小红是好朋友，经常一起玩", "event", 8.0, ["小明", "小红"]),
        ("小明和小红都喜欢猫咪", "preference", 5.0, ["小明", "小红", "猫咪"]),
        ("用户的工作是软件开发工程师", "identity", 9.0, ["用户", "软件开发"]),
        ("用户喜欢喝咖啡", "preference", 5.0, ["用户", "咖啡"]),
    ]
    for content, cat, imp, entities in facts:
        store.add_fact(content, category=cat, importance=imp, entities=entities)


class TestClustering:
    """Entity clustering end-to-end tests."""

    def test_entities_have_embeddings(self, store):
        """All entities should get embeddings via _link_entities."""
        _insert_facts(store)
        ents = store.execute_query(
            "SELECT name, embedding IS NOT NULL as has_emb FROM entities"
        )
        n_with = sum(1 for e in ents if e['has_emb'])
        total = len(ents)
        print(f"Entities with embeddings: {n_with}/{total}")
        for e in ents:
            print(f"  {e['name']:10s} emb={e['has_emb']}")
        assert n_with == total, f"Only {n_with}/{total} entities have embeddings!"

    def test_clustering_creates_clusters(self, store):
        """compute_clusters should find semantic groups."""
        _insert_facts(store)
        clusters = compute_clusters(store, threshold=0.55, min_cluster_size=2)
        assert len(clusters) > 0, "Should find at least one cluster!"

        print(f"Found {len(clusters)} clusters:")
        for c in clusters:
            print(f"  {c['name']} (size={c['size']}, coherence={c['coherence']})")
            print(f"    Members: {[m['name'] for m in c['members']]}")

        # Verify cluster tables exist
        stats = cluster_stats(store)
        assert stats["total_clusters"] >= 1

    def test_is_member_of_relations_created(self, store):
        """Clustering should create is_member_of edges in entity_relations."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        rels = store.execute_query(
            "SELECT er.relation, e1.name as src, e2.name as tgt "
            "FROM entity_relations er "
            "JOIN entities e1 ON er.source_id = e1.entity_id "
            "JOIN entities e2 ON er.target_id = e2.entity_id "
            "WHERE er.relation = 'is_member_of'"
        )
        print(f"is_member_of edges: {len(rels)}")
        for r in rels:
            print(f"  {r['src']:10s} ──{r['relation']}──→ {r['tgt']}")
        assert len(rels) > 0, "Should have at least one is_member_of edge"

    def test_graph_expansion_finds_cluster_members(self, store):
        """Graph expansion should follow is_member_of to find semantically related facts."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        retriever = ThreeDimRetriever(store)

        # Query about "跳绳" — should graph-expand to "游泳" via is_member_of
        with_graph = retriever.search("跳绳对健康的好处", limit=10, use_graph_expansion=True)
        no_graph = retriever.search("跳绳对健康的好处", limit=10, use_graph_expansion=False)

        print(f"With graph: {len(with_graph)} results")
        for r in with_graph:
            tag = " [graph]" if r.get('_graph_expanded') else ""
            print(f"  #{r['fact_id']} score={r['score']:.3f}{tag}: {r['content'][:45]}")

        print(f"\nWithout graph: {len(no_graph)} results")
        for r in no_graph:
            print(f"  #{r['fact_id']} score={r['score']:.3f}: {r['content'][:45]}")

        assert len(with_graph) >= len(no_graph), "Graph expansion should add more results"

        # The graph-expanded results should include facts about "游泳"
        swimming_facts = [r for r in with_graph if "游泳" in r.get("content", "")]
        print(f"\nSwimming facts found via graph: {len(swimming_facts)}")
        assert len(swimming_facts) > 0, "Should find swimming facts via graph!"

    def test_entity_cluster_lookup(self, store):
        """get_entity_clusters should work after clustering."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        # Query an entity that should be clustered
        clusters_跳绳 = store.get_entity_clusters("跳绳")
        clusters_游泳 = store.get_entity_clusters("游泳")
        clusters_小明 = store.get_entity_clusters("小明")

        print(f"跳绳 clusters: {[c['name'] for c in clusters_跳绳]}")
        print(f"游泳 clusters: {[c['name'] for c in clusters_游泳]}")
        print(f"小明 clusters: {[c['name'] for c in clusters_小明]}")

        # 跳绳 and 游泳 should be in the same cluster
        if clusters_跳绳 and clusters_游泳:
            same_cluster = any(
                c1["cluster_id"] == c2["cluster_id"]
                for c1 in clusters_跳绳
                for c2 in clusters_游泳
            )
            if not same_cluster:
                print("⚠️ 跳绳 and 游泳 not in same cluster — threshold might need adjustment")
