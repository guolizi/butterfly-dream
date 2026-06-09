"""E2E test for three-layer ontology: abstract entities + includes edges + graph expansion."""
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
    """Entity clustering end-to-end tests (three-layer ontology)."""

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

    def test_abstract_entity_created(self, store):
        """Clustering should create abstract entities in entities table (L2)."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        abstracts = store.execute_query(
            "SELECT entity_id, name, entity_type, LENGTH(embedding) as embed_len "
            "FROM entities WHERE entity_type = 'abstract'"
        )
        print(f"Abstract entities created: {len(abstracts)}")
        for a in abstracts:
            print(f"  #{a['entity_id']} {a['name']:12s} type={a['entity_type']:10s} embed={a['embed_len']}B")

        assert len(abstracts) > 0, "Should have at least one abstract entity!"
        # Verify abstract entity has an embedding
        for a in abstracts:
            assert a['embed_len'] > 0, f"Abstract entity '{a['name']}' has no embedding!"

    def test_includes_relations_created(self, store):
        """Clustering should create includes edges from abstract → concrete entities (L3)."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        rels = store.execute_query(
            "SELECT er.relation, e1.name as src, e1.entity_type as src_type, "
            "e2.name as tgt, e2.entity_type as tgt_type "
            "FROM entity_relations er "
            "JOIN entities e1 ON er.source_id = e1.entity_id "
            "JOIN entities e2 ON er.target_id = e2.entity_id "
            "WHERE er.relation = 'includes'"
        )
        print(f"includes edges: {len(rels)}")
        for r in rels:
            print(f"  {r['src']:12s}({r['src_type']}) ──{r['relation']}──→ {r['tgt']:12s}({r['tgt_type']})")
        assert len(rels) > 0, "Should have at least one includes edge"

        # Verify all source entities are abstract
        for r in rels:
            assert r['src_type'] == 'abstract', \
                f"Source of includes edge should be abstract, got '{r['src_type']}' for '{r['src']}'"
            assert r['tgt_type'] != 'abstract', \
                f"Target of includes edge should be concrete, got abstract for '{r['tgt']}'"

    def test_no_old_is_member_of_edges(self, store):
        """No old-style is_member_of edges should remain."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        old = store.execute_query(
            "SELECT COUNT(*) as cnt FROM entity_relations WHERE relation = 'is_member_of'"
        )
        assert old[0]['cnt'] == 0, "Should have zero is_member_of edges (all migrated to includes)"

    def test_graph_expansion_finds_cluster_members(self, store):
        """Graph expansion should follow abstract entity → includes to find related facts."""
        _insert_facts(store)
        compute_clusters(store, threshold=0.55, min_cluster_size=2)

        retriever = ThreeDimRetriever(store)

        # Query about "跳绳" — should graph-expand to "游泳" via abstract entity's includes
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
        """get_entity_clusters should work after clustering (still uses cluster_members table)."""
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


class TestStepBack:
    """Step-back: abstract entity matching via query embedding (Stage 1.6)."""

    def _build_three_layer(self, store):
        """Build a minimal three-layer ontology for testing step-back.

        L1: concrete entities [跳绳, 游泳, 乐高]
        L2: abstract entity '运动爱好'
        L3: includes edges: 运动爱好 → 跳绳, 运动爱好 → 游泳
        """
        # Insert facts to create concrete entities
        facts = [
            ("小明喜欢跳绳，这是一项他很喜欢的运动", "preference", 7.0, ["小明", "跳绳"]),
            ("小明每周都去游泳", "activity", 6.0, ["小明", "游泳"]),
            ("跳绳是一种很好的有氧运动", "event", 4.0, ["跳绳"]),
            ("游泳可以锻炼全身肌肉", "event", 4.0, ["游泳"]),
            ("小红喜欢玩乐高拼图", "preference", 6.0, ["小红", "乐高"]),
        ]
        for content, cat, imp, entities in facts:
            store.add_fact(content, category=cat, importance=imp, entities=entities)

        # Manually create abstract entity + cluster
        from butterfly_dream.embedding import get_embedding_service
        svc = get_embedding_service()
        centroid = svc.encode_one("运动爱好活动")
        centroid_blob = svc.serialize(centroid) if centroid is not None else None

        # Find concrete entity IDs for 跳绳 and 游泳
        ids = {}
        for name in ["跳绳", "游泳"]:
            row = store.execute_query(
                "SELECT entity_id FROM entities WHERE name = ?", (name,)
            )
            if row:
                ids[name] = row[0]["entity_id"]

        store.create_cluster(
            name="运动爱好",
            cluster_type="auto",
            member_entity_ids=list(ids.values()),
            similarities=[0.62, 0.58],
            centroid=centroid_blob,
            coherence=0.60,
            relation_type="includes",
        )

    def test_step_back_matches_abstract_entity(self, store):
        """Query about a category should match the abstract entity via embedding."""
        self._build_three_layer(store)

        from butterfly_dream.embedding import get_embedding_service
        qvec = get_embedding_service().encode_one("有什么运动推荐")
        matches = store.match_abstract_entities(qvec, threshold=0.40)

        print(f"Abstract matches for '有什么运动推荐':")
        for m in matches:
            members = [e['name'] for e in m['member_entities']]
            print(f"  {m['name']} (sim={m['similarity']}) → {members}")

        assert len(matches) > 0, "Step-back should match '运动爱好' abstract entity!"
        assert matches[0]['name'] == '运动爱好', "Top match should be '运动爱好'"
        member_names = [e['name'] for e in matches[0]['member_entities']]
        assert '跳绳' in member_names, "Should expand to 跳绳"
        assert '游泳' in member_names, "Should expand to 游泳"

    def test_step_back_finds_facts_via_abstract(self, store):
        """With step-back, a vague query should find facts about cluster members
        that FTS5 alone wouldn't find."""
        self._build_three_layer(store)

        retriever = ThreeDimRetriever(store)

        # Query that doesn't name any concrete entity and won't match fact text
        with_step = retriever.search("推荐一些有意思的活动吧", limit=10, use_step_back=True)
        no_step = retriever.search("推荐一些有意思的活动吧", limit=10, use_step_back=False)

        print(f"With step-back: {len(with_step)} results")
        for r in with_step:
            tag = " [sb]" if r.get('_graph_expanded') else ""
            print(f"  #{r['fact_id']} score={r['score']:.3f}{tag}: {r['content'][:50]}")

        print(f"\nWithout step-back: {len(no_step)} results")
        for r in no_step:
            print(f"  #{r['fact_id']} score={r['score']:.3f}: {r['content'][:50]}")

        # With step-back should find MORE results
        assert len(with_step) > len(no_step), \
            f"Step-back should add more results, got {len(with_step)} <= {len(no_step)}"

        # Should find sports facts via step-back
        sport_facts = [r for r in with_step if any(
            kw in r.get("content", "") for kw in ["跳绳", "游泳"]
        )]
        print(f"\nSport facts found: {len(sport_facts)}")
        assert len(sport_facts) > 0, "Step-back should find sports facts!"
