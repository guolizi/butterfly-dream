# Butterfly Dream v2 — 六层记忆架构检索算法设计

> **设计目标**：为 Butterfly Dream v2 的六层记忆架构（L0-L5）设计一个全面匹配的检索算法，替代当前简陋的 L0+FTS5 + L1+embedding 线性加权方案。

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY CLASSIFIER (QueryRouter)                    │
│  输入: 用户 query                                                    │
│  输出: {query_type, target_layers, heat_zones, routing_hints}       │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER ROUTER (正交于热度路由)                      │
│  决定: 查哪些层 → 每层用什么检索策略 → 每层查多热的记录                │
└──────┬──────┬──────┬──────┬──────┬──────┬───────────────────────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
    ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
    │ L0 │ │ L1 │ │ L2 │ │ L3 │ │ L4 │ │ L5 │
    └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘
      │      │      │      │      │      │
      ▼      ▼      ▼      ▼      ▼      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FUSION ENGINE (Multi-Source Merger)                │
│  输入: 各层检索结果 (异构)                                            │
│  处理: 归一化 → 加权 → 去重 → 重排序 → 上下文组装                     │
│  输出: 统一排序结果 + 结构化上下文包                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 核心设计原则

| 原则 | 说明 |
|------|------|
| **检索路由与热度路由正交** | 先决定查哪些层，再决定查多热的记录 |
| **渐进激活** | 未激活的层优雅降级，不报错 |
| **异构归一化** | 每层检索结果归一化到统一分数空间 [0,1] |
| **插件式扩展** | 新增层/检索源只需实现 `RetrievalSource` 接口 |
| **永远保留** | 冰数据仅 FTS5 索引，冷数据 fp16 量化 embedding |
| **冷却系数叠加** | 热度受 L1-L5 各层叠加影响 |

---

## 2. Query Classifier (查询分类器)

### 2.1 Query Type Taxonomy

| 类型 | 描述 | 示例 | 目标层 |
|------|------|------|--------|
| `fact` | 事实查询 | "Caroline 喜欢什么颜色？" | L0+L1 |
| `causal` | 因果推理 | "为什么 Caroline 选择做心理咨询师？" | L1+L2+L3 |
| `prediction` | 行为预测 | "Caroline 接下来会做什么？" | L3+L4+L5 |
| `contradiction` | 矛盾检测 | "Caroline 的说法前后矛盾吗？" | L1+L2+L5 |
| `relation` | 关系查询 | "Caroline 和 Melanie 是什么关系？" | L1+L2 |
| `emotion` | 情感理解 | "Caroline 最近心情怎么样？" | L1+L2+L3 |
| `narrative` | 叙事查询 | "Caroline 最近经历了什么？" | L1+L2+L4 |
| `persona` | 人格查询 | "Caroline 是个什么样的人？" | L3+L4+L5 |
| `general` | 通用查询 | 无法分类的开放查询 | L0+L1+L2 |

### 2.2 分类算法

```python
class QueryClassifier:
    """
    两阶段分类:
    1. 规则匹配 (快速路径, regex patterns)
    2. LLM 分类 (慢速路径, 仅当规则匹配置信度 < 阈值时)
    """

    RULES = {
        "fact": [
            r"\bwhat\s+(name|color|type|subject|date)\b",
            r"\bwhen\s+(did|was|were|will)\b",
            r"\bwhere\s+(did|was|is|are)\b",
            r"\bhow\s+(many|much|long|old)\b",
            r"\bwhich\s+(one|of|of the)\b",
            # 中文规则
            r"\b(什么|哪个|谁|哪里|什么时候|多少|怎么)\b",
            r"\b(名字|颜色|类型|日期|时间|地点|原因)\b",
        ],
        "causal": [
            r"\bwhy\s+(did|does|is|are|was|were)\b",
            r"\bwhat\s+(caused|led to|resulted in)\b",
            r"\breason\b",
            r"\bbecause\b",
            # 中文规则
            r"\b(为什么|为何|怎么(会|可能)|原因|导致|造成|引发|促使)\b",
            r"\b(因为|所以|因此|于是|结果)\b",
        ],
        "prediction": [
            r"\b(will|would|going to|likely|probably)\b.*\b(next|future|eventually)\b",
            r"\bwhat\s+(will|would)\b.*\bdo\b",
            r"\bprediction\b",
            r"\bexpect\b",
            # 中文规则
            r"\b(会|将要|接下来|下一步|未来|之后|预测|预计|可能)\b",
            r"\b(会怎样|会怎么做|会如何|会有什么)\b",
        ],
        "contradiction": [
            r"\b(contradict|conflict|inconsistent|contradiction)\b",
            r"\b(change|changed|different)\s+(mind|opinion|view)\b",
            # 中文规则
            r"\b(矛盾|冲突|不一致|前后矛盾|自相矛盾)\b",
            r"\b(变了|改变了|不一样了|说过的|说过的话)\b",
        ],
        "relation": [
            r"\b(relationship|relation|connected|related)\b",
            r"\bhow\s+(are|is)\s+\w+\s+and\s+\w+\s+(related|connected)\b",
            # 中文规则
            r"\b(关系|联系|关联|相关|之间)\b",
            r"\b(朋友|同事|家人|恋人|亲戚|认识)\b",
        ],
        "emotion": [
            r"\b(feel|feeling|emotion|mood|sentiment|happy|sad|angry|anxious)\b",
            r"\bhow\s+(is|are)\s+\w+\s+(feeling|doing)\b",
            # 中文规则
            r"\b(心情|情绪|感觉|感受|情感|态度)\b",
            r"\b(开心|难过|生气|焦虑|伤心|快乐|沮丧|害怕|担心|满意)\b",
        ],
        "narrative": [
            r"\b(experience|story|journey|timeline|history|background)\b",
            r"\bwhat\s+(happened|occurred|transpired)\b",
            r"\btell me about\b",
            # 中文规则
            r"\b(经历|故事|经过|历程|过程|背景|历史|回忆)\b",
            r"\b(发生了|做了什么|怎么了|怎么回事)\b",
        ],
        "persona": [
            r"\b(personality|character|temperament|person|like)\b",
            r"\bwhat\s+kind\s+of\s+(person|personality)\b",
            r"\bdescribe\s+\w+\b",
            # 中文规则
            r"\b(性格|人格|个性|为人|什么样的人|怎样的人|特点|特质)\b",
            r"\b(描述|介绍|评价|觉得|认为)\s+\w+\b",
        ],
    }

    def classify(self, query: str) -> QueryIntent:
        # Phase 1: Rule-based
        scores = {}
        for qtype, patterns in self.RULES.items():
            scores[qtype] = sum(
                1 for p in patterns if re.search(p, query.lower())
            )

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score >= 2:  # 高置信度
            return self._build_intent(best_type, query)
        elif best_score == 1:
            return self._build_intent(best_type, query)
        else:
            # Phase 2: LLM fallback (仅当规则无法确定)
            return self._llm_classify(query)
```

### 2.3 路由决策输出

```python
@dataclass
class QueryIntent:
    query_type: str          # fact | causal | prediction | ...
    target_layers: list[int] # [0,1,2,3,4,5] — 要检索的层
    heat_zones: dict         # {layer: [hot, warm, cold, ice]} — 每层查哪些热度
    routing_hints: dict      # 额外提示 (如因果链深度、实体列表)
    query: str               # 原始 query
    embedding: np.ndarray    # query 的 embedding 向量 (预计算)
    psych_probe: Optional[np.ndarray]  # 心理探针向量 (Phase 2+)
```

**心理探针向量 (Psych Probe)**: 将 query 映射到 11 维心理空间（大五人格5 + 能量动机4 + 情绪效价2），用于检索历史上处于相似心理状态时的记忆。Phase 1 为 None，Phase 2+ 由轻量级分类器或 LLM 生成。

> 受"认知驱动的多维混合检索"方案启发。核心创新：不是搜"相似的句子"，而是搜"相似的心情"。详见 §3.7 L5 检索。

---

## 3. 各层检索方式

### 3.1 检索源接口 (Plugin Architecture)

所有层实现统一接口，方便扩展：

```python
class RetrievalSource(ABC):
    """所有检索源的统一接口"""

    layer_id: int           # 0-5
    name: str               # "L0_WorkingMemory", "L1_FactPool", ...

    @abstractmethod
    def retrieve(
        self,
        intent: QueryIntent,
        heat_zone: str,      # "hot" | "warm" | "cold" | "ice"
        limit: int,
    ) -> list[RetrievalResult]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """渐进激活: 检查该层是否有数据"""
        ...

    @abstractmethod
    def estimate_cost(self, intent: QueryIntent) -> float:
        """估计检索成本 (ms 或 ops), 用于成本感知路由"""
        ...
```

### 3.2 L0 — 工作记忆 (Working Memory)

**数据特性**: 原始对话轮次 + 微事实索引 (FTS5 关键词)

**检索方式**:

| 热度 | 策略 | 说明 |
|------|------|------|
| 🔥 热 | FTS5 + 完整 embedding | 最近 N 轮对话全文搜索 |
| 🌤️ 温 | FTS5 仅 | 关键词匹配 |
| ❄️ 冷 | FTS5 仅 | 关键词匹配 |
| 🧊 冰 | FTS5 仅 | 关键词匹配 (全部历史) |

**排序算法**:
```
score = BM25 × 0.5 + recency_decay × 0.3 + query_embed_sim × 0.2
```
- recency_decay: 指数衰减 (半衰期 = 对话轮次数 / 2)
- query_embed_sim: query embedding 与对话片段的余弦相似度

**实现**:
```python
class L0Retrieval(RetrievalSource):
    layer_id = 0
    name = "L0_WorkingMemory"

    def retrieve(self, intent, heat_zone, limit):
        # FTS5 关键词匹配
        fts_results = self.store.fts_search(
            intent.query, limit=limit * 3
        )

        # 热路径: 补充 embedding 排序
        if heat_zone in ("hot", "warm"):
            embed_results = self.store.embedding_search(
                intent.embedding, limit=limit * 2
            )
            return self._merge_fts_embed(fts_results, embed_results)

        return fts_results
```

### 3.3 L1 — 三池 (Fact Pools)

**数据特性**: 事件记录池 / 静态知识池 / 行为模式池 (统一 facts 表, type 字段区分)

**检索方式**:

| 热度 | 策略 | 说明 |
|------|------|------|
| 🔥 热 | 完整 embedding (float32) + FTS5 | 全量语义搜索 |
| 🌤️ 温 | 完整 embedding (float32) + FTS5 | 全量语义搜索 |
| ❄️ 冷 | fp16 量化 embedding + FTS5 | 精度降低但存储减半 |
| 🧊 冰 | FTS5 仅 | 无 embedding |

**三维评分 (现有 ThreeDimRetriever 的增强版)**:
```
score = (α × relevance + β × recency + γ × importance) × trust × heat_zone_weight
```

其中 `heat_zone_weight` 来自热度权重表（🔥=1.0, 🌤️=0.7, ❄️=0.4, 🧊=0.2），而非冷却系数。冷却系数决定事实处于哪个热度区间，热度区间决定权重。详见 §5 热度与检索的交互。

**池间路由**: 根据 query 类型优先检索特定池

| Query 类型 | 优先池 | 理由 |
|-----------|--------|------|
| fact (事件) | 事件记录池 | 有时间锚点 |
| fact (知识) | 静态知识池 | 去时间化稳定知识 |
| prediction | 行为模式池 | 条件-行为规律 |
| emotion | 事件记录池 + 情感维度 | 情感轨迹 |

**实现**:
```python
class L1Retrieval(RetrievalSource):
    layer_id = 1
    name = "L1_FactPool"

    def retrieve(self, intent, heat_zone, limit):
        # 1. 选择优先池
        pool = self._select_pool(intent.query_type)

        # 2. 根据热度选择检索策略
        if heat_zone in ("hot", "warm"):
            return self._dense_retrieve(intent, pool, limit)
        elif heat_zone == "cold":
            return self._quantized_retrieve(intent, pool, limit)
        else:  # ice
            return self._fts_only_retrieve(intent, pool, limit)

    def _dense_retrieve(self, intent, pool, limit):
        # FTS5 候选池 (limit × 3)
        fts_candidates = self._fts_search(intent, pool, limit * 3)

        # Embedding 候选池 (limit × 3)
        embed_candidates = self._embed_search(intent, pool, limit * 3)

        # 合并 + 三维评分
        candidates = self._merge_candidates(fts_candidates, embed_candidates)
        return self._score_and_sort(candidates, intent)
```

### 3.4 L2 — 关系层 (Relation Layer)

**数据特性**: 时间链 / 因果链 / 实体图 / 溯源 / 情感轨迹

**检索方式**: 图遍历 + 路径搜索

| 关系类型 | 检索策略 | 排序信号 |
|---------|---------|---------|
| 时间链 | 时间窗口 + 排序 | 时间接近度 |
| 因果链 | 四层递进 (符号→短程统计→中程LLM→长程叙事) | 因果强度 |
| 实体图 | PPR (Personalized PageRank) | PPR 分数 |
| 溯源 | 来源追踪 | 来源可信度 |
| 情感轨迹 | 情感路径搜索 | 情感强度 + 方向 |
| **因果子图游走** (Phase 2+) | L1 命中触发 L2 因果回溯 | 因果深度 × 语义相关性 |

**算法**:
```python
class L2Retrieval(RetrievalSource):
    layer_id = 2
    name = "L2_RelationLayer"

    def retrieve(self, intent, heat_zone, limit):
        results = []

        # 1. 实体图 PPR (始终执行)
        if intent.routing_hints.get("entities"):
            ppr_results = self._ppr_search(
                seed_entities=intent.routing_hints["entities"],
                alpha=0.85,
                max_depth=2,
                limit=limit,
            )
            results.extend(ppr_results)

        # 2. 因果链 (仅 causal 类型)
        if intent.query_type == "causal":
            causal_results = self._causal_chain_search(
                intent, limit=limit
            )
            results.extend(causal_results)

        # 3. 时间链 (仅 narrative/emotion 类型)
        if intent.query_type in ("narrative", "emotion"):
            timeline = self._timeline_search(
                intent, limit=limit
            )
            results.extend(timeline)

        # 4. 情感轨迹 (仅 emotion 类型)
        if intent.query_type == "emotion":
            emotion_path = self._emotion_trajectory(
                intent, limit=limit
            )
            results.extend(emotion_path)

        return self._dedup_and_sort(results, limit)

    def _ppr_search(self, seed_entities, alpha, max_depth, limit):
        """Personalized PageRank on entity graph"""
        # 使用现有 store.expand_entities_for_retrieval()
        # 增强: 支持 PPR alpha 参数, seed_scores
        return self.store.expand_entities_for_retrieval(
            seed_entities,
            max_depth=max_depth,
            max_results=limit,
            ppr_alpha=alpha,
        )

    def _causal_chain_search(self, intent, limit):
        """四层递进因果链检索"""
        # Level 1: 符号规则 (快速, 关键词匹配)
        symbolic = self._symbolic_causal(intent)

        # Level 2: 短程统计 (co-occurrence 统计)
        if len(symbolic) < limit:
            statistical = self._statistical_causal(intent, limit - len(symbolic))
            symbolic.extend(statistical)

        # Level 3: 中程 LLM (需要时调用)
        if len(symbolic) < limit and self._should_use_llm():
            llm_causal = self._llm_causal(intent, limit - len(symbolic))
            symbolic.extend(llm_causal)

        # Level 4: 长程叙事 (来自 L4)
        if len(symbolic) < limit:
            narrative_causal = self._narrative_causal(intent)
            symbolic.extend(narrative_causal)

        return symbolic

    def _causal_subgraph_walk(self, anchor_fact, max_depth=2):
        """因果子图游走: 从锚点事件出发, 沿因果边反向游走

        当 L1 检索命中某个高相关性事实时, 触发 L2 因果回溯:
        - 沿着 Caused_By 边回溯前因 (反向)
        - 沿着 Led_To 边追踪后果 (正向)
        - 返回 1~2 跳的因果子图

        评分: 因果深度 × 语义相关性
        - 深度 1 的直接因果: × 1.0
        - 深度 2 的间接因果: × 0.5
        """
        # 获取锚点事实的因果邻域
        causes = self.store.get_causal_predecessors(
            anchor_fact, max_depth=max_depth
        )
        effects = self.store.get_causal_successors(
            anchor_fact, max_depth=max_depth
        )

        subgraph = []
        for cause in causes:
            subgraph.append({
                "source": "L2",
                "type": "causal_subgraph",
                "relation": "caused_by",
                "content": cause["description"],
                "anchor": anchor_fact["description"],
                "depth": cause["depth"],
                "score": 1.0 / cause["depth"],  # 深度越近分数越高
            })

        for effect in effects:
            subgraph.append({
                "source": "L2",
                "type": "causal_subgraph",
                "relation": "led_to",
                "content": effect["description"],
                "anchor": anchor_fact["description"],
                "depth": effect["depth"],
                "score": 1.0 / effect["depth"],
            })

        return subgraph
```

### 3.5 L3 — 抽象层 (Abstraction Layer)

**数据特性**: 管道式处理 (不存储, 产出持久化在 L1 的行为模式池/静态知识池)

**检索方式**: 模式匹配 + 知识检索 + Parent-Child 源事实回溯

| 检索类型 | 策略 | 说明 |
|---------|------|------|
| 模式发现 | 聚类匹配 | query embedding → 最近聚类中心 |
| 知识归纳 | 语义检索 | query → L1 静态知识池 (带抽象标记) |
| **Parent-Child 回溯** | 抽象→源事实映射 | 检索 L3 抽象时，通过 `abstracts_from` 带回关联的 L1 原始事实 |

**Parent-Child 机制**: L3 检索结果不仅返回抽象事实本身，还通过 `abstracts_from` 映射带回其下属的 L1 源事实。这样 Agent 既看到宏观模式，又看到具体证据。映射关系在 `fact_relations` 表中维护。

```
检索命中 L3 抽象事实 A
    ↓
通过 fact_relations.abstracts_from 找到 A 关联的 L1 源事实 [F1, F2, F3]
    ↓
返回结构化包: {abstract: A, source_facts: [F1, F2, F3]}
```

**排序算法**:
```
score = cluster_coherence × semantic_similarity × (1 + 0.2 × log(source_count))
```
- `cluster_coherence`: 聚类内平均相似度（GMM 分量的平均后验概率）
- `semantic_similarity`: query embedding 与聚类中心的相似度
- `source_count`: 该抽象事实关联的 L1 源事实数量（更多源事实 → 更可靠的抽象）

**聚类匹配使用马氏距离 (Mahalanobis distance)** 替代余弦相似度，考虑 GMM 各分量的协方差结构——一个狭长的聚类在某个方向上应"容忍"更大的距离。详见下方算法。

**算法**:
```python
class L3Retrieval(RetrievalSource):
    layer_id = 3
    name = "L3_AbstractionLayer"

    def retrieve(self, intent, heat_zone, limit):
        # L3 不直接存储数据, 产出在 L1 池中
        # 检索时: 从 L1 中筛选出 "抽象" 标记的事实

        # 1. 找到 query 所属的聚类 (马氏距离)
        clusters = self._find_relevant_clusters_mahalanobis(
            intent.embedding, top_k=3
        )

        # 2. 从聚类中提取抽象事实 + Parent-Child 回溯
        abstract_facts = []
        for cluster in clusters:
            abstract = self.store.get_cluster_abstract(cluster.cluster_id)
            if not abstract:
                continue

            # Parent-Child: 带回 L1 源事实
            source_facts = self.store.get_abstract_source_facts(
                abstract.fact_id  # 通过 fact_relations.abstracts_from
            )

            abstract_facts.append({
                "abstract": abstract,
                "source_facts": source_facts,
                "cluster_id": cluster.cluster_id,
                "mahalanobis_dist": cluster.distance,
            })

        # 3. 排序: 聚类相干性 × 语义相似度 × 源事实数量增益
        return self._score_abstracts(abstract_facts, intent, limit)

    def _find_relevant_clusters_mahalanobis(self, query_embed, top_k):
        """使用马氏距离找到与 query 最相关的 GMM 聚类

        马氏距离考虑 GMM 各分量的协方差结构:
            d(x, μ_k) = sqrt((x - μ_k)^T Σ_k^{-1} (x - μ_k))

        相比余弦相似度:
        - 狭长聚类在短轴方向更"严格"，长轴方向更"宽容"
        - 圆形聚类退化为欧氏距离
        - 无协方差信息时回退到余弦相似度
        """
        clusters = self.store.get_all_clusters()

        # 获取 GMM 参数 (每个分量的 μ_k, Σ_k)
        gmm_params = self.store.get_gmm_parameters()

        scored = []
        for c in clusters:
            if not c.get("centroid"):
                continue

            if gmm_params and c.cluster_id in gmm_params:
                mu = gmm_params[c.cluster_id]["mean"]
                cov_inv = gmm_params[c.cluster_id]["cov_inv"]  # 预计算 Σ_k^{-1}
                diff = query_embed - mu
                dist = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
                similarity = 1.0 / (1.0 + dist)  # 距离→相似度转换
            else:
                # 无 GMM 参数时回退到余弦相似度
                similarity = cosine_similarity(query_embed, c["centroid"])

            scored.append((c, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ClusterMatch(c, s) for c, s in scored[:top_k]]
```

### 3.6 L4 — 叙事层 (Narrative Layer)

**数据特性**: 双层叙事 (持久化人生主干 + 按需动态细节)

**检索方式**:

| 检索类型 | 策略 | 说明 |
|---------|------|------|
| 主干检索 | 版本化叙事匹配 | query → 最匹配的叙事主干版本 |
| 动态细节 | 按需构建 | 基于主干 + query → LLM 生成细节 |
| **时间折叠** (Phase 2+) | L4 摘要 → 时间窗 → L1 过滤 | 宏观问题先命中 L4 叙事，再用时间窗过滤 L1 细节 |

**时间折叠机制**: 宏观问题（"我大学四年怎么过的"）→ 先命中 L4 叙事摘要 → 提取时间范围 `[t_start, t_end]` → 作为 L1 检索的额外过滤条件。微观问题（"昨天发生了什么"）→ 直接 L1 检索，不经过 L4。这兼顾了宏观视野（不迷失在细节中）和微观精度，同时降低向量库的检索成本。

```
Query: "我大学四年怎么过的"
    ↓
L4 命中叙事摘要: "2021-2025: 考研与失恋的交织期" (time_range: [2021-09, 2025-06])
    ↓
L1 检索自动附加时间过滤: time BETWEEN '2021-09' AND '2025-06'
    ↓
返回: L4 叙事摘要 + L1 该时间窗内的关键事实
```

**算法**:
```python
class L4Retrieval(RetrievalSource):
    layer_id = 4
    name = "L4_NarrativeLayer"

    def retrieve(self, intent, heat_zone, limit):
        # 1. 获取当前叙事主干 (最新版本)
        narrative = self.store.get_latest_narrative()

        if not narrative:
            return []  # 渐进激活: 无叙事数据

        # 2. 计算 query 与叙事主干的语义相关性
        relevance = self._narrative_relevance(intent, narrative)

        if relevance < 0.3:
            return []  # 不相关, 跳过 L4

        # 3. 时间折叠: 提取叙事时间窗作为 routing_hints
        if narrative.get("time_range"):
            intent.routing_hints["narrative_time_window"] = narrative["time_range"]
            intent.routing_hints["narrative_id"] = narrative.get("id")

        # 4. 如果 query 需要细节, 按需构建
        if intent.query_type in ("narrative", "persona", "causal"):
            details = self._build_dynamic_details(
                narrative, intent, limit
            )
            return details

        # 5. 否则返回叙事主干摘要
        return [{
            "source": "L4",
            "content": narrative["summary"],
            "version": narrative["version"],
            "time_range": narrative.get("time_range"),
            "relevance": relevance,
            "score": relevance,
        }]

    def _narrative_relevance(self, intent, narrative):
        """计算 query 与叙事主干的语义相关性"""
        # 使用 narrative 的 embedding
        narr_embed = narrative.get("embedding")
        if narr_embed is not None:
            return cosine_similarity(intent.embedding, narr_embed)
        return 0.5  # 无 embedding 时中性值
```

### 3.7 L5 — 灵魂层 (Soul Layer)

**数据特性**: 人格模型 (结构化) + 行为预测 (概率分布)

**检索方式**:

| 检索类型 | 策略 | 说明 |
|---------|------|------|
| 人格匹配 | 结构化属性匹配 | query → 人格维度匹配 |
| 行为预测 | 概率分布检索 | query → 最相关的行为预测 |
| 矛盾检测 | 惊讶度计算 | query → 与人格模型的偏离度 |
| 心理探针检索 (Phase 2+) | 心理状态匹配 | query 心理探针 → 历史相似心理状态下的记忆 |
| 反差检索 (Phase 2+) | 预测-实际反例匹配 | 检索历史上预测失败的反例记忆，用于自我修正 |

**心理探针检索 (Psych Probe Retrieval)**: 核心创新——不是搜"相似的句子"，而是搜"相似的心情"。当 `intent.psych_probe` 不为 None 时，检索历史上处于相似心理状态时的记忆和应对结果。

```
用户 query: "我今天好累，什么都不想干"
    ↓
QueryClassifier 生成心理探针: [开放性=0.3, 尽责性=0.2, 外向性=0.1, ...]
    ↓
L5 检索: 找到历史上心理探针最相似的 N 条记忆
    ↓
返回: "上次类似状态(2024-10-05) → 画画释放情绪 → 效果良好"
       "再上次(2024-07-12) → 找Melanie聊天 → 效果更好"
```

**算法**:
```python
class L5Retrieval(RetrievalSource):
    layer_id = 5
    name = "L5_SoulLayer"

    def retrieve(self, intent, heat_zone, limit):
        # 1. 获取人格模型
        persona = self.store.get_persona_model()

        if not persona:
            return []  # 渐进激活

        results = []

        # 2. 人格匹配 (persona 类型)
        if intent.query_type == "persona":
            results.extend(self._match_persona(intent, persona))

        # 3. 行为预测 (prediction 类型)
        if intent.query_type == "prediction":
            results.extend(self._match_prediction(intent, persona))

        # 4. 矛盾检测 (contradiction 类型)
        if intent.query_type == "contradiction":
            results.extend(self._detect_contradiction(intent, persona))

        # 5. 心理探针检索 (Phase 2+, 任意 query 类型)
        if intent.psych_probe is not None:
            psych_results = self._psych_probe_retrieve(
                intent.psych_probe, limit=limit
            )
            results.extend(psych_results)

        # 6. 反差检索 (Phase 2+, prediction/contradiction/emotion 类型)
        if intent.query_type in ("prediction", "contradiction", "emotion"):
            contrast_results = self._contrastive_retrieve(
                intent, persona, limit=limit
            )
            results.extend(contrast_results)

        return results

    def _psych_probe_retrieve(self, psych_probe, limit):
        """心理探针检索: 找到历史上心理状态最相似的记忆

        使用马氏距离计算 query 心理探针与历史记忆心理状态的相似度:
            d(probe, memory) = sqrt((probe - μ_mem)^T Σ^{-1} (probe - μ_mem))

        其中 μ_mem 是记忆发生时记录的心理探针向量，Σ 是全局协方差矩阵。
        """
        # 获取所有带心理探针标记的历史记忆
        memories = self.store.get_psych_probed_memories()

        if not memories:
            return []

        scored = []
        for mem in memories:
            mem_probe = mem.get("psych_probe")
            if mem_probe is None:
                continue

            # 马氏距离 (或余弦相似度兜底)
            dist = mahalanobis_distance(psych_probe, mem_probe)
            similarity = 1.0 / (1.0 + dist)

            scored.append({
                "source": "L5",
                "type": "psych_probe",
                "content": mem["content"],
                "timestamp": mem["timestamp"],
                "psych_similarity": similarity,
                "outcome": mem.get("outcome", ""),  # 应对结果
                "score": similarity,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

**反差检索 (Contrastive Retrieval)**: 检索历史上"预测失败"的反例记忆——当 L5 预测用户应该做 A，但用户实际做了 B，这些 B 就是反差记忆。与矛盾检测互补：矛盾检测输出"当前 query 与人格模型的偏离度"（一个分数），反差检索输出"历史上预测失败的具体记忆"（一组证据）。

```
L5 预测: "Caroline 压力大时会画画放松"（概率 0.8）
当前: Caroline 压力很大，但没有画画

标准检索: 找到画画放松的记忆 → "建议你去画画"
反差检索: 找到上次预测"会画画"但实际没画的记忆
  → "2024-10-05: 压力大但没画画，因为被 deadline 压着"
  → "2024-07-12: 压力大但选择了跑步，因为画画工具不在身边"

结果: Agent 不是盲目建议"去画画"，而是说
  "上次压力大你也没画画，因为被 deadline 压着。这次要不要试试短时间放松？"
```

**触发条件**: `prediction` 类型（自动附带，自我修正预测）、`contradiction` 类型（提供矛盾的历史证据）、`emotion` 类型（可选附带，发现情绪变化的反常模式）。

**反差记忆的存储**: 反差检索需要"预测 vs 实际"的对比数据。在睡眠周期 L5 更新阶段记录：

```sql
CREATE TABLE prediction_counterfactuals (
    id INTEGER PRIMARY KEY,
    person TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    predicted_behavior TEXT NOT NULL,   -- 预测的行为
    predicted_prob REAL NOT NULL,       -- 预测概率
    actual_behavior TEXT NOT NULL,      -- 实际发生的行为
    actual_match REAL NOT NULL,         -- 匹配度 [0, 1]
    trigger_event TEXT,                 -- 触发反差的外部事件
    prediction_context TEXT,            -- 预测时的上下文
    embedding BLOB                      -- 用于检索的向量
);
```

```python
    def _contrastive_retrieve(self, intent, persona, limit):
        """反差检索: 找到历史上预测失败的反例记忆

        核心思路:
        1. 获取当前人格模型的预测分布
        2. 找到历史上"预测概率高但实际行为不同"的记忆
        3. 这些记忆是人格模型的"反例"——能揭示模型的盲区

        评分公式:
            contrast_score = predicted_prob × (1 - actual_match)
                          × recency_weight × trigger_relevance

        其中:
        - predicted_prob: 人格模型对该行为的预测概率
        - actual_match: 实际行为与预测的匹配度 (0=完全不匹配)
        - recency_weight: 时间衰减
        - trigger_relevance: 与当前 query 的语义相关性
        """
        # 获取所有带预测-实际对比标记的记忆
        counterfactuals = self.store.get_prediction_counterfactuals()

        if not counterfactuals:
            return []

        scored = []
        for cf in counterfactuals:
            predicted_prob = cf.get("predicted_prob", 0.5)
            actual_match = cf.get("actual_match", 1.0)

            # 反差分数: 预测越自信、实际越偏离 → 越值得检索
            contrast_score = predicted_prob * (1 - actual_match)

            if contrast_score < 0.2:
                continue  # 低反差价值，跳过

            # 与当前 query 的相关性
            query_relevance = self._contrastive_relevance(
                intent, cf
            )

            scored.append({
                "source": "L5",
                "type": "contrastive",
                "content": cf["description"],
                "predicted": cf["predicted_behavior"],
                "actual": cf["actual_behavior"],
                "trigger": cf.get("trigger_event", ""),
                "contrast_score": contrast_score,
                "query_relevance": query_relevance,
                "score": contrast_score * query_relevance,
                "timestamp": cf["timestamp"],
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _match_persona(self, intent, persona):
        """人格维度匹配"""
        # 将 query 映射到人格维度空间
        query_dims = self._query_to_dimensions(intent.query)
        matched = []

        for dim, value in persona["dimensions"].items():
            if dim in query_dims:
                matched.append({
                    "source": "L5",
                    "dimension": dim,
                    "value": value,
                    "relevance": query_dims[dim],
                    "score": query_dims[dim] * value.get("confidence", 0.5),
                })

        return matched

    def _match_prediction(self, intent, persona):
        """行为预测检索"""
        predictions = persona.get("predictions", [])

        # 找到与 query 最相关的预测
        scored = []
        for pred in predictions:
            sim = self._prediction_relevance(intent, pred)
            scored.append((pred, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [{
            "source": "L5",
            "type": "prediction",
            "content": p["description"],
            "probability": p["probability"],
            "relevance": s,
            "score": s * p["probability"],
        } for p, s in scored[:3]]

    def _detect_contradiction(self, intent, persona):
        """矛盾检测: 惊讶度 × 置信度"""
        # 计算 query 与人格模型的偏离度
        surprise = self._compute_surprise(intent, persona)
        confidence = persona.get("confidence", 0.5)

        return [{
            "source": "L5",
            "type": "contradiction_check",
            "surprise": surprise,
            "confidence": confidence,
            "contradiction_score": surprise * confidence,
            "score": surprise * confidence,
        }]
```

---

## 4. 跨层路由策略

### 4.1 路由矩阵

```
Query Type    | L0  | L1  | L2  | L3  | L4  | L5
--------------|-----|-----|-----|-----|-----|-----
fact          | ✅  | ✅  | ⬜  | ⬜  | ⬜  | ⬜
causal        | ✅  | ✅  | ✅  | ✅  | ⬜  | ⬜
prediction    | ⬜  | ✅  | ✅  | ✅  | ✅  | ✅
contradiction | ⬜  | ✅  | ✅  | ⬜  | ⬜  | ✅
relation      | ⬜  | ✅  | ✅  | ⬜  | ⬜  | ⬜
emotion       | ✅  | ✅  | ✅  | ✅  | ⬜  | ⬜
narrative     | ⬜  | ✅  | ✅  | ✅  | ✅  | ⬜
persona       | ⬜  | ✅  | ⬜  | ✅  | ✅  | ✅
general       | ✅  | ✅  | ✅  | ⬜  | ⬜  | ⬜
```

- ✅ = 必须检索
- ⬜ = 可选 (根据 query 具体内容决定)

### 4.2 渐进激活降级

```python
def resolve_active_layers(intent: QueryIntent, store) -> list[int]:
    """渐进激活: 检查每层是否有数据, 无数据则优雅降级"""
    active = []

    # L0: 始终可用 (工作记忆)
    active.append(0)

    # L1: 始终可用 (facts 表)
    active.append(1)

    # L2: 检查是否有实体关系
    if store.has_entity_relations():
        active.append(2)

    # L3: 检查是否有聚类
    if store.has_clusters():
        active.append(3)

    # L4: 检查是否有叙事主干
    if store.has_narrative():
        active.append(4)

    # L5: 检查是否有人格模型
    if store.has_persona_model():
        active.append(5)

    # 取交集: 目标层 ∩ 可用层
    return [l for l in intent.target_layers if l in active]
```

### 4.3 成本感知路由

```python
def cost_aware_route(intent: QueryIntent, active_layers: list[int]) -> list[int]:
    """
    成本感知路由: 对延迟敏感的场景 (chat) 减少深层检索
    对深度推理场景 (qa, longterm) 启用全量检索
    """
    scenario = intent.routing_hints.get("scenario", "balanced")

    if scenario == "chat":
        # 聊天场景: 快速响应, 仅 L0+L1
        return [l for l in active_layers if l <= 1]

    if scenario == "qa":
        # QA 场景: 需要深度检索
        return active_layers

    if scenario == "longterm":
        # 长期记忆: 启用 L3-L5
        return active_layers

    # balanced: 根据 query 类型智能选择
    return active_layers
```

---

## 5. 热度与检索的交互

### 5.1 热度路由 (正交于检索路由)

```
检索路由: 决定查哪些层
热度路由: 决定查多热的记录 (同一层内)
```

### 5.2 冷却系数叠加

每条事实的冷却系数 = 各层叠加影响（乘性系数，>1 加速冷却，<1 减速冷却）：

```
冷却系数 = 1.0
  × L1_factor:  importance ≤ 3 → 1.5  (低重要性加速冷却)
  × L2_factor:  relation_density_high → 0.5  (高关系密度减速冷却)
  × L3_factor:  has_abstract → 2.0  (已抽象替代加速冷却)
  × L4_factor:  is_narrative_key → 0.3  (叙事关键节点减速冷却)
  × L5_factor:  is_core_trait → 0.3  (核心特质减速冷却)

综合冷却系数 = 各层系数相乘
- 系数 < 1 → 冷却减速（保留在热区更久）
- 系数 > 1 → 冷却加速（更快冷却）
```

**例子：**
```
事实: "Caroline 2023-05-08 参加了支持小组"
  L1: importance=3 → 加速 × 1.5
  L2: 12条关系链 → 减速 × 0.5
  L3: 已抽象替代 → 加速 × 2.0
  L4: 叙事关键节点 → 减速 × 0.3
  ────────────────────────────
  综合: 1.5 × 0.5 × 2.0 × 0.3 = 0.45
  → 冷却很慢，长期保持温/热 ✅
```

> 与主文档 §7.7 冷却系数公式一致。

### 5.3 热度 → 检索策略映射

```python
HEAT_STRATEGY = {
    "hot": {
        "embedding": "float32_full",     # 完整精度 embedding
        "fts": "full_query",              # 完整 query
        "graph": "full_ppr",              # 完整 PPR
        "priority": 1.0,                  # 检索优先级
    },
    "warm": {
        "embedding": "float32_full",
        "fts": "full_query",
        "graph": "pruned_ppr",            # 剪枝 PPR (深度=1)
        "priority": 0.8,
    },
    "cold": {
        "embedding": "float16_quantized", # fp16 量化
        "fts": "full_query",
        "graph": "none",                  # 无图检索
        "priority": 0.4,
    },
    "ice": {
        "embedding": "none",              # 无 embedding
        "fts": "keyword_only",            # 仅关键词
        "graph": "none",
        "priority": 0.1,
    },
}
```

### 5.4 热度晋升/降级

```
热晋升 (查询驱动):
  - 缺失计数 ≥ 3 的冰数据 → 晋升为冷
  - 查询命中率高的冷数据 → 晋升为温

冷晋升 (睡眠周期):
  - 批量 LLM 提取冰数据摘要 → 写入 L1
  - 摘要 embedding (fp16) → 冷区

热路径晋升:
  - importance ≥ 0.9 的事实 → 立即写入 L1 (跳过 L0)

降级:
  - 长期未命中 + 低冷却系数 → 降级
  - 被 L3 抽象替代的原始事实 → 降级
```

---

## 6. 结果融合/排序算法

### 6.1 异构结果归一化

每层检索结果格式:

```python
@dataclass
class RetrievalResult:
    source_layer: int       # 0-5
    content: str            # 事实内容
    score: float            # 层内归一化分数 [0, 1]
    confidence: float       # 置信度 [0, 1]
    metadata: dict          # 层特定元数据
```

### 6.2 跨层融合算法

#### 6.2.1 核心公式

融合引擎将各层异构检索结果归一化后，使用**增强评分公式**：

```
FusionScore = α·relevance + β·recency + γ·importance + δ·mood_resonance(π_curr, π_doc)
```

| 因子 | 来源 | 说明 |
|:----|:----|:----|
| `relevance` | 各层检索的语义相关性 | 归一化到 [0, 1] |
| `recency` | 时间衰减 | 热度权重 (🔥=1.0, 🌤️=0.7, ❄️=0.4, 🧊=0.2) |
| `importance` | 事实重要性 | 来自 L1 三维评分 |
| `mood_resonance` | **心境一致性共振** (Phase 2+) | 当前 GMM 模式与记忆发生时的 GMM 模式的相似度 |

**心境一致性共振 (Mood Resonance)**: 计算当前 query 的 GMM 模式分布 π_curr 与历史记忆发生时的 GMM 模式分布 π_doc 的相似度。当两者模式相似时（如都处于"应激放纵模式"），即使语义稍弱也给予提权。这使 Agent 表现出"共情能力"——回忆起的不仅是事情，更是"当时的心境"。

```
mood_resonance = cosine_similarity(π_curr, π_doc)
```

其中 π 是 GMM 模式的后验概率分布（3~5 维）。Phase 1 无 GMM 数据时，mood_resonance = 0.5（中性值）。

#### 6.2.2 跨层触发机制 (Phase 2+)

FusionEngine 在融合过程中，根据各层检索结果**级联触发**其他层的补充检索：

```
L1 命中高相关性事实 (relevance > 0.8)
    ↓ 自动触发
L2 因果子图游走: 沿 Caused_By / Led_To 边提取 1~2 跳因果子图
    ↓
合并到上下文包: {fact, causes, effects}

L4 命中叙事摘要 (含 time_range)
    ↓ 自动触发
L1 时间窗过滤: 附加 time BETWEEN [t_start, t_end]
    ↓
合并到上下文包: {narrative_summary, time_window_facts}
```

#### 6.2.3 算法

```python
class FusionEngine:
    """融合引擎: 异构结果归一化 → 跨层加权 → 去重 → MMR 重排序 → 结构化输出"""

    LAYER_BASE_WEIGHTS = {0: 0.10, 1: 0.30, 2: 0.25, 3: 0.15, 4: 0.10, 5: 0.10}

    QUERY_TYPE_ADJUSTMENTS = {
        "fact":         {0: 0.15, 1: 0.50, 2: 0.20, 3: 0.10, 4: 0.05, 5: 0.00},
        "causal":       {0: 0.05, 1: 0.25, 2: 0.40, 3: 0.20, 4: 0.10, 5: 0.00},
        "prediction":   {0: 0.00, 1: 0.15, 2: 0.20, 3: 0.25, 4: 0.20, 5: 0.20},
        "contradiction":{0: 0.00, 1: 0.30, 2: 0.30, 3: 0.10, 4: 0.10, 5: 0.20},
        "relation":     {0: 0.00, 1: 0.30, 2: 0.50, 3: 0.10, 4: 0.10, 5: 0.00},
        "emotion":      {0: 0.10, 1: 0.30, 2: 0.30, 3: 0.20, 4: 0.10, 5: 0.00},
        "narrative":    {0: 0.00, 1: 0.20, 2: 0.25, 3: 0.20, 4: 0.35, 5: 0.00},
        "persona":      {0: 0.00, 1: 0.20, 2: 0.05, 3: 0.20, 4: 0.25, 5: 0.30},
        "general":      {0: 0.15, 1: 0.35, 2: 0.25, 3: 0.10, 4: 0.10, 5: 0.05},
    }

    def fuse(
        self,
        layer_results: dict[int, list[RetrievalResult]],
        intent: QueryIntent,
        limit: int,
    ) -> list[RetrievalResult]:
        """融合所有层的结果"""

        # 0. 跨层触发: 检查是否需要级联补充检索
        self._cascade_trigger(layer_results, intent)

        # 1. 获取层权重
        weights = self.QUERY_TYPE_ADJUSTMENTS.get(
            intent.query_type, self.LAYER_BASE_WEIGHTS
        )

        # 2. 层内归一化 (确保每层分数在 [0, 1])
        normalized = {}
        for layer, results in layer_results.items():
            normalized[layer] = self._normalize_layer(results)

        # 3. 跨层加权合并 + 心境一致性共振
        merged = []
        seen_contents = {}  # 去重用

        # 获取当前 GMM 模式分布 (Phase 2+)
        curr_gmm_pi = self._get_current_gmm_pi(intent)

        for layer, results in normalized.items():
            w = weights.get(layer, 0.1)
            for r in results:
                # 心境一致性共振 (Phase 2+)
                mood_boost = 1.0
                if curr_gmm_pi is not None and r.gmm_pi is not None:
                    resonance = cosine_similarity(curr_gmm_pi, r.gmm_pi)
                    mood_boost = 1.0 + 0.3 * resonance  # 最高提权 30%

                # 去重: 内容相似度检查
                content_key = self._content_fingerprint(r.content)
                if content_key in seen_contents:
                    existing = seen_contents[content_key]
                    if r.score > existing.score:
                        existing.score = r.score
                        existing.source_layer = layer
                    continue

                r.score = r.score * w * mood_boost
                seen_contents[content_key] = r
                merged.append(r)

        # 4. MMR 重排序 (多样性与相关性平衡)
        reranked = self._mmr_rerank(merged, intent.embedding, limit)

        # 5. 结构化输出
        return self._assemble_output(reranked, intent)

    def _cascade_trigger(self, layer_results, intent):
        """跨层触发: 根据已有检索结果触发补充检索

        1. L1 高相关性事实 → 触发 L2 因果子图游走
        2. L4 叙事命中 → 触发 L1 时间窗过滤
        """
        # 1. 因果子图游走: L1 高相关性事实触发 L2 回溯
        if 1 in layer_results and 2 in layer_results:
            for r in layer_results[1]:
                if r.score > 0.8 and r.source_layer == 1:
                    l2 = L2Retrieval()
                    subgraph = l2._causal_subgraph_walk(
                        {"description": r.content, "fact_id": r.id}
                    )
                    layer_results.setdefault(2, []).extend(subgraph)

        # 2. 时间折叠: L4 叙事时间窗 → L1 时间过滤
        if 4 in layer_results and 1 in layer_results:
            for r in layer_results[4]:
                if r.metadata and r.metadata.get("time_range"):
                    time_range = r.metadata["time_range"]
                    for l1r in layer_results[1]:
                        if hasattr(l1r, "metadata") and l1r.metadata:
                            l1r.metadata["time_window"] = time_range

    def _mmr_rerank(
        self,
        results: list[RetrievalResult],
        query_embed: np.ndarray,
        limit: int,
        lambda_param: float = 0.7,
    ) -> list[RetrievalResult]:
        """
        Maximal Marginal Relevance 重排序

        score = λ × relevance(query, doc) - (1-λ) × max_similarity(doc, selected)

        λ = 0.7: 偏向相关性
        λ = 0.5: 平衡
        λ = 0.3: 偏向多样性
        """
        if not results:
            return []

        selected = []
        candidates = list(results)

        while len(selected) < limit and candidates:
            best_idx = -1
            best_score = -float("inf")

            for i, candidate in enumerate(candidates):
                # 相关性
                relevance = candidate.score

                # 多样性惩罚: 与已选结果的最大相似度
                max_sim = 0.0
                for sel in selected:
                    sim = self._content_similarity(candidate, sel)
                    max_sim = max(max_sim, sim)

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))

        return selected

    def _assemble_output(
        self,
        results: list[RetrievalResult],
        intent: QueryIntent,
    ) -> dict:
        """组装结构化输出"""
        return {
            "query": intent.query,
            "query_type": intent.query_type,
            "results": results[:10],  # top-10
            "layer_breakdown": self._layer_breakdown(results),
            "context": self._build_context(results, intent),
        }

    def _build_context(self, results, intent):
        """构建结构化上下文包 (供 LLM 使用)"""
        context = {
            "facts": [],       # L1 事实
            "relations": [],   # L2 关系
            "patterns": [],    # L3 模式
            "narrative": None, # L4 叙事
            "persona": None,   # L5 人格
        }

        for r in results:
            if r.source_layer == 1:
                context["facts"].append(r.content)
            elif r.source_layer == 2:
                context["relations"].append(r.metadata)
            elif r.source_layer == 3:
                context["patterns"].append(r.content)
            elif r.source_layer == 4 and context["narrative"] is None:
                context["narrative"] = r.content
            elif r.source_layer == 5 and context["persona"] is None:
                context["persona"] = r.content

        return context
```

### 6.3 内容相似度与去重

```python
def _content_fingerprint(self, content: str) -> str:
    """内容指纹: 用于快速去重"""
    # 使用 jieba 分词后的 token 集合的 hash
    tokens = tokenize(content)
    return hashlib.md5(" ".join(sorted(tokens)).encode()).hexdigest()

def _content_similarity(self, a: RetrievalResult, b: RetrievalResult) -> float:
    """内容相似度: 用于 MMR 多样性计算"""
    # 同层: 使用 embedding 余弦相似度
    if a.source_layer == b.source_layer:
        return cosine_similarity(a.embedding, b.embedding)
    # 跨层: 使用 Jaccard 相似度
    return jaccard_similarity(tokenize(a.content), tokenize(b.content))
```

---

## 7. 完整检索流程

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. QUERY CLASSIFICATION                                              │
│    query → QueryClassifier → QueryIntent                             │
│    {query_type, target_layers, heat_zones, embedding}                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. LAYER RESOLUTION                                                  │
│    QueryIntent.target_layers ∩ active_layers → resolved_layers       │
│    渐进激活检查 + 成本感知路由                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. HEAT ZONE RESOLUTION                                              │
│    对每个 resolved_layer, 根据冷却系数决定查多热的记录                  │
│    hot → warm → cold → ice (按优先级降序)                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. PARALLEL LAYER RETRIEVAL                                          │
│    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                 │
│    │ L0  │ │ L1  │ │ L2  │ │ L3  │ │ L4  │ │ L5  │                 │
│    │FTS5 │ │3D   │ │PPR  │ │Clus-│ │Narr-│ │Pers-│                 │
│    │     │ │Score│ │Caus-│ │ter  │ │ative│ │ona  │                 │
│    │     │ │     │ │al   │ │Match│ │     │ │Match│                 │
│    └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                 │
│       │       │       │       │       │       │                    │
│       └───────┴───────┴───────┴───────┴───────┘                    │
│                              │                                      │
│                              ▼                                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. FUSION                                                            │
│    层内归一化 → 跨层加权 → 去重 → MMR 重排序 → 上下文组装              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT                                                            │
│    {                                                                 │
│      "results": [...],          # 统一排序结果                       │
│      "context": {               # 结构化上下文包                      │
│        "facts": [...],                                               │
│        "relations": [...],                                           │
│        "patterns": [...],                                            │
│        "narrative": "...",                                           │
│        "persona": {...}                                              │
│      },                                                              │
│      "layer_breakdown": {...}   # 每层贡献统计                       │
│    }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. 分阶段实现路径

### Phase 1: MVP (最小可行产品)

**目标**: 替换当前简陋方案, 实现核心六层路由框架

| 组件 | 实现内容 | 工作量 |
|------|---------|--------|
| QueryClassifier | 规则匹配分类器 (8 种类型) | 1-2 天 |
| RetrievalSource 接口 | 抽象基类 + 注册机制 | 0.5 天 |
| L0 检索 | 现有 FTS5 封装为 RetrievalSource | 0.5 天 |
| L1 检索 | 现有 ThreeDimRetriever 封装 + 池间路由 | 1 天 |
| L2 检索 | 现有 PPR 封装 + 时间链检索 | 1 天 |
| FusionEngine v1 | 层内归一化 + 跨层加权 + 简单去重 | 1 天 |
| 渐进激活 | 检查每层是否有数据 | 0.5 天 |
| 热度路由 | 冷却系数计算 + 热度分级检索 | 1 天 |

**Phase 1 交付**: 可运行的检索管道, 支持 fact/causal/relation/general 四种 query 类型

**Phase 1 不包含**:
- L3/L4/L5 检索 (渐进激活降级)
- MMR 重排序
- 成本感知路由
- LLM 分类器

### Phase 2: 增强

**目标**: 增加深层检索 + 重排序 + 成本感知 + 心理探针 + 状态感知

| 组件 | 实现内容 | 工作量 |
|------|---------|--------|
| L3 检索 | 聚类匹配 + Parent-Child 源事实回溯 + 马氏距离 | 2 天 |
| L4 检索 | 叙事主干检索 + 动态细节构建 + 时间折叠 | 2 天 |
| L5 检索 | 人格模型匹配 + 行为预测检索 + 心理探针检索 + 反差检索 | 3 天 |
| MMR 重排序 | 多样性-相关性平衡 | 1 天 |
| 成本感知路由 | 场景感知层选择 | 1 天 |
| 因果链检索 | 四层递进 (符号+统计) + 因果子图游走 | 2 天 |
| 情感轨迹检索 | 情感路径搜索 | 1 天 |
| 心理探针生成 | 轻量级分类器/LLM 生成 query 心理探针向量 | 2 天 |
| 心境一致性共振 | FusionEngine 增加 mood_resonance 评分因子 | 1 天 |
| 跨层触发机制 | FusionEngine 级联触发因果子图 + 时间折叠 | 1 天 |

**Phase 2 交付**: 完整六层检索, 支持所有 query 类型 + 心理状态感知检索 + 跨层级联触发

### Phase 3: 成熟

**目标**: 优化 + 自适应 + 自监督

| 组件 | 实现内容 | 工作量 |
|------|---------|--------|
| LLM Query Classifier | LLM 辅助分类 (低置信度回退) | 2 天 |
| 自适应层权重 | 基于反馈自动调整 QUERY_TYPE_ADJUSTMENTS | 2 天 |
| 自监督热度管理 | 基于检索命中率自动调整冷却系数 | 2 天 |
| 检索缓存 | 相似 query 结果缓存 (LRU) | 1 天 |
| 异步预检索 | 空闲时预计算常见 query 的检索结果 | 2 天 |
| 性能监控 | 每层延迟/召回率/精确率仪表盘 | 1 天 |
| A/B 测试框架 | 检索策略对比测试 | 2 天 |

**Phase 3 交付**: 自适应的智能检索系统

---

## 9. 与现有架构的整合点

### 9.1 现有代码复用

| 现有组件 | 新架构中的角色 | 修改程度 |
|---------|--------------|---------|
| `ThreeDimRetriever` | L1 检索的核心评分引擎 | 封装为 RetrievalSource |
| `store.expand_entities_for_retrieval()` | L2 PPR 检索 | 直接复用 |
| `clustering.py` | L3 聚类信息 | 增加抽象事实检索方法 |
| `embedding.py` | 所有层的 embedding 服务 | 直接复用 |
| `retrieval.py` 的 query 分类逻辑 | QueryClassifier 的规则部分 | 迁移到新类 |
| `store.py` 的 facts 表 | L1 数据源 | 无需修改 |
| `store.py` 的 entity_relations 表 | L2 数据源 | 无需修改 |

### 9.2 新增存储需求

| 新增数据 | 用途 | 存储位置 |
|---------|------|---------|
| `narratives` 表 | L4 叙事主干 | 新表 |
| `persona_models` 表 | L5 人格模型 | 新表 |
| `behavior_predictions` 表 | L5 行为预测 | 新表 |
| `retrieval_cache` 表 | Phase 3 检索缓存 | 新表 |
| `heat_metadata` 列 | 冷却系数追踪 | facts 表新增列 |

### 9.3 向后兼容

- 新检索算法通过 `ButterflyDreamProvider` 的 `search()` 方法暴露
- 现有 `ThreeDimRetriever.search()` 保持不动 (作为 L1 的内部实现)
- 新增 `retrieval_v2.py` 模块, 不修改现有 `retrieval.py`
- 通过配置开关 `use_v2_retrieval: bool` 控制新旧切换

---

## 10. 性能考虑

### 10.1 延迟预算

| 阶段 | 目标延迟 | 说明 |
|------|---------|------|
| Query Classification | < 5ms | 规则匹配 |
| Layer Resolution | < 1ms | 配置查找 |
| Heat Zone Resolution | < 1ms | 冷却系数计算 |
| L0 Retrieval | < 10ms | FTS5 |
| L1 Retrieval | < 50ms | FTS5 + embedding |
| L2 Retrieval | < 100ms | PPR + 图遍历 |
| L3 Retrieval | < 20ms | 聚类匹配 |
| L4 Retrieval | < 30ms | 叙事检索 |
| L5 Retrieval | < 20ms | 人格匹配 |
| Fusion | < 20ms | 归一化 + 重排序 |
| **Total** | **< 250ms** | 全量六层 |

### 10.2 并行策略

```python
# 各层检索并行执行
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {
        executor.submit(source.retrieve, intent, heat, limit):
        layer for layer, source in sources.items()
    }
    for future in as_completed(futures):
        layer = futures[future]
        layer_results[layer] = future.result()
```

### 10.3 缓存策略

```python
class RetrievalCache:
    """LRU 检索缓存 (Phase 3)"""

    def __init__(self, max_size=1000, ttl_seconds=300):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds

    def get(self, query: str, query_type: str) -> Optional[dict]:
        key = self._make_key(query, query_type)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["time"] < self.ttl:
                self.cache.move_to_end(key)
                return entry["result"]
            del self.cache[key]
        return None

    def set(self, query: str, query_type: str, result: dict):
        key = self._make_key(query, query_type)
        self.cache[key] = {"result": result, "time": time.time()}
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
```

---

## 11. 附录: 关键数据结构

### 11.1 QueryIntent

```python
@dataclass
class QueryIntent:
    query_type: str                     # fact | causal | prediction | ...
    target_layers: list[int]            # [0,1,2,3,4,5]
    heat_zones: dict[int, str]          # {0: "hot", 1: "warm", ...}
    routing_hints: dict                 # {scenario, entities, ...}
    query: str                          # 原始 query
    embedding: Optional[np.ndarray]     # 预计算 embedding
```

### 11.2 RetrievalResult

```python
@dataclass
class RetrievalResult:
    source_layer: int                   # 0-5
    source_name: str                    # "L1_FactPool"
    content: str                        # 事实内容
    score: float                        # 归一化分数 [0, 1]
    confidence: float                    # 置信度 [0, 1]
    embedding: Optional[np.ndarray]     # 内容 embedding
    metadata: dict                      # 层特定元数据
    fact_id: Optional[int]              # 关联的事实 ID
```

### 11.3 FusionOutput

```python
@dataclass
class FusionOutput:
    results: list[RetrievalResult]      # 统一排序结果
    context: dict                       # 结构化上下文包
    layer_breakdown: dict[int, int]     # 每层贡献了多少结果
    query_type: str                     # 分类结果
    latency_ms: float                   # 总延迟
```
