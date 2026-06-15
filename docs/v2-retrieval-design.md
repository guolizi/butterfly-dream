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
```

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
```

### 3.5 L3 — 抽象层 (Abstraction Layer)

**数据特性**: 管道式处理 (不存储, 产出持久化在 L1 的行为模式池/静态知识池)

**检索方式**: 模式匹配 + 知识检索

| 检索类型 | 策略 | 说明 |
|---------|------|------|
| 模式发现 | 聚类匹配 | query embedding → 最近聚类中心 |
| 知识归纳 | 语义检索 | query → L1 静态知识池 (带抽象标记) |

**算法**:
```python
class L3Retrieval(RetrievalSource):
    layer_id = 3
    name = "L3_AbstractionLayer"

    def retrieve(self, intent, heat_zone, limit):
        # L3 不直接存储数据, 产出在 L1 池中
        # 检索时: 从 L1 中筛选出 "抽象" 标记的事实
        # 使用现有的聚类信息 (clustering.py)

        # 1. 找到 query 所属的聚类
        clusters = self._find_relevant_clusters(intent.embedding, top_k=3)

        # 2. 从聚类中提取抽象事实
        abstract_facts = []
        for cluster in clusters:
            facts = self.store.get_cluster_abstracts(cluster.cluster_id)
            abstract_facts.extend(facts)

        # 3. 排序: 聚类相干性 × 语义相似度
        return self._score_abstracts(abstract_facts, intent, limit)

    def _find_relevant_clusters(self, query_embed, top_k):
        """找到与 query 最相关的实体聚类"""
        # 使用聚类中心 embedding 进行最近邻搜索
        clusters = self.store.get_all_clusters()
        scored = []
        for c in clusters:
            if c.get("centroid"):
                sim = cosine_similarity(query_embed, c["centroid"])
                scored.append((c, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
```

### 3.6 L4 — 叙事层 (Narrative Layer)

**数据特性**: 双层叙事 (持久化人生主干 + 按需动态细节)

**检索方式**:

| 检索类型 | 策略 | 说明 |
|---------|------|------|
| 主干检索 | 版本化叙事匹配 | query → 最匹配的叙事主干版本 |
| 动态细节 | 按需构建 | 基于主干 + query → LLM 生成细节 |

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

        # 3. 如果 query 需要细节, 按需构建
        if intent.query_type in ("narrative", "persona", "causal"):
            details = self._build_dynamic_details(
                narrative, intent, limit
            )
            return details

        # 4. 否则返回叙事主干摘要
        return [{
            "source": "L4",
            "content": narrative["summary"],
            "version": narrative["version"],
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

        return results

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

```python
class FusionEngine:
    """
    多源异构结果融合引擎

    核心思路:
    1. 每层结果先层内归一化到 [0, 1]
    2. 跨层加权合并 (层权重取决于 query 类型)
    3. 去重 (内容相似度 ≥ 0.85 合并)
    4. 重排序 (MMR: Maximal Marginal Relevance)
    5. 上下文组装 (按层结构化输出)
    """

    # 每层的基础权重 (query 类型可调整)
    LAYER_BASE_WEIGHTS = {
        0: 0.10,  # L0 工作记忆 (低权重, 临时性)
        1: 0.30,  # L1 事实池 (核心)
        2: 0.25,  # L2 关系层
        3: 0.15,  # L3 抽象层
        4: 0.10,  # L4 叙事层
        5: 0.10,  # L5 灵魂层
    }

    # Query 类型 → 层权重调整
    QUERY_TYPE_ADJUSTMENTS = {
        "fact":         {0: 0.20, 1: 0.50, 2: 0.20, 3: 0.05, 4: 0.03, 5: 0.02},
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

        # 1. 获取层权重
        weights = self.QUERY_TYPE_ADJUSTMENTS.get(
            intent.query_type, self.LAYER_BASE_WEIGHTS
        )

        # 2. 层内归一化 (确保每层分数在 [0, 1])
        normalized = {}
        for layer, results in layer_results.items():
            normalized[layer] = self._normalize_layer(results)

        # 3. 跨层加权合并
        merged = []
        seen_contents = {}  # 去重用

        for layer, results in normalized.items():
            w = weights.get(layer, 0.1)
            for r in results:
                # 去重: 内容相似度检查
                content_key = self._content_fingerprint(r.content)
                if content_key in seen_contents:
                    # 保留更高分的
                    existing = seen_contents[content_key]
                    if r.score > existing.score:
                        existing.score = r.score
                        existing.source_layer = layer
                    continue

                r.score = r.score * w  # 应用层权重
                seen_contents[content_key] = r
                merged.append(r)

        # 4. MMR 重排序 (多样性与相关性平衡)
        reranked = self._mmr_rerank(merged, intent.embedding, limit)

        # 5. 结构化输出
        return self._assemble_output(reranked, intent)

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

**目标**: 增加深层检索 + 重排序 + 成本感知

| 组件 | 实现内容 | 工作量 |
|------|---------|--------|
| L3 检索 | 聚类匹配 + 抽象事实检索 | 2 天 |
| L4 检索 | 叙事主干检索 + 动态细节构建 | 2 天 |
| L5 检索 | 人格模型匹配 + 行为预测检索 | 2 天 |
| MMR 重排序 | 多样性-相关性平衡 | 1 天 |
| 成本感知路由 | 场景感知层选择 | 1 天 |
| 因果链检索 | 四层递进 (符号+统计) | 2 天 |
| 情感轨迹检索 | 情感路径搜索 | 1 天 |

**Phase 2 交付**: 完整六层检索, 支持所有 query 类型

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
