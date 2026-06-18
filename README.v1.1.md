# 🦋 Butterfly Dream v1.1 — 语义检索 + 三层本体 + PPR 图展开

> **v1.1 分支**在 v1（main）的 HRR 向量编码 + 三维检索基础上，引入了 **神经语义嵌入**、**三层本体架构** 和 **PPR 多跳图展开**，从根本上解决了 v1 的语义鸿沟问题。

---

## 目录

- [v1.1 新增特性](#v11-新增特性)
- [架构总览](#架构总览)
- [三层本体（Three-layer Ontology）](#三层本体three-layer-ontology)
- [神经语义嵌入](#神经语义嵌入)
- [Step-back 抽象匹配](#step-back-抽象匹配)
- [PPR 多跳图展开](#ppr-多跳图展开)
- [检索流水线](#检索流水线)
- [配置与代码](#配置与代码)
- [v1 vs v1.1 对比](#v1-vs-v11-对比)
- [性能](#性能)

---

## v1.1 新增特性

| 特性 | v1 (main) | v1.1 |
|------|-----------|----|
| **实体表示** | 扁平实体（全部 type=unknown） | **三层本体**：具体实体 → 抽象实体（type=abstract）→ 关系边 |
| **聚类** | 成员↔成员 `is_member_of`（O(n²)） | 抽象→成员 `includes` 边（O(n)） |
| **语义编码** | HRR（1024 维，无语义） | **bge-small-zh-v1.5** 本地神经嵌入（512 维，语义密集） |
| **检索权重** | FTS + Jaccard + HRR | FTS + Jaccard + **Embedding** + HRR（embedding 主导） |
| **实体发现** | 仅文本匹配 | **Step-back**：query embedding 匹配抽象实体 |
| **图展开** | BFS（depth≤2 平等入池） | **PPR**（Personalized PageRank，距离自然衰减） |
| **中文支持** | jieba 分词 | 同左 + bge-small-zh-v1.5 原生中文语义 |

---

## 架构总览

```
┌──────────────────────────────────────────────────────┐
│                    search(query)                      │
├──────────────────────────────────────────────────────┤
│  Stage 1:    FTS5 全文搜索                            │
│  Stage 1.5:  语义分类候选                             │
│  Stage 1.6:  Step-back — 抽象实体 embedding 匹配 🔥  │
│  Stage 1.75: PPR 图展开 🔥🔥                         │
│  Stage 2:    三维评分 (rel×rec×imp) × trust           │
└──────────────────────────────────────────────────────┘
```

---

## 三层本体（Three-layer Ontology）

v1.1 将 flat 实体关系重组为语义层次：

```
L1: 具体实体（entities, type='unknown'）
  ── 跳绳, 游泳, 乐高, 小明, 小红 ...

L2: 抽象实体（entities, type='abstract'）
  ── 运动爱好, 益智游戏 ...

L3: includes 关系（entity_relations, relation='includes'）
  ── 运动爱好 ─includes──→ 跳绳
  ── 运动爱好 ─includes──→ 游泳
```

### 好处

- **FTS5 可命中抽象实体名**：搜"运动"直接命中"运动爱好"抽象实体
- **图展开 O(n) vs O(n²)**：旧 `is_member_of` 需要成员间两两连边
- **抽象实体 embedding 代表类别语义**：聚类 centroid 自动编码为向量
- **与 step-back 完美配合**：query embedding 直接比对抽象实体

### 表结构

v1.1 中以下表是核心，所有列名均使用 `*_id` 模式（`fact_id`, `entity_id`, `cluster_id`, `relation_id` 等）：

```sql
-- facts 表 — 核心事实存储（v1.1 新增 embedding 列）
CREATE TABLE facts (
    fact_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL UNIQUE,         -- 事实文本
    category         TEXT DEFAULT 'general',       -- place/time/person/event/activity/identity/...
    tags             TEXT DEFAULT '',              -- 逗号分隔关键词
    importance       REAL DEFAULT 5.0,             -- 1.0 ~ 10.0
    trust_score      REAL DEFAULT 0.5,
    retrieval_count  INTEGER DEFAULT 0,            -- 检索次数统计
    helpful_count    INTEGER DEFAULT 0,            -- 有用反馈统计
    is_persistent    INTEGER DEFAULT 0,            -- 1 = 长期记忆，不会被预取过滤
    content_date     TEXT,                         -- 事件日期 (YYYY-MM-DD)
    created_at       TEXT DEFAULT (datetime('now')),
    updated_at       TEXT DEFAULT (datetime('now')),
    hrr_vector       BLOB,                         -- v1 旧编码（回退用）
    embedding        BLOB                          -- 🔥 v1.1 主力 — bge-small-zh 512-dim float32
);

-- entities 表 — 实体（L1 具体 + L2 抽象）
CREATE TABLE entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,               -- 实体名
    entity_type TEXT DEFAULT 'unknown',             -- 'unknown'（具体）或 'abstract'（抽象）
    aliases     TEXT DEFAULT '',                    -- 预留别名字段（当前未使用）
    created_at  TEXT DEFAULT (datetime('now')),
    embedding   BLOB                               -- 🔥 v1.1 — 实体名嵌入向量
);

-- entity_relations 表 — 关系边（L3：co_occur + includes）
CREATE TABLE entity_relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation    TEXT DEFAULT 'related_to',          -- 'co_occur' | 'includes'
    weight      REAL DEFAULT 1.0,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation)
);

-- clusters 表 — 聚类管理（后台管理用，检索走 entities + includes 边）
CREATE TABLE clusters (
    cluster_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    cluster_type TEXT DEFAULT 'auto' CHECK(cluster_type IN ('auto', 'manual', 'abstract')),
    member_count INTEGER DEFAULT 0,
    centroid     BLOB,                              -- 聚类中心向量
    coherence    REAL DEFAULT 0.0,
    created_at   TEXT DEFAULT (datetime('now')),
    updated_at   TEXT DEFAULT (datetime('now'))
);

-- cluster_members 表 — 聚类成员关系
CREATE TABLE cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    entity_id  INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    similarity REAL DEFAULT 0.0,                    -- 成员-中心相似度
    PRIMARY KEY (cluster_id, entity_id)
);

-- fact_entities 表 — 事实↔实体关联
CREATE TABLE fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

-- merge_log 表 — 事实合并审计
CREATE TABLE merge_log (
    merge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kept_fact_id     INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    absorbed_fact_id INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
    merged_content   TEXT,
    merge_reason     TEXT DEFAULT 'auto',
    created_at       TEXT DEFAULT (datetime('now'))
);
```

> 注：`merge` 路径（`_merge_exact_match` / `_merge_semantic`）也会自动计算 embedding（`COALESCE(?, embedding)`），确保合并后的事实也有向量用于检索。

---

## 神经语义嵌入

### 技术选型

- **模型**：[BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- **大小**：33 MB，纯 CPU 推理
- **维度**：512（vs HRR 的 1024）
- **语义**：双语（中+英），原生中文支持
- **引擎**：FastEmbed（ONNX Runtime），~5ms/次

### 嵌入替换

```
HRR 向量（v1）         神经嵌入（v1.1）
1024 维, 无语义        512 维, 语义密集
余弦≈0.5 随便都是      cosine(跳绳,游泳)=0.50 ← 运动类
只能用 HRR decode      cosine(小明,小红)=0.62 ← 人名类
                      cosine(跳绳,咖啡)=0.10 ← 完全无关
```

### 检索权重分配

```
fts_weight=0.35    ← FTS5 全文搜索
jaccard_weight=0.20 ← 关键词重叠
embed_weight=0.35  ← 神经嵌入语义相似度  🔥
hrr_weight=0.10   ← 旧数据回退
```

---

## Step-back 抽象匹配

**问题**：v1 中 query 必须包含已知实体名才能触发图展开。写意查询"有什么运动推荐"无实体命中 → 0 结果。

**v1.1 方案**：Stage 1.6 将 query embedding 与所有抽象实体做余弦匹配：

```
query: "有什么运动推荐"
→ q_embed = bge("有什么运动推荐")
→ 与所有抽象 entity.embedding 做批量余弦
→ "运动爱好" cosine=0.66 ✅ > threshold=0.50
→ includes → [跳绳, 游泳]
→ 加入 PPR 展开池 → 捞到 4 条相关事实
```

**参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_step_back` | `True` | 开关 |
| `step_back_threshold` | `0.50` | 匹配阈值 |

---

## PPR 多跳图展开

**问题**：v1 用 BFS，depth≤2 内所有事实平等入池，不分远近。

**v1.1 方案**：Personalized PageRank（幂迭代），每个实体得到 visit probability → 事实按 PPR 降权。

```
PPR 画像（种子=跳绳, α=0.85）:
  跳绳(seed)     ppr=0.8603  → boost=0.93×  (全额)
  小明(1-hop)    ppr=0.0691  → boost=0.53×  (中等)
  游泳(2-hop)    ppr=0.0092  → boost=0.50×  (最低)

  好处:
  - 多路径累积: 跳绳→小明 + 跳绳→运动爱好→小明, 小明得分更高
  - 无硬限制: PPR 自然收敛，不会漏掉 depth=3 但有价值的内容
  - 噪声隔离: 远距离实体 PPR ≈ 0，自动过滤
```

### 算法

```python
# 标准化列随机矩阵 M（所有边双向）
# 种子向量 p（种子均匀分布）
# 反复: r_new = (1-α) · M @ r + α · p
# 直到 L1 差 < 1e-6 或 100 迭代
```

**参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_ppr` | `True` | 开关 |
| `ppr_alpha` | `0.85` | teleport 概率（低=探索，高=聚焦种子） |
| `min_ppr` | `0.005` | 最低 PPR 过滤阈值 |

---

## 检索流水线

```
query: "有什么运动推荐？"
  │
Stage 1 — FTS5 全文搜索
  → 麦：搜不到（无字面匹配）
  │
Stage 1.5 — 语义分类候选
  → 麦：无匹配
  │
Stage 1.6 — Step-back 🔥
  → q_embed vs 抽象 entities (batch cosine)
  → "运动爱好" cosine=0.66 ✅
  → includes → [跳绳, 游泳]
  │
Stage 1.75 — PPR 图展开 🔥🔥
  → 种子: [跳绳, 游泳]
  → PPR 幂迭代:
      跳绳 = 0.86
      游泳 = 0.85
      小明 = 0.12
      运动爱好 = 0.10
  → 所有 ppr > 0.005 的实体入池
  → 每个事实带 _ppr_score
  │
Stage 2 — 三维评分
  → relevance × recency × importance
  → × PPR 距离降权 (0.5 + 0.5 × ppr)
  → × trust
  │
结果:
  #1 小明喜欢跳绳 (ppr=0.86, score=0.30)
  #2 跳绳是一种有氧运动 (ppr=0.86, score=0.26)
  #3 小明每周都去游泳 (ppr=0.07, score=0.28)
  #4 游泳可以锻炼全身肌肉 (ppr=0.01, score=0.25)
```

---

## 配置与代码

### 新的 search 参数

```python
retriever.search(
    "有什么运动推荐",
    limit=10,
    use_step_back=True,         # 开启 step-back 🔥
    step_back_threshold=0.50,   # 抽象实体匹配阈值
    use_ppr=True,               # 开启 PPR 🔥🔥
    ppr_alpha=0.85,             # teleport 概率
)
```

### 独立调用抽象实体匹配

```python
from butterfly_dream.embedding import get_embedding_service

svc = get_embedding_service()
qvec = svc.encode_one("有什么运动推荐")
matches = store.match_abstract_entities(qvec, threshold=0.50)

for m in matches:
    print(f"{m['name']} (sim={m['similarity']})")
    print(f"  → members: {[e['name'] for e in m['member_entities']]}")
```

### 独立调用 PPR

```python
# 直接调用 PPR
ppr_scores = store.compute_ppr(
    seed_entity_ids=[1, 2],  # 实体 ID
    alpha=0.85,
)

# 带事实捞取的 PPR 展开
result = store.expand_entities_for_retrieval(
    ["跳绳", "游泳"],
    ppr_alpha=0.85,
    min_ppr=0.005,
)
```

### 手动构建三层本体

```python
from butterfly_dream.embedding import get_embedding_service

svc = get_embedding_service()
centroid_blob = svc.serialize(svc.encode_one("运动爱好活动"))

store.create_cluster(
    name="运动爱好",
    cluster_type="auto",
    member_entity_ids=[1, 2],   # 跳绳, 游泳
    similarities=[0.62, 0.58],
    centroid=centroid_blob,
    coherence=0.60,
    relation_type="includes",   # ← 关键：用 includes 代替 is_member_of
)
```

---

## v1 vs v1.1 对比

| 场景 | v1 (main) | v1.1 |
|------|-----------|-----|
| "有什么运动推荐" | 0 结果 | **4 结果** 🎉 |
| "推荐一些活动" (无实体名) | 0 结果 | **通过 step-back 找到** ✅ |
| "跳绳相关" (有实体名) | 游泳/跳绳同等权重 | **跳绳优先，游泳自然降权** ✅ |
| "小明" (有实体名且有聚类) | 仅共现实体 | **共现 + 抽象实体成员全部展开** ✅ |
| 孤立实体（无关系边） | FTS5 兜底 | FTS5 + embedding 双保险 |

---

## 性能

| 操作 | v1 | v1.1 |
|------|----|----|
| 单次嵌入编码 | — | ~5ms (CPU, bge-small-zh) |
| FTS5 搜索 (~1000 facts) | ~3ms | ~3ms |
| PPR 计算 (50 nodes) | — | ~8ms (50×50 密集矩阵) |
| PPR 计算 (200 nodes) | — | ~35ms (200×200) |
| 全流水线检索 | ~12ms | ~15-25ms |
| 冷启动（jieba + 模型加载） | ~2.6s | ~3.2s (+0.6s bge 首次加载) |

> PPR 矩阵规模 = N×N (N=实体总数)，对于典型场景（<500 实体）完全可接受。

---

## 🗄️ 数据库设计

### 表全景

| 表 | 写入时机 | 读取时机 | v1.1 新增 |
|----|---------|---------|--------|
| `facts` | `add_fact()` | `search()` FTS5 + 评分 | `embedding` 列 |
| `facts_fts` | 同步 trigger | `MATCH` 查询 | — |
| `entities` | `_link_entities()` / `create_cluster()` | 实体匹配 + step-back + PPR | `embedding` 列 + `entity_type='abstract'` |
| `fact_entities` | `_link_entities()` | 实体→事实关联 | — |
| `entity_relations` | `_link_entities()` (co_occur) / `create_cluster()` (includes) | PPR + BFS 图展开 | `includes` 关系类型 |
| `clusters` | `create_cluster()` | 管理后台 + `match_abstract_entities()` 查成员 | ✅ **新表** |
| `cluster_members` | `create_cluster()` | `match_abstract_entities()` 查成员 | ✅ **新表** |
| `media_attachments` | 多媒体写入 | 多媒体检索 | — |
| `merge_log` | 事实合并 | 审计 | — |

### 实体关系图（三层本体）

```
                       ┌───────────────────┐
                       │    运动爱好        │  L2 抽象实体
                       │  (type=abstract)   │    entity_type='abstract'
                       │  embedding=<centr> │
                       └────────┬──────────┘
                    includes/   │   \includes
                     weight=0.62│    \weight=0.58
                               │     \
                    ┌──────────▼┐    ┌▼──────────┐
                    │  跳绳      │    │  游泳      │  L1 具体实体
                    │(entity #11)│    │(entity #13)│    entity_type='unknown'
                    └─────┬─────┘    └────┬───────┘
                          │  co_occur    │
                    0.7   │/0.7   0.6\  │/0.6
                          │/           \│/
                    ┌─────▼─────┐    ┌───▼───────┐
                    │  小明      │    │  小红      │
                    │(entity #10)│    │(entity #12)│
                    └─────┬─────┘    └─────┬─────┘
                          │                │
                     fact_entities    fact_entities
                          │                │
                    ┌─────▼─────┐    ┌─────▼─────────┐
                    │ 事实#42    │    │ 事实#43        │
                    │ 小明喜欢   │    │ 小红喜欢       │
                    │ 跳绳...    │    │ 玩乐高拼图     │
                    └───────────┘    └───────────────┘
```

### 写入流程：`add_fact("小明喜欢跳绳…")`

```python
store.add_fact(
    content="小明喜欢跳绳，这是一项他很喜欢的运动",
    category="preference",
    importance=7.0,
    entities=["小明", "跳绳"],  # ← 外部传入或 LLM 提取
)
```

**Step 1 — 写入 `facts` 表**

```sql
INSERT INTO facts (content, category, tags, importance, is_persistent, content_date, hrr_vector, embedding)
VALUES ('小明喜欢跳绳…', 'preference', '小明,跳绳,运动', 7.0, 1, '2023-01-19', <hrr>, <bge-embedding>);
```

`facts` 表字段说明（v1.1 新增字段加 🔥 标记）：

| 列 | 类型 | 说明 |
|---|---|---|
| `fact_id` | INTEGER | 主键自增 |
| `content` | TEXT | 事实文本，UNIQUE |
| `category` | TEXT | 分类（place/time/person/event/activity/identity/preference/...） |
| `tags` | TEXT | 逗号分隔关键词 |
| `importance` | REAL | 1.0 ~ 10.0 |
| `trust_score` | REAL | 信任度，默认 0.5 |
| `is_persistent` | INTEGER | 1 = 长期记忆 |
| `content_date` | TEXT | 事件日期 YYYY-MM-DD |
| `hrr_vector` | BLOB | 8192 bytes — v1 旧编码（回退用） |
| `embedding` 🔥 | BLOB | 2048 bytes — **v1.1 主力** — bge-small-zh 512-dim float32 |

**Step 2 — `_link_entities()` 创建/关联实体**

```python
for name in entity_names:  # ["小明", "跳绳"]
    # 2a) INSERT OR IGNORE INTO entities (name)
    # 2b) 新实体 → 立即算 embedding
    vec = bge-small-zh.encode("小明")  # → 512-dim
    UPDATE entities SET embedding=? WHERE name='小明';
    # entity_type stays 'unknown' — only clusters write 'abstract'

    # 2c) 关联事实↔实体
    INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
    VALUES (42, 10);   -- 事实42 → 小明
    INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
    VALUES (42, 11);   -- 事实42 → 跳绳

# 2d) 共现关系提取
INSERT INTO entity_relations (source_id, target_id, relation, weight)
VALUES (10, 11, 'co_occur', 0.7)   -- importance 7.0/10 = 0.7
ON CONFLICT UPDATE weight = MIN(weight + 0.35, 10.0);
```

### 聚类流程：`compute_clusters()`

```python
from butterfly_dream.clustering import compute_clusters
clusters = compute_clusters(store, threshold=0.55, min_cluster_size=2)
```

```
Step 1 — 加载所有有 embedding 的实体
  SELECT ... FROM entities WHERE embedding IS NOT NULL
  → ["小明"(10), "小红"(12), "跳绳"(11), "游泳"(13)]

Step 2 — 全对 cosine 相似度矩阵
  cos(小明, 小红) = 0.62 ≥ 0.55 ✅ → 同一聚类

Step 3 — 连通分量 → 找到聚类
  聚类1: [小明, 小红] (coherence=0.62)

Step 4 — 自动命名（成员中最中心的 + "类"后缀）
  小明 vs 小红 → "小明" 是 centroid → "小明类"

Step 5 — 写入 create_cluster()
```

`create_cluster()` 内部：

```sql
-- 5a) clusters 表（后台管理）
INSERT INTO clusters (name, cluster_type, member_count, centroid, coherence)
VALUES ('小明类', 'auto', 2, <centroid>, 0.60);

-- 5b) entities 表 → 抽象实体（L2）
INSERT OR IGNORE INTO entities (name, entity_type, embedding)
VALUES ('小明类', 'abstract', <centroid>);
-- entity_type = 'abstract'  ✅

-- 5c) cluster_members 表
INSERT OR IGNORE INTO cluster_members (cluster_id, entity_id, similarity)
VALUES (1, 10, 0.62), (1, 12, 0.58);

-- 5d) entity_relations → includes 边（L3）
INSERT INTO entity_relations (source_id, target_id, relation, weight)
VALUES (18, 10, 'includes', 0.62),  -- 小明类 → 小明
       (18, 12, 'includes', 0.58);  -- 小明类 → 小红
```

### 检索流程：`search("有什么运动推荐")`

```
query → bge-small-zh → qvec(512-dim)

Stage 1   — FTS5 全文搜索        (fts_rank)
Stage 1.5 — 语义分类候选         (category filter)
Stage 1.6 — Step-back 抽象匹配   (qvec vs abstract entities)
Stage 1.75 — PPR 图展开          (graph distance weighting)
Stage 2   — 三维评分             (rel × rec × imp × trust × ppr_boost)
```

PPR 画像（种子=跳绳, α=0.85）：

```
  跳绳(seed) ppr=0.860 → boost=0.93×
  小明(1-hop) ppr=0.069 → boost=0.53×
  游泳(2-hop) ppr=0.009 → boost=0.50×
```

### v1.1 编码链路

```
事实文本 ──bge-small-zh──→ embedding(512-dim) ──serialize──→ facts.embedding (BLOB)
实体名称 ──bge-small-zh──→ embedding(512-dim) ──serialize──→ entities.embedding (BLOB)
聚类 centroid ──均值合并──→ embedding(512-dim) ──serialize──→ clusters.centroid + 抽象实体.embedding

查询文本 ──bge-small-zh──→ qvec(512-dim)
  ├── FTS5 命中 → fts_rank（传统全文）
  ├── vs 事实 embedding → embed_sim（语义相似度）
  └── vs 抽象实体 embedding → step-back（写意查询兜底）
         └── includes → 具体实体 → PPR → 距离加权事实
```

---

## 开发者

- **guolizi** — 架构设计 & 实现

> 🤝 v1.1 的三层本体与 PPR 方案受知识图谱推理研究启发，step-back 抽象匹配受「先抽象后具体」的 LLM reasoning 思路影响。
