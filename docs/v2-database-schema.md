# 🦋 Butterfly Dream v2 — 数据库 Schema 设计

> 开始时间：2026-06-17
> 状态：设计稿
> 范围：六层记忆架构（L0-L5）所有持久化存储的完整数据库 Schema 设计
> 数据库：SQLite（WAL 模式，`PRAGMA journal_mode=WAL`）

---

## 目录

- [一、设计原则](#一设计原则)
- [二、L0 — 工作记忆](#二l0--工作记忆)
- [三、L1 — 事实池体系](#三l1--事实池体系)
- [四、L2 — 关系层](#四l2--关系层)
- [五、L3 — 抽象层](#五l3--抽象层)
- [六、L4 — 叙事层](#六l4--叙事层)
- [七、L5 — 灵魂层](#七l5--灵魂层)
- [八、情感维度](#八情感维度)
- [九、检索系统](#九检索系统)
- [十、系统管理](#十系统管理)
- [十一、索引汇总](#十一索引汇总)
- [十二、迁移路径](#十二迁移路径)

---

## 一、设计原则

### 1.1 核心原则

| 原则 | 说明 |
|:----|:-----|
| **永远保留** | 数据不删除，只有冷热分级。用户删除权通过隐私擦除层实现（二次确认后硬删除） |
| **WAL 模式** | 支持并发读（在线对话）和写（睡眠周期） |
| **外键约束** | `PRAGMA foreign_keys = ON`，确保引用完整性 |
| **GENERATED 列** | 情感维度使用 GENERATED 派生列从 JSON 主存储提取固定字段 |
| **FTS5 全文索引** | 中文使用 jieba 分词预分割后索引 |
| **统一时间格式** | 所有时间字段使用 ISO 8601 文本格式（`datetime('now')` 本地时间） |
| **JSON 扩展** | SQLite 的 `json_extract()` 用于灵活字段 |

### 1.2 命名约定

- 表名：`snake_case`，复数形式
- 主键：`{table}_id` 自增 INTEGER
- 时间字段：`created_at` / `updated_at` TEXT
- 外键：引用表名去掉复数后缀 + `_id`（如 `emotion_events.event_id`）
- 索引：`idx_{table}_{column}`

### 1.3 数据库文件

```
$HERMES_HOME/memories/memory_store.db
```

所有 L0-L5 数据存储在同一个 SQLite 文件中。不同用户通过 `person` 字段隔离（以人为中心的记忆模型）。

---

## 二、L0 — 工作记忆

### 2.1 对话轮次表

```sql
-- 原始对话轮次
CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,              -- 所属人物
    session_id  TEXT NOT NULL,              -- 会话 ID
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,              -- 对话内容
    turn_order  INTEGER NOT NULL,           -- 轮次序号
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_turns_person_session
    ON conversation_turns(person, session_id, turn_order);

-- L0 FTS5 全文索引（对话轮次检索）
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_turns_fts
    USING fts5(content, person, content=conversation_turns, content_rowid=turn_id);

CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(rowid, content, person)
        VALUES (new.turn_id, jieba_segment(new.content), new.person);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(conversation_turns_fts, rowid, content, person)
        VALUES ('delete', old.turn_id, old.content, old.person);
END;

CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON conversation_turns BEGIN
    INSERT INTO conversation_turns_fts(conversation_turns_fts, rowid, content, person)
        VALUES ('delete', old.turn_id, old.content, old.person);
    INSERT INTO conversation_turns_fts(rowid, content, person)
        VALUES (new.turn_id, jieba_segment(new.content), new.person);
END;
```

### 2.2 微事实索引

```sql
-- L0 微事实索引（关键词 → turn_id → snippet）
-- 纯本地构建：jieba 分词 + 停用词过滤 + 名词性短语规则
CREATE TABLE IF NOT EXISTS micro_facts (
    micro_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    keyword     TEXT NOT NULL,              -- 关键词（jieba 分词结果）
    turn_id     INTEGER NOT NULL REFERENCES conversation_turns(turn_id) ON DELETE CASCADE,
    snippet     TEXT NOT NULL,              -- 原文片段
    promoted    INTEGER DEFAULT 0,          -- 0=未晋升, 1=已晋升到 L1
    miss_count  INTEGER DEFAULT 0,          -- 热晋升缺失计数
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_micro_person_keyword
    ON micro_facts(person, keyword);
CREATE INDEX IF NOT EXISTS idx_micro_promoted
    ON micro_facts(promoted) WHERE promoted = 0;
```

### 2.3 热晋升标记队列

```sql
-- 热晋升标记队列（查询驱动，仅标记不执行）
CREATE TABLE IF NOT EXISTS promotion_queue (
    queue_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    keyword     TEXT NOT NULL,              -- 关键词
    miss_count  INTEGER DEFAULT 1,          -- 缺失计数
    turn_ids    TEXT NOT NULL,              -- 关联的 turn_id 列表（JSON 数组）
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    updated_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(person, keyword)
);
```

---

## 三、L1 — 事实池体系

### 3.1 统一事实表（四池合一）

四个池（事件记录池、静态知识池、行为模式池、情感事件池）中，前三个池共用 `facts` 表，通过 `type` 字段区分。情感事件池独立为 `emotion_events` 表。

```sql
-- 统一事实表（事件记录池 / 静态知识池 / 行为模式池）
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,              -- 所属人物
    content         TEXT NOT NULL,              -- 事实内容
    type            TEXT NOT NULL DEFAULT 'event'
                    CHECK(type IN ('event', 'knowledge', 'behavior')),
        -- 'event'     = 事件记录池（有时间锚点的具体事件）
        -- 'knowledge' = 静态知识池（去时间化的稳定知识）
        -- 'behavior'  = 行为模式池（条件-行为规律）

    -- 池间公共字段
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    importance      REAL DEFAULT 0.5,          -- 0.0~1.0 连续标度
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    is_persistent   INTEGER DEFAULT 0,         -- 1 = 用户要求"记住这个"

    -- 时间字段
    content_date    TEXT,                       -- 事件发生日期（ISO 格式）
    created_at      TEXT DEFAULT (datetime('now','localtime')),
    updated_at      TEXT DEFAULT (datetime('now','localtime')),

    -- 热度管理
    heat_zone       TEXT DEFAULT 'hot'
                    CHECK(heat_zone IN ('hot', 'warm', 'cold', 'ice')),
    cooling_factor  REAL DEFAULT 1.0,           -- 综合冷却系数

    -- 情感关联
    emotion_tag     TEXT,                       -- 轻量情感标签（null=中性）
        -- null      = 中性/无情感
        -- "开心"    = 具体情感标签
        -- "positive" = 正向（valence > 0.3）
        -- "negative" = 负向（valence < -0.3）
        -- "mixed"   = 矛盾情感

    -- 抽象层级
    abstract_level  INTEGER DEFAULT 0,          -- 0=原始, 1=L3 一级抽象, 2=二级, 3=三级
    is_abstract     INTEGER DEFAULT 0,          -- 1 = L3 抽象产物

    -- 嵌入向量
    embedding       BLOB,                       -- 512-dim float32 稠密向量

    -- 结构化事件字段（符号规则引擎 Layer 0 使用，可选）
    structured_data TEXT,                       -- JSON: {"subject":"Caroline","action":"报名","object":"课程","time":"2024-03","location":null}

    UNIQUE(person, content)
);

-- Facts 表索引
CREATE INDEX IF NOT EXISTS idx_facts_person_type
    ON facts(person, type);
CREATE INDEX IF NOT EXISTS idx_facts_importance
    ON facts(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created
    ON facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_content_date
    ON facts(content_date);
CREATE INDEX IF NOT EXISTS idx_facts_heat_zone
    ON facts(heat_zone) WHERE heat_zone IN ('hot', 'warm');
CREATE INDEX IF NOT EXISTS idx_facts_person_abstract
    ON facts(person, is_abstract) WHERE is_abstract = 1;
```

### 3.2 行为模式池特有字段

行为模式池使用 `facts` 表（`type='behavior'`），额外数据存储在关联表：

```sql
-- 行为模式生命周期（与 facts 表 1:1 关联）
CREATE TABLE IF NOT EXISTS behavior_patterns (
    pattern_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id             INTEGER NOT NULL UNIQUE REFERENCES facts(fact_id) ON DELETE CASCADE,
    person              TEXT NOT NULL,

    -- 生命周期
    status              TEXT DEFAULT 'tentative'
                        CHECK(status IN ('tentative', 'confirming', 'confirmed', 'superseded', 'evolved')),
        -- tentative  = 待验证（conf ≥ 0.3）
        -- confirming = 验证中（conf ≥ 0.6）
        -- confirmed  = 有效（conf ≥ 0.8）
        -- superseded = 被证伪
        -- evolved    = 已演化

    confidence          REAL DEFAULT 0.0,       -- 0.0~1.0

    -- 时间窗口
    valid_from          TEXT,                   -- 模式开始有效的时间
    valid_until         TEXT,                   -- NULL = 当前仍然有效

    -- 模式类型
    pattern_type        TEXT DEFAULT 'routine'
                        CHECK(pattern_type IN ('routine', 'emotion-driven', 'value-driven')),
        -- 'routine'        = 例行行为，无情感驱动
        -- 'emotion-driven' = 情感驱动的行为
        -- 'value-driven'   = 价值观驱动的行为

    -- 演化链
    evolved_from_pattern_id INTEGER REFERENCES behavior_patterns(pattern_id),

    -- 源事实
    source_fact_ids     TEXT,                   -- 支撑该模式的 L1 事实 ID 列表（JSON 数组）

    created_at          TEXT DEFAULT (datetime('now','localtime')),
    updated_at          TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_bp_person_status
    ON behavior_patterns(person, status);
CREATE INDEX IF NOT EXISTS idx_bp_confidence
    ON behavior_patterns(confidence DESC);
```

### 3.3 实体体系

```sql
-- 实体表
CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,              -- 所属人物（以人为中心）
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',            -- 别名列表（逗号分隔）
    embedding   BLOB,                       -- 512-dim float32
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(person, name)
);

CREATE INDEX IF NOT EXISTS idx_entities_person_name
    ON entities(person, name);

-- 事实-实体关联
CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);
```

### 3.4 事实间关系

```sql
-- 事实间关系（多对多映射）
-- 用途：abstracts_from（L3 抽象→源事实）、contradicted_by（矛盾检测）
CREATE TABLE IF NOT EXISTS fact_relations (
    relation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    source_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    target_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL
                  CHECK(relation_type IN ('abstracts_from', 'contradicted_by', 'supports', 'evolved_from')),
        -- abstracts_from  = L3 抽象事实 → 源事实
        -- contradicted_by = 事实 A 被事实 B 矛盾
        -- supports        = 事实 A 支持事实 B
        -- evolved_from    = 行为模式演化（冗余，方便跨表查询）
    context       TEXT,                     -- 抽象角度说明（可选）
    created_at    TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(source_fact_id, target_fact_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_fr_target
    ON fact_relations(target_fact_id);
CREATE INDEX IF NOT EXISTS idx_fr_source
    ON fact_relations(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_fr_type
    ON fact_relations(relation_type);
```

### 3.5 合并日志

```sql
-- 事实合并日志
CREATE TABLE IF NOT EXISTS merge_log (
    merge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person           TEXT NOT NULL,
    kept_fact_id     INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    absorbed_fact_id INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
    merged_content   TEXT,
    merge_reason     TEXT DEFAULT 'auto',
    created_at       TEXT DEFAULT (datetime('now','localtime'))
);
```

### 3.6 媒体附件

```sql
-- 媒体附件（多媒体记忆）
CREATE TABLE IF NOT EXISTS media_attachments (
    media_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    storage_type  TEXT NOT NULL DEFAULT 'file' CHECK(storage_type IN ('file', 'url')),
    file_path     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    file_size     INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    caption       TEXT DEFAULT '',
    transcript    TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now','localtime'))
);
```

---

## 四、L2 — 关系层

### 4.1 实体关系图

```sql
-- 实体间关系
CREATE TABLE IF NOT EXISTS entity_relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,              -- 所属人物
    source_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation    TEXT DEFAULT 'related_to',  -- 关系类型
        -- 'related_to'    = 一般关联
        -- 'friend_of'     = 朋友
        -- 'family_of'     = 家人
        -- 'colleague_of'  = 同事
        -- 'partner_of'    = 伴侣
        -- 'belongs_to'    = 隶属
        -- 'located_at'    = 位于
    weight      REAL DEFAULT 1.0,           -- 关系强度
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_er_source_target
    ON entity_relations(source_id, target_id);
```

### 4.2 因果链

```sql
-- 因果链主表（支持多因一果）
CREATE TABLE IF NOT EXISTS causal_relations (
    relation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person        TEXT NOT NULL,
    effect_fact_id  INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    cause_fact_ids  TEXT NOT NULL,             -- 原因列表 [1, 2, 3]（JSON 数组）
    causality_type  TEXT NOT NULL
                    CHECK(causality_type IN ('direct', 'indirect', 'inferred', 'rule')),
        -- direct   = 短程（时间邻近）
        -- indirect = 中程（L3 抽象桥接）
        -- inferred = 长程（L4 叙事补全）
        -- rule     = 符号规则
    confidence      REAL DEFAULT 0.5,
    layer           INTEGER DEFAULT 0,         -- 0=符号规则, 1=短程, 2=中程, 3=长程
    abstracted_by   INTEGER REFERENCES facts(fact_id) ON DELETE SET NULL,
    signals_json    TEXT,                       -- 多信号评分详情（JSON）
    validated_by    TEXT DEFAULT 'statistical'
                    CHECK(validated_by IN ('statistical', 'llm', 'rule', 'llm_only')),
    rule_match_count INTEGER DEFAULT 0,         -- 符号规则匹配数（Layer 0 专用）
    rule_ids        TEXT,                       -- 匹配的规则 ID 列表（JSON 数组）
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 因果链辅助索引表（加速反向查询）
CREATE TABLE IF NOT EXISTS causal_relation_members (
    relation_id   INTEGER NOT NULL REFERENCES causal_relations(relation_id) ON DELETE CASCADE,
    cause_fact_id INTEGER NOT NULL,
    PRIMARY KEY (relation_id, cause_fact_id)
);

CREATE INDEX IF NOT EXISTS idx_crm_cause
    ON causal_relation_members(cause_fact_id);
CREATE INDEX IF NOT EXISTS idx_cr_effect
    ON causal_relations(effect_fact_id);
```

### 4.3 时间链

```sql
-- 时间链（时序关系）
CREATE TABLE IF NOT EXISTS timeline_relations (
    relation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person         TEXT NOT NULL,
    earlier_fact_id INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    later_fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    time_gap_days  REAL,                     -- 时间间隔（天）
    confidence     REAL DEFAULT 1.0,
    created_at     TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(earlier_fact_id, later_fact_id)
);
```

### 4.4 溯源

```sql
-- 溯源（事实来源追踪）
CREATE TABLE IF NOT EXISTS provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    source_type   TEXT NOT NULL
                  CHECK(source_type IN ('llm_extraction', 'l0_promotion', 'l3_abstraction',
                                        'l4_narrative', 'user_input', 'historical_import')),
    source_session_id TEXT,                  -- 来源会话 ID
    source_turn_id    INTEGER,              -- 来源对话轮次
    confidence        REAL DEFAULT 0.7,      -- 来源可信度
    created_at        TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_prov_fact
    ON provenance(fact_id);
```

---

## 五、L3 — 抽象层

### 5.1 聚类

```sql
-- 聚类（GMM 聚类参数存储）
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person       TEXT NOT NULL,
    name         TEXT NOT NULL,
    cluster_type TEXT DEFAULT 'auto'
                 CHECK(cluster_type IN ('auto', 'manual', 'abstract')),
    member_count INTEGER DEFAULT 0,
    centroid     BLOB,                      -- 聚类中心向量
    coherence    REAL DEFAULT 0.0,          -- 聚类内平均相似度
    -- GMM 参数（JSON，用于马氏距离计算）
    gmm_mean     TEXT,                      -- JSON: 均值向量 μ_k
    gmm_cov_inv  TEXT,                      -- JSON: 协方差逆矩阵 Σ_k^{-1}（预计算）
    gmm_weight   REAL DEFAULT 0.0,          -- 分量权重 π_k
    created_at   TEXT DEFAULT (datetime('now','localtime')),
    updated_at   TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(person, name)
);

-- 聚类成员
CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    fact_id    INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    similarity REAL DEFAULT 0.0,
    PRIMARY KEY (cluster_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_cm_fact
    ON cluster_members(fact_id);
```

---

## 六、L4 — 叙事层

### 6.1 叙事主干

```sql
-- 叙事主干（持久化人生主线）
CREATE TABLE IF NOT EXISTS narratives (
    narrative_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    person        TEXT NOT NULL,
    version       INTEGER NOT NULL,          -- 版本号
    summary       TEXT NOT NULL,              -- 叙事摘要
    chapters      TEXT NOT NULL,              -- 章节列表（JSON 数组）
        -- [{"title": "寻找自我", "time_range": ["2023-05", "2023-06"], "summary": "..."}, ...]
    time_range    TEXT,                       -- 覆盖时间范围（JSON: [start, end]）
    embedding     BLOB,                       -- 叙事摘要的 embedding
    source_ids    TEXT,                       -- 构建该叙事的 L1-L3 事实 ID 列表（JSON 数组）
    is_active     INTEGER DEFAULT 1,          -- 1=当前版本
    created_at    TEXT DEFAULT (datetime('now','localtime')),
    updated_at    TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_narr_person_active
    ON narratives(person, is_active) WHERE is_active = 1;
```

### 6.2 叙事关键节点标记

```sql
-- 叙事关键情感节点（L4 标记，影响情感冷却）
CREATE TABLE IF NOT EXISTS narrative_emotion_nodes (
    node_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    event_id    INTEGER NOT NULL REFERENCES emotion_events(event_id) ON DELETE CASCADE,
    narrative_id INTEGER NOT NULL REFERENCES narratives(narrative_id) ON DELETE CASCADE,
    node_type   TEXT NOT NULL
                CHECK(node_type IN ('turning_point', 'climax', 'resolution', 'key_memory')),
    reason      TEXT,                       -- 标记理由
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(event_id, narrative_id)
);
```

---

## 七、L5 — 灵魂层

### 7.1 人格模型

```sql
-- 人格模型（结构化存储）
CREATE TABLE IF NOT EXISTS persona_models (
    model_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL UNIQUE,
    version     INTEGER NOT NULL,

    -- 人格维度（JSON）
    dimensions  TEXT NOT NULL,
        -- {
        --   "openness": {"value": 0.7, "confidence": 0.8},
        --   "conscientiousness": {"value": 0.6, "confidence": 0.7},
        --   "extraversion": {"value": 0.3, "confidence": 0.6},
        --   "agreeableness": {"value": 0.8, "confidence": 0.7},
        --   "neuroticism": {"value": 0.4, "confidence": 0.6}
        -- }

    -- 人格特质（结构化）
    traits      TEXT,
        -- {
        --   "core_values": ["家庭", "成长", "社群"],
        --   "emotion_regulation": "创作释放",
        --   "decision_tendency": "谨慎但勇于改变",
        --   "relationship_pattern": "深度少数",
        --   "life_stage": "职业探索期"
        -- }

    -- GMM 参数（Phase 2+）
    gmm_params  TEXT,
        -- {
        --   "k": 3,
        --   "components": [
        --     {"mean": [...], "covariance": [...], "weight": 0.4, "label": "创作态"},
        --     ...
        --   ]
        -- }

    -- 行为向量空间参数
    behavior_space TEXT,
        -- {
        --   "dimensions": 15,
        --   "core_dimensions": 12,
        --   "gmm_dimensions": 3,
        --   "population_cov": [...]
        -- }

    -- 元数据
    confidence      REAL DEFAULT 0.5,
    sample_count    INTEGER DEFAULT 0,
    is_active       INTEGER DEFAULT 1,
    created_at      TEXT DEFAULT (datetime('now','localtime')),
    updated_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 人格模型版本快照（用于回滚）
CREATE TABLE IF NOT EXISTS persona_snapshots (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    person       TEXT NOT NULL,
    version      INTEGER NOT NULL,
    model_snapshot TEXT NOT NULL,             -- 完整人格模型 JSON
    reason       TEXT,                        -- 快照原因（更新前/回滚前）
    created_at   TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_ps_person_version
    ON persona_snapshots(person, version DESC);
```

### 7.2 行为预测日志

```sql
-- 预测日志（统一替代 prediction_counterfactuals）
CREATE TABLE IF NOT EXISTS behavior_predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,

    -- 预测内容
    predicted_behavior          TEXT NOT NULL,    -- 预测的行为
    predicted_prob              REAL NOT NULL,    -- 预测概率
    predicted_accompanying_emotion TEXT,          -- 预测的情绪（可选）
    predicted_emotion_vad       TEXT,             -- 预测时用户情绪 VAD（JSON）

    -- 情感变化预期
    expected_emotion_shift      TEXT,             -- 预测的行为会带来的情感变化（JSON 数组）

    -- 预测时的上下文
    context_snapshot            TEXT,             -- 上下文快照（JSON）
    pattern_relation_id         INTEGER,          -- 关联的情感-行为模式

    -- 实际结果（行为发生后回填）
    actual_behavior             TEXT,             -- 实际行为（null=未发生）
    actual_match                REAL,             -- 行为匹配度 0~1
    actual_emotion_vad          TEXT,             -- 行为时的实际情绪（JSON）
    post_emotion_vad            TEXT,             -- 行为后的情绪（JSON）
    emotion_shift               TEXT,             -- 实际情感变化（JSON 数组）
    shift_match                 REAL,             -- 预期 vs 实际情感变化匹配度

    -- 结果分类
    outcome TEXT DEFAULT 'unobserved'
            CHECK(outcome IN ('fulfilled', 'partial', 'failed', 'unobserved')),
        -- fulfilled  = 行为发生 + 情绪变化匹配预期
        -- partial    = 行为发生但情绪变化不匹配
        -- failed     = 行为未发生
        -- unobserved = 尚未观察到结果

    -- 情感反馈
    emotion_outcome TEXT CHECK(emotion_outcome IN ('improved', 'worsened', 'unchanged', NULL)),

    -- 干预标记（自我实现预言隔离）
    intervened      INTEGER DEFAULT 0,       -- 1 = Agent 向用户表达过该预测
    intervened_prob REAL,                    -- 干预前的预测概率

    -- 反差检索索引
    embedding       BLOB,

    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_bp_person_outcome
    ON behavior_predictions(person, outcome);
CREATE INDEX IF NOT EXISTS idx_bp_timestamp
    ON behavior_predictions(timestamp DESC);
```

---

## 八、情感维度

### 8.1 情感事件池

```sql
-- 情感事件池（L1 第 4 个池，独立表）
CREATE TABLE IF NOT EXISTS emotion_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,              -- 情感事件发生时间

    -- 情感模型
    emotion_model   TEXT DEFAULT 'vad-3d'
                    CHECK(emotion_model IN ('vad-3d', 'emotion-21d')),
        -- 'vad-3d'     = VAD 三维（当前）
        -- 'emotion-21d' = 21 维情感空间（未来）

    -- 唯一主存储：通用情感向量
    emotion_vector  TEXT NOT NULL,               -- JSON 数组
        -- VAD 3D: [valence, arousal, dominance]
        -- 21 维: [...21 个值...]

    -- GENERATED 派生列（VAD 3D 时有效，21D 时自动 NULL）
    valence   REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[0]')
        END
    ),
    arousal   REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[1]')
        END
    ),
    dominance REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[2]')
        END
    ),

    -- 情感标签（可选，方便人类阅读）
    emotion_label   TEXT,

    -- 情感对象
    emotion_target  TEXT,
        -- null         = 无对象（mood，如"莫名焦虑"）
        -- 'self'       = 对自己
        -- 'person:xxx' = 对某人
        -- 'entity:xxx' = 对实体/概念
        -- 'event:xxx'  = 对事件
        -- 'place:xxx'  = 对场所

    -- 强度（GENERATED 列，从 emotion_vector 自动计算）
    intensity       REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN sqrt(
            (json_extract(emotion_vector, '$[0]') * json_extract(emotion_vector, '$[0]') +
             json_extract(emotion_vector, '$[1]') * json_extract(emotion_vector, '$[1]') +
             json_extract(emotion_vector, '$[2]') * json_extract(emotion_vector, '$[2]')) / 3.0
        )
        END
    ),

    -- 事实关联
    primary_fact_id     INTEGER REFERENCES facts(fact_id),  -- 主要触发事实
    related_fact_ids    TEXT,                    -- 关联事实列表（JSON 数组）

    -- 来源
    source          TEXT NOT NULL DEFAULT 'user'
                    CHECK(source IN ('user', 'assistant', 'l0_promotion', 'inferred')),
        -- user       = 来自用户的真实情感
        -- assistant  = LLM 回复中表达的情感
        -- l0_promotion = 从 L0 晋升
        -- inferred   = 从模式推断

    -- 重要性（LLM 初始标注）
    initial_importance  REAL DEFAULT 0.5,        -- 0.0~1.0，LLM 初始标注
    significance_reason TEXT,                    -- 重要性理由（可选）

    -- 触发话题
    trigger_topics      TEXT,                    -- JSON 数组

    -- 认知评价维度（Phase 2+，可选）
    appraisal_dimensions TEXT,                   -- JSON 对象
        -- {
        --   "goal_congruence": -0.8,
        --   "agency": "other",
        --   "ego_involvement": "self-esteem",
        --   "certainty": 0.9,
        --   "novelty": 0.3,
        --   "coping_potential": 0.4,
        --   "norm_compatibility": -0.5
        -- }

    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now','localtime')),
    updated_at      TEXT DEFAULT (datetime('now','localtime'))
);

-- 情感事件索引
CREATE INDEX IF NOT EXISTS idx_ee_person_time
    ON emotion_events(person, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ee_source
    ON emotion_events(source);
CREATE INDEX IF NOT EXISTS idx_ee_primary_fact
    ON emotion_events(primary_fact_id);
```

### 8.2 情感触发关联（静态知识池）

```sql
-- 情感触发关联（归属静态知识池，人物画像的一部分）
CREATE TABLE IF NOT EXISTS emotion_triggers (
    trigger_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    trigger_type    TEXT NOT NULL
                    CHECK(trigger_type IN ('topic', 'entity', 'event_type', 'location')),
    trigger_value   TEXT NOT NULL,

    -- 关联的情感向量（当前情感状态）
    associated_vector   TEXT,                    -- JSON 数组（VAD 或通用）
    emotion_label       TEXT,                    -- 关联的情感标签（可选）

    -- 统计
    trigger_count       INTEGER DEFAULT 1,
    last_triggered_at   TEXT,
    recent_count        INTEGER DEFAULT 1,       -- 最近 5 次事件中匹配次数
    recent_consistency  REAL DEFAULT 0.0,        -- 最近 5 次 VAD 余弦相似度均值

    -- 置信度
    confidence          REAL DEFAULT 0.0,
        -- min(recent_count / 5, 1.0) × recent_consistency

    -- 源事件
    source_event_ids    TEXT,                    -- 关联的 emotion_events ID 列表（JSON 数组）

    created_at          TEXT DEFAULT (datetime('now','localtime')),
    updated_at          TEXT DEFAULT (datetime('now','localtime')),

    UNIQUE(person, trigger_type, trigger_value)
);

CREATE INDEX IF NOT EXISTS idx_et_person_type
    ON emotion_triggers(person, trigger_type);
CREATE INDEX IF NOT EXISTS idx_et_confidence
    ON emotion_triggers(confidence DESC);
```

### 8.3 情感状态节点（Phase 2 物化）

```sql
-- 情感状态节点（Phase 2 从 emotion_events 迁移）
CREATE TABLE IF NOT EXISTS emotion_states (
    state_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    timestamp       TEXT NOT NULL,

    emotion_model   TEXT DEFAULT 'vad-3d'
                    CHECK(emotion_model IN ('vad-3d', 'emotion-21d')),
    emotion_vector  TEXT NOT NULL,               -- JSON 数组（唯一主存储）

    -- GENERATED 派生列
    valence   REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[0]') END
    ),
    arousal   REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[1]') END
    ),
    dominance REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN json_extract(emotion_vector, '$[2]') END
    ),

    emotion_label   TEXT,
    emotion_target  TEXT,
    trigger_topics      TEXT,                    -- JSON 数组
    intensity       REAL GENERATED ALWAYS AS (
        CASE WHEN emotion_model = 'vad-3d'
        THEN sqrt(
            (json_extract(emotion_vector, '$[0]') * json_extract(emotion_vector, '$[0]') +
             json_extract(emotion_vector, '$[1]') * json_extract(emotion_vector, '$[1]') +
             json_extract(emotion_vector, '$[2]') * json_extract(emotion_vector, '$[2]')) / 3.0
        )
        END
    ),
    primary_fact_id     INTEGER REFERENCES facts(fact_id),
    related_fact_ids    TEXT,
    source              TEXT NOT NULL DEFAULT 'user'
                    CHECK(source IN ('user', 'assistant', 'l0_promotion', 'inferred')),
    initial_importance  REAL DEFAULT 0.5,
    significance_reason TEXT,
    appraisal_dimensions TEXT,                   -- 认知评价维度（Phase 2+，可选）
    notes               TEXT,
    created_at          TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_es_person_time
    ON emotion_states(person, timestamp DESC);
```

### 8.4 情感转变（Phase 2 物化）

```sql
-- 情感转变
CREATE TABLE IF NOT EXISTS emotion_transitions (
    transition_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    from_state_id   INTEGER REFERENCES emotion_states(state_id),
    to_state_id     INTEGER REFERENCES emotion_states(state_id),
    transition_type TEXT NOT NULL
                    CHECK(transition_type IN (
                        'positive_breakthrough', 'negative_impact',
                        'cumulative_sublimation', 'resilience_recovery',
                        'gradual_shift', 'sudden_flip'
                    )),
    vector_delta    TEXT,                       -- 后向量 - 前向量（JSON 数组）
    delta_magnitude REAL,                       -- 变化幅度（标量）
    trigger_fact_ids TEXT,                      -- 触发事实 ID 列表（JSON 数组）
    pattern_id      INTEGER,                    -- 关联的情感模式（可选）
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);
```

### 8.5 情感模式（Phase 2 物化）

```sql
-- 情感模式
CREATE TABLE IF NOT EXISTS emotion_patterns (
    pattern_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person              TEXT NOT NULL,
    description         TEXT NOT NULL,
    vector_region       TEXT,                   -- 情感空间中的区域描述（JSON）
        -- VAD 3D: {"valence": [min, max], "arousal": [min, max], "dominance": [min, max]}
    confidence          REAL DEFAULT 0.0,
    source_transition_ids TEXT,                 -- 关联的 transition ID 列表（JSON 数组）
    created_at          TEXT DEFAULT (datetime('now','localtime')),
    updated_at          TEXT DEFAULT (datetime('now','localtime'))
);
```

### 8.6 情感-行为模式关联

```sql
-- 情感模式 ↔ 行为模式的多对多关联
CREATE TABLE IF NOT EXISTS pattern_relations (
    relation_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion_pattern_id  INTEGER REFERENCES emotion_patterns(pattern_id) ON DELETE CASCADE,
    behavior_pattern_id INTEGER REFERENCES behavior_patterns(pattern_id) ON DELETE CASCADE,
    correlation         REAL DEFAULT 0.0,       -- 关联强度 0~1
    sample_count        INTEGER DEFAULT 0,      -- 观察次数
    last_observed_at    TEXT,
    confidence          REAL DEFAULT 0.0,       -- min(sample_count / 10, 1.0) × correlation
    created_at          TEXT DEFAULT (datetime('now','localtime')),
    updated_at          TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(emotion_pattern_id, behavior_pattern_id)
);
```

---

## 九、检索系统

### 9.1 检索缓存（Phase 3）

```sql
-- 检索缓存（LRU）
CREATE TABLE IF NOT EXISTS retrieval_cache (
    cache_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key   TEXT NOT NULL UNIQUE,         -- hash(query + query_type)
    query       TEXT NOT NULL,
    query_type  TEXT NOT NULL,
    result      TEXT NOT NULL,                -- 检索结果 JSON
    embedding   BLOB,                        -- query embedding（用于相似 query 匹配）
    hit_count   INTEGER DEFAULT 1,
    created_at  TEXT DEFAULT (datetime('now','localtime')),
    expires_at  TEXT                          -- TTL 过期时间
);

CREATE INDEX IF NOT EXISTS idx_rc_expires
    ON retrieval_cache(expires_at);
```

### 9.2 热度元数据

```sql
-- 热度追踪（facts 表的补充）
-- 冷却系数已存储在 facts.cooling_factor，此处记录热度变更历史
CREATE TABLE IF NOT EXISTS heat_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id     INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    person      TEXT NOT NULL,
    old_zone    TEXT,
    new_zone    TEXT NOT NULL CHECK(new_zone IN ('hot', 'warm', 'cold', 'ice')),
    reason      TEXT,                        -- 变更原因
    created_at  TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_hl_fact
    ON heat_log(fact_id);
CREATE INDEX IF NOT EXISTS idx_hl_person
    ON heat_log(person, created_at DESC);
```

---

## 十、系统管理

### 10.1 用户屏蔽表

```sql
-- 用户屏蔽（不提，不删）
CREATE TABLE IF NOT EXISTS user_blocks (
    block_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    person      TEXT NOT NULL,
    block_type  TEXT NOT NULL
                CHECK(block_type IN (
                    'trigger_topic', 'trigger_entity',
                    'fact', 'emotion_event',
                    'behavior_pattern', 'emotion_pattern',
                    'topic'
                )),
    block_value TEXT NOT NULL,               -- 屏蔽的值
    blocked_at  TEXT DEFAULT (datetime('now','localtime')),
    reason      TEXT,                        -- 用户提供的原因（可选）
    UNIQUE(person, block_type, block_value)
);
```

### 10.2 睡眠周期日志

```sql
-- 睡眠周期执行日志
CREATE TABLE IF NOT EXISTS sleep_cycle_log (
    cycle_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    person          TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    status          TEXT DEFAULT 'running'
                    CHECK(status IN ('running', 'completed', 'interrupted', 'failed')),
    phases_completed TEXT,                   -- 已完成阶段列表（JSON 数组）
    stats           TEXT,                    -- 统计信息（JSON）
        -- {
        --   "new_facts": 15,
        --   "new_patterns": 3,
        --   "narrative_updated": true,
        --   "personality_updated": false,
        --   "llm_tokens": 5000,
        --   "duration_ms": 120000
        -- }
    checkpoint      TEXT,                    -- 中断恢复点（JSON）
        -- {"phase": "phase2_cold_promotion", "last_processed_micro_fact_id": 370}
    created_at      TEXT DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_scl_person_status
    ON sleep_cycle_log(person, status);
```

### 10.3 配置表

```sql
-- 系统配置（键值对）
CREATE TABLE IF NOT EXISTS system_config (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    description TEXT,
    updated_at  TEXT DEFAULT (datetime('now','localtime'))
);

-- 默认配置
INSERT OR IGNORE INTO system_config(key, value, description) VALUES
    ('schema_version', '2.0', '数据库 Schema 版本'),
    ('emotion_decay_alpha', '0.1', '幂律衰减基础衰减率（/月）'),
    ('emotion_decay_beta_base', '0.5', '幂律衰减基准指数'),
    ('heat_hot_days', '7', '🔥 热区保留天数'),
    ('heat_warm_days', '30', '🌤️ 温区保留天数'),
    ('heat_cold_days', '90', '❄️ 冷区保留天数'),
    ('sleep_cycle_interval_hours', '6', '睡眠周期最小间隔（小时）'),
    ('sleep_cycle_min_new_events', '20', '触发睡眠周期的最小新增事件数'),
    ('sleep_cycle_force_interval_hours', '24', '强制睡眠周期间隔（小时）');
```

### 10.4 全文搜索索引

```sql
-- FTS5 全文索引（事实表）
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, person, content=facts, content_rowid=fact_id);

-- FTS5 同步触发器
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags, person)
        VALUES (new.fact_id, jieba_segment(new.content), jieba_segment(new.tags), new.person);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags, person)
        VALUES ('delete', old.fact_id, old.content, old.tags, old.person);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags, person)
        VALUES ('delete', old.fact_id, old.content, old.tags, old.person);
    INSERT INTO facts_fts(rowid, content, tags, person)
        VALUES (new.fact_id, jieba_segment(new.content), jieba_segment(new.tags), new.person);
END;

-- 媒体附件 FTS5 索引
CREATE VIRTUAL TABLE IF NOT EXISTS media_attachments_fts
    USING fts5(description, caption, transcript, content=media_attachments, content_rowid=media_id);

CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
        VALUES (new.media_id, jieba_segment(new.description), jieba_segment(new.caption), jieba_segment(new.transcript));
END;

CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
        VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
END;

CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
        VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
    INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
        VALUES (new.media_id, jieba_segment(new.description), jieba_segment(new.caption), jieba_segment(new.transcript));
END;
```

---

## 十一、索引汇总

| 表 | 索引 | 用途 |
|:--|:----|:-----|
| `conversation_turns` | `idx_turns_person_session` | 按人物+会话查询对话轮次 |
| `conversation_turns` | `conversation_turns_fts` | FTS5 全文索引（BM25 排序） |
| `micro_facts` | `idx_micro_person_keyword` | 按人物+关键词查微事实 |
| `micro_facts` | `idx_micro_promoted` | 筛选未晋升的微事实 |
| `facts` | `idx_facts_person_type` | 按人物+池类型过滤 |
| `facts` | `idx_facts_importance` | 按重要性排序 |
| `facts` | `idx_facts_created` | 按创建时间排序 |
| `facts` | `idx_facts_content_date` | 按事件日期过滤 |
| `facts` | `idx_facts_heat_zone` | 按热度分区检索 |
| `facts` | `idx_facts_person_abstract` | 筛选 L3 抽象事实 |
| `behavior_patterns` | `idx_bp_person_status` | 按人物+状态过滤 |
| `behavior_patterns` | `idx_bp_confidence` | 按置信度排序 |
| `entities` | `idx_entities_person_name` | 按人物+名称查实体 |
| `entity_relations` | `idx_er_source_target` | 实体图 PPR 检索 |
| `causal_relations` | `idx_cr_effect` | 按结果查原因 |
| `causal_relation_members` | `idx_crm_cause` | 按原因查结果 |
| `emotion_events` | `idx_ee_person_time` | 按人物+时间查情感轨迹 |
| `emotion_events` | `idx_ee_source` | 按来源过滤 |
| `emotion_events` | `idx_ee_primary_fact` | 按触发事实关联 |
| `emotion_states` | `idx_es_person_time` | 按人物+时间查情感状态 |
| `emotion_triggers` | `idx_et_person_type` | 按人物+类型查触发关联 |
| `emotion_triggers` | `idx_et_confidence` | 按置信度排序 |
| `behavior_predictions` | `idx_bp_person_outcome` | 按人物+结果过滤 |
| `behavior_predictions` | `idx_bp_timestamp` | 按时间排序 |
| `narratives` | `idx_narr_person_active` | 获取当前叙事版本 |
| `persona_snapshots` | `idx_ps_person_version` | 按版本查询快照 |
| `sleep_cycle_log` | `idx_scl_person_status` | 查询最近睡眠周期 |
| `heat_log` | `idx_hl_fact` | 查事实热度变更历史 |
| `heat_log` | `idx_hl_person` | 查人物热度变更记录 |
| `retrieval_cache` | `idx_rc_expires` | 清理过期缓存 |
| `fact_relations` | `idx_fr_target` | 反向查抽象源事实 |
| `fact_relations` | `idx_fr_source` | 正向查抽象目标事实 |
| `fact_relations` | `idx_fr_type` | 按关系类型过滤 |
| `provenance` | `idx_prov_fact` | 按事实查来源 |
| `cluster_members` | `idx_cm_fact` | 按事实查所属聚类 |

---

## 十二、迁移路径

### Phase 1（当前 → MVP）

```sql
-- 1. 现有 v1 表保持不变
--    facts, entities, fact_entities, entity_relations, merge_log,
--    media_attachments, clusters, cluster_members

-- 2. 新增 v2 表
--    conversation_turns, micro_facts, promotion_queue
--    conversation_turns_fts（FTS5 虚拟表 + 同步触发器）
--    emotion_events, emotion_triggers
--    fact_relations, provenance, timeline_relations
--    user_blocks, sleep_cycle_log, system_config

-- 3. facts 表新增列（ALTER TABLE）
ALTER TABLE facts ADD COLUMN type TEXT DEFAULT 'event';
ALTER TABLE facts ADD COLUMN person TEXT DEFAULT '';
ALTER TABLE facts ADD COLUMN heat_zone TEXT DEFAULT 'hot';
ALTER TABLE facts ADD COLUMN cooling_factor REAL DEFAULT 1.0;
ALTER TABLE facts ADD COLUMN emotion_tag TEXT;
ALTER TABLE facts ADD COLUMN abstract_level INTEGER DEFAULT 0;
ALTER TABLE facts ADD COLUMN is_abstract INTEGER DEFAULT 0;

-- 4. importance 标度迁移（1.0~10.0 → 0.0~1.0）
UPDATE facts SET importance = (importance - 1.0) / 9.0;
```

### Phase 2（三表物化）

```sql
-- 1. 新增三表
--    emotion_states（从 emotion_events 迁移）
--    emotion_transitions（物化推导结果）
--    emotion_patterns（物化聚类结果）
--    behavior_patterns（行为模式生命周期）
--    pattern_relations（情感-行为关联）
--    causal_relations, causal_relation_members（因果链）
--    narratives, narrative_emotion_nodes（叙事层）

-- 2. 数据迁移
INSERT INTO emotion_states (person, timestamp, emotion_model, emotion_vector, ...)
    SELECT person, timestamp, emotion_model, emotion_vector, ...
    FROM emotion_events;
```

### Phase 3（高级能力）

```sql
-- 1. 新增表
--    persona_models, persona_snapshots（L5 人格模型）
--    behavior_predictions（预测日志）
--    retrieval_cache（检索缓存）
--    heat_log（热度变更历史）

-- 2. 废弃旧表
--    prediction_counterfactuals（由 behavior_predictions 替代）
```

### Phase 4（成熟）

```sql
-- 1. 新增表
--    无新增表，仅索引优化和分区策略调整

-- 2. 情感模型切换（如 vad-3d → emotion-21d）
--    INSERT 只写 emotion_vector，GENERATED 列自动返回 NULL
--    查询代码自然 fallback 到 emotion_vector
```

---

## 附录：表关系图

```
L0 工作记忆:
  conversation_turns ──→ micro_facts
  conversation_turns_fts (FTS5 虚拟表)
       │
       │ (promotion_queue → 热晋升标记)
       ▼
L1 事实池:
  facts (type=event/knowledge/behavior)
    ├── behavior_patterns (1:1, type=behavior 时)
    ├── entities ←── fact_entities ──→ facts
    ├── fact_relations (abstracts_from / contradicted_by)
    ├── merge_log
    └── media_attachments
       │
       │ (emotion_tag 轻量关联)
       ▼
    emotion_events (独立表, L1 第 4 池)
       ├── emotion_triggers (静态知识池)
       ├── emotion_states (Phase 2 物化)
       ├── emotion_transitions (Phase 2 物化)
       └── emotion_patterns (Phase 2 物化)
            └── pattern_relations ──→ behavior_patterns
       │
       ▼
L2 关系层:
  entity_relations ──→ entities
  causal_relations ──→ causal_relation_members ──→ facts
  timeline_relations ──→ facts
  provenance ──→ facts
       │
       ▼
L3 抽象层:
  clusters ──→ cluster_members ──→ facts
       │
       ▼
L4 叙事层:
  narratives ──→ narrative_emotion_nodes ──→ emotion_events
       │
       ▼
L5 灵魂层:
  persona_models ──→ persona_snapshots
  behavior_predictions
       │
       ▼
系统管理:
  user_blocks
  sleep_cycle_log
  system_config
  retrieval_cache
  heat_log
  facts_fts (FTS5 虚拟表)
  media_attachments_fts (FTS5 虚拟表)
```
