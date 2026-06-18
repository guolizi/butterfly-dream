# 🦋 Butterfly Dream v2 — 六层记忆架构

> *"昔者庄周梦为蝴蝶，栩栩然蝴蝶也。"*
>
> 记忆如蝶，翩跹于时间、意义与关联的多维空间。

**Butterfly Dream v2** 是下一代记忆系统，采用六层架构（L0-L5），以 **agent 的认知层** 为视角，记录从对话中观察到和学习到的一切。

---

## 🏗️ 架构总览

```
L0 ─ 工作记忆 ──── 原始对话轮次 + 微事实索引
  ↓ 晋升（热/冷）
L1 ─ 事实池 ────── 事件 / 知识 / 行为模式 / 情感事件
  ↓ 抽象
L2 ─ 关系层 ────── 实体关系图 / 因果链 / 时间链
  ↓ 聚类
L3 ─ 抽象层 ────── GMM 聚类 / 概念抽象
  ↓ 叙事化
L4 ─ 叙事层 ────── 人生主线 / 关键节点
  ↓ 人格化
L5 ─ 灵魂层 ────── 人格模型 / 行为预测
```

### 核心设计原则

| 原则 | 说明 |
|:----|:-----|
| **永远保留** | 数据不删除，只有冷热分级 |
| **以人为中心** | 所有对话参与者都是平等主体，每人都有自己的 L1-L5 |
| **agent 也是主体** | agent 拥有自己的事实、情感和人格模型（L5） |
| **遗忘 = 检索策略分级** | 永不删，只按热度降级检索优先级 |
| **并行提取** | VAD 情感评价和事实维度并行提取，非串行推导 |
| **算法优先** | 确定性代码优先于 LLM prompt |

---

## 🧱 六层详解

### L0 — 工作记忆

原始对话轮次存储 + 关键词索引。`person` 字段记录实际说话人，不需要 `role` 字段。

- `conversation_turns` — 原始对话轮次
- `micro_facts` — 关键词 → turn_id 索引（jieba 分词）
- `promotion_queue` — 热晋升标记队列

### L1 — 事实池

四个池：事件记录池、静态知识池、行为模式池、情感事件池。前三个共用 `facts` 表，通过 `type` 区分。

- `facts` — 统一事实表（event / knowledge / behavior）
- `behavior_patterns` — 行为模式生命周期
- `emotion_events` — 情感事件（VAD 三维 + GENERATED 列）
- `emotion_triggers` — 情感触发关联
- `entities` / `fact_entities` — 实体体系
- `fact_relations` — 事实间关系（抽象/矛盾/支持）

### L2 — 关系层

- `entity_relations` — 实体关系图（PPR 检索）
- `causal_relations` — 三层因果链（短程/中程/长程）
- `timeline_relations` — 时间链
- `provenance` — 溯源追踪

### L3 — 抽象层

- `clusters` / `cluster_members` — GMM 聚类

### L4 — 叙事层

- `narratives` — 叙事主干 + 版本管理
- `narrative_emotion_nodes` — 关键情感节点

### L5 — 灵魂层

- `persona_models` — 人格模型（大五 + 特质 + GMM）
- `persona_snapshots` — 人格版本快照
- `behavior_predictions` — 行为预测日志

---

## 📚 文档

| 文档 | 说明 |
|:----|:-----|
| [`docs/v2-architecture-discussion.md`](docs/v2-architecture-discussion.md) | 架构设计讨论 |
| [`docs/v2-database-schema.md`](docs/v2-database-schema.md) | 完整数据库 Schema |
| [`docs/v2-retrieval-design.md`](docs/v2-retrieval-design.md) | 检索系统设计 |
| [`docs/v2-emotion-dimension.md`](docs/v2-emotion-dimension.md) | 情感维度设计 |
| [`docs/v2-behavior-prediction.md`](docs/v2-behavior-prediction.md) | 行为预测设计 |
| [`docs/v2-implementation-issues.md`](docs/v2-implementation-issues.md) | 实现问题记录 |
| [`docs/v2-db-viewer.md`](docs/v2-db-viewer.md) | 数据库查看器 |

---

## 🧪 评测

评测体系基于业界标准基准：

| 基准 | 来源 | 题数 | 类别 |
|:----|:----:|:----:|:----|
| **LoCoMo** | ACL 2024 | 1986 | 单跳 / 多跳 / 跨会话 / 时序推理 / 对抗 |

详见 [`eval/README.md`](eval/README.md)。

---

## 📄 许可证

MIT
