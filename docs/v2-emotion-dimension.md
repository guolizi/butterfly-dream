# 🦋 Butterfly Dream v2 — 情感维度设计

> 开始时间：2026-06-14
> 最后更新：2026-06-17
> 状态：讨论中，待续
> 范围：情感维度的设计目标、数据结构、VAD 模型、冷却机制、与各层关系、LLM 情感记忆服务

---

## 目录

- [一、设计目标](#一设计目标)
- [二、VAD 三维模型](#二vad-三维模型)
- [三、数据结构](#三数据结构)
- [四、VAD 数值来源](#四vad-数值来源)
- [五、情感重要性机制](#五情感重要性机制)
- [六、情感触发关联机制](#六情感触发关联机制)
- [七、情感冷却设计](#七情感冷却设计)
- [八、情感维度与各层的关系](#八情感维度与各层的关系)
- [九、LLM 情感记忆服务接口设计](#九llm-情感记忆服务接口设计)
- [十、情感模式与行为模式池的协同](#十情感模式与行为模式池的协同)
- [十一、初期实现建议](#十一初期实现建议)
- [十二、待讨论的问题](#十二待讨论的问题)
- [附录：讨论记录](#附录讨论记录)

---

## 一、设计目标

### 1.1 核心使命

> **为记忆系统提供结构化的情感信号，让上层（L3/L4/L5）和 LLM 对话能够基于情感数据理解用户、回忆共情、规避伤害。**

情感维度是**信号层，不是推理层**。它记录"发生了什么情感"，不负责"为什么"和"接下来会怎样"——因果推理是 L2 因果链的职责，行为预测是 L5 的职责。

### 1.2 四大核心能力

```
┌─────────────────────────────────────────────────────────────┐
│                    情感维度核心能力                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ❶ 情感记录（溯源）                                          │
│     记录"谁在什么时间因为什么事产生了什么情感"                   │
│     → 支撑：情感轨迹检索、L4 叙事弧线                          │
│                                                             │
│  ❷ 情感模式发现（抽象）                                       │
│     从多次情感转变中归纳"这个人的情感反应规律"                   │
│     → 支撑：L5 人格模型（情绪调节模式）、行为预测                │
│                                                             │
│  ❸ 情感状态感知（实时）                                       │
│     当前对话中用户/角色的情感状态                              │
│     → 支撑：mood_resonance、psych_probe、L3 情绪转折检测       │
│                                                             │
│  ❹ 情感记忆服务（LLM 对话）                                   │
│     为 LLM 对话提供情感上下文，让回复有温度                      │
│     ├─ 情感重要性标记 → "这个事件对用户很重要"                   │
│     ├─ 情感触发关联 → "这个话题会触发用户的负面情感"             │
│     └─ 主动检索接口 → LLM 在对话中按需查询                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 能力 ❹ 的两个关键场景

| 场景 | 效果 | 依赖的数据 |
|:----|:-----|:----------|
| **情感记忆召回** | 用户提到压力大 → LLM 知道上次压力大时通过跑步缓解了 → "上次你压力大的时候去跑了 5 公里，这次要不要试试？" | 情感重要性标记 + 情感-事件关联 |
| **情感回避** | 用户有身材焦虑 → LLM 知道"身高/体重"话题关联强烈负面情感 → 主动避开或小心处理 | 话题-情感触发关联 |

### 1.4 情感维度不做什么

```
❌ 不做情感生成 — 不负责"角色现在应该有什么情感"
   （那是 LLM 的角色扮演能力，不是记忆系统的职责）

❌ 不做情感推理 — 不负责"因为 A 所以 B 情感变化"
   （因果推理是 L2 因果链的职责，情感维度只提供原始数据）

❌ 不做情感预测 — 不负责"接下来会有什么情感"
   （那是 L5 行为预测的职责，情感维度只提供历史模式）
```

---

## 二、VAD 三维模型

### 2.1 VAD 定义

| 维度 | 范围 | 说明 |
|:----|:----|:-----|
| **Valence（愉悦度）** | -1.0 ~ 1.0 | 愉快 ↔ 不愉快 |
| **Arousal（唤醒度）** | 0.0 ~ 1.0 | 激动 ↔ 平静 |
| **Dominance（支配度）** | 0.0 ~ 1.0 | 掌控 ↔ 无力 |

### 2.2 常见情感在 VAD 空间中的位置

| 情感 | Valence | Arousal | Dominance |
|:----|:-------:|:-------:|:---------:|
| 开心 | +0.8 | 0.7 | 0.7 |
| 兴奋 | +0.7 | 0.9 | 0.6 |
| 平静 | +0.5 | 0.2 | 0.5 |
| 焦虑 | -0.6 | 0.8 | 0.3 |
| 悲伤 | -0.7 | 0.3 | 0.3 |
| 愤怒 | -0.5 | 0.9 | 0.6 |
| 惊讶 | 0.0 | 0.8 | 0.4 |
| 无聊 | -0.4 | 0.2 | 0.4 |
| 自豪 | +0.7 | 0.6 | 0.8 |
| 羞愧 | -0.5 | 0.4 | 0.2 |

### 2.4 情感模型兼容性

当前使用 **VAD 三维模型**（valence, arousal, dominance）。架构设计预留了未来切换更精细情感模型的能力（如 21 维情感空间模型）。

**兼容策略：**

```sql
emotion_model TEXT DEFAULT 'vad-3d'
  -- 'vad-3d'     = 当前 VAD 三维（valence, arousal, dominance）
  -- 'emotion-21d' = 未来 21 维情感空间
  -- 其他模型可扩展
```

- 存储层用 `emotion_model` 字段标记当前模型，查询时根据模型选择对应的距离计算方式
- intensity 公式通用化：`intensity = √(Σeᵢ²) / √dim`（任意维度适用）
- 推导层（转变/模式）基于向量空间，不依赖固定维度数
- 迁移路径：Phase 1 固定字段 → Phase 2 双写 → Phase 3 统一向量

详见 §三 存储模型和 §九 初期实现建议。

1. **更丰富的情感表达**
   - 单标签+强度: "开心 0.8"
   - VAD: valence=+0.8, arousal=0.7, dominance=0.7
   - → 可以区分"兴奋的开心"和"平静的开心"

2. **情感轨迹更精确**
   - 单标签: "开心 → 焦虑"（跳跃，中间状态丢失）
   - VAD: (0.8,0.7,0.7) → (0.5,0.6,0.5) → (0.0,0.7,0.4) → (-0.6,0.8,0.3)
   - → 可以看到情感在 VAD 空间中的连续变化路径

3. **情感模式更丰富**
   - 单标签: "她压力大时会画画"
   - VAD: "当 valence 下降 + arousal 上升时，她倾向于做高支配度活动"
   - → 模式发现可以基于 VAD 空间中的区域，而非固定标签

4. **跨文化通用**
   - 情感标签有文化差异（中文的"郁闷"没有精确英文对应）
   - VAD 是跨文化的连续空间

---

## 三、存储模型

### 3.1 情感事件池（L1 第 4 个池）

情感事件作为 L1 的第 4 个池，与事件记录池、静态知识池、行为模式池并列。

```
L1 池体系:

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  事件记录池    │  │  静态知识池    │  │  行为模式池    │  │  情感事件池    │
│  (文本事实)    │  │  (去时间化)    │  │  (条件-行为)   │  │  (VAD 序列)   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       ↓                                                      ↓
  推导层（非存储，查询时构建或睡眠周期物化）
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  情感转变      │  │  情感模式      │  │  情感触发关联   │
  │  (transitions)│  │  (patterns)   │  │  (triggers)   │
  └──────────────┘  └──────────────┘  └──────────────┘
```

**为什么独立成池：**
- 情感池的核心数据是**连续数值**（VAD 三维），不是文本，FTS5/embedding 检索不适用
- 冷却规则不同（importance 特殊处理）
- 单条情感事件没有独立意义，必须放在时间序列中才有价值
- 与事件池通过 `context_fact_id` 关联，互不依赖存储

**情感池 vs 事件池的关系：**

```
事件池: "Caroline 的宠物去世了" (fact_id=1001)
    ↓ context_fact_id
情感池: valence=-0.85, arousal=0.7, dominance=0.2, importance=0.95
    ↑ 两条记录通过 context_fact_id 关联
    ↑ 情感池是连续的线，事件池是线上的点
```

**触发事实 vs 情感对象：**

`primary_fact_id` 和 `emotion_target` 是两个正交维度，共同描述一次情感体验：

```
primary_fact_id（触发事实）    emotion_target（情感对象）

"老板在会议上批评了我"          对"老板"生气
"朋友说我妈病了"               对"妈妈"担心
"考试通过了"                   对"自己"自豪
"莫名焦虑"                     null（无对象，mood）
```

触发事实回答"因为什么事"，情感对象回答"对谁/什么"。两者可以相同（对老板生气是因为老板做的事），也可以不同（被老板骂了但对自己生气）。

### 3.2 数据结构

#### 情感事件池（Phase 1 一表）

```sql
emotion_events:
  event_id, person, timestamp,
  emotion_model TEXT DEFAULT 'vad-3d',   -- 情感模型类型
    -- 'vad-3d'     = VAD 三维（当前）
    -- 'emotion-21d' = 21 维情感空间（未来）
  valence, arousal, dominance,          -- VAD 三维核心（当前模型专用）
  emotion_vector JSON,                  -- 通用情感向量（未来扩展）
    -- 当前：emotion_vector = [valence, arousal, dominance]
    -- 未来 21 维：emotion_vector = [...21 个值...]
    -- 写入时与固定字段同步，查询时优先使用固定字段（性能）
  emotion_label,                        -- 可选，方便人类阅读
  emotion_target TEXT,                   -- 情感对象
    -- null         = 无对象（mood，如"莫名焦虑"）
    -- 'self'       = 对自己（自尊、自豪、羞愧）
    -- 'person:老板' = 对某人
    -- 'entity:工作' = 对实体/概念
    -- 'event:考试'  = 对事件
    -- 'place:办公室' = 对场所
  intensity,                            -- 通用公式：√(Σeᵢ²) / √dim
  primary_fact_id,                      -- 主要触发事实（非空）
  related_fact_ids,                     -- 关联事实列表（可选，TEXT[]）
    -- 支持多因情感：同一天升职+朋友搬走 → 矛盾情感
    -- primary_fact_id 是主要触发因素，related_fact_ids 是辅助因素
  source (llm_extraction / l0_promotion / inferred),
  importance (0.0~1.0),                -- 情感重要性
  significance_reason,                  -- LLM 标注的重要性理由（可选）
  trigger_topics (TEXT[]),              -- 提取的情感触发话题
  notes
```

#### 事件池（新增 emotion_tag 字段）

```sql
event_facts:
  fact_id, person, timestamp, content,
  emotion_tag TEXT,                    -- 新增：轻量情感标签
    -- null      = 中性/无情感（默认）
    -- "开心"    = 具体情感标签
    -- "positive" = 正向情感（valence > 0.3，自动派生）
    -- "negative" = 负向情感（valence < -0.3，自动派生）
    -- "mixed"   = 矛盾情感
  ...
```

**emotion_tag 的作用：**
- 让 L3 抽象层能**直接过滤**情感相关事实，无需每次 JOIN 情感池
- 只存标签，不存 VAD 数值（避免冗余）
- 需要详细 VAD 时 → 通过 `primary_fact_id` JOIN 到情感池

**同步时机（两阶段）：**
- **第一阶段（L0→L1，实时）：** LLM 多维度提取时直接填 emotion_tag。这是"最佳努力"估值，可能漏标，但 LLM 对话立即可用。
- **第二阶段（L3，睡眠周期）：** L3 回顾情感池中新增的记录，如果事件池对应事实的 emotion_tag 为 null，回填。详见 §四 情感提取策略。

#### 三表结构（最终形态，Phase 2 物化）

```sql
-- 情感状态节点
emotion_states:
  state_id, person, timestamp,
  emotion_model TEXT DEFAULT 'vad-3d',
  valence, arousal, dominance,
  emotion_vector JSON,
  emotion_label,
  emotion_target TEXT,                   -- 情感对象（同 emotion_events）
  intensity,
  primary_fact_id,
  related_fact_ids,
  source,
  importance,
  significance_reason,
  notes

-- 情感转变
emotion_transitions:
  transition_id, person,
  from_state_id, to_state_id,
  transition_type (正向突破/负向冲击/累积升华/韧性恢复/...),
  vector_delta JSON,                    -- 通用：后向量 - 前向量（逐元素差）
  delta_magnitude FLOAT,                -- 变化幅度（标量，任意维度通用）
  trigger_fact_ids,
  pattern_id (可选)

-- 情感模式
emotion_patterns:
  pattern_id, person, description,
  vector_region JSON,                   -- 在情感空间中的区域描述（通用）
  confidence, source_transition_ids,
  created_at, updated_at

-- 情感触发关联
emotion_triggers:
  trigger_id, person,
  trigger_type (topic / entity / event_type / location / ...),
  trigger_value,
  associated_vector JSON,               -- 关联的典型情感向量（通用）
  trigger_count,
  last_triggered_at,
  confidence,
  source_state_ids
```

### 3.3 三个层次

```
情感状态节点（时间点 → 情感状态 + 强度）
    ↓ 关联
情感转变（从什么状态到什么状态 + 触发因素）
    ↓ 抽象
情感模式（多次转变中归纳的规律）
```

### 3.4 查询时动态推导（Phase 1）

```sql
-- transitions = 按时间排序相邻 event 的 VAD 变化
-- patterns = 聚类相似 VAD 变化路径
-- triggers = 按 topic 聚合的 VAD 均值
```

冷却：简单时间衰减（按 timestamp 降权），importance ≥ 0.8 永久保温

---

## 四、情感提取策略

### 4.1 核心思路：搭便车，不独立触发

情感提取**不做独立触发判断**，而是跟随 L0→L1 多维度提取一起执行。每次 LLM 多维度提取时，情感维度作为五个输出维度之一同步输出。

**为什么可以这样：**
- LLM 已经被多维度提取调用了，多输出几个 VAD token 的成本几乎为零
- 不需要前置关键词规则来"判断是否要提取情感"
- 真正需要控制的是**是否写入存储**，而不是是否调用 LLM

### 4.2 两阶段写入

```
第一阶段（L0→L1，实时）:
┌─────────────────────────────────────────────────────────────┐
│  LLM 多维度提取时同步输出情感维度                              │
│                                                              │
│  LLM 输出控制：                                              │
│  ├─ 有情感信号 → 完整 VAD + importance + trigger_topics      │
│  │              + emotion_tag                                │
│  │              → 写入情感池 + 事件池 emotion_tag             │
│  │                                                           │
│  └─ 无情感信号 → 中性默认值                                   │
│     valence=0.0, arousal=0.5, dominance=0.5                 │
│     importance=null, trigger_topics=[]                       │
│     → 不写入情感池，事件池 emotion_tag=null                   │
│                                                              │
│  这是"最佳努力"估值，不一定精确，但 LLM 对话立即可用            │
└─────────────────────────────────────────────────────────────┘
       ↓
第二阶段（L3 抽象层，睡眠周期）:
┌─────────────────────────────────────────────────────────────┐
│  L3 有更多上下文（前后事件、重复模式），可以做：                │
│                                                              │
│  ① 修正 VAD 值                                               │
│     "单次提取 valence=-0.4，但连续 3 天都是这个情绪"           │
│     → 调高 importance                                        │
│                                                              │
│  ② 合并同类情感事件                                           │
│     "连续 5 条'压力大'记录 → 合并为一条持续情感状态"            │
│                                                              │
│  ③ 标记 false positive                                       │
│     "用户说'笑死我了'但实际是中性 → 标记为误报"                 │
│                                                              │
│  ④ 回填 emotion_tag（如果 LLM 漏标了）                        │
│     "情感池有记录但事件池 emotion_tag=null → 补上"             │
│                                                              │
│  ⑤ 发现情感模式（emotion_patterns）                           │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 与现有架构的一致性

两阶段写入与现有架构的热/冷路径模式完全一致：

```
事实晋升：  热路径（实时）→ 冷路径（睡眠周期 refine）
检索路由：  快速路径（关键词）→ LLM 兜底（低置信度时）
情感提取：  实时写入（初步）→ L3 refine（睡眠周期修正）
```

### 4.4 LLM 提取格式

```json
{
  "dimension": "event",
  "content": "用户说最近工作压力很大",
  "emotion_tag": "焦虑",
  ...
}
{
  "dimension": "emotion",
  "emotion_model": "vad-3d",
  "emotion_vector": [-0.4, 0.7, 0.4],
  "valence": -0.4,
  "arousal": 0.7,
  "dominance": 0.4,
  "emotion_label": "焦虑",
  "emotion_target": null,               -- 无对象（mood）
  "importance": 0.5,
  "significance_reason": "用户主动表达压力，但未提及具体事件影响",
  "trigger_topics": ["工作"]
}
```

注意：`emotion_tag` 是事件维度的字段（放在事件池），VAD/emotion_vector 是情感维度的字段（放在情感池）。同一 LLM 调用同时输出两个维度。`emotion_vector` 与固定字段同步，未来切换模型时固定字段可为 NULL，以 `emotion_vector` 为准。

### 4.5 热路径（已有）

对话中出现强烈情感表达（intensity ≥ 0.9 或 importance ≥ 0.9）→ 触发热路径晋升，不等睡眠周期。已在 §4-L0 热路径晋升中定义。

---

## 五、情感重要性机制

### 5.1 intensity vs importance

| 维度 | 含义 | 计算方式 | 用途 |
|:----|:-----|:---------|:-----|
| **intensity** | 情感有多强烈 | 通用公式：√(Σeᵢ²) / √dim（VAD 三维下 = √(v²+a²+d²)/√3） | 冷却、检索排序 |
| **importance** | 这个情感事件对用户人生有多重要 | LLM 标注（0~1） | 永久保温、LLM 情感记忆召回 |

### 5.2 importance 标注规则

| importance 范围 | 含义 | 例子 | 冷却行为 |
|:--------------|:-----|:-----|:---------|
| ≥ 0.9 | 人生里程碑级 | 宠物去世、结婚、升职、重大挫折 | 永久保温 |
| 0.7 ~ 0.9 | 重要情感事件 | 重要考试通过、与挚友争吵 | 减速冷却 |
| 0.4 ~ 0.7 | 日常情感事件 | 工作小成就、普通约会 | 正常冷却 |
| < 0.4 | 轻微情感波动 | 被路人踩脚、看到好天气 | 加速冷却 |

### 5.3 LLM 标注时机

在多维度提取时同步标注。importance 的判断依据：
- 事件对用户人生轨迹的潜在影响
- 事件涉及的核心价值（亲情/友情/事业/健康等）
- 用户在该事件上的情感投入程度（intensity × 持续时间）

---

## 六、情感触发关联机制

### 6.1 目标

建立"话题/实体 → 情感 VAD"的映射，支撑 LLM 情感回避和情感记忆召回。

### 6.2 归属：静态知识池

emotion_triggers **归属静态知识池**，而非情感池。

**理由：**
- triggers 的本质是**用户的稳定心理特征**（"这个用户对什么话题有什么情感反应"），和"用户喜欢跑步"是同一类数据
- 生命周期独立：情感事件会冷却遗忘，但"用户对身高话题敏感"这个知识不应随情感事件冷却而消失
- 查询路径最短：LLM 对话中查"这个话题安全吗" → 直接查静态知识池，不经过情感池
- 数据来源与归属分离：数据从情感池聚合而来，但沉淀为静态知识

**数据流：**

```
情感池（emotion_events）
  ↓ 聚合 trigger_topics（睡眠周期或查询时）
静态知识池（emotion_triggers）
  ↓ LLM 对话中直接查询
情感回避 / 情感记忆召回
```

这与 L1→L2 的晋升机制类似——情感池负责记录，静态知识池负责沉淀为稳定知识。

### 6.3 构建方式（渐进式）

```
Phase 1（初期）:
  从 emotion_events 的 trigger_topics 字段聚合
  同一 topic 出现 ≥ 3 次 → 建立触发关联（不要求 VAD 一致性）
  关联置信度 = min(recent_count / 5, 1.0) × recent_consistency
    recent_count: 最近 5 次事件中匹配的次数
    recent_consistency: 最近 5 次事件的 VAD 余弦相似度均值
    → 高 confidence = 稳定情感反应
    → 低 confidence = 情感在演化/不稳定（本身就是信息）

Phase 2+:
  睡眠周期中主动挖掘隐性触发关联
  L3 抽象层发现"话题 A 总是伴随情感 B"的模式
  写入 emotion_triggers 表
```

**为什么不用全历史均值：**

emotion_triggers 的 `associated_vector` 是**当前状态**（用于情感回避），不是历史总结。情感演化由 emotion_events 的时序数据覆盖：

```
情感回避（emotion_triggers）:
  取最近 5 次事件的 VAD 加权平均（时间越近权重越高）
  → 回答"这个话题现在安全吗？"

情感演化（emotion_events 时序）:
  query_emotion_timeline(topic="前任") 
  → 从 emotion_events 按 trigger_topics 过滤，返回时间序列
  → 回答"用户对 X 的情感怎么演变的？"
```

**低 confidence 的信号价值：**

| confidence | 含义 | 情感回避行为 |
|:----------:|:-----|:------------|
| ≥ 0.7 | 稳定情感反应 | 正常使用 VAD 判断 |
| 0.3 ~ 0.7 | 情感在演化/不稳定 | 取最近 1 次 VAD（最新状态），标注低置信度 |
| < 0.3 | 数据不足或高度矛盾 | 不触发自动回避，交由 LLM 自行判断 |

### 6.4 LLM 对话中的使用方式

```
用户输入 → LLM 检测到敏感话题
    ↓
LLM 调用情感触发查询: "话题 X → 该用户 → 情感 VAD"
    ↓
if confidence >= 0.7:
    if valence < -0.6:
        主动回避或谨慎处理该话题
    elif importance > 0.8:
        可自然提及相关情感记忆（如果上下文合适）
elif confidence >= 0.3:
    取最近 1 次 VAD，标注低置信度，由 LLM 自行判断
else:
    不触发自动行为，交由 LLM 自行判断
```

### 6.5 触发关联的冷却

不参与冷热分级。触发关联是累积统计，只增不减。但如果长时间（> 6 个月）无新触发，confidence 逐渐衰减。

---

## 七、情感冷却设计

情感维度参与整体冷热分级，但有独立的冷却规则：

### 7.1 emotion_states

| 冷却依据 | 加速条件 | 减速条件 |
|:--------|:---------|:---------|
| 时间衰减为主 | 关联事实冷却 | importance ≥ 0.7 |
| | 低频人物 | 被 L4 标记为叙事关键情感节点 |
| | | 被 L5 标记为人格特质相关 |

### 7.2 emotion_transitions

| 冷却依据 | 加速条件 | 减速条件 |
|:--------|:---------|:---------|
| 跟随源状态 | 源状态冷却 | 被 L3 标记为情感模式的一部分 |

### 7.3 emotion_patterns

| 冷却依据 | 加速条件 | 减速条件 |
|:--------|:---------|:---------|
| 重要性为主 | 低频/已被替代 | 高频出现 |
| | 低 confidence | 高 confidence |

### 7.4 加热规则

- 被检索命中 → 热度 +1 级
- importance ≥ 0.8 → 永久保温
- 被 L4 标记为叙事关键情感节点 → 永久保温
- 被 L5 标记为人格特质相关情感 → 永久保温

---

## 八、情感维度与各层的关系

```
L0→L1 晋升（多维度提取）:
  → 情感维度输出（VAD + importance + trigger_topics）→ 写入 emotion_events
  → 高 importance（≥ 0.9）立即触发热路径晋升

L1 事件记录池:
  → 事件触发情感 → context_fact_id 关联
  ← 情感状态为事件提供情感上下文（LLM 对话中使用）

L2 关系层:
  → 时间链关联情感转变
  → 因果链解释情感转变的原因
  ← 情感轨迹为因果链提供情感信号
  ← 情感触发关联为因果链提供"话题→情感"的模式输入

L3 抽象层:
  → 情绪转折检测（VAD 空间中的大位移 → 实时触发抽象管道）
  → 从 emotion_transitions 归纳 emotion_patterns
  → **睡眠周期中 refine 情感数据**（修正 VAD、合并同类事件、标记 false positive、回填 emotion_tag）
  ← 情感模式为行为预测提供输入

L4 叙事层:
  → 情感轨迹是故事线的情感弧线
  → 高 importance 情感节点是叙事的关键锚点
  ← 叙事关键情感节点 → 情感冷却减速

L5 灵魂层:
  → 情感模式是人格模型的一部分（情绪调节模式）
  → 情感触发关联是人格特质的外在表现
  ← 人格特质相关情感 → 情感冷却减速
  ← VAD 3 维（psych_probe）直接从情感池取值，无需映射

LLM 对话（能力 ❹）:
  → 情感记忆服务：按需查询高 importance 情感事件
  → 情感回避：查询当前话题的情感触发关联
  ← 情感数据让 LLM 回复更有温度

遗忘机制:
  → 情感数据参与冷热分级
  ← 各层冷却系数叠加影响情感热度
  ← importance ≥ 0.8 的情感事件永久保温（即使关联事实已冷却）
```

---

## 九、LLM 情感记忆服务接口设计

### 9.1 设计原则

```
Phase 1: 自动注入兜底（零 LLM 工作量）
Phase 2+: 自动注入 + 工具调用补充（可选深度查询）
```

**为什么自动注入是基础：**
- 情感回避是**安全需求**——不能让 LLM 在不知道话题敏感性的情况下回复
- 情感记忆召回是**质量需求**——自动注入保证 LLM 不会在"裸奔"状态下回复情感话题
- 工具调用是**延迟换深度**——只在 LLM 明确需要更多数据时才触发

### 9.2 Phase 1：自动注入（分级递进检索）

LLM 不感知情感检索的存在。检索管道在 LLM 回复之前自动完成：

```
用户输入
    ↓
QueryClassifier 识别 query_type
    ↓
Layer Router 决定查哪些层
    ↓
各层检索（含 §8 分级递进情感检索）
    ↓
Fusion Engine 组装上下文
    ↓
LLM 回复（上下文中已包含情感数据）
```

**自动注入的内容：**

| 数据 | 来源 | 触发条件 |
|:----|:-----|:---------|
| 当前话题的情感触发关联 | emotion_triggers（静态知识池） | 关键词检测到情感信号 |
| 高 importance 情感记忆（top-3） | emotion_events | triggers 显示强关联（confidence ≥ 0.7） |
| 最近 VAD 状态 | emotion_events（最近一条） | 任何情感信号 |

**LLM 视角：** 它看到的上下文里已经有了"这个话题用户之前有过负面反应"或"用户最近情绪不太好"的信息，它自然就知道怎么回复。

### 9.3 Phase 2+：工具调用（可选深度查询）

LLM 在对话中可以主动调用以下工具获取更详细的情感数据：

```python
# 工具 1：按话题查历史情感事件
def query_emotion_memory(
    topic: str,           # 话题关键词
    person: str,          # 目标人物
    limit: int = 5,       # 返回条数
    min_importance: float = 0.0  # 最低重要性过滤
) -> list[dict]:
    """
    返回: [
        {timestamp, emotion_label, valence, arousal, dominance,
         intensity, importance, context_fact_content, ...}
    ]
    用途: LLM 想深入了解"用户对某个话题的情感历史"
    """

# 工具 2：查情感轨迹（时间序列）
def query_emotion_timeline(
    person: str,
    time_range: tuple[str, str],  # (start, end) ISO 时间
    granularity: str = "day"      # "hour" / "day" / "week"
) -> list[dict]:
    """
    返回: [
        {time_bucket, avg_valence, avg_arousal, avg_dominance,
         dominant_emotion, event_count, ...}
    ]
    用途: LLM 分析"用户最近情绪变化趋势"
    """

# 工具 3：查用户情感模式
def query_emotion_pattern(
    person: str
) -> list[dict]:
    """
    返回: [
        {pattern_id, description, vector_region,
         confidence, frequency, ...}
    ]
    用途: LLM 了解"这个人的情感反应规律"
    """
```

**触发时机：** LLM 在自动注入的数据不够用时，自行决定是否调用工具。例如：

```
用户: "我最近是不是情绪波动很大？"

自动注入 → 最近 VAD 状态 + 最近几条情感事件
LLM 觉得不够 → 调用 query_emotion_timeline("user", ("2026-05-01", "2026-06-16"))
            → 拿到完整时间序列 → 分析波动趋势 → 回复
```

### 9.4 自动注入 vs 工具调用的分工

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM 情感记忆服务                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  自动注入（Phase 1 起）                    工具调用（Phase 2+ 起）      │
│  ─────────────────────                    ────────────────────      │
│  • 情感回避（安全需求）                      • 深度情感历史查询          │
│  • 最近情感状态（上下文感知）                  • 情感轨迹趋势分析          │
│  • 高 importance 情感记忆                    • 情感模式探索              │
│  • 当前话题触发关联                          • 跨时间范围对比            │
│                                                                     │
│  延迟：< 200ms（分级递进）                   延迟：100ms ~ 1s            │
│  触发：自动（LLM 无感知）                    触发：LLM 主动调用          │
│  频率：每次对话                             频率：按需                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.5 不做什么

```
❌ 不做情感注入的"智能判断"——LLM 不需要判断"我现在需要情感数据吗"
   （检索管道自动处理，LLM 只负责用）

❌ 不做情感上下文的"精简"——自动注入的数据已经是分级递进后的最小集
   （第一层无信号 → 零注入；第二层弱关联 → 仅 triggers；第三层强关联 → 完整）

❌ 不做 LLM 的情感数据缓存——每次对话独立检索
   （情感数据变化快，缓存可能过时）
```

---

## 十、情感模式与行为模式池的协同

### 10.1 问题

情感模式（emotion_patterns）和行为模式（behavior_patterns）各自独立发现和存储，但天然相关：

```
行为模式："压力大时 → 去跑步"
情感模式："当 valence 下降 + arousal 上升时，倾向于做高支配度活动"
```

它们描述的是**同一个规律的不同视角**，但目前没有交叉引用。情感模式缺少具体的消费场景，行为模式缺少情感上下文。

### 10.2 核心思路：情感模式作为行为模式的上下文选择器

```
Phase 3 协同架构：

┌─────────────────────────────────────────────────────────────────┐
│                    L3 睡眠周期（统一发现管道）                      │
│                                                                   │
│  事件池事实 + 情感池 VAD 序列                                      │
│    ↓                                                              │
│  统一模式发现（一次 LLM 调用，同时输出情感和行为两个视角）           │
│    ├─ 行为模式："压力大时 → 去跑步"                                │
│    └─ 情感模式："valence↓ + arousal↑ → 高支配度活动"              │
│    ↓                                                              │
│  两个模式通过关联表交叉引用                                         │
│    └─ pattern_relations（多对多）                                  │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    L5 行为预测（使用协同）                         │
│                                                                   │
│  用户当前情感状态（VAD）                                           │
│    ↓                                                              │
│  匹配情感模式 → 找到关联的行为模式（多对多）                        │
│    ↓                                                              │
│  行为模式作为 GMM 的上下文先验                                      │
│    → P(mode_k | emotion_pattern) 提升匹配模式的权重                │
│    → 预测更精准                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 10.3 数据结构：多对多关联表

```sql
-- 情感模式 ↔ 行为模式的多对多关联
pattern_relations:
  relation_id,
  emotion_pattern_id REFERENCES emotion_patterns(pattern_id),
  behavior_pattern_id REFERENCES behavior_patterns(pattern_id),
  correlation FLOAT,              -- 关联强度 0~1
    -- 0.0 = 无关联（默认）
    -- 0.3 = 弱关联（偶发共存）
    -- 0.7 = 强关联（高频伴随）
    -- 1.0 = 必然伴随（理论上限）
  sample_count INT,               -- 观察次数（correlation 的置信度基础）
  last_observed_at,
  confidence,                     -- min(sample_count / 10, 1.0) × correlation
  created_at, updated_at

-- 行为模式池新增字段
behavior_patterns:
  ...现有字段...
  pattern_type: 'routine' | 'emotion-driven' | 'value-driven',
    -- 'routine'        = 例行行为，无情感驱动（如"每周日游泳"）
    -- 'emotion-driven' = 情感驱动的行为（如"焦虑时去跑步"）
    -- 'value-driven'   = 价值观驱动的行为（如"坚持环保"）

-- 情感模式保持现有结构不变
emotion_patterns:
  ...现有字段...  -- 无需新增字段，关联通过 pattern_relations 表
```

**为什么用关联表而非单字段引用：**

```
❌ 单字段引用（emotion_context → emotion_patterns.id）
   情感模式 A → 行为模式 1（一对一，丢失了"焦虑也可能画画"的信息）

✅ 多对多关联表
   情感模式 A（焦虑）→ 行为模式 1（跑步）、行为模式 2（画画）
   情感模式 B（兴奋）→ 行为模式 1（跑步）
   → 完整保留"一种情感有多种应对方式"的多样性
```

### 10.4 关联的发现方式

```python
# Phase 3 L3 睡眠周期：统一模式发现
def discover_patterns():
    """
    LLM 输入: 事件池事实 + 情感池 VAD 序列
    LLM 输出: [
        {
            "behavior": "压力大时去跑步",
            "emotion_context": "valence↓ + arousal↑",
            "emotion_label": "焦虑",
            "pattern_type": "emotion-driven",
            "confidence": 0.85
        },
        {
            "behavior": "每周日去游泳",
            "emotion_context": null,   # 无情感关联
            "pattern_type": "routine",
            "confidence": 0.9
        },
        ...
    ]
    """
    # 1. 从事件池 + 情感池提取候选模式对
    # 2. 去重：与已有 emotion_patterns / behavior_patterns 匹配
    # 3. 新建或更新 pattern_relations
    # 4. 更新 correlation = 新观察次数 / 总观察次数
```

**关联强度的统计积累：**

```
同一情感模式和行为模式同时出现 1 次 → correlation = 0.1, confidence = 0.1
同一情感模式和行为模式同时出现 5 次 → correlation = 0.5, confidence = 0.5
同一情感模式和行为模式同时出现 10 次 → correlation = 0.8, confidence = 0.8
```

correlation 不是 LLM 一次性标注的，而是**多次观察的统计积累**，置信度随观察次数自然增长。

### 10.5 L5 行为预测中的使用

```
用户当前状态:
  VAD = (-0.5, 0.8, 0.3)  -- 焦虑

Step 1: 匹配情感模式
  → 匹配到 "焦虑模式" (pattern_id=3)
    vector_region = {valence: [-0.7, -0.3], arousal: [0.6, 0.9], dominance: [0.2, 0.5]}

Step 2: 通过 pattern_relations 找到关联的行为模式（多对多）
  → pattern_relations WHERE emotion_pattern_id = 3:
    ├─ behavior_pattern_id=7  "去跑步"    correlation=0.8, confidence=0.8
    ├─ behavior_pattern_id=12 "画画"      correlation=0.5, confidence=0.5
    └─ behavior_pattern_id=5  "暴饮暴食"  correlation=0.3, confidence=0.3

Step 3: 提升 GMM 对应模式的先验概率
  → P(mode_跑步 | context) += 0.2  (correlation × confidence = 0.64)
  → P(mode_画画 | context) += 0.1  (correlation × confidence = 0.25)
  → P(暴饮暴食 | context) += 0.05  (correlation × confidence = 0.09)
  → 预测结果更倾向于"跑步"而非其他行为
```

### 10.6 生命周期协同

两个模式走同一套生命周期状态机，但独立演进：

```
发现（L3 睡眠周期）
  ↓ tentative（置信度 < 0.7）
  ↓ 多次验证（同一模式出现 ≥ 3 次）
confirming（置信度 0.7~0.9）
  ↓ 时间稳定（持续 ≥ 2 个睡眠周期）
confirmed（置信度 ≥ 0.9）
  ↓ 被新模式替代
evolved（通过 evolved_from 链接到新模式）
  or
superseded（被证伪，保留历史记录）
```

**关联的生命周期跟随两个模式中置信度较低的那一个：**

```
emotion_pattern confidence = 0.9, behavior_pattern confidence = 0.6
  → pattern_relations.confidence = min(0.9, 0.6) = 0.6
  → 行为模式还在 tentative 阶段，关联不可用
```

### 10.7 不做什么

```
❌ 不合并两个池 — 情感模式和行为模式存储结构不同，合并反而耦合
   （情感模式是 VAD 空间中的区域，行为模式是条件-行为文本描述）

❌ 不做情感→行为的强制映射 — 不是所有行为都有情感驱动
   （routine 类型的行为如"每周日游泳"与情感无关）

❌ 不做双向因果推理 — 不负责"因为情感模式 A 所以行为模式 B"
   （因果推理是 L2 因果链的职责，这里只做关联映射）

❌ 不做跨人关联 — 不比较不同用户的情感-行为关联模式
   （那是社会模拟场景的职责，不在本设计范围内）
```

---

## 十一、初期实现建议

```
Phase 1（一表 + 基础能力 — 当前）:
  emotion_events 表（含 VAD 固定字段 + emotion_model + emotion_vector）
  查询时动态推导 transitions / patterns / triggers
  冷却：简单时间衰减 + importance ≥ 0.8 永久保温
  LLM 对话：自动注入情感上下文（分级递进检索），零 LLM 工作量
  情感模型：VAD 三维（emotion_model='vad-3d'）

Phase 2（三表物化 + 兼容准备）:
  emotion_states（从 emotion_events 迁移）
  emotion_transitions（物化推导结果，通用 vector_delta）
  emotion_patterns（物化聚类结果，通用 vector_region）
  emotion_triggers（物化触发关联，通用 associated_vector）
  冷却：各层冷却系数叠加
  情感触发关联：睡眠周期中主动挖掘
  双写：写入固定字段同时写 emotion_vector JSON

Phase 3（高级能力 + 模型切换）:
  情感模式与行为模式池协同（pattern_relations 多对多关联表，统一发现管道）
  L5 预测中情感模式作为 GMM 上下文先验（correlation × confidence 提权）
  跨人情感模式对比（社会模拟场景）
  情感轨迹预测（作为 L5 行为预测的子模块）
  可切换情感模型（如 21 维），查询逻辑根据 emotion_model 路由
```

---

## 十二、待讨论的问题

以下问题尚未深入，留待后续讨论：

- [x] **情感提取的触发策略** — ✅ 已解决：搭便车（跟随多维度提取）+ 两阶段写入（L0→L1 实时写入 + L3 睡眠周期 refine）。详见 §四。
- [x] **情感重要性的定义** — ✅ 已解决：LLM 标注 importance，区分 intensity（多强烈）和 importance（多重要）。详见 §五。
- [x] **情感触发关联的归属** — ✅ 已解决：归属静态知识池。triggers 是用户的稳定心理特征，不随情感事件冷却而消失。详见 §六。
- [x] **主动检索的触发时机** — ✅ 已解决：分级递进检索（关键词检测 → emotion_triggers → 完整情感检索）。详见 `v2-retrieval-design.md` §8。
- [x] **VAD 到 L5 的接口** — ✅ 已解决：psych_probe 直接用 VAD 3 维（取代旧 2 维情绪效价），零映射。行为预测核心维度从 11 维→12 维。详见 `v2-behavior-prediction.md` §2.1。
- [x] **LLM 情感记忆服务的接口设计** — ✅ 已解决：Phase 1 自动注入兜底（分级递进检索），Phase 2+ 增加工具调用（query_emotion_memory / query_emotion_timeline / query_emotion_pattern）。详见 §九。
- [x] **情感模式与行为模式池的协同** — ✅ 已解决：多对多关联表（pattern_relations），情感模式作为 GMM 上下文先验。详见 §十。
- [x] **多人物情感管理** — ✅ 已解决：emotion_events 新增 emotion_target 字段（TEXT），区分 mood（null）/ 对自己（self）/ 对他人（person:xxx）/ 对实体（entity:xxx）/ 对事件（event:xxx）/ 对场所（place:xxx）。与 primary_fact_id（触发事实）正交。详见 §三。
- [x] **情感轨迹预测** — ✅ 已解决：不需要独立模块。改为统一的 behavior_predictions 预测日志表，记录预测时的情绪 + 行为后的情感变化，形成情感反馈闭环。详见 `v2-behavior-prediction.md` §十。
- [x] **情感触发关联的隐私问题** — ✅ 已解决：新增 user_blocks 屏蔽表（不提，不删）。用户说"别提这个" → 记录屏蔽，检索时跳过，数据永远保留。真实删除需二次确认。详见主架构文档 §4-遗忘机制。

---

## 附录：讨论记录

| 日期 | 议题 | 决策/内容 |
|:----|:-----|:----------|
| 2026-06-14 | 初始讨论 | 问题分析、VAD 模型引入、三表结构、冷却框架、与各层关系 |
| 2026-06-16 | 深度分析 | 四大核心能力明确、VAD 定稿、importance 机制、emotion_triggers、LLM 情感记忆服务。详见主架构文档 §十 讨论记录 |
| 2026-06-16 | 存储模型定稿 | 情感事件作为 L1 第 4 个池（独立于事件池）。emotion_events 支持多事实关联（primary_fact_id + related_fact_ids）。事件池新增 emotion_tag 轻量标签。详见 §三 |
| 2026-06-16 | 情感提取策略定稿 | 搭便车（跟随多维度提取，不独立触发）+ 两阶段写入（L0→L1 实时写入 + L3 睡眠周期 refine）。emotion_tag 由 LLM 直接填，L3 回填漏标。详见 §四 |
| 2026-06-16 | 情感模型兼容性设计 | 新增 emotion_model 字段标记情感模型类型。存储层用 JSON 通用向量 + 固定字段双写。推导层改为通用 vector_delta/vector_region。intensity 公式通用化。迁移路径：Phase 1 固定字段 → Phase 2 双写 → Phase 3 统一向量。详见 §二、§三、§九 |
| 2026-06-16 | 情感触发关联归属定稿 | emotion_triggers 归属静态知识池（非情感池）。triggers 是用户的稳定心理特征，不随情感事件冷却而消失。数据流：情感池聚合 → 静态知识池沉淀 → LLM 对话直接查询。详见 §六 |
| 2026-06-16 | 主动检索触发时机定稿 | 分级递进检索：零成本关键词检测 → emotion_triggers（10~30ms）→ 完整情感检索（50~200ms）。详见 `v2-retrieval-design.md` §8 |
| 2026-06-16 | VAD 到 L5 接口定稿 | psych_probe 直接用 VAD 3 维（取代旧 2 维情绪效价），零映射。VAD 从 emotion_events 取最近真实记录，不参与线性投影训练。行为预测核心维度从 11 维→12 维。详见 `v2-behavior-prediction.md` §2.1 |
| 2026-06-16 | LLM 情感记忆服务接口定稿 | Phase 1 自动注入兜底（分级递进检索），Phase 2+ 增加工具调用（query_emotion_memory / query_emotion_timeline / query_emotion_pattern）。详见 §九 |
|| 2026-06-16 | 情感模式与行为模式池协同定稿 | 多对多关联表（pattern_relations），情感模式作为 GMM 上下文先验。统一发现管道（L3 睡眠周期一次 LLM 调用输出两个视角）。关联强度统计积累，生命周期跟随低置信度方。详见 §十 |
|| 2026-06-16 | 情感对象字段定稿 — emotion_target | emotion_events 新增 emotion_target TEXT 字段，区分 mood（null）/ 对自己（self）/ 对他人（person:xxx）/ 对实体（entity:xxx）/ 对事件（event:xxx）/ 对场所（place:xxx）。与 primary_fact_id（触发事实）正交。LLM 提取时同步输出，零额外成本。详见 §三 |
|| 2026-06-16 | 情感轨迹预测定稿 — 预测日志 + 情感反馈闭环 | 不需要独立的情感轨迹预测模块。改为统一的 behavior_predictions 表（替代 prediction_counterfactuals），记录预测时的情绪 + 行为后的情感变化，形成反馈闭环。详见 `v2-behavior-prediction.md` §十 |
|| 2026-06-16 | 隐私问题定稿 — user_blocks 屏蔽表 | 新增 user_blocks 屏蔽表（不提，不删）。用户说"别提这个" → 记录屏蔽，检索时跳过，数据永远保留。真实删除需二次确认。详见主架构文档 §4-遗忘机制 |
|| 2026-06-17 | emotion_triggers 聚合策略修正 | 从"全历史均值 + 要求 VAD 一致性"改为"最近 5 次加权 + 不要求一致性"。低 confidence 本身就是信号（情感在演化）。情感演化由 emotion_events 时序 + query_emotion_timeline 覆盖。详见 §六 |
