# 🦋 Butterfly Dream — 全维记忆插件 for Hermes Agent

> *"昔者庄周梦为蝴蝶，栩栩然蝴蝶也。"*
> 记忆如蝶，翩跹于时间、意义与关联的多维空间。

**Butterfly Dream** 是一个为 [Hermes Agent](https://hermes-agent.nousresearch.com) 设计的高级记忆插件，在 Holographic 架构上构建了**LLM 自动提取（带重要性/分类/标签）+ 三维检索（Relevance × Recency × Importance）+ 实体图谱 + 多媒体存储 + 事实合并 + 反思 + 熔断保护**的完整记忆系统。从纯文本事实到图片/音频/视频，从单轮搜索到多跳推理，让 Agent 的记忆既有宽度又有深度——同时确保可靠性。

## ✨ 特性

### 🧠 核心记忆

- **🗣️ LLM 自动提取** — LLM 在对话压缩和会话结束时**自动**分析对话、提取值得记住的事实，带重要性/分类/标签，无需手动调用
- **🧠 三维检索** — 同时衡量语义相关性、时间衰减和事实重要性，告别单纯向量搜索
- **🧬 事实合并** — 同实体/同主题事实自动归并，冲突检测标记矛盾，杜绝冗余堆积
- **🕰️ 时间感知** — 基于指数衰减的时效性评分，近期事实自然权重更高
- **⭐ 重要性评分** — LLM 自动评估每条事实的重要性（1-10），关键信息永不沉没
- **📊 信任度反馈** — 用户可标记有用/无用，好事实上升、坏事实下沉
- **🔗 实体关系图** — 自动提取实体并建立关系，支持多跳推理

### 🖼️ 多媒体

- **🖼️ 多媒体记忆** — 图片/音频/视频文件的存储与检索，支持描述/字幕/语音转写的 FTS5 全文搜索
- **📦 自动压缩** — 图片(Pillow→JPEG)、视频(ffmpeg→H.264)、音频(ffmpeg→MP3)自动压缩，默认开启，可配置质量/码率/分辨率
- **🛡️ 大文件保护** — 超过 `max_size_mb`（默认 100MB）的文件跳过压缩，避免长时间卡顿

### 🛡️ 可靠性

- **🔇 琐事消息过滤** — 自动跳过 "ok"、"好的"、"👍" 等无信息量对话，LLM 只处理有内容的对话
- **⚡ 熔断保护** — 连续 3 次 LLM 提取失败后自动冷却 120 秒，避免 API 异常导致级联失败
- **🧵 线程安全** — 异步提取使用锁保护共享状态，多线程环境零竞态

### 🧠 元认知

- **🧠 反思 Reflection** — 每 5 次提取后自动分析全部存储事实，发现模式、矛盾、演化趋势，生成元事实写回库中

### 🔧 通用

- **🔄 增量更新** — 新事实无缝加入，无需重建索引
- **🧩 HRR 向量编码** — 基于 Holographic Reduced Representations 的高密度语义编码
- **🔌 即插即用** — 标准 Hermes MemoryProvider 接口，一行配置启用

## 📦 安装

### 方式一：直接复制（推荐）

```bash
git clone https://github.com/guolizi/butterfly-dream.git
cp -r butterfly-dream/src/butterfly_dream $HERMES_HOME/plugins/butterfly-dream
```

### 方式二：pip 安装

```bash
pip install butterfly-dream
```

## ⚙️ 配置

在 `config.yaml` 中启用：

```yaml
plugins:
  butterfly-dream:
    enabled: true
    db_path: $HERMES_HOME/memory_store.db
    llm_extract: true
    extraction_model:
      provider: deepseek
      model: deepseek-v4-flash
    retrieval:
      relevance_weight: 0.4
      recency_weight: 0.3
      importance_weight: 0.3
      recency_half_life_days: 30
    min_trust_threshold: 0.3
    default_trust: 0.5
    trivial_filter: true           # 琐事消息过滤开关
    circuit_breaker:               # 熔断保护配置
      max_failures: 3              # 连续失败次数触发熔断
      cooldown_seconds: 120        # 冷却时间（秒）
    reflection: true               # 反思（元认知分析）开关
    compression:
      enabled: true                # 媒体压缩总开关
      max_size_mb: 100             # 超过此大小(MB)跳过压缩
      timeout: 600                 # ffmpeg 超时秒数
      image:
        quality: 85                # JPEG quality
        max_dim: 1920              # 最大尺寸（缩放）
      video:
        bitrate: "1M"              # 视频码率
        max_dim: 1280
      audio:
        bitrate: "128k"            # 音频比特率
        sample_rate: 44100
```

然后设置 memory provider：

```yaml
memory:
  provider: butterfly-dream
```

## 🎯 三维检索说明

Butterfly Dream 在检索记忆时综合计算三个维度的分数：

```
final_score = (α × relevance + β × recency + γ × importance) × trust
```

| 维度 | 含义 | 计算方式 | 默认权重 |
|:----|:----|:--------|:--------|
| **Relevance** | 与当前话题的语义相关度 | FTS5 + Jaccard + HRR 混合 | α = 0.4 |
| **Recency** | 时间的"新鲜度" | `0.5^(age_days / half_life)` | β = 0.3 |
| **Importance** | 事实本身的重要程度 | LLM 自动打分 (1-10) | γ = 0.3 |
| **Trust** | 可信度乘数 | 用户反馈训练 | 0.0 ~ 1.0 |

### 🔄 场景权重建议

| 场景 | α 相关 | β 时效 | γ 重要 | 说明 |
|:----|:------|:-------|:-------|:----|
| 日常对话 | 0.4 | 0.4 | 0.2 | 最近聊的更重要 |
| 技术项目 | 0.5 | 0.2 | 0.3 | 技术决策重关联和重要性 |
| 长期记忆 | 0.3 | 0.1 | 0.6 | 重要的就是重要的，不管多久 |
| 快速问答 | 0.6 | 0.3 | 0.1 | 最相关的最优先 |

## 🗣️ LLM 自动提取

Butterfly Dream 会在以下时机**自动**调用 LLM 从对话历史中提取值得记忆的事实：

| 时机 | 触发条件 | 执行方式 |
|:----|:--------|:--------|
| **上下文压缩前** (`on_pre_compress`) | Hermes Agent 准备压缩对话历史时 | 异步 daemon 线程，不阻塞 |
| **会话结束时** (`on_session_end`) | 用户结束会话 (`/new` 或超时) | 异步 daemon 线程，不阻塞 |
| **记忆写入镜像** (`on_memory_write`) | 用户/Agent 调用 `memory(action='add')` | 同步，自动映射到 fact_store |

### 提取流程

```
对话消息 → 琐事过滤(跳过"ok/好的/👍"等)
    → 截断保护(≤1M字符, 首尾各500K)
    → 熔断检查(连续失败? → 冷却120s)
    → LLM 调用(含重要性评分Prompt)
    → JSON 解析 → 合法性校验(≥10字符, ≤400字符)
    → 分类映射 → 重要性归一化(1-10)
    → 存储到事实库 → 成功计数+1
    → 反思检查(每5次触发 → 元分析)
```

### LLM 提取 Prompt 覆盖范围

- 用户偏好、习惯、个人信息
- 项目决策、架构选择、技术理由
- 工具配置、搭建步骤、环境细节
- 对话中达成的关键约定与共识
- 跨会话值得记住的任何信息

### 输出结构

每条提取的事实包含：
- **content** — 事实陈述（纯文本，≤400 字符）
- **category** — 分类：`user_pref` / `project` / `tool` / `general`
- **tags** — 逗号分隔的搜索标签
- **importance** — 重要性 1-10（9-10 关键信息，5-6 有用上下文，1-2 琐碎细节）

提取后的事实自动经过**事实合并**管道（同实体归并 + 矛盾检测），确保库中无重复堆积。

### 🔇 琐事消息过滤

自动跳过不包含值得记忆信息的消息，减少不必要的 LLM 调用：

| 类型 | 匹配示例 |
|:----|:--------|
| 英文确认 | `ok`, `okay`, `thanks`, `got it`, `yes`, `no`, `yep`, `nope` |
| 中文确认 | `好的`, `好`, `嗯`, `是的`, `对`, `可以`, `行`, `明白`, `收到` |
| 中英文组合 | `好的吧`, `好吧`, `先这样`, `差不多了` |
| 表情 | `👍`, `👌`, `😊`, `😄` |
| 可跟随标点 | 支持末尾 `.`, `!`, `?`, `~`, `。`, `！`, `？`, `～`, `,` 等 |

配置开关：
```yaml
plugins:
  butterfly-dream:
    trivial_filter: true     # 设为 false 禁用
```

### ⚡ 熔断保护

当 LLM 提取连续失败时，自动进入冷却状态，避免无效重试浪费 token：

| 参数 | 默认值 | 说明 |
|:----|:-------|:----|
| `max_failures` | 3 | 连续失败次数上限 |
| `cooldown_seconds` | 120 | 达到上限后冷却的秒数 |

- 成功一次即重置计数器
- 冷却期内所有提取和反思均跳过
- 冷却期结束后自动重置，恢复正常工作
- 线程安全：锁保护共享状态，多线程无竞态

配置示例：
```yaml
plugins:
  butterfly-dream:
    circuit_breaker:
      max_failures: 3
      cooldown_seconds: 120
```

### 🧠 反思 Reflection

受 [Generative Agents](https://arxiv.org/abs/2304.03442) 启发，Butterfly Dream 会在每 `N` 次提取后自动对全部存储事实进行元分析：

| 参数 | 默认值 | 说明 |
|:----|:-------|:----|
| `reflection_interval` | 5 | 每 N 次提取触发一次反思 |

**反思内容**：
1. **模式** — 识别用户重复出现的偏好、行为模式
2. **矛盾** — 发现事实之间潜在冲突
3. **演化** — 跟踪用户偏好或决策随时间的变化
4. **空白** — 指出值得进一步了解的方向

**输出**：生成带 `reflection` 标签的元事实，重要性自动设为 ≥5，存储在常规事实库中，后续检索可被正常召回。

配置开关：
```yaml
plugins:
  butterfly-dream:
    reflection: true     # 设为 false 禁用
```


当新提取的事实与已有事实描述同一实体或主题时，自动归并而非简单堆叠。

### 三级合并策略

```
新事实进入 → Level 1: 内容完全一致？
              ├─ 是 → 合并（取最高 importance，合并 tags）
              └─ 否 → Level 2: 共享实体 + 语义相似度 ≥ 0.4？
                        ├─ 是 → 智能合并内容
                        └─ 否 → Level 3: 新建事实
```

### 内容合并算法（`_combine_fact_content`）

| 场景 | 行为 | 示例 |
|:----|:----|:-----|
| **一者包含另一者** | 保留较长版本 | `"用户用 VS Code"` + `"用户用 VS Code 写 Python"` → `"用户用 VS Code 写 Python"` |
| **矛盾检测** | 用 ⚡ 标记双方 | `"用户喜欢喝咖啡"` + `"用户不喝咖啡"` → `"用户喜欢喝咖啡 ⚡用户不喝咖啡"` |
| **补充新信息** | `；` 拼接 | `"项目用 FastAPI"` + `"项目用 PostgreSQL"` → `"项目用 FastAPI；项目用 PostgreSQL"` |
| **无新信息** | 保留原样 | `"天气不错"` + `"今天天气好"` → `"天气不错"` |

### 矛盾检测机制

基于否定词启发式：当两条事实共享多个关键词但一方含否定词时判定为矛盾。

```python
否定词 = {"not", "don't", "never", "no", "不喜欢", "不要", "不是", ...}
共享关键词 ≥ 3 且 否定状态不同 → 标记矛盾
```

### 合并日志

所有合并操作记录在 `merge_log` 表中，可溯源：

```sql
merge_log (merge_id, kept_fact_id, absorbed_fact_id, merged_content, merge_reason)
```

配置开关：

```yaml
plugins:
  butterfly-dream:
    merge_facts: true     # 设为 false 禁用语义合并（仅保留精确去重）
```

## 🏗️ 项目结构

```
butterfly-dream/
├── README.md                  # 本文件
├── LICENSE                    # MIT 许可证
├── pyproject.toml             # Python 包配置
├── .gitignore
├── src/
│   └── butterfly_dream/
│       ├── __init__.py        # MemoryProvider 入口 + 工具 handler(fact_store, media_attach, etc.)
│       ├── store.py           # SQLite 存储层 + 媒体附件表
│       ├── retrieval.py       # 三维检索器（含媒体 FTS5 并行搜索）
│       ├── holographic.py     # HRR 向量引擎
│       ├── media_utils.py     # 缩略图生成 + 文件 GC
│       ├── media_compressor.py# 图片/视频/音频自动压缩
│       └── plugin.yaml        # 插件元数据
└── tests/
    └── ...
```

## 🔄 与 Holographic 对比

| 功能 | Holographic (upstream) | Butterfly Dream |
|:----|:---------------------|:---------------|
| 存储引擎 | SQLite + HRR | SQLite + HRR |
| 检索维度 | 相关性 × 信任度 | **三维：相关性 × 时效性 × 重要性 × 信任度** |
| **LLM 自动提取** | ✅ 基础版（内容+分类+标签） | ✅ **增强版（+重要性评分 + 反思触发）** |
| 提取 Prompt | 基础版 | 增强版（含重要性评分 1-10 评分表） |
| 重要性和评分 | ❌ | ✅ LLM 自动评分 1-10，检索加权 |
| 分类/标签 | ✅ 4 类（user_pref/project/tool/general） | ✅ 同上 |
| 提取时机 | 压缩前 + 会话结束时 | 压缩前 + 会话结束时 + **记忆写入镜像** |
| **🔇 琐事消息过滤** | ❌ | ✅ 跳过 "ok/好的/👍" 等无信息内容 |
| **⚡ 熔断保护** | ❌ | ✅ 连续 3 次失败后冷却 120 秒 |
| **🧠 反思 Reflection** | ❌ | ✅ 每 5 次提取后元分析 → 元事实 |
| **🧵 线程安全** | ❌ 无锁 | ✅ 锁保护共享状态 |
| CJK 实体提取改进 | ❌ | ✅ 混合脚本规则 + 停用词过滤 |
| FTS5 搜索修复 | ❌ | ✅ OR 组合语义修正 |
| 时效衰减 | ⚠️ 可选（默认关闭） | ✅ **默认启用，可配半衰期** |
| 实体图谱 | ✅ 实体解析 + HRR 代数推理 | ✅ **增强：实体关系图 + SQL 交叉检索** |
| 事实合并/冲突解决 | ❌ | ✅ 同实体自动归并 + 矛盾检测 |
| 三维检索 + 场景权重 | ❌ | ✅ Relevance × Recency × Importance + 5 种预设 |
| 多媒体附件 | ❌ | ✅ 图片/音频/视频 CAS 存储 + FTS5 |
| 媒体自动压缩 | ❌ | ✅ Pillow + ffmpeg，可配置 |

## 📝 License

MIT License — 详见 [LICENSE](LICENSE) 文件。

## 🌟 致谢

- [NousResearch](https://nousresearch.com/) — Hermes Agent 框架
- [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al.) — 三维检索思想来源
- [Holographic Reduced Representations](https://arxiv.org/abs/2210.10853) (Plate, 1995)
- 庄周 — 蝴蝶梦的灵感
