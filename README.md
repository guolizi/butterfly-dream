# 🦋 Butterfly Dream — 全维记忆插件 for Hermes Agent

> *"昔者庄周梦为蝴蝶，栩栩然蝴蝶也。"*
> 记忆如蝶，翩跹于时间、意义与关联的多维空间。

**Butterfly Dream** 是一个为 [Hermes Agent](https://hermes-agent.nousresearch.com) 设计的全维记忆插件，基于 Holographic 的 HRR 向量引擎和 SQLite 存储层，**完整实现了从 LLM 自动提取到三维检索的全链路**：重要性评分、分类标签、中文分词、实体关系图、事实合并、多媒体存储、反射/熔断/过滤等。从纯文本事实到图片/音频/视频，从单轮搜索到多跳推理，让 Agent 的记忆既有宽度又有深度——同时确保可靠性。

## ✨ 特性

### 🧠 核心记忆

- **🗣️ LLM 自动提取** — LLM 在对话压缩和会话结束时**自动**分析对话、提取值得记住的事实，带重要性/分类/标签，无需手动调用
- **🧠 三维检索** — 同时衡量语义相关性、时间衰减和事实重要性，告别单纯向量搜索
- **🧬 事实合并** — 同实体/同主题事实自动归并，冲突检测标记矛盾，杜绝冗余堆积
- **🕰️ 时间感知** — 基于指数衰减的时效性评分，近期事实自然权重更高
- **⭐ 重要性评分** — LLM 自动评估每条事实的重要性（1-10），关键信息永不沉没
- **🏷️ 持久标记** — LLM 自动判断事实是否跨会话持久（身份/架构/配置=true，临时状态=false），配合 `persistent_only` 过滤实现精准检索
- **📊 信任度反馈** — 用户可标记有用/无用，好事实上升、坏事实下沉
- **🔗 实体关系图** — 自动提取实体并建立关系，支持多跳推理
- **📅 时间线查询** — 按创建时间正序查看某实体的全部事实变更历史，追踪偏好和决策的演化过程

### 🖼️ 多媒体

- **🖼️ 多媒体记忆** — 图片/音频/视频文件的存储与检索，支持描述/字幕/语音转写的 FTS5 全文搜索
- **📦 自动压缩** — 图片(Pillow→JPEG)、视频(ffmpeg→H.264)、音频(ffmpeg→MP3)自动压缩，默认开启，可配置质量/码率/分辨率
- **🛡️ 大文件保护** — 超过 `max_size_mb`（默认 100MB）的文件跳过压缩，避免长时间卡顿

### 🛡️ 可靠性

- **🔇 琐事消息过滤** — 自动跳过 "ok"、"好的"、"👍" 等无信息量对话，LLM 只处理有内容的对话
- **⚡ 熔断保护** — 连续 3 次 LLM 提取失败后自动冷却 120 秒，避免 API 异常导致级联失败
- **🧵 线程安全** — 异步提取使用锁保护共享状态，多线程环境零竞态
- **🔗 跨源去重** — LLM 自动提取 + on_memory_write 镜像 + 反思生成之间自动去重 (Jaccard ≥ 0.7 跳过写入)
- **🔤 中文分词** — FTS5 全文搜索接入 jieba 分词引擎，混合中文/英文内容正确索引，告别 CJK 不可搜索

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
```

> 💡 **提示**：启用 `llm_extract: true` 后，LLM 会在每次对话压缩时自动提取事实。确保 `extraction_model` 填写的是有 JSON mode 支持的模型。

### 依赖

```bash
pip install numpy          # HRR 向量编码（可选，但强烈推荐）
pip install Pillow         # 图片压缩缩略图
pip install ffmpeg-python  # 音视频压缩
pip install jieba          # 中文分词（FTS5 全文搜索）
```

## 🔗 FTS5 中文检索说明

Butterfly Dream 使用 SQLite FTS5 作为全文搜索引擎。FTS5 默认的 `unicode61` tokenizer 以空格分词，因此中文文本（无空格）原本会被当做单个 token 索引，完全不可搜索。

解决方案是在 **索引侧** 和 **查询侧** 都接入 jieba 分词：

1. **索引侧**：FTS5 sync trigger 调用 `jieba_segment()` SQLite 函数，在内容进入 FTS5 索引前，用 jieba 分词并在词间插入空格。
2. **查询侧**：`_sanitize_fts_query()` 对用户查询同样进行 jieba 分词，确保查询 token 与索引 token 匹配。

**效果**：
- `"用户喜欢喝咖啡和编程"` → jieba 分为 `["用户", "喜欢", "喝咖啡", "和", "编程"]` → 搜索 `"喜欢 编程"` 命中 ✓
- 英中混合 `"love 猫咪"` → 英文和中文 token 均可独立搜索 ✓
- 单一 jieba 复合词（如 `"喝咖啡"`）按完整词索引，子串 `"咖啡"` 无法命中——但 Agent 搜索时知道完整概念，可搜 `"喝咖啡"` 或 `"coffee"`

> ⚠️ jieba 首次加载约需 0.6 秒（载入词典），后续调用几乎无开销。首次初始化 `MemoryStore` 时会有一次载入延迟，不影响后续搜索性能。

## 🧠 三维检索使用示例

```python
from butterfly_dream import ButterflyDreamMemoryProvider

# ... 正常使用 Hermes agent ...

# 1. 存储事实
await agent.tools.fact_store(action="add", content="用户喜欢用 VS Code 写 Python", importance=7)

# 2. 三维检索（默认 balanced 权重）
results = await agent.tools.fact_store(
    action="search",
    query="用户偏好编辑器",
    scenario="chat",       # chat / technical / longterm / qa / balanced
    limit=5,
)

# 3. 实体中心检索
await agent.tools.fact_store(action="probe", entity="VS Code")

# 4. 排序推理
await agent.tools.fact_store(action="reason", entities=["VS Code", "Python"])

# 5. 矛盾检测
await agent.tools.fact_store(action="contradict")

# 6. 时间线查询
await agent.tools.fact_store(action="timeline", entity="VS Code", min_importance=3)

# 7. 实体摘要卡
await agent.tools.fact_store(action="summarize", entity="VS Code")

# 8. 反馈训练
await agent.tools.fact_feedback(action="helpful", fact_id=1)
```

## 📊 与上游 Holographic 的功能对比

| 特性 | Holographic main | **Butterfly Dream** |
|:-----|:---------------:|:------------------:|
| HRR 向量编码 | ✅ | ✅ |
| SQLite 持久存储 | ✅ | ✅ |
| 实体自动提取 (regex) | ✅ | ✅ (增强 CJK 括号) |
| **LLM 自动提取** | ❌ | ✅ |
| **重要性评分 (1-10)** | ❌ | ✅ |
| **分类标签 (category/tags)** | ❌ | ✅ |
| **三维检索 (相关×时效×重要)** | ❌ | ✅ |
| **OR 语义 + 前缀匹配 FTS5** | ❌ | ✅ |
| **jieba 中文分词** | ❌ | ✅ |
| 实体关系图 | ❌ | ✅ |
| 多跳推理 | ❌ | ✅ |
| 事实合并 (精确+语义) | ❌ | ✅ |
| 指数衰减时效 | ❌ | ✅ |
| 信任度反馈训练 | ❌ | ✅ |
| 琐事消息过滤 | ❌ | ✅ |
| 熔断保护 | ❌ | ✅ |
| 线程安全 | ❌ | ✅ |
| 异步提取 (+ 压缩触发) | ❌ | ✅ |
| 反思 (元认知) | ❌ | ✅ |
| 多媒体 (图片/音频/视频) | ❌ | ✅ |
| 自动压缩 (图片/视频/音频) | ❌ | ✅ |
| 持久标记 | ❌ | ✅ |
| 跨源去重 | ❌ | ✅ |
| 时间线查询 | ❌ | ✅ |
| 实体摘要卡 (零 LLM 成本) | ❌ | ✅ |
| 完整评测体系 (检索+提取) | ❌ | ✅ |
| 代码行数 (估算) | ~400 | ~1800+ |

## 🧪 评测集

评测包含 **检索** 和 **提取** 两个维度，覆盖中英文、多长度多领域场景。详见 [`eval/README.md`](eval/README.md)。

| 类型 | 场景数 | 查询数 | 说明 |
|:-----|:------:|:------:|:-----|
| 🔍 检索评测 | 46 | 114 | 预置事实 → 自然语言查询，测试 FTS5 + HRR + Jaccard + 重要性三维评分 |
| 🗣️ 提取评测 | 20 | 95 | 真实 LLM 端到端：对话 → 提取 → 存储 → 查询，含中英文场景 |
| 🔤 原版 Holographic 对比 | — | 16 | 同事实集，英文 0/8→**5/8**，中文 0/8→**8/8**（根因见下方） |

### 核心结论

- **检索 R@1**：**0.691**（46 场景，114 查询，黑盒端到端）
- **提取通过率**：**~62%**（20 场景，95 查询，LLM 非确定性 ±5-15%）
- **与原版 Holographic 对比**：FTS5 AND 语义 + 停用词全灭 → Butterfly Dream 的 **OR 展开 + 前缀匹配 + jieba 分词** 解决。完整对比见 `eval/README.md` 和 `eval/en_compare.py`

```bash
# 快速运行
python3 eval/run_eval.py              # 检索评测
python3 eval/test_extraction.py        # 提取评测
python3 eval/en_compare.py             # 与原版 Holographic 对比
```

## 📄 许可证

MIT
