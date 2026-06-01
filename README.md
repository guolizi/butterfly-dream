# 🦋 Butterfly Dream — 三维记忆插件 for Hermes Agent

> *"昔者庄周梦为蝴蝶，栩栩然蝴蝶也。"*
> 记忆如蝶，翩跹于时间、意义与关联的三维空间。

**Butterfly Dream** 是一个为 [Hermes Agent](https://hermes-agent.nousresearch.com) 设计的高级记忆插件，基于 Holographic 记忆架构重构，引入**三维检索**（Relevance × Recency × Importance）让 Agent 的记忆更贴近人类的认知方式。

## ✨ 特性

- **🧠 三维检索** — 同时衡量语义相关性、时间衰减和事实重要性，告别单纯向量搜索
- **🧬 事实合并** — 同实体/同主题事实自动归并，冲突检测标记矛盾，杜绝冗余堆积
- **🕰️ 时间感知** — 基于指数衰减的时效性评分，近期事实自然权重更高
- **⭐ 重要性评分** — LLM 自动评估每条事实的重要性（1-10），关键信息永不沉没
- **📊 信任度反馈** — 用户可标记有用/无用，好事实上升、坏事实下沉
- **🔗 实体关系图** — 自动提取实体并建立关系，支持多跳推理
- **🖼️ 多媒体记忆** — 图片/音频/视频文件的存储与检索，支持描述/字幕/语音转写的 FTS5 全文搜索
- **📦 自动压缩** — 图片(Pillow→JPEG)、视频(ffmpeg→H.264)、音频(ffmpeg→MP3)自动压缩，默认开启，可配置质量/码率/分辨率
- **🛡️ 大文件保护** — 超过 `max_size_mb`（默认 100MB）的文件跳过压缩，避免长时间卡顿
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
    compression:
      enabled: true          # 媒体压缩总开关
      max_size_mb: 100       # 超过此大小(MB)跳过压缩
      timeout: 600           # ffmpeg 超时秒数
      image:
        quality: 85           # JPEG quality
        max_dim: 1920         # 最大尺寸（缩放）
      video:
        bitrate: "1M"         # 视频码率
        max_dim: 1280
      audio:
        bitrate: "128k"       # 音频比特率
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

## 🧬 事实合并（Fact Merging）

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

| 功能 | Holographic | Butterfly Dream |
|:----|:-----------|:---------------|
| 存储引擎 | SQLite + HRR | SQLite + HRR |
| 检索维度 | 相关性 × 信任度 | **三维：相关性 × 时效性 × 重要性 × 信任度** |
| 重要性评分 | ❌ | ✅ LLM 自动评分 |
| 时效衰减 | ⚠️ 可选（默认关闭） | ✅ 默认启用，可配置半衰期 |
| 实体图谱 | ✅ 基础实体解析 | ✅ 增强实体关系 |
| 事实合并/冲突解决 | ❌ | ✅ 同实体事实自动归并 |
| 多媒体附件 | ❌ | ✅ 图片/音频/视频 CAS 存储 + FTS5 搜索 |
| 媒体自动压缩 | ❌ | ✅ Pillow + ffmpeg，可配置 |
| 多场景权重 | ❌ | ✅ 按场景预设权重模板 |

## 📝 License

MIT License — 详见 [LICENSE](LICENSE) 文件。

## 🌟 致谢

- [NousResearch](https://nousresearch.com/) — Hermes Agent 框架
- [Generative Agents](https://arxiv.org/abs/2304.03442) (Park et al.) — 三维检索思想来源
- [Holographic Reduced Representations](https://arxiv.org/abs/2210.10853) (Plate, 1995)
- 庄周 — 蝴蝶梦的灵感
