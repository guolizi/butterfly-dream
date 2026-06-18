# 🦋 butterfly-v2db — v2 数据库 CLI 查看工具

`butterfly-v2db` 是 Butterfly Dream v2 数据库的交互式查看工具，支持按层（L0~L5）浏览和检索数据。

## 安装

脚本位于 `~/.local/bin/butterfly-v2db`，已加入 `PATH`，直接使用即可。

```bash
butterfly-v2db --help
```

## 全局选项

| 选项 | 说明 |
|:----|:----|
| `--db <path>` | 指定数据库路径（默认: `~/butterfly-dream/eval/dbs/locomo/conv26_v2.db`） |
| `--no-color` | 禁用彩色输出 |

## 命令结构

```
butterfly-v2db <层> <命令> [选项]
```

---

## L0 — 工作记忆层

L0 层存储原始对话轮次、FTS5 全文索引和微事实关键词索引。

### 1. `search` — FTS5 全文搜索

最灵活的搜索方式，支持多种匹配模式。

```bash
# 基本搜索
butterfly-v2db l0 search adoption

# 前缀搜索（匹配 adoption, adopt, adopting...）
butterfly-v2db l0 search "adop*"

# 精确短语搜索
butterfly-v2db l0 search '"support group"'

# 布尔 AND（同时包含）
butterfly-v2db l0 search "adoption AND family"

# 布尔 OR（任一包含）
butterfly-v2db l0 search "piano OR violin"

# 排除（包含 adoption 但不含 agency）
butterfly-v2db l0 search "adoption NOT agency"

# 限制条数
butterfly-v2db l0 search "adoption AND family" -l 3
```

**FTS5 语法速查：**

| 语法 | 示例 | 说明 |
|:----|:----|:----|
| 单词 | `adoption` | 精确匹配 |
| 前缀 | `adop*` | 匹配所有以 adop 开头的词 |
| 短语 | `"support group"` | 精确短语匹配 |
| AND | `adoption AND family` | 同时包含两个词 |
| OR | `piano OR violin` | 包含任一词 |
| NOT | `adoption NOT agency` | 包含前者但不含后者 |
| 组合 | `(adoption OR foster) AND family` | 括号分组 |

### 2. `keyword` — 微事实关键词检索

精确匹配关键词，速度最快（O(1) 哈希查找）。

```bash
butterfly-v2db l0 keyword dancing
butterfly-v2db l0 keyword adoption -l 5
```

### 3. `multi` — 多关键词交集

找**同时**聊到多个话题的对话轮次（命中 ≥2 个关键词才显示）。

```bash
butterfly-v2db l0 multi adoption lgbtq family
butterfly-v2db l0 multi adoption lgbtq family -l 3
```

### 4. `session` — 查看会话内容

查看某次会话的完整对话记录。

```bash
butterfly-v2db l0 session session_1
butterfly-v2db l0 session session_5
```

### 5. `turns` — 最近对话

查看最近 N 轮对话。

```bash
butterfly-v2db l0 turns        # 最近 10 轮
butterfly-v2db l0 turns 5      # 最近 5 轮
```

### 6. `stats` — 数据库统计

显示 L0 层的数据概况。

```bash
butterfly-v2db l0 stats
```

输出内容：
- 对话轮次总数（按人物分布）
- 会话数
- FTS5 索引行数
- 微事实总数和唯一关键词数
- 晋升队列条数
- 最热关键词 Top 10（带柱状图）
- 会话列表（带轮次柱状图）

---

## 颜色标记说明

| 标记 | 含义 |
|:----|:----|
| `Caroline` | 说话人/事实所属人（绿色高亮） |
| `【关键词】` | FTS5 命中高亮 |
| `#关键词` | 微事实标签 |
| `session_N:M` | 第 N 次会话的第 M 轮 |
| `<event>` | L1 事实类型（event / knowledge / behavior） |
| `[event]` | L1 事实分类（event / state / opinion / goal ...） |
| `HOT` | 热区标记（HOT / WARM / COLD / ICE） |

---

## L1 — 长期记忆层

L1 层存储提取后的事实（facts），按 person 区分归属，支持类型（event/knowledge/behavior）、分类、热区等维度筛选。

### 1. `list` — 列出 facts

```bash
# 列出所有 facts
butterfly-v2db l1 list

# 按人物筛选
butterfly-v2db l1 list -p Caroline
butterfly-v2db l1 list -p Melanie

# 按类型筛选
butterfly-v2db l1 list -t event
butterfly-v2db l1 list -t knowledge
butterfly-v2db l1 list -t behavior

# 按分类筛选
butterfly-v2db l1 list -c opinion
butterfly-v2db l1 list -c goal

# 按热区筛选
butterfly-v2db l1 list --heat-zone hot
butterfly-v2db l1 list --heat-zone cold

# 仅显示抽象事实
butterfly-v2db l1 list --abstract

# 组合筛选
butterfly-v2db l1 list -p Caroline -t event -c event

# 最低重要性
butterfly-v2db l1 list -i 0.6

# 限制条数
butterfly-v2db l1 list -l 5
```

### 2. `show` — 查看 fact 详情

显示事实的所有字段，以及关联的实体、来源（provenance）、关系。

```bash
butterfly-v2db l1 show 112
butterfly-v2db l1 show 112 -j   # JSON 格式
```

输出内容：
- 人物、内容、类型、分类
- 重要性、信任分、热区、冷却因子
- 抽象层、情感标签
- 持久标记、检索/有帮助计数
- 关联实体
- 来源记录（session → turn）
- 事实关系

### 3. `search` — FTS5 全文搜索

```bash
# 基本搜索
butterfly-v2db l1 search support

# 按人物筛选
butterfly-v2db l1 search support -p Caroline

# 按类型筛选
butterfly-v2db l1 search painting -t knowledge

# 前缀搜索
butterfly-v2db l1 search "adop*"

# 精确短语
butterfly-v2db l1 search '"support group"'

# 布尔 AND
butterfly-v2db l1 search "painting AND relax"

# 布尔 OR
butterfly-v2db l1 search "counseling OR adoption"
```

### 4. `stats` — 事实统计

显示 L1 层的数据概况。

```bash
butterfly-v2db l1 stats
```

输出内容：
- 事实总数（按人物分布）
- 类型分布（event / knowledge / behavior）
- 分类分布（带柱状图）
- 热区分布（hot / warm / cold / ice）
- 重要性分布
- 行为模式统计
- 情感事件统计
- 合并日志统计

### 5. `entities` — 列出实体

```bash
# 所有实体
butterfly-v2db l1 entities

# 按人物筛选
butterfly-v2db l1 entities -p Caroline

# JSON 格式
butterfly-v2db l1 entities -j
```

### 6. `entity` — 实体详情

```bash
butterfly-v2db l1 entity Caroline
butterfly-v2db l1 entity Melanie
```

输出内容：实体基本信息 + 关联 facts 列表 + 关联实体。

### 7. `relations` — 事实关系

```bash
# 所有关系
butterfly-v2db l1 relations

# 按人物筛选
butterfly-v2db l1 relations -p Caroline

# 按关系类型筛选
butterfly-v2db l1 relations -t supports
```

### 8. `provenance` — 来源记录

```bash
# 所有来源
butterfly-v2db l1 provenance

# 按人物筛选
butterfly-v2db l1 provenance -p Caroline

# 按来源类型
butterfly-v2db l1 provenance --source-type llm_extract
```

### 9. `patterns` — 行为模式

```bash
# 所有模式
butterfly-v2db l1 patterns

# 按人物筛选
butterfly-v2db l1 patterns -p Melanie

# 最低置信度
butterfly-v2db l1 patterns --min-confidence 0.6
```

### 10. `emotions` — 情感事件

```bash
butterfly-v2db l1 emotions
butterfly-v2db l1 emotions -p Caroline
```

### 11. `triggers` — 情感触发器

```bash
butterfly-v2db l1 triggers
butterfly-v2db l1 triggers -p Melanie
```

### 12. `merge-log` — 合并日志

```bash
butterfly-v2db l1 merge-log
butterfly-v2db l1 merge-log -p Caroline
```

---

## 示例速查

```bash
# L0: 搜索对话轮次
butterfly-v2db l0 search adoption

# L0: 找同时聊到 LGBTQ 和收养的轮次
butterfly-v2db l0 multi adoption lgbtq

# L0: 看第一次会话完整内容
butterfly-v2db l0 session session_1

# L0: 看数据概况
butterfly-v2db l0 stats

# L1: 列出 Caroline 的所有 event 类型事实
butterfly-v2db l1 list -p Caroline -t event

# L1: 搜索关于 painting 的事实
butterfly-v2db l1 search painting

# L1: 查看 fact 详情
butterfly-v2db l1 show 112

# L1: 数据统计
butterfly-v2db l1 stats

# L1: 查看实体详情
butterfly-v2db l1 entity Caroline

# 用不同数据库
butterfly-v2db --db /path/to/other.db l1 stats
```

---

## 扩展指南

后续添加新层（L1~L5、emotion）时，在 `~/.local/bin/butterfly-v2db` 中：

1. 写 `add_lX_commands(sub)` — 注册该层的子命令
2. 写 `run_lX(args)` — 路由到具体实现函数
3. 在 `main()` 中加一行：
   ```python
   lX = sub.add_parser("lX", help="...")
   lX_sub = lX.add_subparsers(dest="lX_cmd", required=True)
   add_lX_commands(lX_sub)
   ```
4. 在路由部分加：
   ```python
   elif args.layer == "lX":
       run_lX(args)
   ```

---

## 已知问题

详见 `docs/v2-implementation-issues.md`，当前 L0 层已记录的问题：

- FTS5 外部内容表查询语法（不能用别名）
- jieba 英文缩写误标为关键词
- FTS5 snippet 列索引从 0 开始
- 英文停用词表不完整
- jieba 对英文 POS 标注的局限性
