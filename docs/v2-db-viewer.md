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
- 对话轮次总数（user/assistant 分布）
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
| 🧑 | user（当前记忆主体，如 Caroline） |
| 🤖 | assistant（对话对方，如 Melanie） |
| `【关键词】` | FTS5 命中高亮 |
| `#关键词` | 微事实标签 |
| `session_N:M` | 第 N 次会话的第 M 轮 |

---

## 示例速查

```bash
# 搜索 Caroline 聊收养的事
butterfly-v2db l0 search adoption

# 找同时聊到 LGBTQ 和收养的轮次
butterfly-v2db l0 multi adoption lgbtq

# 看第一次会话完整内容
butterfly-v2db l0 session session_1

# 看对话结尾
butterfly-v2db l0 turns 5

# 看数据概况
butterfly-v2db l0 stats

# 搜特定短语
butterfly-v2db l0 search '"support group"'

# 用不同数据库
butterfly-v2db --db /path/to/other.db l0 stats
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
