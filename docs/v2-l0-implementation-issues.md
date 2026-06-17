# L0 层实现问题记录

> 基于 conv-26 (LoCoMo) 数据建库过程中发现的问题，供后续代码实现时参考解决。

## 1. FTS5 外部内容表查询

### 问题
使用 `content=conversation_turns` 外部内容表时，查询语法与普通 FTS5 表不同。

### 错误写法
```sql
-- ❌ 不能用别名
SELECT snippet(c.fts, 0, '【', '】')
FROM conversation_turns_fts c
WHERE c.fts MATCH 'adoption'
```

### 正确写法
```sql
-- ✅ 直接用表名 MATCH
SELECT snippet(conversation_turns_fts, 0, '【', '】')
FROM conversation_turns_fts
WHERE conversation_turns_fts MATCH 'adoption'
```

### 解决方案
- 查询时直接使用虚拟表名，不要用别名
- JOIN 外部表时用 `conversation_turns_fts.rowid = t.turn_id`

---

## 2. jieba 英文缩写 POS 误标

### 问题
jieba 将英文缩写（contractions）错误标注为名词/专名：

| 词 | jieba 标注 | 实际 |
|:--|:----------|:----|
| `what` | `eng` (保留) | 疑问词 |
| `re` | `eng` (保留) | "are" 缩写 |
| `ve` | `eng` (保留) | "have" 缩写 |
| `ll` | `eng` (保留) | "will" 缩写 |
| `m` | `x` (过滤) | "am" 缩写 |
| `s` | `eng` (保留) | "is/has" 缩写 |
| `t` | `eng` (保留) | "not" 缩写 |
| `don` | `eng` (保留) | "don't" 缩写 |
| `didn` | `eng` (保留) | "didn't" 缩写 |

### 影响
仅 conv-26 就产生了 190 条垃圾关键词。

### 解决方案
1. 在停用词表中加入所有常见英文缩写形式
2. 增加长度过滤：纯英文且长度 ≤ 3 的词直接过滤（除非在 EXTRA_KEEP 中）
3. 考虑在分词前先正则展开缩写（`'re` → ` are`, `'ve` → ` have` 等）

---

## 3. FTS5 snippet 列索引从 0 开始

### 问题
`snippet(table, column_index, ...)` 的 `column_index` 参数从 0 开始计数，不是 1。

### 错误写法
```sql
-- ❌ column_index=1 会返回 person 列的内容
snippet(conversation_turns_fts, 1, '<b>', '</b>', '...', 40)
-- 结果只显示 "Caroline" 而不是对话内容
```

### 正确写法
```sql
-- ✅ column_index=0 对应 content 列
snippet(conversation_turns_fts, 0, '<b>', '</b>', '...', 40)
```

### 解决方案
- FTS5 列索引从 0 开始，与 SQLite 常规列索引（从 1 开始）不同
- 建表时确认列顺序：`fts5(content, person)` → content=0, person=1

---

## 4. jieba_segment 自定义函数

### 问题
FTS5 触发器中使用了 `jieba_segment()` 函数，但纯 SQLite 中不存在此函数。

### 当前做法
```python
# 注册一个恒等函数作为占位
conn.create_function("jieba_segment", 1, lambda x: x)
```

### 解决方案（后续实现）
1. 在 Python 层注册真正的 jieba 分词函数
2. 或者不在触发器层面分词，改为批量插入后统一更新 FTS5 索引
3. 或者使用 FTS5 的 tokenizer 参数（如 `tokenize=porter unicode61` 对英文已足够）

---

## 5. 英文停用词表不完整

### 问题
初始停用词表只覆盖了基础英文虚词，遗漏了大量常见词。

### 遗漏类型
- 英文缩写（contractions）：`don`, `didn`, `isn`, `aren`, `wasn`, `weren`, `haven`, `hasn`, `hadn`, `won`, `wouldn`, `couldn`, `shouldn`, `mightn`, `mustn`, `needn`, `daren`, `ain`
- 常见动词：`get`, `go`, `say`, `make`, `know`, `think`, `take`, `see`, `come`, `want`, `look`, `use`, `find`, `give`, `tell`, `like`
- 时间词：`day`, `days`, `time`, `times`, `year`, `years`, `month`, `months`, `week`, `weeks`, `now`, `today`, `tomorrow`, `yesterday`
- 程度词：`very`, `quite`, `really`, `actually`, `definitely`, `probably`, `maybe`, `perhaps`
- 中文虚词：`的`, `了`, `在`, `是`, `我`, `有`, `和`, `就`, `不`, `人`, `都`, `一`, `一个` 等

### 解决方案
- 维护一个全面的停用词表（中英文）
- 考虑使用 nltk/spaCy 的停用词表作为基础
- 对英文词，长度 ≤ 2 且不在 EXTRA_KEEP 中的直接过滤

---

## 6. EXTRA_KEEP 白名单维护

### 问题
纯 POS 标签过滤会漏掉很多重要的活动名、情感词、专名。需要手动维护白名单。

### 需要保留但 POS 标签不匹配的词类型
- 动名词作活动名：`dancing`, `painting`, `running`, `swimming`, `hiking`, `camping`
- 情感形容词：`happy`, `sad`, `excited`, `grateful`, `proud`, `inspired`
- 复合概念：`lgbtq`, `transgender`, `mental health`, `support group`
- 月份/星期：`january`, `february`, `monday`, `tuesday`
- 地点：`paris`, `rome`, `london`

### 解决方案
- 维护 EXTRA_KEEP 集合，覆盖活动、情感、专名
- 后续可考虑用 LLM 从对话中提取关键词来动态扩充
- 或者用 TF-IDF 自动识别重要短语

---

## 7. 关键词长度过滤

### 问题
2 字符以下的词基本没有检索价值（`a`, `i`, `is`, `it`, `to`, `in`, `on`, `at`, `by`, `of` 等）。

### 当前规则
```python
if len(word) < 2:
    return False
```

### 建议
- 英文词：长度 ≥ 3（除非在 EXTRA_KEEP 中）
- 中文词：长度 ≥ 2（单个汉字无意义）
- 数字：全部过滤

---

## 8. 纯数字/标点过滤

### 问题
jieba 可能将纯数字或纯标点标注为名词。

### 当前规则
```python
if re.match(r'^[\d\s\.\,\!\?\;\:\'\"\-\(\)\[\]\{\}]+$', word):
    return False
```

### 建议
- 使用正则确保关键词至少包含一个字母或中文字符
- 对纯数字串直接过滤

---

## 9. 数据集限制

### 问题
LoCoMo conv-26 数据集有 35 个 session 的日期时间字段，但只有前 19 个 session 有实际对话内容（session_20 ~ session_35 只有 `_date_time` 字段，没有对话数据）。

### 影响
- 实际导入 19 个 session，419 轮对话
- 不是数据库或导入逻辑的问题

### 解决方案
- 导入时只处理有 `session_N`（不含 `_date_time` 后缀）的键
- 按 session 编号排序确保顺序正确

---

## 10. jieba 对英文文本的 POS 标注局限性

### 问题
jieba 主要针对中文分词设计，对英文文本的 POS 标注不够准确。

### 示例
```
"I'm starting a dance studio"
→ I(x) '(x) m(x) (x) starting(eng) (x) a(x) (x) dance(eng) (x) studio(eng)
```
- 所有英文词都被标为 `eng` 或 `x`
- 无法区分名词/动词/形容词

### 影响
- 英文对话中大量有意义的词（如 `dance`, `studio`, `counseling`）被当作 `eng` 处理
- 依赖 EXTRA_KEEP 白名单来补偿

### 解决方案
1. **短期**：维护 EXTRA_KEEP 白名单
2. **中期**：对英文文本使用 NLTK/spaCy 的 POS tagger
3. **长期**：考虑混合分词策略——检测语言后选择不同的分词器
4. 或者：对所有 `eng` 标签的词不做 POS 过滤，仅做停用词和长度过滤

---

## 总结：代码实现时的 TODO

| 优先级 | 问题 | 解决方案 |
|:-----:|:----|:--------|
| 🔴 P0 | FTS5 外部表查询语法 | 查询时直接用表名 MATCH，不用别名 |
| 🔴 P0 | jieba 英文缩写误标 | 停用词表 + 长度过滤 |
| 🔴 P0 | FTS5 snippet 列索引 | column_index 从 0 开始 |
| 🟡 P1 | jieba_segment 函数 | Python 层注册或改用 tokenizer |
| 🟡 P1 | 英文停用词表不完整 | 补充缩写/动词/时间词/程度词 |
| 🟡 P1 | EXTRA_KEEP 白名单 | 维护活动/情感/专名集合 |
| 🟢 P2 | 关键词长度过滤 | 英文 ≥ 3，中文 ≥ 2 |
| 🟢 P2 | 纯数字/标点过滤 | 正则确保含字母或中文 |
| 🟢 P2 | jieba 英文 POS 局限 | 混合分词策略或放宽 eng 过滤 |
| ⚪ P3 | 数据集限制 | 按 session 编号排序导入 |
