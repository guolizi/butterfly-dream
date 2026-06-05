# LoCoMo Benchmark Issues

记录评测中发现的数据集问题和系统问题，方便后续排查和改进。

---

## Q5 (conv-26): Gold 答案标错

| 字段 | 内容 |
|---|---|
| 问题 | When did Melanie run a charity race? |
| Gold | The sunday before 25 May 2023 |
| 证据 (D2:1) | "I ran a charity race for mental health **last Saturday**" |
| Session 2 日期 | 2023-05-25 (Thursday) |
| 推算 | "last Saturday" = 2023-05-20 (Saturday ✅) |
| 我们的回答 | 2023-05-20 ✅ 正确 |
| 两次得分 | 2, 2 |

**问题**: 原文说 "Saturday"，gold 标成 "sunday"，是 LoCoMo 数据集标注错误。
**影响**: 正确答案被 judge 扣分，不可修复（除非修改 gold）。

---

## Q9 (conv-26): 提取共指消解错误

| 字段 | 内容 |
|---|---|
| 问题 | When did Caroline meet up with her friends, family, and mentors? |
| Gold | The week before 9 June 2023 |
| 证据 (D3:11) | "My friends, family and mentors are my rocks… Here's a pic from when **we** met up last week!" |
| Session 3 日期 | 2023-06-09 |
| 提取结果 | "Caroline met up with **Melanie** last week, around June 2, 2023" ❌ |
| 我们的回答 | I don't have enough information |
| 两次得分 | 1, 1 |

**问题**: 提取 LLM 把 "we met up" 的 "we" 错误指向对话对象 Melanie，实际指的是上一句的 friends/family/mentors。
**影响**: 检索时用 "meet up" 搜到的是错误的实体，正确事实根本没被提取。
**改进方向**: 提取 prompt 增加上下文窗口、多候选实体、speaker-aware 共指消解。

---

## Q24 (conv-26): Gold 答案提及对话中不存在的书名

| 字段 | 内容 |
|---|---|
| 问题 | What books has Melanie read? |
| Gold | "Nothing is Impossible", "Charlotte's Web" |
| 证据 | Charlotte's Web 在 Session 6 明确提及；"Nothing is Impossible" 在全 19 个 session 中**从未出现** |
| 最长匹配 | Session 17: "Been reading that book you recommended a while ago" — 但未提书名 |
| 模型回答 | 只答出 Charlotte's Web |
| 得分 | 3 |

**问题**: Gold 答案中包含对话文本中不存在的信息。"Nothing is Impossible" 在全部 218 条提取事实中无对应，属于 LoCoMo 数据集标注噪音。

**影响**: 此题不可能答对满分，系统不受影响。

**改进方向**: 无需修复（数据集问题）。分析时直接跳过。

---

## Q27 (conv-26): Gold 答案提及对话中不存在的书名（同 Q24）

| 字段 | 内容 |
|---|---|
| 问题 | When did Melanie read the book **"nothing is impossible"**? |
| Gold | **2022** |
| 证据 | "Nothing is Impossible" 在全 19 个 session 中**从未出现**；年份 `2022` 也未在对话中出现 |
| 最长匹配 | Session 7: "This book I read last year reminds me to always pursue my dreams" — 未提书名 |
| 模型回答 | I don't have enough information |
| 得分 | 2 |

**问题**: 同 Q24，Gold 答案中包含对话文本中不存在的信息。书名和年份均无法从对话中获取。

**影响**: 此题不可能答对，系统不受影响。

**改进方向**: 无需修复（数据集问题）。分析时直接跳过。

---

## 待归档模板

```
## Q{题号} (conv-{id}): {简述}

| 字段 | 内容 |
|---|---|
| 问题 | {question} |
| Gold | {gold answer} |
| 证据 | {evidence} |
| 我们的回答 | {hypothesis} |
| 得分 | {scores across runs} |

**问题**: {描述}
**影响**: {影响}
**改进方向**: {方向}
```
