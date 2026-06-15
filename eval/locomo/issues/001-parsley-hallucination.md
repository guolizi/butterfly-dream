# Issue 001: "parsley" 系统性幻觉

## 概要

提取 LLM 将对话中出现的食物名称 "parsley"（欧芹）系统性地幻觉化为一个
泛用 placeholder 词，替代了大量本应是 "family"、"partner"、"children"、
"friends" 等概念的语义位置。同时导致宠物名 **"Bailey"** 被完全丢失，
替换为 "Melanie has a cat named parsley"。

## 模型信息

- **提取模型**: `openai/gpt-oss-120b:free` (OpenRouter)
- **回答模型**: 同上
- **评测基准**: LoCoMo (ACL 2024), conv-26
- **DB**: `eval/dbs/locomo/conv-26.db` (458 facts)

## 证据

### 原始对话上下文

Session 13 (2023-08-23), turn 2-4:

```
[2] Caroline: I do- Oscar, my guinea pig. He's been great. How are your pets?
[3] Melanie:  they're good- we got another cat named **Bailey** too.
              Here's a pic of Oliver. Can you show me one of Oscar?
[4] Caroline: He's so cute! ... check out this pic of him [Oscar] eating **parsley**!
              Veggies are his fave!
```

### DB 中的状态

| 预期提取 | DB 中的实际内容 | 问题 |
|----------|----------------|------|
| Melanie has a cat named **Bailey** | `Melanie has a cat named **parsley**` (id=201) | **Bailey 丢失** |
| Oscar likes eating **parsley** | `Oscar likes eating **parsley**` (id=203) | ✅ 这条正确 |
| Melanie and her **family** | `Melanie and her **parsley**` (×30+ facts) | 幻觉 |
| Caroline's **family** / **children** | `Caroline's **parsley**` (×15+ facts) | 幻觉 |
| **family** time | `**parsley** time` (×5 facts) | 幻觉 |
| Caroline loves **parsley** | `Caroline loves **parsley**` (id=207) | 模棱两可 |

### "parsley" 的分布

全文搜索显示 "parsley" 出现在 **50+ 条事实** 中，作为独立实体被注册
（entity_id 关联 `parsley`），跨越 Melanie 和 Caroline 两个主要人物的
事实。但在原始对话中，"parsley" 仅出现一次——作为 Oscar（豚鼠）的食物。

## 影响

1. **Bailey 完全丢失** — 正确答案应是 "Oliver, Luna, Bailey"，但 Bailey
   从未被写入 DB。Q53 理论上永远答不对。
2. **语义污染** — 50+ 条事实中 "parsley" 替代了多个不同概念
   (family/partner/children)，导致检索时这些事实被 "parsley" 关键词关联，
   干扰 FTS5 和 embedding 的语义匹配。
3. **实体图膨胀** — `parsley` 作为独立实体关联大量事实，扭曲了实体间
   的共现关系。

## 建议修复

### 短期（当前 DB）

1. 如果原始对话可用，重提取涉及 Bailey 和 family/partner 概念的事实
2. 或将 DB 中的 "parsley" 作为实体别名合并到正确的目标概念

### 长期（提取管线）

1. 考虑在提取 prompt 中增加对专有名称（宠物名/人名）的保真检查
2. 或使用更强大的提取模型（当前 gpt-oss-120b:free 是免费模型）
3. 提取后增加一轮实体消歧验证，过滤明显不合逻辑的值
