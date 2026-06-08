# 语义鸿沟问题汇总

> 创建日期: 2026-06-08
> 来源: v0.3-baseline 评测分析

## 问题描述

用户问题中的用词与 DB 中存储的事实用语不一致，导致 FTS5 查询无法匹配到相关的 gold 事实。这些问题都属于"查询词 vs DB用语"之间的语义鸿沟。

## 影响范围

v0.3-baseline 评测的 4 个成功对话（conv-26/30/41/42）中，Cat1 的 42 个低分题大部分与此相关。

---

## 具体问题清单

### 1. conv-41_q3 — 上位词vs下位词

- **Q**: "What martial arts has John done?"
- **Gold**: Kickboxing, Taekwondo
- **FTS5查询词**: martial*, soldierly*, warlike*, arts*, humanities*, artistry*, John*, toilet*, bathroom*
- **DB事实**: "John does **kickboxing**", "John is going to do some **taekwondo**"
- **根因**: 查询用上位词"martial arts"，DB存下位词"kickboxing/taekwondo"。WordNet同义词(soldierly/warlike/humanities)无法桥接此鸿沟。gold 事实只匹配 John*（1个词），而误导事实"John asked if it was a martial arts place"匹配 martial*+art*+John*（3个词）。
- **Gold FTS5排名**: #4, #25, #26（在候选池内但被排序挤出）

---

### 2. conv-41_q26 — 泛指vs具体国家名

- **Q**: "What European countries has Maria been to?"
- **Gold**: Spain, England
- **FTS5查询词**: European*, countries*, state*, nation*, country*, land*, Maria*, mare*
- **DB事实**: "Maria took a solo trip to **Spain**", "Maria traveled to **England**"
- **根因**: 查询含"European"和"countries"等泛指词，但DB中具体国家名"Spain/England"不在查询中。European*匹配0条，countries*只匹配到John相关事实。gold 事实仅靠 Maria*（1个词）匹配。
- **Gold FTS5排名**: #65, #66, #106, #107（候选池截断）

---

### 3. conv-42_q43 — 泛指电影vs具体片名

- **Q**: "What movies have both Joanna and Nate seen?"
- **Gold**: "Little Women", "Lord of the Rings"
- **FTS5查询词**: movies*, movie*, film*, picture*, Joanna*, Nate*, see*
- **DB事实**: 8条相关事实，含"Joanna recommended the movie **Little Women**"、"Joanna watched **The Lord of the Rings**"
- **根因**: 查询用泛指词"movies"，但期待的具体片名"Little Women"和"Lord of the Rings"不在查询中。gold 事实只匹配 movie+人名（2-3词），而"Joanna asked if he had seen any good movies lately"匹配 movie*+see*+人名（5词），BM25更高。
- **Gold FTS5排名**: #53, #66（候选池截断）

---

### 4. conv-42_q52 — 别名/不同表述

- **Q**: "When did Nate **get Tilly** for Joanna?"
- **Gold**: 25 May, 2022
- **FTS5查询词**: Nate*, get*, receive*, obtain*, acquire*, Tilly*, Joanna*
- **DB事实**: "Nate **gave** Joanna a **stuffed animal puppy** as a gift on 25 May 2022"
- **根因**: 问题用"get Tilly"但DB事实用"gave stuffed animal puppy"。`get*`不匹配"gave"（不同词根），且事实中无"Tilly"一词（提取时用了"stuffed animal puppy"而非昵称）。gold 事实仅匹配 Nate*+Joanna*（2个词），在几百条同类事实中无法区分。
- **Gold FTS5排名**: 不在top-200（完全匹配不到）

---

### 5. conv-42_q57 — 误导事实的语义干扰

- **Q**: "What animal do both Nate and Joanna like?"
- **Gold**: Turtles
- **FTS5查询词**: animal*, beast*, creature*, Joanna*, Nate*, like*, both*
- **DB事实**: "Nate likes holding his **turtles**"（匹配 like*+Nate* 2词）、"Joanna wishes she could get two **turtles**"（匹配 Joanna*+wish* 2词）
- **误导事实**: "Nate gave Joanna a **stuffed animal** puppy"（匹配 animal*+Joanna*+Nate* 3词）
- **根因**: turtle 事实只匹配 like*/人名（1-2词），而 stuffed animal 事实匹配 animal*+人名（3个词）。3D排序后误导事实排名更高，模型据此推断"喜欢狗"而非"乌龟"。
- **Gold FTS5排名**: #15, #19, #22（在候选池内但被误导事实挤出top-15）

---

## 根因分类

| 类型 | 代表题 | 说明 |
|------|--------|------|
| **上位词→下位词** | q3, q26 | 用户用泛指问，DB存具体实例 |
| **具体实体名缺失** | q43, q52 | 片名/别名不在查询或事实中 |
| **同义表述差异** | q52 | 同义动词不同词根（get/gave） |
| **误导事实干扰** | q57 | 相似语义但错误的竞争事实排序更高 |

## 可能的修复方向

1. **FTS5查询扩展**: 同义词展开从WordNet改为更针对领域的词库（如 martial→kickboxing）
2. **查询改写**: 对泛指词做LLM驱动的查询分解（"movies"→"Little Women, LOTR, Inception..."）
3. **实体别名索引**: 提取时将别名/昵称也加入 tags 字段（如 Tilly 作为 stuffed animal puppy 的别名）
4. **语义匹配层**: 在FTS5之后引入embedding相似度重排序
5. **词根泛化**: get*改为同时匹配 got/gave/gotten/received
