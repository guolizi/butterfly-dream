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

## Q34 (conv-26): 多义性——多个事实匹配同一问题，gold 只认一个

| 字段 | 内容 |
|---|---|
| 问题 | When did Caroline go to a pride parade **during the summer**? |
| Gold | The week before 3 July 2023 |
| 检索结果 (rank 2) | Caroline attended a pride parade a few weeks before July 15, 2023 (around late June 2023) ✅ |
| 检索结果 (rank 4) | Caroline attended a pride parade last Friday, August 11, 2023 ❌ |
| 模型回答 | August 11, 2023（选了日期更具体的那条） |
| 得分 | 2 |

**问题**: 正确事实（rank 2, score 0.4389）分数比错误事实（rank 4, score 0.4275）**更高**，但模型倾向选日期更具体的 "August 11" 而非模糊的 "around late June"。这是推理偏误，不是检索问题。

**影响**: 模型在多候选答案中选了最具体但不是 gold 的那个。

**改进方向**: 可在 answer prompt 加"如有多个匹配事实，优先最早的"规则，但治标不治本。本质是数据集的多义性。

---

## Q35 (conv-26): 提问用词和对话原文的词汇鸿沟

| 字段 | 内容 |
|---|---|
| 问题 | What events has Caroline participated in to help **children**? |
| Gold | Mentoring program, school speech |
| 对话原文 | "encouraged **students**" (Session 3), "mentorship program for LGBTQ **youth**" + "mentor a transgender **teen**" (Session 9) |
| 提取事实 | "gave a school talk...encouraging **students**..." / "joined a mentorship program for LGBTQ **youth**..." |
| 模型回答 | 只提到 adoption 相关活动（"children" 关键词匹配） |
| 得分 | 2 |

**问题**: 提问用词 "children" 与原文关键词 "students/youth/teen" 不匹配，FTS5 精确匹配无法跨过这个词汇鸿沟。提取事实本身准确无误。

**影响**: 即使检索到正确事实（mentorship rank 15, school talk rank 20+），也因为不包含 "children" 关键词被 adoption 类事实（含 "children" 字面）挤到 top-10 之外。

**改进方向**: 需要语义匹配能力（query expansion / embedding retrieval）才能根本解决。目前属于系统能力边界，非 bug。

---

## Q37 (conv-26): 提取中相对日期未绝对化

| 字段 | 内容 |
|---|---|
| 问题 | When did Caroline join a mentorship program? |
| Gold | The weekend before 17 July 2023 |
| 对话原文 | Session 9 (July 17, 2023): "**Last weekend** I joined a mentorship program..." |
| 提取结果 | "Caroline joined a mentorship program for LGBTQ youth in **July 2023**"（模糊） |
| 得分 | 3 |

**问题**: 提取 LLM 的 extraction prompt 已经要求根据 session timestamp 解析相对日期，但 LLM 实际输出中仍写成 "July 2023" 而非 "July 15-16, 2023"。属于 LLM 未严格遵守指令。

**影响**: 检索时精确度损失，但 gold 也接受模糊答案 "The weekend before 17 July 2023"，score=3 够用。

**修复**: 已在 547454b 中加强 extraction prompt 的日期分辨率指令。需重跑 extraction 后才能验证。

---

## Q39 (conv-26): 词汇鸿沟——family vs kids

| 字段 | 内容 |
|---|---|
| 问题 | What activities has Melanie done with her family? |
| Gold | Pottery, painting, camping, museum, swimming, hiking |
| 最终得分 | 3 (v2-entity, owl-alpha) |

**问题**: 查询词 "family" 无法匹配事实中使用的 "kids"（"Melanie took her kids to the pottery workshop"），BM25 精确匹配导致 3 个活动掉出 top-10。

**修复**: synonyms.py + 查询时同义词扩展（family → kid, children）。修复后 painting 进入 top-2，4/6 gold 活动在 top-10（pottery rank 16, museum rank 9 仍需进一步优化）。

---

## Q40 (conv-26): 词汇鸿沟 + 3D scoring 压缩

| 字段 | 内容 |
|---|---|
| 问题 | In what ways is Caroline participating in the LGBTQ community? |
| Gold | Joining activist group, going to pride parades, participating in an art show, mentoring program |
| 最终得分 | 3 (v2-entity, owl-alpha) |

**问题**: FTS5 候选池（30条）中包含所有 4 个 gold 事实（activist group rank 10, mentorship rank 13, art show rank 15-18, pride parade rank 27-29），但 3D scoring 后全部掉出 top-10。根因：1) importance=10 的泛泛 identity 事实（"Caroline is a transgender woman"）被过度推高；2) fts_rank 被 sigmoid 压缩到 0.85-0.99 之间，差距仅 0.03-0.04。

**修复**: synonyms.py 中 participate → join, attend, go 扩展。修复后 mentorship 从 outside top-10 提到 rank 6，pride parade rank 12。

---

## Q41 (conv-26): 推理粒度——"camping at the beach" 是否算 beach trip？

| 字段 | 内容 |
|---|---|
| 问题 | How many times has Melanie gone to the beach in 2023? |
| Gold | 2 |
| 证据 | D6:16 (camping at the beach) + D10:8 (went to the beach recently) |
| 提取事实 | ① "camping at the beach" ✅ ② "went to the beach recently" ✅ ③ "once or twice a year" |
| 得分 | 2 |

**问题**: 三个相关事实都在 top-10（rank 2/3/10），模型也全看到了。但模型认为 "camping at the beach" 是 camping trip 而非 beach trip，只数出 1 个确认的 beach trip。Gold 标准把两个都算上。

**根因**: 提取把事件标记为 "camping at the beach"（强调 camping），而非 "went to the beach"（强调 beach）。模型按字面理解只认了后者。这不是检索问题，是提取的事实措辞 + 推理粒度的 mismatch。

**改进方向**: 提取时将复合事件拆成多维度事实（既有 camping 属性又有 beach 属性）。低优先级，gold 本身也有歧义。

---

## Q44 (conv-26): 提取缺失 "abstract art" 风格标签

| 字段 | 内容 |
|---|---|
| 问题 | What kind of art does Caroline make? |
| Gold | abstract art |
| 原文证据 | [D11:12, D11:8, D9:14] → 描述主题（inclusivity, diversity, LGBTQ+ advocacy），未提风格 |
| 实际提及 "abstract" 的位置 | Session 17 (D17:13): Caroline 说 "I've been trying out **abstract** stuff recently" |
| 提取结果 | "Caroline has been trying out abstract painting recently" (ID=199, imp=5) |
| 检索排名 | FTS5 rank 35/141 — 未进 top-30 候选池 |
| 模型回答 | 描述了 art 的主题（inclusivity, diversity, LGBTQ+ advocacy），没说 abstract |
| 得分 | 2 ❌ |

**根因**: 提取没有把 "abstract" 和 "art" 放到同一个描述性事实里。ID=199 说 "trying out abstract painting"（像在描述一个临时尝试），而检索时 `abstract*` 不匹配 `art*` 或 `painting*`。提取也缺少一个概括性事实 "Caroline makes abstract art"。

**修复**: extraction prompt 加规则——提取艺术类事实时显式标注风格标签。**用户决定跳过，不需要改。**

---

## Q47 (conv-26): 词汇鸿沟——"ally" 未匹配 "support"/"friend" 事实

| 字段 | 内容 |
|---|---|
| 问题 | Would Melanie be considered an ally to the transgender community? |
| Gold | Yes, she is supportive |
| 证据 | [] （category 3 推理题，无直接证据引用） |
| 旧 top-10 | 全为 Caroline 的 trans identity 事实，无 Melanie 支持态度 |
| 旧模型回答 | "Likely no" ❌ |
| 得分 | 1 |

**问题**: 查询词 "ally" 无法匹配事实中的 "support"/"friend"（ID=86 "close friends who support each other"），该事实 FTS5 rank 41/123，未进 top-30 候选池。

**修复**: synonyms.py 加 ally → support, friend, advocate。修复后事实 ID=86 进入 FTS5 rank 11，3D top-10 rank 3。

---

## Q49 (conv-26): 提取遗漏 "cup"——对方 speaker 提及的细节未提取

| 字段 | 内容 |
|---|---|
| 问题 | What types of pottery have Melanie and her kids made? |
| Gold | bowls, cup |
| 证据 | D5:8 + D8:2 + D12:14（D12:14 与 pottery 无关，gold 可能标错） |
| 提取 bowl | ✅ ID=44 "Melanie made a black and white bowl" (D5:8) |
| 提取 pots | ✅ ID=70 "kids made their own pots" (D8:2) |
| 提取 cup | ❌ 缺失 |
| 对话原文 cup 位置 | D8:5 **Caroline** 说 "That **cup** is so cute!" ← 对方 speaker 提及 |
| 得分 | 3 |

**根因**: "cup" 出现在 Caroline 的话里，而非 Melanie。提取 LLM 只提取了 Melanie 的 "made their own pots"，没有提取 Caroline 说的 "cup"。这是提取的 speaker 关注偏差。

**修复**: 低优先级，用户决定跳过。

---

## Q53 (conv-26): 宠物事实 importance 偏低导致被挤出 top-10

| 字段 | 内容 |
|---|---|
| 问题 | What are Melanie's pets' names? |
| Gold | Oliver, Luna, Bailey |
| 模型回答 | Bailey only ("no other pets mentioned") |
| 得分 | 2 |
| 检索排名 | fact[125] (Bailey) rank 4, fact[66] (Luna+Oliver) rank 11 ❌ 差0.0001分被挤出 |

**根因**: 提取 LLM 给宠物事实打了 imp=5（归类为"有用背景"），而"Melanie has children"等家庭事实得到 imp=7（归类为"重要关系"）。3D评分中 importance 贡献差 0.067，导致 fact[66] 排在 #11 刚好掉出 top-10。

**验证**: 手动将 fact[66] imp=5→7 后，#11→#2 跃升，两跳进入 top-2。

**修复**: ✅ **已修复** — 提取 prompt 明确标注 pets/animals 与 family members 同等重要（7-8 分）。commit 0e61a2a。

---

## 系统性限制：FTS5 词袋模型无法处理语义理解

**根因**: 当前检索链路是 `问题文本 → FTS5（词袋匹配）→ 评分（rel×rec×imp）`，本质在**词层面**匹配。查询和事实的措辞由不同 LLM 调用产生，当用词不一致时 FTS5 无能为力。这不是加同义词或调权重能根本解决的问题。

**影响**: 所有涉及「查询措辞 ≠ 提取措辞」或「一词多义」的题目都会失败。目前 conv-41 已发现 3 题（1 分），conv-26/30 也有类似题。

**共同改进方向（待优化）**:
- (A) LLM query rewrite：检索前加 LLM 调用，将查询扩展为多组消歧/同义关键词
- (B) Answer prompt 内嵌 rewrite：不单独加 LLM 调用，在 answer prompt 让 LLM 先用关键词检索
- (C) Embedding 双路检索：事实入库时存 embedding，查询时语义召回 + FTS5 混合
- (D) Extraction 措辞规范：规范化地理/实体等信息的提取格式（如 "has been to [place]"）

**当前检索耗时**: 平均 157ms（中位 146ms），加 LLM rewrite 预估 ~300-500ms。

---

### Q8 (conv-41): child 歧义——childhood vs children

| 字段 | 内容 |
|---|---|
| 问题 | What items does John mention having as a child? |
| Gold | A doll, a film camera |
| 关键事实 | [38] John remembers a childhood doll (imp=5, preference) |
|  | [24] John remembers using a film camera as a kid (imp=5, preference) |
|  | [53] John enjoys taking beach photographs with his childhood film camera (imp=5, preference) |
| FTS5 排名 | doll[38] rank 4/50 (fts_rank=0.495), camera[53] rank 8/50 (0.441), camera[24] rank 14/50 (0.381) |
| 最终排名 | doll[38] rank **18**, camera[53] rank **21**, camera[24] rank **24** |
| 模型回答 | "No information available." |
| 得分 | 1 |

**机制**: 查询词 `child` 同时匹配 `childhood`（正确）和 `children`/`kids`（噪声——"son named Kyle"、"has children" 等）。正确事实 imp=5 被孩子事实 imp=6-7 压过。FTS5 无法在词层面区分「童年」vs「孩子」两个语义。

---

### Q22 (conv-41): 「been to [area]」与「went on a road trip exploring…」的词汇鸿沟

| 字段 | 内容 |
|---|---|
| 问题 | What areas of the U.S. has John been to or is planning to go to? |
| Gold | Pacific northwest, east coast |
| 关键事实 | [108] John went on a road trip in 2022 exploring the Pacific Northwest coast... (imp=6, event) |
|  | [127] John is planning a trip to the East Coast (imp=5, goal) |
| East Coast 检索 | FTS5 ✅ rank ~28/50 (overlap: {john, planning, to, is, the} = 5/8) → 最终 rank **13** |
| Pacific NW 检索 | FTS5 ❌ **不在前50名** (overlap: {john, the} = 仅2/16) |
| 模型回答 | "No information available." |
| 得分 | 1 |

**机制**: 提取 LLM 写成 "went on a road trip exploring the Pacific Northwest coast"，查询 LLM 问 "areas of the U.S. has John been to"。FTS5 只有 {john, the} 两个 token 共通，Pacific NW 事实完全不在候选池。

---

### Q26 (conv-41): 「European countries」与「Spain」/「London」的语义鸿沟

| 字段 | 内容 |
|---|---|
| 问题 | What European countries has Maria been to? |
| Gold | Spain, England |
| 证据 | D13:24 Maria: "took a solo trip...in Spain"；D8:15-17 Maria: "trip to England...It was in London." |
| Spain 事实 | [149] Maria took a solo trip to Spain in 2022 (imp=7) |
| London 事实 | [89] Maria visited London a few years ago... (imp=5) ← 提取丢掉了 "England" 这个词 |
| Spain 检索 | FTS5 rank **0.0012**（最后一名），overlap: {maria, to} = 仅2/7 |
| London 检索 | FTS5 ❌ **不在前50名**，overlap: {maria} = 仅1/21 |
| 模型回答 | "No information available." |
| 得分 | 1 |

**机制双层暴击**:
1. **提取弄丢 "England"**：原文 "trip to England" 被提取改写成 "visited London"，FTS5 找不到 `England`
2. **词汇鸿沟**：查询 `European countries` / `been to` 与事实 `Spain` / `trip` / `London` 完全无共通 token

---

### Q12 (conv-41): 提取焦点偏差——交互相关事实被故事内容淹没

| 字段 | 内容 |
|---|---|
| 问题 | What people has Maria met and helped while volunteering? |
| Gold | David, Jean, Cindy, Laura |
| 证据 | D7:5 "While **volunteering** yesterday, I **met**...**Jean**"；D6:5 "conversation with **David**"；D27:8 "resident **Cindy**"；D21:19 "resident **Laura**" |
| 提取事实 | [67] conversation with David (⚠️ 关键词不全) / [57] Jean experienced divorce (❌ 无 met/volunteering) / [269] Cindy wrote note (❌) / [193] Laura wrote letter (❌) |
| FTS5 | ❌ **全部 6 条不在 top-50** — overlap 最少才 0/14 |
| 模型回答 | "No information available." |
| 得分 | 1 |

**机制**: 提取 LLM 倾向于提取"对方的故事内容"而非"说话者与对方的交互关系"。Jean 的原文同时有 `met` + `volunteering` + `Jean` 三个查询词，但提取只写了 "Jean experienced a divorce"。David/Cindy/Laura 同理。

**修复**: ✅ **已修复** (commit `f07ebe1`) — 在 extraction prompt 新增 **INTERACTION FACTS** 规则，要求遇到 met/helped/connected 场景时独立提取一条交互关系事实（如 "Maria met Jean while volunteering at a homeless shelter"），对方的故事作为独立事实。需重跑 extraction 后验证效果。

---

## Q56 (conv-26): 查询类型检测 + 动态 3D 权重调整

| 字段 | 内容 |
|---|---|
| 问题 | What subject have Caroline and Melanie both painted? |
| Gold | Sunsets |
| 模型回答 | "no specific shared subject is mentioned" |
| 得分 | 2 |
| 旧检索 | ❌ 0/6 条日落事实进入 top-20（3D 评分被 imp=8-10 身份事实淹没） |
| 新检索 | ✅ 3/6 条进入 top-20，1 条进入 top-10（rank 6） |

**根因**: 查询词中没有 "sunset"（自然现象——问的是"共同画了什么"，答案词不在查询中）。FTS5 通过 "painted" + 人名找到日落事实（FTS#4-11），但 importance 权重（0.3）让 imp=8-10 的无关身份事实在 3D 评分中反超。

**修复**: ✅ **已修复** — 新增 `_detect_query_type()`，对 `fact` 类查询（what subject/name/when/how many...）将 importance 权重降至 0.05，差值移至 relevance。opinion 类查询（would/be considered）保持默认权重。commit ae5d413。

---

```
|**Importance scoring (1-10):**
- 7-8: Important relationships (family, **pets** — knowing someone's pets/animals is as important as knowing their family members)...
---

## Q3 (conv-41): 上位词「martial arts」vs 下位词「kickboxing/taekwondo」

| 字段 | 内容 |
|---|---|
| 问题 | What martial arts has John done? |
| Gold | Kickboxing, Taekwondo |
| FTS5查询 | (martial* OR soldierly* OR warlike*) OR (arts* OR humanities* OR artistry*) OR (John* OR toilet* OR bathroom*) |
| DB关键事实 | [34] John does **kickboxing** / [6] John practices **kickboxing** / [33] John is going to do some **taekwondo** |
| 误导事实 | [449] John asked Maria if the place she mentioned was a **martial arts** place or a yoga studio ✅ 匹配 martial*+art*+John* |
| Gold FTS5排名 | #4, #25, #26（在候选池内） |
| 最终排名 | 掉出 top-15（3D排序被挤出） |
| 模型回答 | "No martial arts are mentioned" |
| 得分 | 1 |

**根因**: 查询用上位词"martial arts"，DB存下位词"kickboxing/taekwondo"。WordNet同义词展开（soldierly/warlike/humanities）无法桥接此鸿沟。gold 事实仅靠 John*（1个词）匹配，而误导事实同时匹配 martial*+art*+John*（3个词），BM25 更高。虽然 gold 在候选池内，但 3D 排序后掉出 top-15。

---

## Q43 (conv-42): 泛指词「movies」vs 具体片名「Little Women/Lord of the Rings」

| 字段 | 内容 |
|---|---|
| 问题 | What movies have both Joanna and Nate seen? |
| Gold | "Little Women", "Lord of the Rings" |
| FTS5查询 | (movies* OR movie* OR film* OR picture*) OR Joanna* OR Nate* OR see* |
| DB关键事实 | [59] Joanna recommended the movie "**Little Women**" to Nate / [196] Joanna watched "**The Lord of the Rings**" Trilogy / [418] Nate watched "**Little Women**" recently |
| Gold FTS5排名 | #53 (BM25=-3.66), #66 (BM25=-3.40) — 候选池外 ❌ |
| 竞争事实 | [D8] "Joanna asked Nate if he had **seen** any good **movies** lately" → 命中 movie*+see*+人名（5词）BM25=-11.63 |
| 模型回答 | "No specific movies are identified" |
| 得分 | 1 |

**根因**: 查询用泛指词"movies"，但期待的具体片名"Little Women"和"Lord of the Rings"不在查询中。gold 事实只匹配 movie+人名（2-3词），而竞争事实（如"asked if he had seen any good movies lately"）匹配 movie*+see*+人名（5词），BM25 远高于 gold 事实。gold 事实排名 #53/#66，被候选池（45条）截断。

---

## Q52 (conv-42): 别名/用词不一致「get Tilly」vs「gave stuffed animal puppy」

| 字段 | 内容 |
|---|---|
| 问题 | When did Nate get Tilly for Joanna? |
| Gold | 25 May, 2022 |
| FTS5查询 | Nate* OR (get* OR receive* OR obtain* OR acquire*) OR Tilly* OR Joanna* |
| DB关键事实 | [265] Nate **gave** Joanna a **stuffed animal puppy** as a **gift** on **25 May 2022** |
| Gold FTS5排名 | ❌ 不在 top-200（仅匹配 Nate*+Joanna* 2个词） |
| 模型回答 | "The timing is not stated" |
| 得分 | 1 |

**根因**: 三层语义鸿沟叠加：① `get*` 不匹配 "gave"（不同词根）；② 问题说 "Tilly" 但事实写 "stuffed animal puppy"（提取时未用昵称）；③ gold 事实没有 "get"/"receive"/"Tilly" 任何关键词，仅靠 Nate*+Joanna* 匹配，在几百条同类事实中 BM25 平局，完全无法区分。

---

## Q57 (conv-42): 误导事实「stuffed animal puppy」干扰正确「turtles」证据

| 字段 | 内容 |
|---|---|
| 问题 | What animal do both Nate and Joanna like? |
| Gold | Turtles. |
| FTS5查询 | (animal* OR beast* OR creature*) OR both* OR Nate* OR Joanna* OR (like* OR wish* OR care* OR similar*) |
| DB乌龟事实 | [D15] Nate **likes** holding his **turtles**（like*+Nate* 2词）BM25=-4.17 |
| DB乌龟事实 | [D19] Joanna wishes she could get two **turtles**（Joanna*+wish* 2词）BM25=-4.37 |
| 误导事实 | [D12] Nate gave Joanna a **stuffed animal** puppy（animal*+Joanna*+Nate* **3词**）BM25=-4.88 |
| Gold FTS5排名 | #15, #19, #22（在候选池内 ✅） |
| 最终排名 | 误导事实排进 top-3，乌龟事实掉出 top-15 |
| 模型回答 | "Both like dogs (a puppy)" ❌ |
| 得分 | 1 |

**根因**: 乌龟事实只匹配 like*/人名（1-2词），而 stuffed animal 事实匹配 animal*+人名（3个词）。3D排序后误导事实排名更高（#3），模型根据检索到的"stuffed animal puppy"错误推断两人都喜欢狗。这是第一个 gold 在候选池内但因**误导事实干扰**导致答错的案例。
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
