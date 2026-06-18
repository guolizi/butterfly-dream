# L1 事实提取 Prompt（v2 — 优化版）

## 任务
从以下对话中提取 `{person}` 的 L1 事实。对话是 `{person}` 和 `{other_person}` 在 **{session_time}** 的聊天。

## 五个维度，每个维度有独立的字段格式

### 1. 事件（dimension: "event"）
有时间锚点的具体事件。**content_date 必填**，根据会话时间解析相对日期。
- 包含：过去发生的事、未来的计划/承诺
- 每个独立的事件单元都提取（如"参加小组"和"听到故事"是两个独立事件）
```json
{"dimension": "event", "content": "Caroline 昨天参加了LGBTQ互助小组", "type": "event", "category": "social", "tags": "LGBTQ,support_group", "importance": 0.8, "content_date": "2023-05-07", "emotion_tag": "positive", "entities": ["Caroline", "LGBTQ互助小组"], "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ互助小组", "time": "2023-05-07", "location": null}}
```

### 2. 知识/属性（dimension: "knowledge"）
去时间化的稳定知识。**没有 content_date**。
- 包含：偏好、价值观、特质、观点、自我认知、情感影响
```json
{"dimension": "knowledge", "content": "Caroline 对心理咨询或心理健康领域感兴趣", "type": "knowledge", "category": "career", "tags": "counseling,mental_health", "importance": 0.7, "emotion_tag": "positive", "entities": ["Caroline"]}
```

### 3. 行为模式（dimension: "behavior"）
条件-行为规律。**必须提取！**
- 检查对话中是否有"当 X 时，我会做 Y"的条件关系
- 如果没有明确的条件关系，不要强行构造
```json
{"dimension": "behavior", "content": "当 Caroline 感到被接纳时，她会去探索职业方向", "type": "behavior", "category": "psychology", "tags": "acceptance,career_exploration", "importance": 0.6, "emotion_tag": null, "entities": ["Caroline"]}
```

### 4. 情感（dimension: "emotion"）
情感状态。**只包含 emotion 特有字段**。
- 提取有明确情感信号的轮次
- 同一个话题的连续情感可以合并
```json
{"dimension": "emotion", "content": "Caroline 在互助小组中感到开心和感激", "emotion_vector": [0.9, 0.8, 0.6], "emotion_label": "开心感激", "emotion_target": "event:support_group", "source": "user", "importance": 0.7, "significance_reason": "用户主动表达开心和感激", "trigger_topics": ["LGBTQ互助小组", "跨性别故事"], "timestamp": "2023-05-08T13:56:00"}
```

### 5. 关系（dimension: "relation"）
人物关系。**只包含 relation 特有字段**。
```json
{"dimension": "relation", "relation": "friend_of", "source": "Caroline", "target": "Melanie", "weight": 0.8}
```

## 重要规则
1. 所有日期时间根据会话时间解析相对表达（"昨天"→会话前一天, "去年"→会话前一年）
2. **不要遗漏重要事实**：每个独立的信息单元都考虑提取
3. 情感事件提取 2-4 条，只提取有明确情感信号的
4. 每个维度只输出该维度定义的字段，不要混入其他维度的字段
5. 事件(content_date)和情感(timestamp)的时间字段不要留空
6. 只提取 `{person}` 相关的事实，不提取 `{other_person}` 的

## 输出格式
纯 JSON 数组，不要加 markdown 代码块标记或其他文字。
