# L1 事实提取 Prompt（v1）

## 任务
从以下对话中提取 Caroline 的 L1 事实。对话是 Caroline 和 Melanie 之间的聊天记录。

## 提取维度（一次调用，多维度输出）

### 1. 事件（event）
有时间锚点的具体事件。写入 `facts` 表，`type='event'`。
- 尽量提取结构化字段：subject, action, object, time, location
- content_date：事件发生日期（ISO 格式）

### 2. 属性/知识（knowledge）
去时间化的稳定知识（偏好、价值观、特质、观点）。写入 `facts` 表，`type='knowledge'`。

### 3. 行为模式（behavior）
条件-行为规律（"当 X 时，Caroline 会做 Y"）。写入 `facts` 表，`type='behavior'`。
- 有时间锚点+周期性/条件性 → behavior
- 无时间锚点+条件性行为 → knowledge（自述行为倾向）

### 4. 情感（emotion）
情感状态 + VAD + importance。写入 `emotion_events` 表。
- emotion_vector: [valence, arousal, dominance]，范围[-1,1]×[0,1]×[-1,1]
- source: 'user'（从 Caroline 的消息提取）/ 'assistant'（从 Melanie 的消息提取）
- importance: 0.0~1.0
- trigger_topics: 触发话题列表
- emotion_label: 中文情感标签
- emotion_target: null / 'self' / 'person:xxx' / 'event:xxx'

### 5. 关系（relation）
人物之间的关系。写入实体图。

## 事实字段说明
- content: 事实内容（中文，简洁完整的一句话）
- type: event / knowledge / behavior
- category: social / career / psychology / hobby / value / family / health / education / general
- tags: 逗号分隔的英文标签
- importance: 0.0~1.0
  - ≥0.9: 人生里程碑
  - 0.7~0.9: 重要事件
  - 0.4~0.7: 日常
  - <0.4: 轻微
- content_date: 事件日期（ISO，event 类型必填）
- emotion_tag: null(中性) / 具体标签 / positive / negative / mixed
- structured_data: JSON，含 subject/action/object/time/location（event 类型尽量填）

## 输出格式
请输出 JSON 数组，每个元素包含一个事实：
```json
[
  {
    "dimension": "event",
    "content": "Caroline 昨天参加了LGBTQ互助小组",
    "type": "event",
    "category": "social",
    "tags": "LGBTQ,support_group",
    "importance": 0.8,
    "content_date": "2023-05-07",
    "emotion_tag": "开心",
    "entities": ["Caroline", "LGBTQ互助小组"],
    "structured_data": {"subject": "Caroline", "action": "参加", "object": "LGBTQ互助小组", "time": "2023-05-07", "location": null}
  },
  {
    "dimension": "emotion",
    "content": "Caroline 在互助小组中感到开心和感激",
    "emotion_vector": [0.8, 0.7, 0.6],
    "emotion_label": "开心",
    "emotion_target": "event:support_group",
    "source": "user",
    "importance": 0.8,
    "significance_reason": "用户主动表达强烈正面情感",
    "trigger_topics": ["LGBTQ互助小组", "跨性别故事"],
    "timestamp": "2023-05-07T14:00:00"
  },
  {
    "dimension": "relation",
    "relation": "friend_of",
    "source": "Caroline",
    "target": "Melanie",
    "weight": 0.8
  }
]
```

## 对话内容
```
{session_text}
```
