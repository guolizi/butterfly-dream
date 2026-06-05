# Known Issues

## [检索] QA 场景下 importance 权重导致正确答案被挤出 top 10

**发现日期:** 2026-06-04
**影响范围:** LoCoMo 评测 (可能影响所有 QA 评测)
**严重程度:** 中

### 现象

LoCoMo conv-26 的两道题 Q8/Q9，FTS 候选中包含正确答案，但经过 `ThreeDimRetriever.search()` 的三维打分后，正确答案被排到 top 10 之外，模型收到的 facts 里没有答案。

### 根因

`ThreeDimRetriever.search()` 的打分公式：
```
score = (0.4×relevance + 0.4×recency + 0.2×importance) × trust
```

- **relevance**（FTS rank）：正确答案的 FTS rank 最高（最相关）
- **recency**：所有 facts 同一天提取，recency 几乎无差异
- **importance**：正确答案（具体事件 fact）的 importance 被标低（5.0），而概括性 fact importance 更高（7.0-8.0）

结果：importance 加分让概括性 fact 排到了具体事件 fact 前面，但问答需要的是具体事件。

### 具体案例

| Fact | FTS rank | importance | 总分 | 排名 |
|------|----------|-----------|------|------|
| "Caroline met up with friends the week of June 2, 2023" | 0.9987 | 5.0 (0.44) | 0.3240 | #22 |
| "Caroline's transition changed her relationships" | 0.9641 | 8.0 (0.78) | 0.3546 | #2 |

### 可能的修复方向

1. QA 场景下加大 relevance 权重（如 relevance=0.6, importance=0.1）
2. 提取阶段对事件类 fact 提高 importance
3. 增大 FTS 候选数量（当前 limit×3=30），给更多 fact 参与打分的机会
4. 对 QA 场景使用不同的检索策略（如只用 FTS rank，不做三维重排）

### 验证方法

直接用相同 facts 调用 LLM，模型能正确回答。确认是检索排序问题，不是模型能力问题。
