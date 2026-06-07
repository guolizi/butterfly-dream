# 回答 Prompt 对 QA 准确率的影响 — 实验记录

## 背景

在 Butterfly Dream 的 LoCoMo 评测管线中，回答模型使用 GPT-OSS-120B (OpenRouter free) 生成答案，温度 0.1，max_tokens=1024。对于 Cat1-4 使用 CoT prompt，Cat5 使用对抗检测 prompt。

观察发现 Cat1 低分题的主要模式是"检索命中但答不准确"——gold 关键词在 top-15 中但模型答偏或说 No info。尤其是 Q24，17/20 条检索事实都包含 gold 关键词，但模型只用了 #1 事实（notebook）就答了。

## 实验 1: 强制列出所有相关事实

### 方法

将 CoT prompt 从"Think step by step"改为强制在回答前先列出所有相关事实：

```
STEP 1 — 理解问题
STEP 2 — 逐条审查每条事实（#1 到 #15），不跳过任何一条
STEP 3 — 列出所有相关事实（编号+内容）
STEP 4 — 综合所有相关事实给出完整答案
```

系统 prompt 也从"You analyze facts carefully..."改为"You carefully examine each fact one by one before answering..."

### 数据集
- conv-50（Calvin & Dave）
- Cat1 单跳题 32 道
- DB: temp=0.2 提取（608 facts）
- 回答模型: openai/gpt-oss-120b:free, temp=0.1

### 结果

| 指标 | 旧版 CoT | 新版 "list facts" | 变化 |
|---|---|---|---|
| **正确率 (score≥4)** | **50%** (16/32) | **41%** (13/32) | **-3题 ❌** |
| **均分** | **3.28** | **3.00** | **-0.28 ❌** |
| score=5 | 13 | 11 | -2 |
| score=4 | 3 | 2 | -1 |
| score=3 | 5 | 3 | -2 |
| score=2 | 2 | **8** | **+6 ❌** |
| score=1 | 9 | 8 | -1 |

### 分析

新 prompt 不但没改进，反而显著退步了。原因：

1. **Q24 锚定问题未解决**：强制列事实后，模型依然只列了 #1（notebook），完全没看到 #6（documentaries）。GPT-OSS-120B 的注意力机制在 #1 找到"creative process"完美匹配后就停止了。

2. **格式混乱**：模型开始输出 "**Relevant facts** - #1 ... - #2 ..." 之类的清单，然后 judge 对这种非标准回答格式的评分不稳定。

3. **过度列事实导致答偏**：score=2 从 2 题飙升到 8 题。模型列了一堆"相关"事实后，综合出来的答案反而包含了不相关信息或遗漏了关键信息。

### 结论

- **GPT-OSS-120B 的注意力锚定问题不是改 prompt 措辞能解决的**
- #1 有高匹配度关键词时，模型会锁定它，无视后面的事实
- 强制列事实的指令反而让模型答偏更多题
- ⚠️ **当前最佳 prompt = 旧版 CoT（Think step by step）**，已还原

### 可能的后续方向

1. **调整检索排序**：把关键但排名靠后的事实提上来（比如 Q24 的 #6 documentaries 提到 #2 位置）
2. **分段 context**：把 15 条事实分成 3 组，每 5 条让模型看一次，汇总后再回答
3. **换更强模型**：Claude/GPT-4 级别的模型可能没有这个注意力缺陷
4. **两阶段回答**：第一阶段找出相关事实，第二阶段用这些事实回答问题（分两次 LLM 调用）
