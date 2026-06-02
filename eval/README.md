# 🧪 Butterfly Dream 综合记忆评测

> 公平、全面、不迁就被测系统的记忆系统评测框架。

## 设计原则

1. **公平公正** — 查询使用自然语言问句，不迁就被测系统的检索策略
2. **全面覆盖** — 测试检索、提取、推理、时序、跨语言、压力等多维度能力
3. **可复现** — 场景文件固定（LLM 一次性生成后固化），仅 LLM 提取有非确定性方差

## 文件结构

```
eval/
├── README.md                       ← 本文档
├── run_eval.py                     ← 检索评测运行器
├── test_extraction.py              ← 提取评测运行器（真实 LLM 调用）
├── gen_long_scenarios.py           ← 多长度多领域场景生成器
├── en_compare.py                   ← 中英文自然语言查询对比脚本
│
├── scenarios.json                  ← 检索评测场景集 (46 场景, 114 查询)
├── baseline.json                   ← 检索评测基线
├── scenarios_all_en.json           ← 提取评测场景集 (20 场景, 95 查询)
├── baseline_extraction.json        ← 提取评测基线
│
├── en_short_work.json              ← 英文短场景
├── en_med_work.json                ← 英文中场景
└── gen_stress_test.py              ← 大规模压力测试生成器
```

## 检索评测 (黑盒端到端)

检索评测测试 **存储→检索** 管道的端到端表现：预置固定事实，用自然语言查询验证检索质量。

| 维度 | 场景数 | 查询数 | 来源基准 |
|:-----|:------:|:------:|:---------|
| 🔍 基础检索 (EN/ZH/混合) | 3 | 5 | 通用 |
| 🧬 实体探针 | 1 | 1 | MEMOBENCH |
| 🕰️ 时间线查询 | 1 | 1 | LoCoMo |
| 🌀 遗忘曲线 (渐进式干扰) | 2 | 6 | LongMemEval, MEMOBENCH |
| 🔗 联想记忆 (线索→目标) | 1 | 5 | MEMOBENCH |
| ✏️ 记忆编辑 (信息更新) | 1 | 4 | MemoryBench, MemGPT |
| 🔄 同义改写鲁棒性 | 2 | 8 | MemoryBench |
| 🔗 跨会话检索 (分阶段信息) | 1 | 3 | LoCoMo, LoTa-Bench |
| 📊 大规模压力 (50 事实) | 1 | 5 | LongMemEval |
| 📊 多事实聚合检索 | 3 | 4 | RULER Multi-Key, HELMET Aggregation |
| 🛡️ 干扰鲁棒性 | 3 | 3 | RULER Multi-NIAH, LoCoMo 对抗 |
| ⏰ 事实时效优先级 | 4 | 5 | RULER Variable Tracking, LongMemEval |
| 🎯 其他 (去重/持久/重要/矛盾) | 4 | 4 | — |
| **检索总计** | **27** | **54** | — |

### 快速运行

```bash
# 标准检索评测 (46 场景)
python3 eval/run_eval.py

# JSON 输出 (用于对比历史基线)
python3 eval/run_eval.py --json

# 大规模压力测试 (500 事实)
python3 eval/gen_stress_test.py --count 500
python3 eval/run_eval.py --name "500" --extra-scenarios eval/stress_test.json
```

### 基线结果

| 指标 | 值 | 说明 |
|:-----|:--:|:-----|
| 场景数 | 56 | 含 3 个新维度 (聚合/干扰/时效) |
| 查询数 | 126 | |
| **R@1** | **0.702** | 首条命中率 |
| R@3 | 0.889 | 前三命中率 |
| 精确率 | 0.421 | OR 宽召回带来的噪声 |
| 平均延迟 | 7.0ms | 含完整三维评分 |

## 提取评测 (真实 LLM 端到端)

提取评测使用 **真实 LLM 调用**，从自然对话中自动提取事实并存储，再用自然语言查询验证。`test_extraction.py` 支持 `--model`/`--provider`/`--compare` 参数自由切换模型。

**评测流程**：对话 → 真实 LLM 提取 → 存储 → 自然语言查询 → 验证

| 维度 | 场景数 | 查询数 | 语言 |
|:-----|:------:|:------:|:----:|
| 🀄 中文场景 (内置) | 5 | 16 | 纯中文 |
| 🀄 中文场景 (自动生成) | 13 | 71 | 中文(混英文术语) |
| 🔤 英文场景 (自动生成) | 2 | 8 | 纯英文 |
| **提取总计** | **20** | **95** | — |

### 快速运行

```bash
# 全量提取评测 (20 场景, 95 查询)
python3 eval/test_extraction.py

# 仅跑英文场景
python3 eval/test_extraction.py --scenarios eval/en_short_work.json
python3 eval/test_extraction.py --scenarios eval/en_med_work.json

# 加载外部场景 (自动生成长度覆盖场景)
python3 eval/gen_long_scenarios.py --output eval/scenarios_gen.json
python3 eval/test_extraction.py --scenarios eval/scenarios_gen.json

# 模型对比
python3 eval/test_extraction.py --compare
python3 eval/test_extraction.py --model gpt-4o-mini --provider openai
python3 eval/test_extraction.py --model deepseek-v4-flash --json > baseline_extraction.json
```

### 基线结果 (deepseek-v4-flash)

| 指标 | 值 | 说明 |
|:-----|:--:|:-----|
| 场景数 | 20 | 5 内置 + 13 自动生成中文 + 2 自动生成英文 |
| 查询数 | 95 | 混合长度和领域 |
| **通过率** | **~62%** | 端到端 LLM 提取+检索通吃 (每次运行 ±5-15%) |
| 提取延迟 | ~6.8s/轮 | deepseek-v4-flash |
| 平均提取事实 | 6.1 条/场景 | LLM 自动判断 |

## 与原版 Holographic 对比

使用同 9 条事实（3 EN + 3 ZH + 3 mixed），分别用英文和中文自然语言问句测试：

| 场景 | 原版 Holographic | Butterfly Dream | 原因 |
|:----|:---------------:|:--------------:|:-----|
| 🔤 英文自然语言查询 (8 条) | **0/8 (0%)** | **5/8 (62.5%)** | FTS5 AND vs OR |
| 🀄 中文自然语言查询 (8 条) | **0/8 (0%)** | **8/8 (100%)** | FTS5 AND vs OR |

**根因**：原版 Holographic 使用 FTS5 默认 AND 语义 → 自然语言问句要求**所有词**出现在索引中 → 停用词（what/are/the/的/了/吗）导致全灭。Butterfly Dream 的 **OR 展开 + 前缀匹配 + jieba 分词** 让任意词匹配即可召回候选，再由三维评分精排。

```bash
python3 eval/en_compare.py  # 复现对比
```

## 指标说明

| 指标 | 计算方式 |
|:-----|:---------|
| **R@k** | 前 k 条结果中，包含所有预期子串的比例 |
| **精确率** | 前 10 条结果中，相关结果(匹配预期且非预期排除)的比例 |
| **F1** | R@5 与精确率的调和平均 |
| **延迟** | 每次查询/提取的 wall-clock 时间 (ms) |
| **提取通过率** | 查询的全部预期关键词在检索结果中被找到的比例 |

## 扩展指南

### 添加检索场景
1. 在 `scenarios.json` 中添加 JSON 对象（含 `name`、`setup`、`queries`）
2. 查询可使用 `extra_setup` 在查询间添加额外事实
3. 运行 `python3 eval/run_eval.py --json > eval/baseline.json` 更新基线

### 添加提取场景
1. 生成或手写 JSON 场景文件（含 `conversation`、`golden_facts`、`queries`）
2. 运行 `python3 eval/test_extraction.py --scenarios your_scenarios.json`
3. 结果保存到 `baseline_extraction.json`
