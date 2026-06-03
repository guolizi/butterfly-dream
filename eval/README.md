# 🧪 Butterfly Dream 评测

公平、全面的记忆系统评测框架。

## 目录结构

```
eval/
├── README.md                       ← 本文档
│
├── bd_eval/                             ← Butterfly Dream 自有评测
│   ├── run_eval.py                 ← 检索评测运行器
│   ├── test_extraction.py          ← 提取评测运行器（真实 LLM 调用）
│   ├── en_compare.py               ← 中英文自然语言查询对比脚本
│   ├── gen_long_scenarios.py       ← 多长度多领域场景生成器
│   ├── gen_stress_test.py          ← 大规模压力测试生成器
│   ├── scenarios.json              ← 检索评测场景集 (77 场景, 151 查询)
│   ├── baseline.json               ← 检索评测基线
│   ├── scenarios_all_en.json       ← 提取评测场景集 (20 场景, 95 查询)
│   ├── baseline_extraction.json    ← 提取评测基线
│   └── ...                         ← 各长度/领域子集 JSON
│
└── longmemeval/                    ← LongMemEval 外部基准 (ICLR 2025)
│   ├── run_longmemeval.py          ← Butterfly Dream × LongMemEval 适配器
│   ├── debug_one.py                ← 单题诊断脚本
│   ├── data/
│   │   ├── longmemeval_oracle.json ← 500 题 Oracle 版
│   │   └── longmemeval_s.json      ← 500 题 S 版
│   └── results_*.jsonl             ← 评测结果
│
└── personamem/                     ← PersonaMem 外部基准 (COLM 2025)
    ├── run_personamem.py           ← Butterfly Dream × PersonaMem 适配器
    ├── data/
    │   ├── questions_32k.csv       ← 589 题 (32K tokens)
    │   └── shared_contexts_32k.jsonl
    └── results_*.jsonl             ← 评测结果
```

## 快速运行

```bash
# ── Butterfly Dream 自有评测 ──

# 检索评测 (77 场景, 151 查询)
python3 eval/bd_eval/run_eval.py
python3 eval/bd_eval/run_eval.py --json              # JSON 输出
python3 eval/bd_eval/run_eval.py --name "中文"        # 只跑名字含"中文"的场景

# 提取评测 (20 场景, 95 查询)
python3 eval/bd_eval/test_extraction.py
python3 eval/bd_eval/test_extraction.py --compare     # 对比多个模型

# 压力测试
python3 eval/bd_eval/gen_stress_test.py --count 500
python3 eval/bd_eval/run_eval.py --extra-scenarios eval/bd_eval/stress_test.json

# ── LongMemEval 外部基准 ──

# 12 题采样测试
python3 eval/longmemeval/run_longmemeval.py --data eval/longmemeval/data/longmemeval_sample12.json

# 500 题全量 (预计 ~5h)
python3 eval/longmemeval/run_longmemeval.py --subset oracle

# ── PersonaMem 外部基准 ──

# 5 题快速验证
python3 eval/personamem/run_personamem.py --limit 5

# 按题型筛选
python3 eval/personamem/run_personamem.py --type recall_user_shared_facts --limit 50

# 589 题全量
python3 eval/personamem/run_personamem.py
```

## 评测维度概览

### 检索评测 (bd_eval/run_eval.py)

| 维度 | 场景数 | 查询数 |
|:-----|:------:|:------:|
| 🔍 基础检索 (EN/ZH/混合) | 5 | 11 |
| 🧬 实体探针 | 4 | 5 |
| 🕰️ 时间线查询 | 4 | 5 |
| 🌀 遗忘曲线 | 2 | 8 |
| 🔗 联想记忆 / 跨会话 | 5 | 19 |
| ✏️ 记忆编辑/更新 | 5 | 16 |
| 🔄 同义改写鲁棒性 | 2 | 8 |
| 📊 大规模压力 / 多事实聚合 | 8 | 13 |
| 🛡️ 干扰鲁棒性 | 6 | 8 |
| ⏰ 事实时效优先级 | 7 | 9 |
| 🧩 多跳推理 / 时序比较 | 5 | 13 |
| 🎯 对抗相似 / 矛盾检测 | 9 | 14 |
| 🌐 跨语言 / 去重 / 持久标记 / 重要性 | 14 | 14 |
| **总计** | **77** | **151** |

### 提取评测 (bd_eval/test_extraction.py)

| 维度 | 场景数 | 查询数 | 语言 |
|:-----|:------:|:------:|:----:|
| 中文场景 (内置 + 自动生成) | 18 | 87 | 中文 |
| 英文场景 (自动生成) | 2 | 8 | 英文 |
| **总计** | **20** | **95** | — |

### LongMemEval (longmemeval/run_longmemeval.py)

| 维度 | 题数 |
|:-----|:----:|
| temporal-reasoning | 133 |
| multi-session | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |
| **总计** | **500** |

### PersonaMem (personamem/run_personamem.py)

| 维度 | 题数 |
|:-----|:----:|
| track_full_preference_evolution | 139 |
| recall_user_shared_facts | 129 |
| recalling_the_reasons_behind_previous_updates | 99 |
| suggest_new_ideas | 93 |
| generalizing_to_new_scenarios | 57 |
| provide_preference_aligned_recommendations | 55 |
| recalling_facts_mentioned_by_the_user | 17 |
| **总计** | **589** |
