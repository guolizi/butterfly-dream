# 🧪 Butterfly Dream 评测

公平、全面的记忆系统评测框架。

## 目录结构

```
eval/
├── README.md                       ← 本文档
│
├── locomo/                         ← LoCoMo 外部基准 (ACL 2024)
│   ├── run_locomo.py               ← Butterfly Dream × LoCoMo 适配器
│   └── data/locomo10.json          ← 1986 道题 (10 个长对话)
│
├── longmemeval/                    ← LongMemEval 外部基准 (ICLR 2025)
│   ├── run_longmemeval.py          ← Butterfly Dream × LongMemEval 适配器
│   ├── data/
│   │   ├── longmemeval_oracle.json ← 500 题 Oracle 版
│   │   └── longmemeval_s.json      ← 500 题 S 版
│
├── personamem/                     ← PersonaMem 外部基准 (COLM 2025)
│   ├── run_personamem.py           ← Butterfly Dream × PersonaMem 适配器
│   ├── data/
│   │   ├── questions_32k.csv       ← 589 题 (32K tokens)
│   │   └── shared_contexts_32k.jsonl
│
├── eval_utils.py                   ← 共享 LLM 调用工具
├── model_config.yaml               ← 模型配置（提取/回答/评判）
├── dbs/                            ← 评测数据库
└── runs/                           ← 评测运行结果
```

## 快速运行

```bash
# ── LongMemEval 外部基准 ──
python3 eval/longmemeval/run_longmemeval.py --subset oracle

# ── PersonaMem 外部基准 ──
python3 eval/personamem/run_personamem.py

# ── LoCoMo 外部基准 ──
python3 eval/locomo/run_locomo.py
```

## 约定

所有 eval 脚本从 `model_config.yaml` 读取模型设置，换模型只改这一个文件。
每个 role（extraction/answer/judge）可独立配置 provider 和 model。
