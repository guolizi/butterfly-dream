#!/usr/bin/env python3
"""Butterfly Dream 扩展场景生成器 — 构造多维度、多难度、带干扰的评测集。

用法：
    python3 eval/gen_scenarios.py                      # 生成全部扩展场景
    python3 eval/gen_scenarios.py --name "遗忘"         # 只生成特定维度
    python3 eval/gen_scenarios.py --merge              # 合并到 scenarios.json

生成的场景遵循：公平、自然语言查询、不迁就被测系统。
"""

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42


def seed_name(base: str, variant: int) -> str:
    return f"{base} #{variant}"


# ────────────────────────────────────────────────────────
# 1. 实体探针扩展
# ────────────────────────────────────────────────────────
def gen_probe_scenarios():
    scenarios = []

    # 1a. 多实体重叠探针
    scenarios.append({
        "name": seed_name("实体探针-多实体重叠", 1),
        "setup": [
            {"action": "add_fact", "content": "用户喜欢VS Code编辑器", "tags": "工具", "category": "user_pref", "importance": 6, "entities": ["编辑器"]},
            {"action": "add_fact", "content": "用户用VS Code写Python和TypeScript", "tags": "语言", "category": "user_pref", "importance": 7, "entities": ["编辑器", "VS Code"]},
            {"action": "add_fact", "content": "用户的项目使用TypeScript", "tags": "项目", "category": "project", "importance": 6, "entities": ["TypeScript"]},
            {"action": "add_fact", "content": "用户用Neovim管理配置文件", "tags": "工具", "category": "user_pref", "importance": 5, "entities": ["编辑器", "Neovim"]},
        ],
        "queries": [
            {"query": "VS Code", "type": "probe", "params": {}, "expected": ["VS Code写Python"], "recall_at_k": [1, 3]},
            {"query": "编辑器", "type": "probe", "params": {}, "expected": ["VS Code编辑器", "VS Code写Python", "Neovim管理配置文件"], "recall_at_k": [1, 3, 5]},
        ]
    })

    # 1b. 单实体大量事实
    scenarios.append({
        "name": seed_name("实体探针-单实体大量事实", 2),
        "setup": [
            {"action": "add_fact", "content": f"用户对实体X的第{i}条记录", "importance": 3 + i % 5, "tags": "测试", "category": "general", "entities": ["实体X"]}
            for i in range(1, 11)
        ] + [
            {"action": "add_fact", "content": f"其他实体Y的第{i}条记录", "importance": 3, "tags": "测试", "category": "general", "entities": ["实体Y"]}
            for i in range(1, 4)
        ],
        "queries": [
            {"query": "实体X", "type": "probe", "params": {}, "expected": [f"第{i}条记录" for i in range(1, 11)], "recall_at_k": [1, 3, 5]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 2. 基础检索扩展
# ────────────────────────────────────────────────────────
def gen_retrieval_scenarios():
    scenarios = []

    # 2a. 英文复杂句子+同义词
    scenarios.append({
        "name": seed_name("基础检索-英文同义词变体", 4),
        "setup": [
            {"action": "add_fact", "content": "Alice enjoys hiking in the mountains every weekend", "importance": 5, "tags": "hobby", "category": "user_pref"},
            {"action": "add_fact", "content": "Bob loves swimming at the beach during summer", "importance": 5, "tags": "hobby", "category": "user_pref"},
            {"action": "add_fact", "content": "Charlie prefers reading science fiction novels", "importance": 5, "tags": "hobby", "category": "user_pref"},
            {"action": "add_fact", "content": "Alice works as a software engineer at Microsoft", "importance": 7, "tags": "job", "category": "user_pref"},
        ],
        "queries": [
            {"query": "What does Alice do on weekends?", "type": "search", "params": {"scenario": "chat"},
             "expected": ["hiking"], "unexpected": ["Bob", "Charlie"], "recall_at_k": [1, 3]},
            {"query": "Where does Alice work?", "type": "search", "params": {"scenario": "chat"},
             "expected": ["Microsoft"], "unexpected": ["hiking", "reading"], "recall_at_k": [1, 3]},
            {"query": "What kind of books does Charlie like?", "type": "search", "params": {"scenario": "chat"},
             "expected": ["science fiction"], "unexpected": ["Alice", "beach"], "recall_at_k": [1, 3]},
        ]
    })

    # 2b. 中文复杂长句+干扰
    scenarios.append({
        "name": seed_name("基础检索-中文长句+干扰", 5),
        "setup": [
            {"action": "add_fact", "content": "用户上周末和朋友们一起去爬了黄山，天气非常好", "importance": 4, "tags": "旅行", "category": "general"},
            {"action": "add_fact", "content": "用户打算下个月带家人去三亚度假，已经订好了酒店", "importance": 6, "tags": "旅行,计划", "category": "general"},
            {"action": "add_fact", "content": "用户在阿里巴巴做后端开发，主要用Java和Go", "importance": 8, "tags": "工作,技术", "category": "user_pref"},
            {"action": "add_fact", "content": "用户最近在学习Kubernetes和Docker的部署方案", "importance": 6, "tags": "学习,技术", "category": "user_pref"},
            {"action": "add_fact", "content": "用户去年参加了一个全马比赛，成绩是4小时30分", "importance": 4, "tags": "运动", "category": "general"},
            {"action": "add_fact", "content": "用户养了一只布偶猫和三只金鱼", "importance": 3, "tags": "宠物", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户上次爬山去了哪里？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["黄山"], "unexpected": ["三亚", "Kubernetes", "布偶猫"], "recall_at_k": [1, 3]},
            {"query": "用户在哪家公司做什么工作？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["阿里巴巴", "后端开发"], "unexpected": ["黄山", "全马"], "recall_at_k": [1, 3]},
            {"query": "用户养了什么宠物？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["布偶猫", "金鱼"], "unexpected": ["黄山", "Docker"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 3. 时间线扩展
# ────────────────────────────────────────────────────────
def gen_timeline_scenarios():
    scenarios = []

    # 3a. 长期时间线
    events = [
        "用户2018年大学毕业", "用户2019年入职第一家公司",
        "用户2020年学会了Python", "用户2021年晋升为高级工程师",
        "用户2022年开始管理团队", "用户2023年跳槽到新公司",
        "用户2024年买了房子", "用户2025年开始学习AI",
    ]
    scenarios.append({
        "name": seed_name("时间线查询-长期职业发展", 3),
        "setup": [
            {"action": "add_fact", "content": e, "importance": 6 + (8 - i), "tags": "时间线", "category": "general", "entities": ["时间线_A"]}
            for i, e in enumerate(events)
        ],
        "queries": [
            {"query": "时间线_A", "type": "timeline", "params": {},
             "expected": [e.split("用户")[1] for e in events], "recall_at_k": [1, 3, 5]},
        ]
    })

    # 3b. 双实体交叉事件
    scenarios.append({
        "name": seed_name("时间线查询-双实体交叉", 4),
        "setup": [
            {"action": "add_fact", "content": "项目A在1月立项", "importance": 6, "tags": "项目", "category": "project", "entities": ["项目A"]},
            {"action": "add_fact", "content": "项目B在2月立项", "importance": 6, "tags": "项目", "category": "project", "entities": ["项目B"]},
            {"action": "add_fact", "content": "项目A在3月完成开发", "importance": 7, "tags": "项目", "category": "project", "entities": ["项目A"]},
            {"action": "add_fact", "content": "项目B在4月完成开发", "importance": 7, "tags": "项目", "category": "project", "entities": ["项目B"]},
            {"action": "add_fact", "content": "项目A在5月上线", "importance": 8, "tags": "项目", "category": "project", "entities": ["项目A"]},
            {"action": "add_fact", "content": "项目B在6月上线", "importance": 8, "tags": "项目", "category": "project", "entities": ["项目B"]},
        ],
        "queries": [
            {"query": "项目A", "type": "timeline", "params": {},
             "expected": ["1月立项", "3月完成开发", "5月上线"], "recall_at_k": [1, 3, 5]},
            {"query": "项目B", "type": "timeline", "params": {},
             "expected": ["2月立项", "4月完成开发", "6月上线"], "recall_at_k": [1, 3, 5]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 4. 去重/持久/重要性/矛盾扩展
# ────────────────────────────────────────────────────────
def gen_filter_scenarios():
    scenarios = []

    # 4a. 三重去重
    scenarios.append({
        "name": seed_name("跨源去重-三重相似", 2),
        "setup": [
            {"action": "add_fact", "content": "User likes VS Code for Python", "dedup_threshold": 0.7},
            {"action": "add_fact", "content": "User loves VS Code for Python", "dedup_threshold": 0.7},
            {"action": "add_fact", "content": "User adores VS Code for Python", "dedup_threshold": 0.7},
            {"action": "add_fact", "content": "User prefers PyCharm for Java", "dedup_threshold": 0.7},
        ],
        "queries": [
            {"query": "", "type": "dedup_check", "params": {"expected_count": 2},
             "expected": {"exact_count": 2}, "unexpected": [], "recall_at_k": []},
        ]
    })

    # 4b. 持久标记+分类过滤
    scenarios.append({
        "name": seed_name("持久标记-分类组合过滤", 2),
        "setup": [
            {"action": "add_fact", "content": "用户的永久地址是北京朝阳区", "is_persistent": True, "importance": 8, "category": "user_pref"},
            {"action": "add_fact", "content": "用户当前的临时项目是数据库迁移", "is_persistent": False, "importance": 4, "category": "project"},
            {"action": "add_fact", "content": "用户的永久邮箱是user@example.com", "is_persistent": True, "importance": 7, "category": "user_pref"},
            {"action": "add_fact", "content": "正在调试的Bug是登录超时", "is_persistent": False, "importance": 5, "category": "project"},
        ],
        "queries": [
            {"query": "用户的永久信息", "type": "search", "params": {"persistent_only": True},
             "expected": ["北京朝阳区", "user@example.com"], "unexpected": ["数据库迁移", "登录超时"], "recall_at_k": [1, 3]},
        ]
    })

    # 4c. 重要性排序-中文
    scenarios.append({
        "name": seed_name("重要性排序-中文梯度测试", 2),
        "setup": [
            {"action": "add_fact", "content": "用户最喜欢的食物是火锅", "importance": 3, "tags": "偏好", "category": "user_pref"},
            {"action": "add_fact", "content": "用户对花生严重过敏", "importance": 10, "tags": "健康", "category": "user_pref"},
            {"action": "add_fact", "content": "用户偶尔吃一次烧烤", "importance": 2, "tags": "偏好", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户对食物有什么偏好？", "type": "search", "params": {"scenario": "longterm"},
             "expected": ["花生严重过敏", "最喜欢的食物是火锅"], "unexpected": ["偶尔吃一次烧烤"], "recall_at_k": [1, 3]},
        ]
    })

    # 4d. 三方矛盾
    scenarios.append({
        "name": seed_name("矛盾检测-三方态度变化", 2),
        "setup": [
            {"action": "add_fact", "content": "用户最开始喜欢用Windows系统", "importance": 5, "tags": "系统", "category": "user_pref"},
            {"action": "add_fact", "content": "用户后来觉得Windows不好用", "importance": 6, "tags": "系统", "category": "user_pref"},
            {"action": "add_fact", "content": "用户现在只用macOS和Linux", "importance": 8, "tags": "系统", "category": "user_pref"},
        ],
        "queries": [
            {"query": "", "type": "contradict", "params": {},
             "expected": ["Windows系统", "Windows不好用"], "unexpected": [], "recall_at_k": [1, 3]},
        ]
    })

    # 4e. 矛盾+中立事实
    scenarios.append({
        "name": seed_name("矛盾检测-带中立干扰", 3),
        "setup": [
            {"action": "add_fact", "content": "用户认为React是最好的前端框架", "importance": 6, "tags": "技术", "category": "user_pref"},
            {"action": "add_fact", "content": "用户认为Vue比React更容易上手", "importance": 5, "tags": "技术", "category": "user_pref"},
            {"action": "add_fact", "content": "用户用Svelte写个人项目", "importance": 4, "tags": "技术", "category": "user_pref"},
            {"action": "add_fact", "content": "用户Team用Angular写企业应用", "importance": 5, "tags": "技术", "category": "project"},
        ],
        "queries": [
            {"query": "", "type": "contradict", "params": {},
             "expected": ["React", "Vue"], "unexpected": ["Svelte", "Angular"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 5. 跨会话扩展
# ────────────────────────────────────────────────────────
def gen_cross_session_scenarios():
    scenarios = []

    # 5a. 三阶段级联
    scenarios.append({
        "name": seed_name("跨会话检索-三阶段级联", 2),
        "setup": [
            {"action": "add_fact", "content": "用户计划做一个博客网站", "importance": 6, "tags": "项目", "category": "project"},
            {"action": "add_fact", "content": "博客使用Next.js框架", "importance": 6, "tags": "技术", "category": "project"},
        ],
        "queries": [
            {"query": "博客用什么技术栈？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["Next.js"], "recall_at_k": [1, 3]},
            {"query": "博客数据库用什么？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["SQLite"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "博客数据库选用SQLite", "importance": 6, "tags": "技术,更新", "category": "project"},
             ]},
            {"query": "博客部署在哪？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["Vercel"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "博客部署在Vercel上", "importance": 6, "tags": "部署", "category": "project"},
                 {"action": "add_fact", "content": "用户买了blog.example.com域名", "importance": 5, "tags": "域名", "category": "project"},
             ]},
        ]
    })

    # 5b. 矛盾性跨会话更新
    scenarios.append({
        "name": seed_name("跨会话检索-偏好反转", 3),
        "setup": [
            {"action": "add_fact", "content": "用户最喜欢的颜色是蓝色", "importance": 7, "tags": "偏好", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户喜欢什么颜色？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["蓝色"], "recall_at_k": [1, 3]},
            {"query": "用户现在喜欢什么颜色？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["绿色"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "用户现在最喜欢的颜色变成了绿色", "importance": 8, "tags": "偏好,更新", "category": "user_pref"},
             ]},
            {"query": "用户后来又喜欢什么颜色？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["紫色"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "用户后来又喜欢上了紫色", "importance": 7, "tags": "偏好,更新", "category": "user_pref"},
             ]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 6. 多跳/时序/对抗扩展
# ────────────────────────────────────────────────────────
def gen_reasoning_scenarios():
    scenarios = []

    # 6a. 3跳推理链
    scenarios.append({
        "name": seed_name("多跳推理-三跳链", 2),
        "setup": [
            {"action": "add_fact", "content": "用户工作在蚂蚁集团", "importance": 8, "tags": "工作", "category": "user_pref", "entities": ["蚂蚁集团"]},
            {"action": "add_fact", "content": "蚂蚁集团总部在杭州", "importance": 6, "tags": "地理", "category": "general", "entities": ["蚂蚁集团", "杭州"]},
            {"action": "add_fact", "content": "杭州是浙江省的省会", "importance": 5, "tags": "地理", "category": "general", "entities": ["杭州", "浙江"]},
            {"action": "add_fact", "content": "用户最近搬到了上海", "importance": 7, "tags": "生活", "category": "user_pref", "entities": ["上海"]},
        ],
        "queries": [
            {"query": "用户公司的总部在哪个省？", "type": "search", "params": {"scenario": "qa"},
             "expected": ["浙江省", "杭州"], "unexpected": ["上海", "蚂蚁集团"], "recall_at_k": [1, 3]},
            {"query": "用户现在住在哪个城市？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["上海"], "unexpected": ["杭州", "浙江"], "recall_at_k": [1, 3]},
            {"query": "用户公司的竞争对手在哪里？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["杭州"], "unexpected": [], "recall_at_k": [1, 3]},
        ]
    })

    # 6b. 多跳+干扰事实
    scenarios.append({
        "name": seed_name("多跳推理-带干扰链", 3),
        "setup": [
            {"action": "add_fact", "content": "小明在腾讯工作", "importance": 7, "tags": "工作", "category": "user_pref", "entities": ["小明", "腾讯"]},
            {"action": "add_fact", "content": "腾讯总部在深圳", "importance": 6, "tags": "地理", "category": "general", "entities": ["腾讯", "深圳"]},
            {"action": "add_fact", "content": "小红在华为工作", "importance": 7, "tags": "工作", "category": "user_pref", "entities": ["小红", "华为"]},
            {"action": "add_fact", "content": "华为总部在深圳", "importance": 6, "tags": "地理", "category": "general", "entities": ["华为", "深圳"]},
            {"action": "add_fact", "content": "深圳以科技创新闻名", "importance": 4, "tags": "地理", "category": "general", "entities": ["深圳"]},
        ],
        "queries": [
            {"query": "小明和小红在同一个城市工作吗？", "type": "search", "params": {"scenario": "qa"},
             "expected": ["深圳"], "unexpected": ["腾讯", "华为"], "recall_at_k": [1, 3]},
            {"query": "腾讯在哪个城市？", "type": "search", "params": {"scenario": "qa"},
             "expected": ["深圳"], "unexpected": ["小明", "华为"], "recall_at_k": [1, 3]},
        ]
    })

    # 6c. 时序-复杂范围查询
    scenarios.append({
        "name": seed_name("时序比较-季度范围查询", 2),
        "setup": [
            {"action": "add_fact", "content": "项目第一季度完成需求分析", "importance": 6, "tags": "项目", "category": "project"},
            {"action": "add_fact", "content": "项目第二季度进入开发阶段", "importance": 6, "tags": "项目", "category": "project"},
            {"action": "add_fact", "content": "项目第三季度开始测试", "importance": 6, "tags": "项目", "category": "project"},
            {"action": "add_fact", "content": "项目第四季度正式上线", "importance": 8, "tags": "项目", "category": "project"},
        ],
        "queries": [
            {"query": "项目上半年做了什么？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["第一季度完成需求分析", "第二季度进入开发阶段"],
             "unexpected": ["第三季度开始测试", "第四季度正式上线"], "recall_at_k": [1, 3]},
            {"query": "项目什么时候上线？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["第四季度正式上线"], "unexpected": ["需求分析", "开发阶段"], "recall_at_k": [1, 3]},
        ]
    })

    # 6d. 对抗-精细区分
    scenarios.append({
        "name": seed_name("对抗相似-N对一", 2),
        "setup": [
            {"action": "add_fact", "content": "用户经常阅读Python官方文档", "importance": 5, "tags": "阅读", "category": "user_pref"},
            {"action": "add_fact", "content": "用户从来不读Java文档", "importance": 5, "tags": "阅读,评价", "category": "user_pref"},
            {"action": "add_fact", "content": "用户读过Rust官方文档", "importance": 5, "tags": "阅读", "category": "user_pref"},
            {"action": "add_fact", "content": "用户正在读Go语言文档", "importance": 5, "tags": "阅读", "category": "user_pref"},
            {"action": "add_fact", "content": "用户打算读TypeScript文档", "importance": 4, "tags": "阅读,计划", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户读哪些语言的文档？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["Python官方文档", "Rust官方文档", "Go语言文档"],
             "unexpected": ["从来不读Java", "打算读TypeScript"], "recall_at_k": [1, 3]},
            {"query": "用户不读什么文档？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["从来不读Java"], "unexpected": ["Python", "Rust"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 7. 跨语言扩展
# ────────────────────────────────────────────────────────
def gen_crosslingual_scenarios():
    scenarios = []

    # 7a. 中日英混合
    scenarios.append({
        "name": seed_name("跨语言检索-中日英混合", 2),
        "setup": [
            {"action": "add_fact", "content": "Mount Fuji is 3776 meters tall", "importance": 5, "tags": "geography", "category": "general"},
            {"action": "add_fact", "content": "富士山の高さは3776メートルです", "importance": 5, "tags": "地理", "category": "general"},
            {"action": "add_fact", "content": "The population of Tokyo is 14 million", "importance": 5, "tags": "geography", "category": "general"},
            {"action": "add_fact", "content": "用户去过日本的京都和大阪", "importance": 4, "tags": "旅行", "category": "general"},
        ],
        "queries": [
            {"query": "富士山的高度", "type": "search", "params": {"scenario": "qa"},
             "expected": ["3776 meters tall", "3776メートル"], "unexpected": ["Tokyo", "京都"], "recall_at_k": [1, 3]},
            {"query": "How tall is Mount Fuji?", "type": "search", "params": {"scenario": "qa"},
             "expected": ["3776 meters tall", "3776メートル"], "unexpected": ["Tokyo", "京都"], "recall_at_k": [1, 3]},
            {"query": "Tokyo population", "type": "search", "params": {"scenario": "qa"},
             "expected": ["14 million"], "unexpected": ["3776", "京都"], "recall_at_k": [1, 3]},
        ]
    })

    # 7b. 代码+自然语言混合
    scenarios.append({
        "name": seed_name("跨语言检索-代码+自然语言", 3),
        "setup": [
            {"action": "add_fact", "content": "用户用Python写了个爬虫脚本", "importance": 5, "tags": "代码", "category": "project"},
            {"action": "add_fact", "content": "def hello(name): print(f'Hello, {name}!')", "importance": 3, "tags": "code", "category": "general"},
            {"action": "add_fact", "content": "用户熟悉RESTful API设计规范", "importance": 6, "tags": "技术", "category": "user_pref"},
        ],
        "queries": [
            {"query": "Python script for web scraping", "type": "search", "params": {"scenario": "technical"},
             "expected": ["爬虫脚本"], "unexpected": ["hello", "RESTful"], "recall_at_k": [1, 3]},
            {"query": "用户熟悉什么API设计？", "type": "search", "params": {"scenario": "technical"},
             "expected": ["RESTful"], "unexpected": ["爬虫", "hello"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 8. 更新精度扩展
# ────────────────────────────────────────────────────────
def gen_update_scenarios():
    scenarios = []

    # 8a. 级联更新
    scenarios.append({
        "name": seed_name("更新精度-级联三次更新", 2),
        "setup": [
            {"action": "add_fact", "content": "用户的手机是iPhone 14", "importance": 5, "tags": "设备", "category": "user_pref", "entities": ["手机"]},
            {"action": "add_fact", "content": "用户的电脑是MacBook Pro", "importance": 5, "tags": "设备", "category": "tool", "entities": ["电脑"]},
        ],
        "queries": [
            {"query": "用户用什么手机？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["iPhone 14"], "unexpected": ["MacBook"], "recall_at_k": [1, 3]},
            {"query": "用户换了什么新手机？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["iPhone 15"], "unexpected": ["iPhone 14", "MacBook"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "用户换成了iPhone 15", "importance": 6, "tags": "设备,更新", "category": "user_pref", "entities": ["手机"]},
             ]},
            {"query": "用户的电脑换了吗？", "type": "search", "params": {"scenario": "chat"},
             "expected": ["MacBook Pro"], "unexpected": ["iPhone 15", "iPhone 14"], "recall_at_k": [1, 3]},
        ]
    })

    # 8b. 批量更新+验证不相关事实
    scenarios.append({
        "name": seed_name("更新精度-批量验证隔离", 3),
        "setup": [
            {"action": "add_fact", "content": "用户的住址是北京海淀区", "importance": 7, "tags": "地址", "category": "user_pref", "entities": ["地址"]},
            {"action": "add_fact", "content": "用户的公司地址是上海浦东新区", "importance": 6, "tags": "地址", "category": "user_pref", "entities": ["地址"]},
            {"action": "add_fact", "content": "用户的出生日期是1990年1月1日", "importance": 8, "tags": "个人", "category": "user_pref"},
            {"action": "add_fact", "content": "用户的血型是A型", "importance": 4, "tags": "个人", "category": "user_pref"},
            {"action": "add_fact", "content": "用户的紧急联系人电话是13800000000", "importance": 7, "tags": "联系", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户的住址", "type": "search", "params": {"scenario": "chat"},
             "expected": ["北京海淀区"], "unexpected": ["上海浦东", "1990年", "A型", "138"], "recall_at_k": [1, 3]},
            {"query": "用户的住址变了", "type": "search", "params": {"scenario": "chat"},
             "expected": ["深圳南山区"], "unexpected": ["北京海淀区", "上海浦东", "1990年"], "recall_at_k": [1, 3],
             "extra_setup": [
                 {"action": "add_fact", "content": "用户搬到了深圳南山区", "importance": 8, "tags": "地址,更新", "category": "user_pref", "entities": ["地址"]},
             ]},
            {"query": "用户的公司地址", "type": "search", "params": {"scenario": "chat"},
             "expected": ["上海浦东新区"], "unexpected": ["深圳", "北京海淀", "1990年"], "recall_at_k": [1, 3]},
            {"query": "用户的出生日期", "type": "search", "params": {"scenario": "chat"},
             "expected": ["1990年1月1日"], "unexpected": ["深圳", "上海浦东", "A型"], "recall_at_k": [1, 3]},
            {"query": "用户的紧急联系", "type": "search", "params": {"scenario": "chat"},
             "expected": ["13800000000"], "unexpected": ["深圳", "北京", "A型"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 9. 大规模遗忘曲线扩展
# ────────────────────────────────────────────────────────
def gen_long_forgetting_scenarios():
    scenarios = []

    # 9a. 更多干扰层级的遗忘曲线
    distractors_100 = [
        {"action": "add_fact", "content": f"干扰事实{i}: 用户在某天做了某件事", "importance": 3}
        for i in range(1, 101)
    ]

    scenarios.append({
        "name": seed_name("遗忘曲线-100干扰深层衰减", 3),
        "setup": [
            {"action": "add_fact", "content": "用户的核心身份是数据科学家", "importance": 9, "tags": "身份", "category": "user_pref", "is_persistent": True},
            {"action": "add_fact", "content": "用户擅长用PyTorch做深度学习", "importance": 8, "tags": "技能", "category": "user_pref"},
        ],
        "queries": [
            {
                "query": "用户的职业是什么？", "type": "search", "params": {"scenario": "longterm"},
                "expected": ["数据科学家"], "unexpected": ["PyTorch"],
                "recall_at_k": [1, 3, 5],
            },
            {
                "query": "用户的职业是什么？", "type": "search", "params": {"scenario": "longterm"},
                "expected": ["数据科学家"], "unexpected": ["PyTorch"],
                "recall_at_k": [1, 3, 5],
                "extra_setup": distractors_100[:10],
            },
            {
                "query": "用户的职业是什么？", "type": "search", "params": {"scenario": "longterm"},
                "expected": ["数据科学家"], "unexpected": ["PyTorch"],
                "recall_at_k": [1, 3, 5],
                "extra_setup": distractors_100[:50],
            },
            {
                "query": "用户的职业是什么？", "type": "search", "params": {"scenario": "longterm"},
                "expected": ["数据科学家"], "unexpected": ["PyTorch"],
                "recall_at_k": [1, 3, 5],
                "extra_setup": distractors_100,
            },
            {
                "query": "用户擅长什么深度学习框架？", "type": "search", "params": {"scenario": "technical"},
                "expected": ["PyTorch"], "unexpected": ["数据科学家"],
                "recall_at_k": [1, 3],
            },
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 10. 联想记忆扩展
# ────────────────────────────────────────────────────────
def gen_associative_scenarios():
    scenarios = []

    # 10a. 联想记忆+干扰
    scenarios.append({
        "name": seed_name("联想记忆-带干扰的线索匹配", 2),
        "setup": [
            {"action": "add_fact", "content": "用户的座右铭是\"知行合一\"", "importance": 5, "tags": "格言", "category": "user_pref"},
            {"action": "add_fact", "content": "用户最喜欢的作家是村上春树", "importance": 6, "tags": "阅读", "category": "user_pref"},
            {"action": "add_fact", "content": "用户最喜欢的电影导演是诺兰", "importance": 6, "tags": "电影", "category": "user_pref"},
            {"action": "add_fact", "content": "用户最喜欢的歌手是周杰伦", "importance": 5, "tags": "音乐", "category": "user_pref"},
            {"action": "add_fact", "content": "用户的偶像是Elon Musk", "importance": 5, "tags": "人物", "category": "user_pref"},
            {"action": "add_fact", "content": "用户喜欢的球队是巴萨", "importance": 4, "tags": "体育", "category": "user_pref"},
            {"action": "add_fact", "content": "用户最喜欢的动漫是《进击的巨人》", "importance": 5, "tags": "动漫", "category": "user_pref"},
        ],
        "queries": [
            {"query": "用户的座右铭", "type": "search", "params": {"scenario": "chat"},
             "expected": ["知行合一"], "unexpected": ["村上春树", "诺兰", "巴萨"], "recall_at_k": [1, 3]},
            {"query": "用户最喜欢的作家", "type": "search", "params": {"scenario": "chat"},
             "expected": ["村上春树"], "unexpected": ["诺兰", "周杰伦", "Musk"], "recall_at_k": [1, 3]},
            {"query": "用户最喜欢的导演", "type": "search", "params": {"scenario": "chat"},
             "expected": ["诺兰"], "unexpected": ["知行合一", "巴萨", "巨人"], "recall_at_k": [1, 3]},
            {"query": "用户喜欢的歌手", "type": "search", "params": {"scenario": "chat"},
             "expected": ["周杰伦"], "unexpected": ["诺兰", "巴萨", "巨人"], "recall_at_k": [1, 3]},
            {"query": "用户的偶像", "type": "search", "params": {"scenario": "chat"},
             "expected": ["Elon Musk"], "unexpected": ["周杰伦", "村上春树", "巴萨"], "recall_at_k": [1, 3]},
        ]
    })

    return scenarios


# ────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────
ALL_GENERATORS = {
    "实体探针": gen_probe_scenarios,
    "基础检索": gen_retrieval_scenarios,
    "时间线": gen_timeline_scenarios,
    "过滤": gen_filter_scenarios,
    "跨会话": gen_cross_session_scenarios,
    "推理": gen_reasoning_scenarios,
    "跨语言": gen_crosslingual_scenarios,
    "更新": gen_update_scenarios,
    "遗忘曲线": gen_long_forgetting_scenarios,
    "联想记忆": gen_associative_scenarios,
}


def main():
    parser = argparse.ArgumentParser(description="生成扩展评测场景")
    parser.add_argument("--name", default="", help="只生成名字包含此关键词的维度")
    parser.add_argument("--output", default="", help="输出文件 (默认 stdout)")
    parser.add_argument("--merge", action="store_true", help="合并到 scenarios.json")
    args = parser.parse_args()

    all_scenarios = []
    for dim_name, gen_func in ALL_GENERATORS.items():
        if args.name and args.name not in dim_name and args.name not in [s["name"] for s in all_scenarios]:
            # Check if any generated scenario name matches
            dim_scenarios = gen_func()
            matching = [s for s in dim_scenarios if args.name in s["name"]]
            if not matching:
                continue
            all_scenarios.extend(matching)
        else:
            all_scenarios.extend(gen_func())

    output = json.dumps(all_scenarios, ensure_ascii=False, indent=2)

    if args.merge:
        base_path = Path(__file__).resolve().parent / "scenarios.json"
        with open(base_path, encoding="utf-8") as f:
            existing = json.load(f)
        total_before = len(existing)
        existing.extend(all_scenarios)
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"✅ 合并完成: {total_before} → {len(existing)} 场景 (+{len(all_scenarios)})")
        print(f"   总计查询数: {sum(len(s['queries']) for s in existing)}")
    elif args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"✅ 已写入 {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
