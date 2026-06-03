#!/usr/bin/env python3
"""大规模压力测试 — 生成 500+ 事实的数据库，测试检索性能。

用法：
    python3 eval/gen_stress_test.py            # 默认 500 事实
    python3 eval/gen_stress_test.py --count 1000  # 1000 事实

输出：
    stress_test.json — 可直接被 run_eval.py 读取的场景文件

设计参考 LongMemEval / MEMOBENCH 的大规模检索评估方法。
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# 模板生成源
NAMES = ["用户", "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
CITIES = ["北京", "上海", "深圳", "杭州", "成都", "广州", "南京", "武汉", "西安", "重庆",
          "Tokyo", "New York", "London", "Berlin", "Paris", "Singapore", "Seoul"]
LANGUAGES = ["Python", "Rust", "Go", "TypeScript", "Java", "C++", "Ruby", "Kotlin",
             "Scala", "Elixir", "Zig", "Haskell"]
FRAMEWORKS = ["React", "Vue", "Django", "FastAPI", "Spring", "Flask", "Next.js",
              "Svelte", "PyTorch", "TensorFlow", "JAX", "Rocket"]
TOOLS = ["Docker", "Kubernetes", "Neovim", "VS Code", "tmux", "Git", "PostgreSQL",
         "Redis", "Nginx", "Prometheus", "Grafana", "Figma"]
HOBBIES = ["打羽毛球", "游泳", "跑步", "弹钢琴", "摄影", "品茶", "徒步", "滑雪",
           "烹饪", "画画", "写作", "读书", "旅行"]
FOODS = ["寿司", "拉面", "火锅", "披萨", "咖喱", "意面", "饺子", "烤肉", "沙拉",
         "海鲜", "甜点", "三明治"]
MUSIC = ["爵士乐", "古典乐", "摇滚", "电子音乐", "民谣", "嘻哈", "R&B", "流行"]

CATEGORIES = ["user_pref", "tool", "project", "general"]
TAGS_MAP = {
    "user_pref": ["偏好", "习惯", "技能", "爱好"],
    "tool": ["工具", "软件", "硬件", "配置"],
    "project": ["项目", "技术栈", "开发"],
    "general": ["经历", "旅行", "生活", "学习"],
}

TEMPLATES = [
    # 用户偏好
    lambda: f"{random.choice(NAMES)}最喜欢的编程语言是{random.choice(LANGUAGES)}",
    lambda: f"{random.choice(NAMES)}喜欢用{random.choice(TOOLS)}开发",
    lambda: f"{random.choice(NAMES)}住在{random.choice(CITIES)}",
    lambda: f"{random.choice(NAMES)}的爱好是{random.choice(HOBBIES)}",
    lambda: f"{random.choice(NAMES)}喜欢吃{random.choice(FOODS)}",
    lambda: f"{random.choice(NAMES)}喜欢听{MUSIC[random.randint(0, len(MUSIC)-1)]}",
    lambda: f"{random.choice(NAMES)}早上{random.randint(6, 9)}点起床",
    lambda: f"{random.choice(NAMES)}使用{random.choice(TOOLS)}作为主力工具",
    # 技能
    lambda: f"{random.choice(NAMES)}精通{random.choice(LANGUAGES)}和{random.choice(LANGUAGES)}",
    lambda: f"{random.choice(NAMES)}有{random.randint(3, 15)}年编程经验",
    lambda: f"{random.choice(NAMES)}熟悉{random.choice(FRAMEWORKS)}框架",
    lambda: f"{random.choice(NAMES)}会{random.choice(['说中英日三语', '说中文和英文', '说日文和英文'])}",
    # 项目
    lambda: f"{random.choice(NAMES)}的项目使用{random.choice(FRAMEWORKS)}",
    lambda: f"{random.choice(NAMES)}用{random.choice(TOOLS)}部署生产环境",
    lambda: f"{random.choice(NAMES)}的开源项目有{random.choice(['1000', '5000', '10000', '20000'])}星",
    lambda: f"{random.choice(NAMES)}使用{random.choice(['PostgreSQL', 'Redis', 'MongoDB'])}作为数据库",
    # 生活
    lambda: f"{random.choice(NAMES)}每周去健身房{random.randint(2, 5)}次",
    lambda: f"{random.choice(NAMES)}去过{random.choice(CITIES)}和{random.choice(CITIES)}旅行",
    lambda: f"{random.choice(NAMES)}养了一只{random.choice(['橘猫', '金毛', '布偶猫', '柯基'])}",
    lambda: f"{random.choice(NAMES)}在{random.choice(['阿里巴巴', '腾讯', 'Google', 'Microsoft', '字节跳动'])}工作",
    # 工具配置
    lambda: f"{random.choice(NAMES)}用{random.choice(['Neovim', 'VS Code', 'Vim'])}写代码",
    lambda: f"{random.choice(NAMES)}用{random.choice(['tmux', 'screen', 'zellij'])}管理终端",
    lambda: f"{random.choice(NAMES)}使用{random.choice(['Arch Linux', 'Ubuntu', 'macOS', 'NixOS'])}",
    lambda: f"{random.choice(NAMES)}用{random.choice(['i3', 'Hyprland', 'KDE', 'GNOME'])}窗口管理器",
]


def gen_fact(importance: int = None) -> dict:
    template = random.choice(TEMPLATES)
    content = template()
    cat = random.choice(CATEGORIES)
    return {
        "action": "add_fact",
        "content": content,
        "importance": importance or random.randint(2, 9),
        "tags": random.choice(TAGS_MAP[cat]),
        "category": cat,
    }


def gen_stress_scenario(count: int) -> dict:
    """Generate a stress test scenario with `count` facts."""
    random.seed(42)  # deterministic

    setup = []
    for _ in range(count):
        setup.append(gen_fact())

    # Pick a few specific facts as ground truth queries
    # Use fact indices that are spread across the range
    target_indices = [0, count // 4, count // 2, 3 * count // 4, count - 1]
    queries = []

    for idx in target_indices:
        fact = setup[idx]
        content = fact["content"]
        # Extract first meaningful keyword pair for query
        words = content.split()
        # Use first 2 meaningful tokens
        q_words = words[:3] if len(words) >= 3 else words[:2]
        query = " ".join(q_words)

        queries.append({
            "query": query,
            "type": "search",
            "params": {"scenario": "chat"},
            "expected": [content[:30]],  # match on substring of content
            "unexpected": [],
            "recall_at_k": [1, 3, 5],
        })

    return {
        "name": f"大规模压力-{count}事实检索",
        "setup": setup,
        "queries": queries,
    }


def main():
    parser = argparse.ArgumentParser(description="生成大规模压力测试场景")
    parser.add_argument("--count", type=int, default=500, help="事实数量 (默认: 500)")
    parser.add_argument("--output", default="stress_test.json", help="输出文件名")
    args = parser.parse_args()

    scenario = gen_stress_scenario(args.count)
    output_path = Path(__file__).resolve().parent / args.output

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([scenario], f, ensure_ascii=False, indent=2)

    print(f"✅ 生成 {args.count} 事实的压力测试场景 → {output_path}")
    print(f"   查询数: {len(scenario['queries'])}")


if __name__ == "__main__":
    main()
