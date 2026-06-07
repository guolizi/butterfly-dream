"""bd_eval scenarios: conversation test cases for extraction + retrieval.

Each scenario has a conversation (single or multi-session), golden facts
that should be extracted, and natural-language queries with expected answers.

These are used by test_bd_scenarios.py for both:
- Unit tests (mocked LLM, testing pipeline logic)
- Integration tests (real LLM, tagged @pytest.mark.integration)
"""

# Scenarios that test single-session extraction
SINGLE_SESSION_SCENARIOS = [
    {
        "name": "用户偏好-编程语言偏好变化",
        "conversation": [
            {"role": "user", "content": "我最近在学习Rust，感觉比Python快很多"},
            {"role": "assistant", "content": "Rust的性能确实很出色，你之前主要用什么语言？"},
            {"role": "user", "content": "之前一直用Python写后端，上个月开始切换到Rust了"},
        ],
        "golden_facts": [
            "用户之前用Python写后端开发",
            "用户上个月从Python切换到Rust",
            "用户认为Rust比Python快很多",
        ],
        "queries": [
            ("用户之前用什么语言写后端？", ["Python"]),
            ("用户现在用什么语言？", ["Rust"]),
            ("用户为什么切换语言？", ["快"]),
        ],
    },
    {
        "name": "用户信息-工作与居住地",
        "conversation": [
            {"role": "user", "content": "我刚搬到东京，在Google开始新工作了"},
            {"role": "assistant", "content": "恭喜！东京是个很不错的城市。Google在东京有办公室吗？"},
            {"role": "user", "content": "有的，我在涩谷的办公室工作，团队主要做搜索相关"},
        ],
        "golden_facts": [
            "用户搬到了日本东京生活",
            "用户在Google东京办公室工作",
            "用户的团队在涩谷做搜索相关开发",
        ],
        "queries": [
            ("用户住在哪个城市？", ["东京"]),
            ("用户在哪个公司工作？", ["Google"]),
            ("用户的团队做什么？", ["搜索"]),
        ],
    },
    {
        "name": "项目信息-技术栈",
        "conversation": [
            {"role": "user", "content": "我们新项目准备用FastAPI写后端，前端用React"},
            {"role": "assistant", "content": "很好的技术选型。数据库选了什么？"},
            {"role": "user", "content": "PostgreSQL，部署在AWS上，用Docker管理"},
            {"role": "assistant", "content": "你们用Kubernetes吗？"},
            {"role": "user", "content": "暂时不用K8s，直接用ECS部署，以后可能会迁移"},
        ],
        "golden_facts": [
            "新项目后端使用FastAPI框架",
            "新项目前端使用React框架",
            "项目使用PostgreSQL数据库",
            "项目部署在AWS ECS上使用Docker",
        ],
        "queries": [
            ("项目后端用什么框架？", ["FastAPI"]),
            ("项目用什么数据库？", ["PostgreSQL"]),
            ("项目部署在哪里？", ["AWS"]),
        ],
    },
    {
        "name": "多轮对话-偏好积累",
        "conversation": [
            {"role": "user", "content": "我养了一只橘猫，超级可爱"},
            {"role": "assistant", "content": "橘猫很亲人！叫什么名字？"},
            {"role": "user", "content": "叫小胖，已经三岁了，最喜欢吃鱼"},
            {"role": "assistant", "content": "哈哈，橘猫果然爱吃。你还有其他宠物吗？"},
            {"role": "user", "content": "还有一条金毛，叫旺财，它们相处得很好"},
        ],
        "golden_facts": [
            "用户养了一只三岁的橘猫叫小胖",
            "用户还养了一条金毛犬叫旺财",
            "小胖最喜欢吃鱼",
        ],
        "queries": [
            ("用户养了什么宠物？", ["橘猫", "金毛"]),
            ("橘猫叫什么名字？", ["小胖"]),
            ("金毛叫什么名字？", ["旺财"]),
        ],
    },
    # ── English ──
    {
        "name": "User Preferences - Language Preference Shift",
        "conversation": [
            {"role": "user", "content": "I've been learning Rust lately, and it feels much faster than Python"},
            {"role": "assistant", "content": "Rust is known for its performance. What language were you using before?"},
            {"role": "user", "content": "I used to write backend services in Python, but I switched to Rust last month"},
        ],
        "golden_facts": [
            "User previously used Python for backend development",
            "User switched from Python to Rust last month",
            "User thinks Rust is faster than Python",
        ],
        "queries": [
            ("What language did the user write backend in before?", ["Python"]),
            ("What language does the user use now?", ["Rust"]),
            ("Why did the user switch languages?", ["fast"]),
        ],
    },
    {
        "name": "User Info - Work and Location",
        "conversation": [
            {"role": "user", "content": "I just moved to Tokyo and started a new job at Google"},
            {"role": "assistant", "content": "Congratulations! Tokyo is a great city. Does Google have an office there?"},
            {"role": "user", "content": "Yes, I work in the Shibuya office, the team mainly works on search"},
        ],
        "golden_facts": [
            "User moved to Tokyo, Japan",
            "User works at Google's Tokyo office",
            "User's team works on search in Shibuya",
        ],
        "queries": [
            ("What city does the user live in?", ["Tokyo"]),
            ("What company does the user work at?", ["Google"]),
            ("What does the user's team work on?", ["search"]),
        ],
    },
    {
        "name": "Project Info - Tech Stack",
        "conversation": [
            {"role": "user", "content": "We're building the new project with a FastAPI backend and React frontend"},
            {"role": "assistant", "content": "Good tech choices. What database are you using?"},
            {"role": "user", "content": "PostgreSQL, deployed on AWS with Docker"},
            {"role": "assistant", "content": "Are you using Kubernetes?"},
            {"role": "user", "content": "Not yet, just ECS for now, might migrate later"},
        ],
        "golden_facts": [
            "New project backend uses FastAPI framework",
            "New project frontend uses React",
            "Project uses PostgreSQL database",
            "Project is deployed on AWS with Docker",
        ],
        "queries": [
            ("What framework does the project backend use?", ["FastAPI"]),
            ("What database does the project use?", ["PostgreSQL"]),
            ("Where is the project deployed?", ["AWS"]),
        ],
    },
    {
        "name": "Multi-turn - Preference Accumulation",
        "conversation": [
            {"role": "user", "content": "I adopted an orange tabby cat, she's adorable"},
            {"role": "assistant", "content": "Orange tabbies are so friendly! What's her name?"},
            {"role": "user", "content": "Her name is Mochi, she's three years old and loves eating fish"},
            {"role": "assistant", "content": "Mochi sounds cute! Do you have any other pets?"},
            {"role": "user", "content": "Yes, a golden retriever named Biscuit, they get along really well"},
        ],
        "golden_facts": [
            "User has a three-year-old orange tabby cat named Mochi",
            "User also has a golden retriever named Biscuit",
            "Mochi loves eating fish",
        ],
        "queries": [
            ("What pets does the user have?", ["tabby", "golden retriever"]),
            ("What is the cat's name?", ["Mochi"]),
            ("What is the dog's name?", ["Biscuit"]),
        ],
    },
    {
        "name": "Health and Dietary Constraints",
        "conversation": [
            {"role": "user", "content": "I have a severe nut allergy, so I have to be really careful when eating out"},
            {"role": "assistant", "content": "That must be challenging. Are there other dietary restrictions?"},
            {"role": "user", "content": "I'm also lactose intolerant, but I can take lactase pills"},
            {"role": "user", "content": "My doctor said my vitamin D levels are low, so I take supplements daily"},
        ],
        "golden_facts": [
            "User has a severe nut allergy",
            "User is lactose intolerant",
            "User takes vitamin D supplements daily",
        ],
        "queries": [
            ("What food allergies does the user have?", ["nut"]),
            ("Can the user eat dairy?", ["lactose"]),
            ("What supplements does the user take?", ["vitamin D"]),
        ],
    },
    {
        "name": "Weekly Schedule and Habits",
        "conversation": [
            {"role": "user", "content": "I work from home on Mondays and Fridays, in-office the rest of the week"},
            {"role": "assistant", "content": "That's a nice hybrid setup. What does your typical morning look like?"},
            {"role": "user", "content": "I usually run 5k before work, then have a protein smoothie"},
            {"role": "user", "content": "On weekends I do long hikes in the mountains with my partner"},
        ],
        "golden_facts": [
            "User works from home on Mondays and Fridays",
            "User runs 5k every morning before work",
            "User drinks a protein smoothie after running",
            "User goes hiking in the mountains on weekends",
        ],
        "queries": [
            ("When does the user work remotely?", ["Monday", "Friday"]),
            ("What exercise does the user do in the morning?", ["run"]),
            ("What does the user do on weekends?", ["hiking"]),
        ],
    },
]

# Multi-session scenarios (cross-session information tracking)
MULTI_SESSION_SCENARIOS = [
    {
        "name": "Cross-session - Job Change",
        "sessions": [
            {
                "conversation": [
                    {"role": "user", "content": "我目前在微软工作，做Azure相关的开发"},
                    {"role": "assistant", "content": "Azure是个很庞大的平台，你主要负责哪部分？"},
                    {"role": "user", "content": "主要是Kubernetes相关的服务"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "我刚跳槽到字节跳动了，现在做TikTok推荐系统"},
                    {"role": "assistant", "content": "从Azure到推荐系统，跨度挺大的"},
                    {"role": "user", "content": "是的，不过技术栈还是用Go和Rust"},
                ],
            },
        ],
        "golden_facts": [
            "用户在微软做Azure开发",
            "用户主要负责Kubernetes相关服务",
            "用户跳槽到了字节跳动做TikTok推荐系统",
            "用户的技术栈包含Go和Rust",
        ],
        "queries": [
            ("用户现在在哪个公司？", ["字节跳动"]),
            ("用户以前在哪个公司？", ["微软"]),
            ("用户用什么编程语言？", ["Go", "Rust"]),
            ("用户现在做什么方向？", ["TikTok", "推荐系统"]),
        ],
    },
    {
        "name": "Cross-session - Job Change (English)",
        "sessions": [
            {
                "conversation": [
                    {"role": "user", "content": "I currently work at Microsoft, doing Azure development"},
                    {"role": "assistant", "content": "Azure is a huge platform, what part do you work on?"},
                    {"role": "user", "content": "Mostly Kubernetes-related services"},
                ],
            },
            {
                "conversation": [
                    {"role": "user", "content": "I just switched to Meta, now working on Instagram recommendation systems"},
                    {"role": "assistant", "content": "That's quite a shift from Azure to recommendations"},
                    {"role": "user", "content": "Yes, but the tech stack still uses Go and Rust"},
                ],
            },
        ],
        "golden_facts": [
            "User worked at Microsoft on Azure",
            "User mainly handled Kubernetes-related services",
            "User switched to Meta working on Instagram recommendation systems",
            "User's tech stack includes Go and Rust",
        ],
        "queries": [
            ("Where does the user work now?", ["Meta"]),
            ("Where did the user work before?", ["Microsoft"]),
            ("What programming languages does the user use?", ["Go", "Rust"]),
            ("What does the user work on now?", ["Instagram", "recommendation"]),
        ],
    },
]
