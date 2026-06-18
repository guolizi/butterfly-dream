#!/usr/bin/env python3
"""Test the §11 extraction prompt (v4 - grouped by dimension) on session_1."""
import re, json, os, sys, yaml, httpx

# 1. Read prompt template
with open('docs/v2-implementation-issues.md') as f:
    content = f.read()
m = re.search(r'```markdown\n(.*?)```', content, re.DOTALL)
template = m.group(1)

# 2. Read session_1 dialog
with open('eval/locomo/data/locomo10.json') as f:
    data = json.load(f)
for item in data:
    sid = item.get('sample_id', '')
    if '26' in str(sid):
        conv = item.get('conversation', {})
        s1 = conv.get('session_1', [])
        lines = [f"[{t['speaker']}]: {t['text']}" for t in s1]
        dialog = '\n'.join(lines)
        break

# 3. Fill template
filled = template.replace('{person}', 'Caroline')
filled = filled.replace('{other_person}', 'Melanie')
filled = filled.replace('{session_time}', '1:56 pm on 8 May, 2023')
full_prompt = filled + '\n\n## 对话内容\n\n[Session date: 2023-05-08]\n' + dialog

# 4. Get API key
api_key = os.environ.get('ARK_CODING_API_KEY', '')
if not api_key:
    with open(os.path.expanduser('~/.hermes/config.yaml')) as f:
        cfg = yaml.safe_load(f)
    custom = cfg.get('custom_providers', [])
    if isinstance(custom, dict):
        for name, info in custom.items():
            if 'coding' in name.lower():
                api_key = info.get('api_key', '')
                break
    elif isinstance(custom, list):
        for item in custom:
            if isinstance(item, dict) and 'coding' in str(item.get('name', '')).lower():
                api_key = item.get('api_key', '')
                break

if not api_key:
    print("ERROR: No API key")
    sys.exit(1)

print(f"Prompt: {len(full_prompt)} chars")
print(f"Template: {len(template)} chars")
print(f"Dialog: {len(dialog)} chars")
print("=" * 60)

# 5. Call API
client = httpx.Client(timeout=180)
resp = client.post(
    'https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions',
    json={
        'model': 'deepseek-v4-flash',
        'messages': [
            {'role': 'system', 'content': 'You are a meticulous fact extractor. Output ONLY valid JSON. No markdown, no explanations, no extra text.'},
            {'role': 'user', 'content': full_prompt},
        ],
        'temperature': 0.2,
        'max_tokens': 8192,
    },
    headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
)

data = resp.json()
content = data['choices'][0]['message']['content']
print(f"Response: {len(content)} chars")
print("=" * 60)
print(content)
