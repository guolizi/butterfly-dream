#!/usr/bin/env python3
"""Extract English + Chinese from translated all_failed_74.md into a clean bilingual file."""
import re

src = "/home/xx/butterfly-dream/eval/runs/2026-06-11_0732_locomo_v8-merge/all_failed_74.md"
dst = "/home/xx/butterfly-dream/eval/runs/2026-06-11_0732_locomo_v8-merge/all_failed_74_双语对照.md"

text = open(src, encoding="utf-8").read()
lines = text.split("\n")

output = []
output.append("# Conv-26 三版全败题目 — 中英对照")
output.append("")
output.append(f"**共 74 道题** — 在 v6-re / v7-free-fix / v8-merge 三次评测中均未通过 (score < 4)")
output.append("")
output.append("---")
output.append("")

def clean_chinese(chinese):
    """Remove [cat_name], [question], etc. tags from Chinese text."""
    return re.sub(r'^\[(cat_name|question|gold|model_answer|answer)\]\s*', '', chinese).strip()

current_cat = ""
for line in lines:
    # Category header
    m = re.match(r'^## (cat\d: .+)（(.+)）（(\d+ 题)）$', line)
    if m:
        eng = m.group(1)
        chn = clean_chinese(m.group(2))
        current_cat = eng
        output.append(f"## {eng}")
        output.append(f"## {chn}")
        output.append(f"（{m.group(3)}）")
        output.append("")
        continue
    
    # Question
    m = re.match(r'^### (\d+\.\s+)(.+)（(.+)）$', line)
    if m:
        num = m.group(1)
        eng = m.group(2)
        chn = clean_chinese(m.group(3))
        output.append(f"### {num}{eng}")
        output.append(f"    {chn}")
        output.append("")
        continue
    
    # Standard answer
    m = re.match(r'^- \*\*标准答案\*\*:\s+(.+)（(.+)）$', line)
    if m:
        eng = m.group(1)
        chn = clean_chinese(m.group(2))
        output.append(f"- **标准答案**: {eng}")
        output.append(f"    {chn}")
        continue
    
    # Model answer
    m = re.match(r'^- \*\*模型回答 \(v8\)\*\*:\s+(.+)（(.+)）$', line)
    if m:
        eng = m.group(1)
        chn = clean_chinese(m.group(2))
        output.append(f"- **模型回答 (v8)**: {eng}")
        output.append(f"    {chn}")
        continue
    
    # Answer line
    m = re.match(r'^(Answer: .+)（(.+)）$', line)
    if m:
        eng = m.group(1)
        chn = clean_chinese(m.group(2))
        output.append(f"{eng}")
        output.append(f"    {chn}")
        continue
    
    # ID and score lines - keep as-is
    if line.startswith('- **ID**:') or line.startswith('- **得分**'):
        output.append(line)
    
    # Blank lines
    if line.strip() == '':
        output.append('')

open(dst, 'w', encoding='utf-8').write('\n'.join(output))
print(f"Written: {dst}")
print(f"Lines: {len(output)}")
