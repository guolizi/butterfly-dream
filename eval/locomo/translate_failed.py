#!/usr/bin/env python3
"""Translate English text in all_failed_74.md to Chinese, adding (中文) in parentheses.
Robust version: uses JSON dict (original -> translation) so count mismatches don't break it.
Rate-limited: one request per 3 seconds, small batches.
"""
import re, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_utils import call_llm

path = Path("/home/xx/butterfly-dream/eval/runs/2026-06-11_0732_locomo_v8-merge/all_failed_74.md")
text = path.read_text(encoding="utf-8")
lines = text.split("\n")

# Extract segments - store as (seg_type, eng_text, line_idx, extra_info)
segments = []
for i, line in enumerate(lines):
    m = re.match(r'^## (cat\d: .+)（(\d+ 题)）$', line)
    if m:
        segments.append(("cat_name", m.group(1), i, {"num": m.group(2)}))
        continue
    m = re.match(r'^### (\d+\.\s+)(.+)$', line)
    if m:
        segments.append(("question", m.group(2), i, {"prefix": m.group(1)}))
        continue
    m = re.match(r'^- \*\*标准答案\*\*:\s+(.+)$', line)
    if m:
        segments.append(("gold", m.group(1), i, {}))
        continue
    m = re.match(r'^- \*\*模型回答 \(v8\)\*\*:\s+(.+)$', line)
    if m:
        segments.append(("model_answer", m.group(1), i, {}))
        continue
    m = re.match(r'^(Answer: .+)$', line)
    if m:
        segments.append(("answer", m.group(1), i, {}))
        continue

print(f"Total segments: {len(segments)}")

# Process in small batches for reliability
batch_size = 20
translations = {}  # eng_text -> chinese

for batch_start in range(0, len(segments), batch_size):
    batch = segments[batch_start:batch_start + batch_size]
    
    # Build prompt: ask for JSON dict
    items_text = []
    for seg_type, eng_text, _, _ in batch:
        items_text.append(f"  \"{eng_text}\"")
    
    prompt = (
        "Translate each of the following English text segments to Chinese (Simplified).\n"
        "Return a JSON object where each key is the EXACT original English text and each value is its Chinese translation.\n"
        "Keep proper nouns (names, book titles, band names, dates) unchanged.\n"
        "For category names like 'single-session single-hop', translate the concept.\n"
        "For 'Answer:' lines, translate the answer content but keep 'Answer:' prefix.\n"
        f"Return EXACTLY {len(batch)} key-value pairs, no more, no less.\n\n"
        + "{\n" + ",\n".join(items_text) + "\n}"
    )
    
    batch_num = batch_start // batch_size + 1
    total_batches = (len(segments) - 1) // batch_size + 1
    print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} items)...")
    
    # Rate limit: wait 3s between requests
    if batch_start > 0:
        print("  Waiting 3s for rate limit...")
        time.sleep(3)
    
    result = call_llm("answer", messages=[
        {"role": "system", "content": "You are a translator. Return ONLY a JSON object, no other text."},
        {"role": "user", "content": prompt},
    ], max_tokens=8192, temperature=0, timeout=180)
    
    if not result:
        print(f"  FAILED (empty)!")
        continue
    
    # Extract JSON object
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if not json_match:
        print(f"  NO JSON: {result[:200]}")
        continue
    
    try:
        batch_translations = json.loads(json_match.group())
        matched = 0
        for seg_type, eng_text, _, _ in batch:
            if eng_text in batch_translations:
                translations[eng_text] = batch_translations[eng_text]
                matched += 1
            else:
                print(f"  MISSING: [{seg_type}] {eng_text[:60]}...")
        print(f"  OK: {matched}/{len(batch)} matched")
    except json.JSONDecodeError as e:
        print(f"  JSON PARSE: {e}")
        # Try to recover: maybe the model returned extra text
        print(f"  Raw: {result[:300]}")

print(f"\nTotal translated: {len(translations)}/{len(segments)}")

if len(translations) == len(segments):
    new_lines = list(lines)
    for seg_type, eng_text, line_idx, extra in segments:
        chinese = translations[eng_text]
        line = new_lines[line_idx]
        
        if seg_type == "cat_name":
            num = extra.get("num", "")
            new_lines[line_idx] = f"## {eng_text}（{chinese}）（{num}）"
        elif seg_type == "question":
            prefix = extra.get("prefix", "")
            new_lines[line_idx] = f"### {prefix}{eng_text}（{chinese}）"
        elif seg_type == "gold":
            new_lines[line_idx] = f"- **标准答案**: {eng_text}（{chinese}）"
        elif seg_type == "model_answer":
            new_lines[line_idx] = f"- **模型回答 (v8)**: {eng_text}（{chinese}）"
        elif seg_type == "answer":
            new_lines[line_idx] = f"{eng_text}（{chinese}）"
    
    path.write_text("\n".join(new_lines), encoding="utf-8")
    print("FILE UPDATED!")
else:
    print(f"MISMATCH: {len(translations)}/{len(segments)}")
    # Save partial for recovery
    out = path.with_suffix(".partial.json")
    out.write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Partial saved to {out}")
