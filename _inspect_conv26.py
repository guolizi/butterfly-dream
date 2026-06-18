#!/usr/bin/env python3
"""Inspect conv-26 details."""
import json

with open('eval/locomo/data/locomo10.json', 'r') as f:
    data = json.load(f)

conv26 = [d for d in data if d["sample_id"] == "conv-26"][0]

print("Speakers:", conv26["conversation"]["speaker_a"], conv26["conversation"]["speaker_b"])

# Count total turns
total_turns = 0
for k, v in conv26["conversation"].items():
    if k.startswith("session_") and isinstance(v, list):
        total_turns += len(v)
print(f"Total sessions: 35 (session_1 to session_35)")
print(f"Total turns: {total_turns}")

# Show session_1
print("\n=== Session 1 ===")
s1 = conv26["conversation"]["session_1"]
print(f"Date: {conv26['conversation']['session_1_date_time']}")
print(f"Turns: {len(s1)}")
for i, turn in enumerate(s1):
    print(f"  [{i+1}] {turn['speaker']}: {turn['text'][:100]}")

# Show session_2
print("\n=== Session 2 ===")
s2 = conv26["conversation"]["session_2"]
print(f"Date: {conv26['conversation']['session_2_date_time']}")
print(f"Turns: {len(s2)}")
for i, turn in enumerate(s2):
    print(f"  [{i+1}] {turn['speaker']}: {turn['text'][:100]}")
