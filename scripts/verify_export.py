#!/usr/bin/env python3
"""Verify export datasets validity and run HVE tests."""
import json
from pathlib import Path

# Check exported dataset
jsonl_path = "unsloth/qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2.jsonl"
print(f"=== Verifying {jsonl_path} ===")

with open(jsonl_path, 'r', newline='') as f:  # Handle CRLF
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print(f"File size: {Path(jsonl_path).stat().st_size} bytes")

# Check first line validity
first_line = json.loads(lines[0])
print(f"\nFirst example metadata:")
for k, v in first_line.get("metadata", {}).items():
    print(f"  {k}: {v}")

# Count examples with messages
examples_with_messages = sum(1 for l in lines if len(json.loads(l).get("messages", [])) > 0)
print(f"\nExamples with messages: {examples_with_messages}/{len(lines)}")
