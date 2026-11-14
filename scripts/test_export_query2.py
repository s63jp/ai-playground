#!/usr/bin/env python3
"""Debug script to test the export query."""
import sqlite3
import json
import sys

db_path = "experiments/playground.db"

# Test 1: Basic query with mode filter only
print("=== TEST 1: Mode filter only ===")
conn = sqlite3.connect(db_path)
query = """
SELECT s.*, r.agent_id as agent_id
FROM sessions s
LEFT JOIN records r ON s.id = r.session_id AND r.kind='experiment' LIMIT 1
WHERE s.mode=?
ORDER BY s.action_count DESC, s.started_at ASC
"""
rows = conn.execute(query, ('exploration',)).fetchall()
print(f"Found {len(rows)} rows")
conn.close()
