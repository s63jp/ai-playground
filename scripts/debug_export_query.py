#!/usr/bin/env python3
"""Debug script to test export queries step by step."""
import sqlite3
import json

db_path = "<repo>/experiments/playground.db"
adapter = "lmstudio_claude_distilled9b"
mode_filter = "exploration"

# Simulate the sessions_by_adapter_model function step by step
print("=== Simulating sessions_by_adapter_model ===")

where_parts = [f"s.mode=?"]
params = [mode_filter]
print(f"Initial where_parts: {where_parts}")
print(f"Initial params: {params}")

if adapter:
    where_parts.append("s.adapter=?")
    params.append(adapter)
    print(f"After adding adapter - where_parts: {where_parts}")
    print(f"After adding adapter - params: {params}")

sql = "SELECT s.*, r.agent_id as agent_id FROM sessions s LEFT JOIN records r ON s.id = r.session_id AND r.kind='experiment' LIMIT 1 WHERE s.mode=? AND s.adapter=? ORDER BY s.action_count DESC, s.started_at ASC"

print(f"Generated SQL: {sql}")
print(f"SQL params: {params}")

conn = sqlite3.connect(db_path)
try:
    rows = conn.execute(sql, tuple(params)).fetchall()
except Exception as e:
    print(f"Error: {e}")
    
    # Try without the LIMIT 1 first
    sql2 = "SELECT s.*, r.agent_id as agent_id FROM sessions s WHERE s.mode=? AND s.adapter=? ORDER BY s.action_count DESC, s.started_at ASC LIMIT 10"
    print(f"\nTrying SQL without pre-LIMIT: {sql2}")
    rows = conn.execute(sql2, tuple(params)).fetchall()

conn.close()

print(f"\nFound {len(rows)} rows")
if rows:
    for i, row in enumerate(rows):
        print(f"Row {i}: id={row['id']} adapter={row.get('adapter')} mode={row.get('mode')[:20]}...")
