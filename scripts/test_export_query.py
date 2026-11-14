#!/usr/bin/env python3
import sqlite3
import json

db_path = "experiments/playground.db"

query = """
SELECT s.*, r.agent_id as agent_id
FROM sessions s
LEFT JOIN records r ON s.id = r.session_id AND r.kind='experiment' LIMIT 1
WHERE s.mode=?
ORDER BY s.action_count DESC, s.started_at ASC
"""

conn = sqlite3.connect(db_path)
rows = conn.execute(query, ('exploration',)).fetchall()
print(f"Found {len(rows)} rows") if rows else print("No rows")
conn.close()
