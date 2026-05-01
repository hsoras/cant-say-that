# load_skills.py
import json
from pathlib import Path

def load_skills(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    skills = {}
    for top_key in ("low_level_skills", "high_level_patterns"):
        section = data.get(top_key) or {}
        for group in section.values():
            for item in group:
                sid = item["id"]
                skills[sid] = {
                    "name": item["name"],
                    "description": item["description"],
                    "turns": item.get("turns", []), # High-level blueprints
                    "example_prompt": item.get("example_prompt", "") # Low-level examples
                }
    return skills
