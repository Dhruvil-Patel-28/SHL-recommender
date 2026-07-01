import json
import os
from rapidfuzz import fuzz
from typing import List, Dict

CATALOG: List[Dict] = []


def load_catalog():
    """Load the SHL catalog from the data/catalog.json file."""
    global CATALOG
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    with open(catalog_path) as f:
        raw = f.read()
    CATALOG = json.loads(raw, strict=False)
    print(f"Loaded {len(CATALOG)} catalog entries.")


def build_catalog_text(items: List[Dict]) -> str:
    """Build a compact text representation of catalog items for LLM context."""
    lines = []
    for item in items:
        keys_str = ",".join(item.get("keys", []))
        langs = item.get("languages", [])
        langs_str = ", ".join(langs[:2])
        if len(langs) > 2:
            langs_str += f" +{len(langs) - 2}"
        duration = item.get("duration", "") or ""
        desc = (item.get("description", "") or "")[:100]
        lines.append(
            f"{item['name']} | {keys_str} | {duration} | {langs_str} | "
            f"{desc} | {item['link']}"
        )
    return "\n".join(lines)



def filter_catalog(
    skills: List[str],
    job_levels: List[str],
    test_types: List[str],
    languages: List[str],
) -> List[Dict]:
    """
    Python-level filtering. No LLM involved.
    Returns a trimmed list of candidate catalog items based on extracted facets.
    """
    results = []

    for item in CATALOG:
        score = 0

        # Skill matching: check if any skill keyword appears in name or description
        item_text = (item["name"] + " " + (item.get("description", "") or "")).lower()
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in item_text:
                score += 10
            elif fuzz.partial_ratio(skill_lower, item_text) > 80:
                score += 5

        # Job level matching
        if job_levels:
            item_levels = [l.lower() for l in item.get("job_levels", [])]
            for level in job_levels:
                if any(level.lower() in il for il in item_levels):
                    score += 3

        # Test type matching (if user explicitly requested a type)
        if test_types:
            item_keys = [k.lower() for k in item.get("keys", [])]
            for t in test_types:
                if any(t.lower() in ik for ik in item_keys):
                    score += 5

        # Language matching
        if languages:
            item_langs = [l.lower() for l in item.get("languages", [])]
            for lang in languages:
                if any(lang.lower() in il for il in item_langs):
                    score += 2

        if score > 0:
            results.append((score, item))

    # Sort by score descending, take top 30
    results.sort(key=lambda x: x[0], reverse=True)
    top_items = [item for _, item in results[:30]]

    # Track entity_ids already included
    included_ids = {item["entity_id"] for item in top_items}

    # Always include Personality & Behavior and Ability & Aptitude items as fallback
    # These are commonly recommended (OPQ32r, Verify G+, DSI, etc.)
    personality_ability = [
        item
        for item in CATALOG
        if any(
            k in ["Personality & Behavior", "Ability & Aptitude"]
            for k in item.get("keys", [])
        )
        and item["entity_id"] not in included_ids
    ]

    combined = top_items + personality_ability[:20]
    return combined
