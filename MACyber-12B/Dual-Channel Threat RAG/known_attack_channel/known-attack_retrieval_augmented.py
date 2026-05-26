"""Known-attack RAG context builder for MACyber generation scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_BASE = Path(__file__).resolve().parent / "known_attack_RAG.json"

CATEGORY_ALIASES = {
    "network traffic security": "network-traffic",
    "network-traffic": "network-traffic",
    "iot security": "iot",
    "iot": "iot",
    "system log security": "log",
    "log": "log",
    "dns security threat": "dns",
    "dns": "dns",
    "web security threat": "url",
    "url": "url",
    "vulnerability intelligence": "vulnerability",
    "vulnerability": "vulnerability",
    "threat intelligence": "threat",
    "threat": "threat",
}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_category(value: Any) -> str:
    raw = _normalize(value)
    return CATEGORY_ALIASES.get(raw, raw)


def load_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_BASE) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"RAG knowledge base must be a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _format_case(item: dict[str, Any], index: int) -> str:
    meta = item.get("meta", {})
    label = item.get("label", {})
    reasoning = item.get("reasoning", {})
    response = item.get("response", {})
    payload = {
        "meta": meta,
        "label": {
            "official": label.get("official"),
            "severity": label.get("severity"),
        },
        "reasoning": {
            "evidence": reasoning.get("evidence", []),
            "analysis": reasoning.get("analysis", ""),
        },
        "response": {
            "action": response.get("action"),
            "reason": response.get("reason", ""),
        },
    }
    return f"Example {index}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def build_rag_context(
    meta: dict[str, Any] | None,
    knowledge_base_path: str | Path = DEFAULT_KNOWLEDGE_BASE,
    top_k: int = 3,
) -> str:
    """Return a prompt-ready reference block for the current sample metadata."""
    if not meta or top_k <= 0:
        return ""

    category = _normalize_category(meta.get("category"))
    subcategory = _normalize(meta.get("subcategory"))
    knowledge_base = load_knowledge_base(knowledge_base_path)

    exact_matches = []
    category_matches = []
    for item in knowledge_base:
        item_meta = item.get("meta", {})
        item_category = _normalize_category(item_meta.get("category"))
        item_subcategory = _normalize(item_meta.get("subcategory"))
        if subcategory and item_subcategory == subcategory:
            exact_matches.append(item)
        elif category and item_category == category:
            category_matches.append(item)

    selected = (exact_matches + category_matches)[:top_k]
    if not selected:
        return ""
    return "\n\n".join(_format_case(item, idx) for idx, item in enumerate(selected, 1))
