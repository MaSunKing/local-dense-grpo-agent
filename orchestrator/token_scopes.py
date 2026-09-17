"""Exact generated-token scopes from a frozen tokenizer and character protocol."""
from __future__ import annotations

import re
from pathlib import Path

from .common import digest, file_sha256


TOKEN_SCOPE_PROTOCOL = "joint_exact_completion_spans_v1"
ACTION_RE = re.compile(
    r'<call_tool\s+name="(?P<name>[^"]+)"(?P<attrs>[^>]*)>'
    r'(?P<body>.*?)</call_tool>', re.DOTALL,
)
QUERY_RE = re.compile(r'\bquery="(?P<query>[^"]*)"')


def tokenizer_identity(model_dir: Path) -> str:
    rows = {}
    for pattern in (
        "tokenizer*.json", "special_tokens_map.json",
        "added_tokens.json", "chat_template.jinja",
    ):
        for path in sorted(Path(model_dir).glob(pattern)):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe tokenizer file: {path}")
            rows[path.name] = file_sha256(path)
    if not rows:
        raise ValueError("tokenizer identity files are missing")
    return digest(rows)


def _overlap(offsets, spans):
    return [
        index for index, (start, end) in enumerate(offsets)
        if end > start and any(start < right and end > left for left, right in spans)
    ]


def _element(raw: str, opening: str, closing: str):
    start = raw.find(opening)
    end_start = raw.rfind(closing)
    if start < 0 or end_start < start or raw.find(opening, start + 1) >= 0:
        raise ValueError(f"non-unique exact element: {opening}")
    return start, end_start + len(closing)


def character_spans(kind: str, raw: str):
    if kind in {"checklist", "state"}:
        return {kind: [(0, len(raw))]}
    if kind == "final":
        return {"final": [_element(raw, "<answer>", "</answer>")]}
    if kind == "stop":
        marker = "FINAL_READY"
        start = raw.find(marker)
        if start < 0 or raw.find(marker, start + 1) >= 0:
            raise ValueError("Stop must contain exactly one FINAL_READY")
        return {"stop": [(start, start + len(marker))]}
    if kind not in {"search", "browse"}:
        raise ValueError(f"unsupported canonical kind: {kind}")
    matches = list(ACTION_RE.finditer(raw))
    if len(matches) != 1:
        raise ValueError("decision must contain exactly one tool action")
    match = matches[0]
    semantic = [(match.start("body"), match.end("body"))]
    name = match.group("name")
    if kind == "search":
        if name not in {"pubmed_search", "medical_web_search"}:
            raise ValueError("Search record has non-search action")
        semantic_channel = "search_query"
    else:
        if name not in {"browse_document", "browse_webpage"}:
            raise ValueError("Browse record has non-browse action")
        queries = list(QUERY_RE.finditer(match.group("attrs")))
        if len(queries) != 1:
            raise ValueError("Browse action requires exactly one focus query")
        base = match.start("attrs")
        semantic.append((base + queries[0].start("query"), base + queries[0].end("query")))
        semantic_channel = "browse_source_focus"
    boundaries = {match.start(), match.end()}
    for left, right in semantic:
        boundaries.update((left, right))
    ordered = sorted(boundaries)
    tool = []
    for left, right in zip(ordered, ordered[1:]):
        if left != right and not any(left >= a and right <= b for a, b in semantic):
            tool.append((left, right))
    return {"tool": tool, semantic_channel: semantic}


def derive(tokenizer, capture: dict, kind: str) -> dict[str, list[int]]:
    encoded = tokenizer(
        capture["audit_completion"], add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if list(encoded["input_ids"]) != capture["output_ids"]:
        raise ValueError("completion token replay mismatch")
    offsets = [tuple(map(int, row)) for row in encoded["offset_mapping"]]
    channels, used = {}, set()
    for channel, spans in character_spans(kind, capture["raw_completion"]).items():
        indices = _overlap(offsets, spans)
        if not indices:
            raise ValueError(f"empty token scope: {channel}")
        if used.intersection(indices):
            raise ValueError(f"overlapping token scope: {channel}")
        used.update(indices)
        channels[channel] = indices
    return channels

