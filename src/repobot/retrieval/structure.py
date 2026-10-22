"""File-structure helpers: which files a question names, their outlines, and file-order sorting.

Semantic search ranks chunks by similarity, so "the second class in model.py" can't be answered
from search results alone: the model sees chunks in relevance order and counts in that order.
The fix is to give it the file's real structure (an outline in line order) and to present code
in file order.
"""

import difflib
import re
from collections import defaultdict
from pathlib import PurePosixPath

from repobot.models import Chunk, ScoredChunk

# Things that look like file names: "model.py", "src/requests/sessions.py", "index.d.ts"
FILE_MENTION_RE = re.compile(r"(?<![\w/.-])([\w./-]*\w\.(?:[a-zA-Z]{1,4}))(?![\w])")
FUZZY_CUTOFF = 0.85  # "model.py" ~ "models.py" (0.94), but not "axios.create" ~ "axios.js"
MAX_FILES = 2


def mentioned_files(question: str, paths: list[str]) -> list[str]:
    """Indexed paths the question refers to by name. Ambiguous names (several matches) are skipped."""
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        by_name[PurePosixPath(path).name.lower()].append(path)

    found: list[str] = []
    for mention in FILE_MENTION_RE.findall(question):
        mention = mention.strip("./").lower()
        if "/" in mention:
            matches = [p for p in paths if p.lower().endswith(mention)]
        else:
            matches = by_name.get(mention) or []
            if not matches:
                close = difflib.get_close_matches(mention, by_name, n=1, cutoff=FUZZY_CUTOFF)
                matches = by_name[close[0]] if close else []
        if len(matches) == 1 and matches[0] not in found:
            found.append(matches[0])
    return found[:MAX_FILES]


def outline(path: str, chunks: list[Chunk]) -> str:
    """Numbered classes and functions of one file in line order (methods listed under classes)."""
    spans: dict[tuple[str, str], list[int]] = {}
    methods: dict[str, list[str]] = defaultdict(list)
    for c in sorted(chunks, key=lambda c: c.start_line):
        if c.parent is None and c.symbol and c.kind in ("class", "function", "type"):
            span = spans.setdefault((c.kind, c.symbol), [c.start_line, c.end_line])
            span[1] = max(span[1], c.end_line)
        elif c.kind == "method" and c.symbol and c.parent:
            name = c.symbol.split(".", 1)[-1]
            if name not in methods[c.parent]:
                methods[c.parent].append(name)

    sections = []
    for kind, title in (("class", "Classes"), ("function", "Functions"), ("type", "Types")):
        items = sorted((start, end, sym) for (k, sym), (start, end) in spans.items() if k == kind)
        if not items:
            continue
        lines = [f"{title} (in file order):"]
        for i, (start, end, sym) in enumerate(items, start=1):
            lines.append(f"  {i}. {sym} (lines {start}-{end})")
            if methods.get(sym):
                lines.append(f"     methods: {', '.join(methods[sym])}")
        sections.append("\n".join(lines))
    body = "\n".join(sections) or "(no classes or functions)"
    return f"### {path}\n{body}"


def in_file_order(chunks: list[ScoredChunk], first_paths: list[str] = ()) -> list[ScoredChunk]:
    """Group chunks by file and sort each group by line number.

    Files listed in `first_paths` come first; the remaining files keep the order of their
    best-ranked chunk, so the most relevant file is still near the top.
    """
    groups: dict[str, list[ScoredChunk]] = {}
    for sc in chunks:  # dicts keep insertion order = order of each file's best hit
        groups.setdefault(sc.chunk.path, []).append(sc)
    order = [p for p in first_paths if p in groups] + [p for p in groups if p not in first_paths]
    return [
        sc
        for path in order
        for sc in sorted(groups[path], key=lambda sc: sc.chunk.start_line)
    ]
