"""Split source files into chunks: by syntax (tree-sitter) when possible, else line windows.

Syntax-aware chunking gives one chunk per function / class / method, so a chunk is a unit a
developer would ask about, and its symbol name ends up in the embedded text. Markdown is split
by heading sections for the same reason: "Getting Started" stays one chunk instead of being cut
across two 60-line windows.
"""

import re
from collections import defaultdict
from functools import cache

from repobot.models import Chunk, ChunkKind

TREE_SITTER_LANGUAGES = {"python", "javascript", "typescript", "tsx"}
MAX_CHUNK_LINES = 120  # longer definitions are split into line windows
CLASS_INLINE_LINES = 80  # classes up to this size stay one chunk; bigger ones split by method
MAX_SECTION_LINES = 80  # markdown sections up to this size (with subsections) stay one chunk

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MD_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")

DEF_KINDS: dict[str, ChunkKind] = {
    # Python
    "function_definition": "function",
    "class_definition": "class",
    # JavaScript / TypeScript
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "method_definition": "function",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "interface_declaration": "type",
    "type_alias_declaration": "type",
    "enum_declaration": "type",
}
FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}


def chunk_file(source: str, rel_path: str, language: str, repo_id: str) -> list[Chunk]:
    if language in TREE_SITTER_LANGUAGES:
        chunks = chunk_with_tree_sitter(source, rel_path, language, repo_id)
        if chunks:
            return chunks
    if language == "markdown":
        chunks = chunk_markdown(source, rel_path, language, repo_id)
        if chunks:
            return chunks
    return chunk_by_lines(source, rel_path, language, repo_id)


def chunk_markdown(source: str, rel_path: str, language: str, repo_id: str) -> list[Chunk]:
    """One chunk per heading section; a section stays whole (with its subsections) when it fits
    in MAX_SECTION_LINES, otherwise its intro and each subsection become separate chunks.

    The symbol is the heading path ("Getting Started > 3. Model Training"), which ends up in
    the embedded text. `#` lines inside code fences are shell comments, not headings.
    """
    lines = source.splitlines()
    headings: list[tuple[int, int, str]] = []  # (row, level, title)
    fence = None
    for row, line in enumerate(lines):
        if match := MD_FENCE_RE.match(line):
            if fence is None:
                fence = match.group(1)
            elif match.group(1) == fence:
                fence = None
            continue
        if fence is None and (heading := MD_HEADING_RE.match(line)):
            headings.append((row, len(heading.group(1)), heading.group(2)))
    if not headings:
        return []

    parents: list[int | None] = []
    stack: list[int] = []
    for i, (_, level, _) in enumerate(headings):
        while stack and headings[stack[-1]][1] >= level:
            stack.pop()
        parents.append(stack[-1] if stack else None)
        stack.append(i)
    children: dict[int | None, list[int]] = defaultdict(list)
    for i, parent in enumerate(parents):
        children[parent].append(i)

    def section_end(i: int) -> int:
        level = headings[i][1]
        later = (row for row, lvl, _ in headings[i + 1 :] if lvl <= level)
        return next(later, len(lines)) - 1

    def make(start: int, end: int, symbol: str | None) -> list[Chunk]:
        kind = "section" if symbol else "module"
        return _span_chunks(lines, start, end, rel_path, language, repo_id, kind, symbol, None)

    chunks: list[Chunk] = []

    def emit(i: int, path: list[str]) -> None:
        row, _, title = headings[i]
        end = section_end(i)
        name = " > ".join([*path, title])
        if end - row + 1 <= MAX_SECTION_LINES or not children[i]:
            chunks.extend(make(row, end, name))
            return
        first_child = headings[children[i][0]][0]
        if "".join(lines[row + 1 : first_child]).strip():  # intro text, not just the heading
            chunks.extend(make(row, first_child - 1, name))
        for child in children[i]:
            emit(child, [*path, title])

    if headings[0][0] > 0:
        chunks.extend(make(0, headings[0][0] - 1, None))  # text before the first heading
    for top in children[None]:
        emit(top, [])
    return chunks


@cache
def _parser(language: str):
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    grammars = {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
        "tsx": tree_sitter_typescript.language_tsx,
    }
    return Parser(Language(grammars[language]()))


def _definition(node):
    """(kind, name, inner_node) if `node` is a definition worth its own chunk, else None.

    Unwraps the wrappers that put a definition inside another node: Python decorators,
    JS/TS `export`, and `const handler = () => {...}`.
    """
    inner = node
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
    elif node.type == "export_statement":
        inner = node.child_by_field_name("declaration")
    if inner is None:
        return None

    if inner.type in ("lexical_declaration", "variable_declaration"):
        declarators = [c for c in inner.named_children if c.type == "variable_declarator"]
        if len(declarators) != 1:
            return None
        value = declarators[0].child_by_field_name("value")
        name = declarators[0].child_by_field_name("name")
        if value is None or name is None:
            return None
        if value.type in FUNCTION_VALUES:
            return "function", name.text.decode(), value
        if value.type == "class":
            return "class", name.text.decode(), value
        return None

    kind = DEF_KINDS.get(inner.type)
    if kind is None:
        return None
    name = inner.child_by_field_name("name")
    return kind, name.text.decode() if name else "<anonymous>", inner


def chunk_with_tree_sitter(source: str, rel_path: str, language: str, repo_id: str) -> list[Chunk]:
    tree = _parser(language).parse(source.encode("utf-8"))
    if tree.root_node.has_error and not tree.root_node.named_children:
        return []
    lines = source.splitlines()

    def make(start_row: int, end_row: int, kind: ChunkKind, symbol=None, parent=None):
        return _span_chunks(lines, start_row, end_row, rel_path, language, repo_id, kind, symbol, parent)

    chunks: list[Chunk] = []
    pending = []  # top-level nodes between definitions (imports, constants, comments, ...)

    def flush_pending():
        if pending:
            chunks.extend(make(pending[0].start_point[0], pending[-1].end_point[0], "module"))
            pending.clear()

    for node in tree.root_node.named_children:
        definition = _definition(node)
        if definition is None:
            pending.append(node)
            continue

        # Comments directly above a definition document it, so they belong to its chunk.
        start_row = node.start_point[0]
        while pending and pending[-1].type == "comment" and pending[-1].end_point[0] >= start_row - 1:
            start_row = pending.pop().start_point[0]
        flush_pending()

        kind, name, inner = definition
        end_row = node.end_point[0]
        if kind == "class" and end_row - start_row + 1 > CLASS_INLINE_LINES:
            chunks.extend(_split_class(inner, name, start_row, end_row, make))
        else:
            chunks.extend(make(start_row, end_row, kind, name))
    flush_pending()
    return chunks


def _split_class(class_node, class_name: str, start_row: int, end_row: int, make) -> list[Chunk]:
    """Big class -> a header chunk (signature, docstring, attributes) + one chunk per method."""
    body = class_node.child_by_field_name("body")
    members = body.named_children if body is not None else []
    methods = [(m, d) for m in members if (d := _definition(m)) and d[0] == "function"]
    if not methods:
        return make(start_row, end_row, "class", class_name)

    chunks = []
    first_method_row = methods[0][0].start_point[0]
    if first_method_row > start_row:
        chunks.extend(make(start_row, first_method_row - 1, "class", class_name))
    method_nodes = {m.id for m, _ in methods}
    for member in members:
        if member.start_point[0] < first_method_row:
            continue  # already in the header chunk
        if member.id in method_nodes:
            _, method_name, _ = _definition(member)
            chunks.extend(
                make(
                    member.start_point[0],
                    member.end_point[0],
                    "method",
                    f"{class_name}.{method_name}",
                    class_name,
                )
            )
        elif member.type != "comment":
            # Attributes declared between methods: keep them, attributed to the class.
            chunks.extend(make(member.start_point[0], member.end_point[0], "class", class_name))
    return chunks


def _span_chunks(
    lines: list[str],
    start_row: int,
    end_row: int,
    rel_path: str,
    language: str,
    repo_id: str,
    kind: ChunkKind,
    symbol: str | None,
    parent: str | None,
) -> list[Chunk]:
    """Chunk(s) for 0-based rows [start_row, end_row]; long spans become line windows."""
    text = "\n".join(lines[start_row : end_row + 1])
    if not text.strip():
        return []
    if end_row - start_row + 1 <= MAX_CHUNK_LINES:
        return [
            Chunk(
                repo_id=repo_id,
                path=rel_path,
                language=language,
                start_line=start_row + 1,
                end_line=end_row + 1,
                text=text,
                kind=kind,
                symbol=symbol,
                parent=parent,
            )
        ]
    pieces = chunk_by_lines(text, rel_path, language, repo_id, window=MAX_CHUNK_LINES, overlap=10)
    for piece in pieces:
        piece.start_line += start_row
        piece.end_line += start_row
        piece.kind, piece.symbol, piece.parent = kind, symbol, parent
    return pieces


def chunk_by_lines(
    source: str,
    rel_path: str,
    language: str,
    repo_id: str,
    window: int = 60,
    overlap: int = 10,
) -> list[Chunk]:
    lines = source.splitlines()
    chunks: list[Chunk] = []
    step = window - overlap
    for start in range(0, max(len(lines), 1), step):
        body = lines[start : start + window]
        if not "".join(body).strip():
            continue
        chunks.append(
            Chunk(
                repo_id=repo_id,
                path=rel_path,
                language=language,
                start_line=start + 1,
                end_line=start + len(body),
                text="\n".join(body),
            )
        )
        if start + window >= len(lines):
            break
    return chunks
