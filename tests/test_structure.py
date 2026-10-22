from repobot.models import Chunk, ScoredChunk
from repobot.retrieval.structure import in_file_order, mentioned_files, outline

PATHS = [
    "src/model.py",
    "scripts/train.py",
    "src/requests/sessions.py",
    "src/requests/models.py",
    "lib/axios.js",
    "a/__init__.py",
    "b/__init__.py",
]


def test_mentions_exact_fuzzy_and_paths():
    assert mentioned_files("What's the second class of model.py file?", PATHS) == ["src/model.py"]
    assert mentioned_files("compare train.py and model.py", PATHS) == [
        "scripts/train.py",
        "src/model.py",
    ]
    # typo / plural variant resolves to the closest name
    assert mentioned_files("what is in sesions.py", PATHS) == ["src/requests/sessions.py"]
    assert mentioned_files("open requests/models.py", PATHS) == ["src/requests/models.py"]


def test_mentions_ignore_ambiguous_and_non_files():
    assert mentioned_files("what does __init__.py export?", PATHS) == []
    assert mentioned_files("what does axios.create return?", PATHS) == []
    assert mentioned_files("how do sessions work?", PATHS) == []


def chunk(symbol, kind, start, end, parent=None, path="m.py"):
    return Chunk("r", path, "python", start, end, "x", kind, symbol, parent)


def test_outline_numbers_definitions_in_line_order():
    chunks = [
        chunk("MobileNetYOLO", "class", 39, 72),
        chunk(None, "module", 1, 3),
        chunk("SimpleYOLO", "class", 5, 37),
        chunk("helper", "function", 74, 80),
    ]
    text = outline("src/model.py", chunks)
    assert "1. SimpleYOLO (lines 5-37)" in text
    assert "2. MobileNetYOLO (lines 39-72)" in text
    assert "Functions (in file order):\n  1. helper (lines 74-80)" in text


def test_outline_merges_split_class_and_lists_methods():
    chunks = [
        chunk("Session", "class", 10, 20),
        chunk("Session.get", "method", 21, 30, parent="Session"),
        chunk("Session", "class", 31, 33),
        chunk("Session.post", "method", 34, 40, parent="Session"),
    ]
    text = outline("s.py", chunks)
    assert "1. Session (lines 10-33)" in text
    assert "methods: get, post" in text


def test_in_file_order_groups_and_sorts():
    def sc(path, start):
        return ScoredChunk(chunk(None, "block", start, start + 5, path=path), 0.5)

    ranked = [sc("b.py", 40), sc("a.py", 30), sc("b.py", 5), sc("a.py", 1)]
    ordered = in_file_order(ranked)
    assert [(s.chunk.path, s.chunk.start_line) for s in ordered] == [
        ("b.py", 5), ("b.py", 40), ("a.py", 1), ("a.py", 30),
    ]
    pinned = in_file_order(ranked, first_paths=["a.py"])
    assert [s.chunk.path for s in pinned][:2] == ["a.py", "a.py"]
