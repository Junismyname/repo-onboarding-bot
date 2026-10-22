from repobot.ingestion import chunker
from repobot.ingestion.chunker import chunk_file

README = """\
Intro line before any heading.

# Project

Short description.

## Getting Started

### 1. Install

```bash
# Create a virtual environment
python -m venv venv
pip install -r requirements.txt
```

### 2. Train

```bash
python scripts/train.py
```

## License
MIT
"""


def spans(chunks):
    return [(c.start_line, c.end_line, c.kind, c.symbol) for c in chunks]


def test_small_document_sections_stay_whole():
    chunks = chunk_file(README, "README.md", "markdown", "r")
    assert spans(chunks) == [
        (1, 2, "module", None),
        (3, 24, "section", "Project"),
    ]


def test_big_sections_split_into_subsections_with_heading_paths(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_SECTION_LINES", 12)
    chunks = chunk_file(README, "README.md", "markdown", "r")
    symbols = [c.symbol for c in chunks]
    assert symbols == [
        None,
        "Project",
        "Project > Getting Started > 1. Install",
        "Project > Getting Started > 2. Train",
        "Project > License",
    ]
    install = chunks[2]
    # the bash comment inside the fence is not a heading, so it stays in the Install section
    assert "# Create a virtual environment" in install.text
    assert "pip install -r requirements.txt" in install.text


def test_heading_only_intro_is_dropped(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_SECTION_LINES", 12)
    chunks = chunk_file(README, "README.md", "markdown", "r")
    # "## Getting Started" has no text of its own before "### 1. Install"
    assert "Project > Getting Started" not in [c.symbol for c in chunks]


def test_markdown_without_headings_falls_back_to_lines():
    chunks = chunk_file("just\nsome\nnotes\n", "notes.md", "markdown", "r")
    assert spans(chunks) == [(1, 3, "block", None)]
