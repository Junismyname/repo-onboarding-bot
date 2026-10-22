import textwrap

import pytest

pytest.importorskip("tree_sitter_python")

from repobot.ingestion import chunker
from repobot.ingestion.chunker import chunk_file

PY_SOURCE = textwrap.dedent(
    '''\
    """Module docstring."""
    import os

    TIMEOUT = 30


    # Builds the greeting.
    @cache
    def greet(name):
        return f"hi {name}"


    class Small:
        x = 1

        def method(self):
            return self.x
    '''
)


def by_symbol(chunks):
    return {c.symbol: c for c in chunks}


def test_python_definitions_become_chunks():
    chunks = chunk_file(PY_SOURCE, "m.py", "python", "r")
    kinds = [(c.kind, c.symbol) for c in chunks]
    assert kinds == [("module", None), ("function", "greet"), ("class", "Small")]

    module, greet, small = chunks
    assert (module.start_line, module.end_line) == (1, 4)
    # decorator and the comment directly above belong to the function
    assert (greet.start_line, greet.end_line) == (7, 10)
    assert greet.text.startswith("# Builds the greeting.\n@cache")
    assert small.text.rstrip().endswith("return self.x")


def test_big_class_is_split_into_header_and_methods(monkeypatch):
    monkeypatch.setattr(chunker, "CLASS_INLINE_LINES", 3)
    source = textwrap.dedent(
        '''\
        class Session:
            """Persists settings."""
            timeout = 5

            def get(self, url):
                return url

            retries = 3

            @property
            def proxies(self):
                return {}
        '''
    )
    chunks = chunk_file(source, "s.py", "python", "r")
    assert [(c.kind, c.symbol, c.parent) for c in chunks] == [
        ("class", "Session", None),
        ("method", "Session.get", "Session"),
        ("class", "Session", None),  # attribute between methods is kept
        ("method", "Session.proxies", "Session"),
    ]
    assert chunks[0].text.splitlines()[-1].strip() == "timeout = 5"
    assert chunks[3].text.lstrip().startswith("@property")


def test_long_function_is_split_but_keeps_symbol(monkeypatch):
    monkeypatch.setattr(chunker, "MAX_CHUNK_LINES", 20)
    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    chunks = chunk_file(f"def big():\n{body}\n", "b.py", "python", "r")
    assert len(chunks) > 1
    assert all(c.symbol == "big" and c.kind == "function" for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 51


def test_typescript_exports_arrow_functions_and_types():
    source = textwrap.dedent(
        """\
        import { db } from "./db";

        export interface User {
          id: string;
        }

        export const getUser = async (id: string): Promise<User> => {
          return db.find(id);
        };

        export function deleteUser(id: string) {
          return db.remove(id);
        }

        export class UserService {
          list() {
            return db.all();
          }
        }
        """
    )
    chunks = chunk_file(source, "users.ts", "typescript", "r")
    assert [(c.kind, c.symbol) for c in chunks] == [
        ("module", None),
        ("type", "User"),
        ("function", "getUser"),
        ("function", "deleteUser"),
        ("class", "UserService"),
    ]


def test_unparseable_language_falls_back_to_lines():
    chunks = chunk_file("a\nb\n", "notes.txt", "text", "r")
    assert [(c.kind, c.start_line, c.end_line) for c in chunks] == [("block", 1, 2)]
