from types import SimpleNamespace

from repobot.models import Chunk, ScoredChunk
from repobot.retrieval import hybrid

PATHS = ["Makefile", "tests/test_api.py", "tests/smoke/package.json", "src/app.py"]


class FakeStore:
    def paths(self, repo_id):
        return PATHS

    def file_chunks(self, repo_id, path):
        return []

    def search(self, repo_id, query, using, limit, scope="all", path=None):
        if path:  # file-scoped search
            return [ScoredChunk(Chunk("r", path, "make", 1, 5, "test:\n\tpytest tests"), 0.4)]
        if scope == "code":
            return [ScoredChunk(Chunk("r", "tests/test_api.py", "python", 1, 9, "def test_x(): ..."), 0.7)]
        return []


def components():
    return SimpleNamespace(
        store=FakeStore(),
        dense=SimpleNamespace(embed_query=lambda q: [0.0]),
        sparse=None,
        reranker=None,
    )


def test_command_questions_get_root_build_files():
    result = hybrid.retrieve("r", "How do I run the tests?", components())
    paths = [sc.chunk.path for sc in result.chunks]
    assert "Makefile" in paths
    assert "tests/smoke/package.json" not in paths  # nested build files are fixtures, not the project's
    makefile = next(sc for sc in result.chunks if sc.chunk.path == "Makefile")
    assert makefile.score == hybrid.BUILD_FILE_SCORE


def test_other_questions_do_not():
    result = hybrid.retrieve("r", "How are cookies persisted?", components())
    assert "Makefile" not in [sc.chunk.path for sc in result.chunks]
