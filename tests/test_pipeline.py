from types import SimpleNamespace

import pytest

from repobot.ingestion import pipeline
from repobot.models import RepoRef


class FakeStore:
    def __init__(self, existing=()):
        self.collections = {name: [] for name in existing}
        self.deleted = []

    def collection_exists(self, name):
        return name in self.collections

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)

    def create_collection(self, name, dim):
        self.collections[name] = []

    def upsert(self, name, chunks, dense, sparse):
        self.collections[name].extend(chunks)


class FakeEmbedder:
    dim = 3

    def embed_documents(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    checkout = tmp_path / "repos" / "o__r__abc"
    checkout.mkdir(parents=True)
    (checkout / "app.py").write_text("def main():\n    return 1\n")
    ref = RepoRef("o", "r", "abc", checkout)
    monkeypatch.setattr(pipeline.settings, "data_dir", tmp_path)
    monkeypatch.setattr(pipeline, "clone_repo", lambda *args: ref)
    return ref


def components(store):
    return SimpleNamespace(store=store, dense=FakeEmbedder(), sparse=FakeEmbedder())


def test_leftover_half_built_collection_is_rebuilt(repo, tmp_path):
    store = FakeStore(existing=[repo.repo_id])  # collection exists, but no repo map: interrupted
    pipeline.ingest_repo("https://github.com/o/r", components(store))
    assert store.deleted == [repo.repo_id]
    assert store.collections[repo.repo_id]  # re-filled
    assert (tmp_path / "maps" / f"{repo.repo_id}.md").exists()
    assert pipeline.is_indexed(store, repo.repo_id)


def test_finished_index_is_reused(repo, tmp_path):
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{repo.repo_id}.md").write_text("map")
    store = FakeStore(existing=[repo.repo_id])
    pipeline.ingest_repo("https://github.com/o/r", components(store))
    assert store.deleted == []
    assert store.collections[repo.repo_id] == []  # nothing re-embedded


def test_failed_embedding_leaves_no_index(repo, tmp_path):
    class Broken(FakeEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("CUDA out of memory")

    store = FakeStore()
    with pytest.raises(RuntimeError):
        pipeline.ingest_repo(
            "https://github.com/o/r",
            SimpleNamespace(store=store, dense=Broken(), sparse=FakeEmbedder()),
        )
    assert not store.collection_exists(repo.repo_id)
    assert not (tmp_path / "maps" / f"{repo.repo_id}.md").exists()
