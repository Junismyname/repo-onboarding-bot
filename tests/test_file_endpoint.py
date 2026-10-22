from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from repobot.api import main

REPO_ID = "owner__repo__abc123def456"


class FakeStore:
    def collection_exists(self, repo_id):
        return repo_id == REPO_ID

    def list_repos(self):
        return [REPO_ID]

    def paths(self, repo_id):
        return ["src/model.py", "../secret.txt"]  # a poisoned index entry must still be refused


@pytest.fixture
def client(tmp_path, monkeypatch):
    checkout = tmp_path / "repos" / REPO_ID / "src"
    checkout.mkdir(parents=True)
    (checkout / "model.py").write_text("class A:\n    pass\n")
    (tmp_path / "repos" / "secret.txt").write_text("do not serve")
    (tmp_path / "maps").mkdir()
    (tmp_path / "maps" / f"{REPO_ID}.md").write_text("map")  # marks ingestion as finished
    monkeypatch.setattr(main.settings, "data_dir", tmp_path)
    monkeypatch.setattr(main, "get_components", lambda: SimpleNamespace(store=FakeStore()))
    return TestClient(main.app)  # not a context manager: skips the model-loading lifespan


def test_serves_indexed_file(client):
    response = client.get(f"/repos/{REPO_ID}/file", params={"path": "src/model.py"})
    assert response.status_code == 200
    body = response.json()
    assert body["content"].startswith("class A:")
    assert body["language"] == "python"
    assert body["github_url"] == "https://github.com/owner/repo/blob/abc123def456/src/model.py"


@pytest.mark.parametrize(
    "path",
    ["../secret.txt", "src/../../secret.txt", "/etc/passwd", "C:/Windows/win.ini", "src/nope.py"],
)
def test_refuses_paths_outside_index_or_checkout(client, path):
    assert client.get(f"/repos/{REPO_ID}/file", params={"path": path}).status_code == 404


def test_unknown_repo_is_404(client):
    assert client.get("/repos/other__repo__x/file", params={"path": "src/model.py"}).status_code == 404


def test_half_built_index_is_not_served(client, tmp_path):
    # Collection exists but the repo map (written last) doesn't: ingestion is still running.
    (tmp_path / "maps" / f"{REPO_ID}.md").unlink()
    assert client.get(f"/repos/{REPO_ID}/file", params={"path": "src/model.py"}).status_code == 404
    assert client.get("/repos").json() == []
