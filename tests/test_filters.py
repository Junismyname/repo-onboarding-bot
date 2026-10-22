from pathlib import Path

from repobot.ingestion.filters import detect_language, iter_source_files


def test_detect_language():
    assert detect_language(Path("a/b.py")) == "python"
    assert detect_language(Path("web/App.tsx")) == "tsx"
    assert detect_language(Path("Dockerfile")) == "dockerfile"
    assert detect_language(Path("logo.png")) is None


def test_iter_source_files_skips_junk(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("x")
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / "bundle.min.js").write_text("x")
    (tmp_path / "blob.py").write_bytes(b"\x00\x01binary")

    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path, 512)}
    assert found == {"src/app.py"}
