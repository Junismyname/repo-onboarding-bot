from repobot.ingestion.chunker import chunk_by_lines


def test_line_windows_cover_file_with_overlap():
    source = "\n".join(f"line {i}" for i in range(1, 131))  # 130 lines
    chunks = chunk_by_lines(source, "a.py", "python", "r", window=60, overlap=10)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert chunks[0].text.splitlines()[0] == "line 1"
    assert chunks[-1].text.splitlines()[-1] == "line 130"


def test_blank_file_produces_no_chunks():
    assert chunk_by_lines("\n\n  \n", "a.py", "python", "r") == []


def test_chunk_ids_are_deterministic():
    a = chunk_by_lines("x = 1", "a.py", "python", "r")[0]
    b = chunk_by_lines("x = 1", "a.py", "python", "r")[0]
    assert a.chunk_id == b.chunk_id
