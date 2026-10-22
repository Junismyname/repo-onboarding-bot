from repobot.metrics import recall_at_k, reciprocal_rank


def test_recall_dedupes_files_before_cutting_at_k():
    retrieved = ["a.py", "a.py", "a.py", "b.py"]  # 3 chunks from a.py, then b.py
    assert recall_at_k(retrieved, ["b.py"], k=2) == 1.0
    assert recall_at_k(retrieved, ["b.py"], k=1) == 0.0


def test_recall_partial():
    assert recall_at_k(["a.py", "c.py"], ["a.py", "b.py"], k=5) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(["x.py", "a.py"], ["a.py"]) == 0.5
    assert reciprocal_rank(["x.py"], ["a.py"]) == 0.0
