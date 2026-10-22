"""Retrieval eval: file-level recall@k and MRR over a golden question set.

Each line of a questions file (eval/questions/*.jsonl):
    {"repo_url": "...", "question": "...", "expected_files": ["path/in/repo.py", ...]}

Scope "all" searches every chunk; "code" excludes prose (markdown/rst/text), which is how
the app retrieves code (docs get their own small, separate budget).

Usage:
    python eval/run_eval.py eval/questions/*.jsonl                       # full ablation grid
    python eval/run_eval.py eval/questions/*.jsonl --chunking syntax --scopes code --modes dense
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from pins import PINNED  # eval answers are tied to these commits, not the latest

from repobot.config import settings
from repobot.indexing.store import DENSE, SPARSE
from repobot.ingestion.pipeline import index_name, ingest_repo
from repobot.metrics import recall_at_k, reciprocal_rank
from repobot.models import ScoredChunk
from repobot.services import Components, get_components

KS = (1, 3, 5, 10)
MODES = ("dense", "bm25", "hybrid", "hybrid+rerank", "dense+rerank")
RESULTS_DIR = Path(__file__).parent / "results"


def load_questions(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def search(
    mode: str, collection: str, question: str, c: Components, scope: str
) -> list[ScoredChunk]:
    """Top chunks for one retrieval configuration. Ranked lists are long so that
    file-level recall@10 still has 10 distinct files to look at."""
    limit = settings.retrieve_k
    if mode in ("dense", "dense+rerank"):
        hits = c.store.search(collection, c.dense.embed_query(question), DENSE, limit, scope)
    elif mode == "bm25":
        return c.store.search(collection, c.sparse.embed_query(question), SPARSE, limit, scope)
    else:
        hits = c.store.hybrid_search(
            collection, c.dense.embed_query(question), c.sparse.embed_query(question), limit, scope
        )
    if mode.endswith("+rerank"):
        return c.reranker.rerank(question, hits, top_k=limit)
    return hits


def evaluate(questions, collections, mode, c: Components, scope: str) -> dict:
    rows = []
    for q in questions:
        start = time.perf_counter()
        hits = search(mode, collections[q["repo_url"]], q["question"], c, scope)
        latency_ms = (time.perf_counter() - start) * 1000
        paths = [h.chunk.path for h in hits]
        rows.append(
            {
                "question": q["question"],
                "expected_files": q["expected_files"],
                "retrieved_files": list(dict.fromkeys(paths))[:10],
                "latency_ms": round(latency_ms, 1),
                "mrr": reciprocal_rank(paths, q["expected_files"]),
                **{f"recall@{k}": recall_at_k(paths, q["expected_files"], k) for k in KS},
            }
        )
    summary = {
        "mrr": statistics.mean(r["mrr"] for r in rows),
        **{f"recall@{k}": statistics.mean(r[f"recall@{k}"] for r in rows) for k in KS},
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in rows),
    }
    return {"summary": summary, "questions": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+", type=Path)
    parser.add_argument("--chunking", nargs="+", default=["lines", "syntax"])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--scopes", nargs="+", default=["all", "code"], choices=["all", "code"])
    args = parser.parse_args()

    c = get_components(use_reranker=any(m.endswith("+rerank") for m in args.modes))
    questions = load_questions(args.questions)
    urls = list(dict.fromkeys(q["repo_url"] for q in questions))
    results = []

    for chunking in args.chunking:
        collections = {}
        for url in urls:
            start = time.perf_counter()
            repo = ingest_repo(url, c, chunking=chunking, sha=PINNED.get(url))
            collections[url] = index_name(repo, chunking)
            count = c.store.client.count(collections[url]).count
            print(f"[{chunking}] {collections[url]}: {count} chunks ({time.perf_counter() - start:.1f}s)")
        for scope in args.scopes:
            for mode in args.modes:
                # keep model warm-up out of the latency numbers
                search(mode, collections[urls[0]], "warmup", c, scope)
                result = evaluate(questions, collections, mode, c, scope)
                result["config"] = {"chunking": chunking, "scope": scope, "mode": mode}
                results.append(result)
                s = result["summary"]
                print(
                    f"  {chunking:7s} {scope:5s} {mode:14s} R@1 {s['recall@1']:.2f}  "
                    f"R@3 {s['recall@3']:.2f}  R@5 {s['recall@5']:.2f}  "
                    f"R@10 {s['recall@10']:.2f}  MRR {s['mrr']:.2f}  "
                    f"p50 {s['p50_latency_ms']:.0f} ms"
                )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"n_questions": len(questions), "runs": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
