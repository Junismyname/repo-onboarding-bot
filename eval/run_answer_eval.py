"""Answer-level eval through the running API: are answers grounded and correctly cited?

For each golden question, asks /repos/{repo_id}/ask and measures:
  - cited:         answer contains at least one [path:start-end] citation
  - cites_expected: some citation points at one of the expected files
  - valid:         share of citations that match a label the API actually sent as a source
                   (anything else was invented by the model)

Usage (API on :8080, repos indexed):
    python eval/run_answer_eval.py eval/questions/*.jsonl
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import httpx
import pins

CITATION_RE = re.compile(r"\[([^\[\]\s]+?):(\d+)-(\d+)\]")
RESULTS_DIR = Path(__file__).parent / "results"


def ask(api: str, repo_id: str, question: str) -> tuple[str, list[str], float]:
    sources, tokens = [], []
    start = time.perf_counter()
    with httpx.stream(
        "POST", f"{api}/repos/{repo_id}/ask", json={"question": question}, timeout=300
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: {"):
                continue
            event = json.loads(line[len("data: "):])
            if "sources" in event:
                sources = [s["label"] for s in event["sources"]]
            elif "token" in event:
                tokens.append(event["token"])
    return "".join(tokens), sources, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    rows = []
    for path in args.questions:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            repo_id = pins.repo_id(q["repo_url"])
            answer, sources, seconds = ask(args.api, repo_id, q["question"])
            cites = [f"{p}:{a}-{b}" for p, a, b in CITATION_RE.findall(answer)]
            cited_files = {c.rsplit(":", 1)[0] for c in cites}
            rows.append(
                {
                    "question": q["question"],
                    "expected_files": q["expected_files"],
                    "answer": answer,
                    "citations": cites,
                    "cited": bool(cites),
                    "cites_expected": bool(cited_files & set(q["expected_files"])),
                    "valid_citations": sum(c in sources for c in cites),
                    "seconds": round(seconds, 2),
                }
            )
            r = rows[-1]
            print(
                f"{'C' if r['cited'] else '-'}{'E' if r['cites_expected'] else '-'} "
                f"{r['valid_citations']}/{len(cites)} valid  {seconds:5.1f}s  {q['question'][:70]}",
                flush=True,
            )

    n_cites = sum(len(r["citations"]) for r in rows)
    summary = {
        "tag": args.tag,
        "n": len(rows),
        "answers_with_citation": sum(r["cited"] for r in rows) / len(rows),
        "answers_citing_expected_file": sum(r["cites_expected"] for r in rows) / len(rows),
        "citation_validity": sum(r["valid_citations"] for r in rows) / n_cites if n_cites else 0,
        "p50_seconds": statistics.median(r["seconds"] for r in rows),
    }
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"answers_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
