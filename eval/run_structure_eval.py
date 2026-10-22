"""Structure eval: ordinal questions about a file ("What's the second class in model.py?").

Ground truth comes from the index itself: tree-sitter chunks carry each definition's symbol and
line range, so the true order of classes / functions in a file is known exactly. Questions are
generated with a fixed seed so runs are comparable.

An answer is correct if, of all the file's class (or function) names, the one it mentions
first is the expected one.

Usage (API on :8080):
    python eval/run_structure_eval.py --tag baseline
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import pins

ORDINALS = ["first", "second", "third", "fourth"]
TEMPLATES = [
    "What's the {ord} {kind} in {file}?",
    "What is the {ord} {kind} defined in the {file} file?",
    "Which {kind} comes {ord} in {file}?",
]
RESULTS_DIR = Path(__file__).parent / "results"


def definitions(qdrant: str, collection: str) -> dict[str, dict[str, list[str]]]:
    """path -> {"class": [names in file order], "function": [...]} for top-level definitions."""
    first_line: dict[tuple[str, str, str], int] = {}
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": ["path", "symbol", "kind", "parent", "start_line"]}
        if offset is not None:
            body["offset"] = offset
        page = httpx.post(f"{qdrant}/collections/{collection}/points/scroll", json=body).json()
        for point in page["result"]["points"]:
            p = point["payload"]
            if p["kind"] in ("class", "function") and p["symbol"] and p["parent"] is None:
                key = (p["path"], p["kind"], p["symbol"])
                first_line[key] = min(first_line.get(key, p["start_line"]), p["start_line"])
        offset = page["result"]["next_page_offset"]
        if offset is None:
            break
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for (path, kind, symbol), line in first_line.items():
        by_file[path][kind].append((line, symbol))
    return {
        path: {kind: [s for _, s in sorted(items)] for kind, items in kinds.items()}
        for path, kinds in by_file.items()
    }


def make_questions(qdrant: str, collections: list[str], per_repo: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    questions = []
    for collection in collections:
        defs = definitions(qdrant, collection)
        basenames = Counter(Path(p).name for p in defs)
        candidates = []
        for path, kinds in sorted(defs.items()):
            if "test" in path.lower() or basenames[Path(path).name] > 1:
                continue  # skip tests and ambiguous names like __init__.py
            for kind, names in sorted(kinds.items()):
                if len(names) < 2 or len(set(names)) < len(names):
                    continue
                for i in range(min(len(names), len(ORDINALS))):
                    candidates.append((path, kind, names, i))
        for path, kind, names, i in rng.sample(candidates, min(per_repo, len(candidates))):
            template = rng.choice(TEMPLATES)
            questions.append(
                {
                    "repo_id": collection,
                    "question": template.format(ord=ORDINALS[i], kind=kind, file=Path(path).name),
                    "path": path,
                    "names": names,
                    "expected": names[i],
                }
            )
    return questions


def first_mentioned(answer: str, names: list[str]) -> str | None:
    """Earliest of `names` in the answer, ignoring file names like `formDataToJSON.js`."""
    positions = {}
    for name in names:
        match = re.search(rf"(?<![\w./]){re.escape(name)}(?!\w|\.[A-Za-z]{{1,4}}\b)", answer)
        if match:
            positions[name] = match.start()
    return min(positions, key=positions.get) if positions else None


def ask(api: str, repo_id: str, question: str) -> tuple[str, list[str]]:
    sources, tokens = [], []
    with httpx.stream(
        "POST", f"{api}/repos/{repo_id}/ask", json={"question": question}, timeout=300
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: {"):
                event = json.loads(line[len("data: "):])
                if "sources" in event:
                    sources = [s["label"] for s in event["sources"]]
                elif "token" in event:
                    tokens.append(event["token"])
    return "".join(tokens), sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--per-repo", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    # Only the pinned eval repos: indexing anything else must not change the question set.
    collections = sorted(pins.repo_id(url) for url in pins.PINNED)
    questions = make_questions(args.qdrant, collections, args.per_repo, args.seed)
    rows = []
    for q in questions:
        answer, _sources = ask(args.api, q["repo_id"], q["question"])
        got = first_mentioned(answer, q["names"])
        rows.append({**q, "answer": answer, "got": got, "correct": got == q["expected"]})
        print(f"{'OK ' if got == q['expected'] else 'BAD'} {q['repo_id'][:24]:24s} {q['question']}"
              f"  -> {got} (expected {q['expected']})", flush=True)

    summary = {
        "tag": args.tag,
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "no_name_mentioned": sum(r["got"] is None for r in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"structure_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
