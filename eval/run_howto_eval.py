"""How-to eval: does the answer give the repository's actual commands?

Each line of eval/questions/howto.jsonl lists commands copied verbatim from the repo's docs
(README, package.json scripts, Makefile); a list entry means "any of these".
Score per question = share of those commands that appear in the answer (whitespace
normalized). A summary like "install the dependencies [README.md:101-160]" scores 0.

Usage (API on :8080):
    python eval/run_howto_eval.py --tag baseline
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import pins
from run_followup_eval import ask

QUESTIONS = Path(__file__).parent / "questions" / "howto.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def present(expected: str | list[str], answer: str) -> bool:
    options = [expected] if isinstance(expected, str) else expected
    return any(normalize(option) in answer for option in options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    rows = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        repo_id = pins.repo_id(q["repo_url"])
        result = ask(args.api, repo_id, q["question"], [])
        answer = normalize(result["answer"])
        # An expected entry is a command, or a list of equivalent commands (any one counts).
        found = [cmd for cmd in q["expected_commands"] if present(cmd, answer)]
        score = len(found) / len(q["expected_commands"])
        rows.append({**q, "answer": result["answer"], "sources": result["sources"],
                     "found": found, "score": score})
        print(f"{score:4.0%}  {q['question']}  missing: "
              f"{[c for c in q['expected_commands'] if c not in found] or '-'}", flush=True)

    summary = {
        "tag": args.tag,
        "n": len(rows),
        "command_recall": statistics.mean(r["score"] for r in rows),
        "fully_correct": sum(r["score"] == 1 for r in rows) / len(rows),
    }
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"howto_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
