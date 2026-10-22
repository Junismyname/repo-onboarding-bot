"""Follow-up eval: does the second turn of a conversation get resolved correctly?

Each line of eval/questions/followups.jsonl has two turns. The first is asked alone; the second
is asked with the first turn (question + the bot's answer) as history. Scoring the second turn:
  - "structure": the answer must name the right definition (true order read from the index)
  - "expected_files": one of the files must be among the sources the API retrieved

--no-history sends the follow-up without history: the behavior before multi-turn support.

Usage (API on :8080):
    python eval/run_followup_eval.py --tag history
    python eval/run_followup_eval.py --tag no-history --no-history
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import pins
from run_structure_eval import definitions, first_mentioned

QUESTIONS = Path(__file__).parent / "questions" / "followups.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def ask(api: str, repo_id: str, question: str, history: list[dict]) -> dict:
    out = {"query": question, "sources": [], "answer": ""}
    tokens = []
    start = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{api}/repos/{repo_id}/ask",
        json={"question": question, "history": history},
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: {"):
                continue
            event = json.loads(line[len("data: "):])
            if "query" in event:
                out["query"] = event["query"]
            elif "sources" in event:
                out["sources"] = [s["path"] for s in event["sources"]]
            elif "token" in event:
                if "first_token_s" not in out:
                    out["first_token_s"] = time.perf_counter() - start
                tokens.append(event["token"])
    out["answer"] = "".join(tokens)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    defs_cache: dict[str, dict] = {}
    rows = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        repo_id = pins.repo_id(case["repo_url"])
        first_q, follow_q = case["turns"]

        first = ask(args.api, repo_id, first_q, [])
        history = [] if args.no_history else [
            {"role": "user", "content": first_q},
            {"role": "assistant", "content": first["answer"]},
        ]
        second = ask(args.api, repo_id, follow_q, history)

        if "structure" in case:
            s = case["structure"]
            if repo_id not in defs_cache:
                defs_cache[repo_id] = definitions(args.qdrant, repo_id)
            names = defs_cache[repo_id][s["path"]][s["kind"]]
            expected = names[s["index"]]
            got = first_mentioned(second["answer"], names)
            correct, detail = got == expected, f"{got} (expected {expected})"
        else:
            correct = bool(set(second["sources"]) & set(case["expected_files"]))
            detail = f"expected {case['expected_files'][0]} in sources"
        rows.append(
            {**case, "first": first, "second": second, "correct": correct, "detail": detail}
        )
        print(f"{'OK ' if correct else 'BAD'} {follow_q!r:58s} -> query {second['query']!r}",
              flush=True)
        print(f"      {detail}", flush=True)

    summary = {
        "tag": args.tag,
        "n": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "follow_up_first_token_p50_s": statistics.median(
            r["second"].get("first_token_s", 0) for r in rows
        ),
        "first_turn_first_token_p50_s": statistics.median(
            r["first"].get("first_token_s", 0) for r in rows
        ),
    }
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"followup_{args.tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "conversations": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
