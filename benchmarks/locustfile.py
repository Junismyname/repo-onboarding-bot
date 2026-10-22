"""Load test for /ask. Reports custom "SSE" entries: ttft (first answer token) and total
(stream finished). Locust's own /ask entry only measures time to response headers.

Questions come from eval/questions/*.jsonl (46 questions over 3 repos), so prompts differ
between requests. With only a handful of repeated questions, vLLM's prefix cache would serve
most of every prompt from cache and make TTFT look unrealistically good.

Usage (API on :8080, the eval repos indexed):
    locust -f benchmarks/locustfile.py --host http://127.0.0.1:8080 \
        --headless -u 8 -r 8 -t 2m --csv benchmarks/results/u8
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
from locust import HttpUser, between, task

QUESTIONS_DIR = Path(__file__).parent.parent / "eval" / "questions"
sys.path.insert(0, str(QUESTIONS_DIR.parent))
import pins  # eval/pins.py: the pinned eval commits

QDRANT_URL = os.environ.get("REPOBOT_QDRANT_URL", "http://localhost:6333")


def load_workload() -> list[tuple[str, str]]:
    """(repo_id, question) pairs for every eval question whose pinned repo is indexed."""
    collections = {
        c["name"] for c in httpx.get(f"{QDRANT_URL}/collections").json()["result"]["collections"]
    }
    workload = []
    for path in sorted(QUESTIONS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if "question" not in q or "expected_files" not in q:
                continue  # only the 46 single-turn retrieval questions (not followups/howto)
            repo_id = pins.repo_id(q["repo_url"])
            if repo_id in collections:
                workload.append((repo_id, q["question"]))
    if not workload:
        raise RuntimeError("No eval repos are indexed; run eval/run_eval.py first")
    return workload


WORKLOAD = load_workload()


class AskUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask(self) -> None:
        repo_id, question = random.choice(WORKLOAD)
        start = time.perf_counter()
        ttft_ms = None
        with self.client.post(
            f"/repos/{repo_id}/ask",
            json={"question": question},
            stream=True,
            catch_response=True,
            name="/ask",
        ) as response:
            # chunk_size=None: yield bytes as they arrive instead of buffering 512 at a time
            for line in response.iter_lines(chunk_size=None):
                if ttft_ms is None and line and b'"token"' in line:
                    ttft_ms = (time.perf_counter() - start) * 1000
            total_ms = (time.perf_counter() - start) * 1000
            if ttft_ms is None:
                response.failure("no tokens streamed")
                return
        for name, value in (("ttft", ttft_ms), ("total", total_ms)):
            self.environment.events.request.fire(
                request_type="SSE", name=name, response_time=value,
                response_length=0, exception=None, context={},
            )
