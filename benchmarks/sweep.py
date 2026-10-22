"""Concurrency sweep: run the Locust load test at several user counts, print a markdown table.

Also reports vLLM's prefix-cache hit rate for each run (scraped from its /metrics endpoint).

Usage (API on :8080, eval repos indexed):
    python benchmarks/sweep.py --users 1 2 4 8 16 --duration 90 --tag prefix-cache-on
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def prefix_cache_counters(vllm_url: str) -> tuple[float, float]:
    """(hits, queries) token counters from vLLM's Prometheus metrics."""
    text = httpx.get(f"{vllm_url}/metrics").text

    def counter(name: str) -> float:
        match = re.search(rf"^vllm:{name}_total\{{[^}}]*\}} ([\d.e+]+)$", text, re.MULTILINE)
        return float(match.group(1)) if match else 0.0

    return counter("prefix_cache_hits"), counter("prefix_cache_queries")


def run(users: int, duration: int, host: str, tag: str, vllm_url: str) -> dict:
    prefix = RESULTS / f"{tag}_u{users}"
    hits_before, queries_before = prefix_cache_counters(vllm_url)
    proc = subprocess.run(
        [
            sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--host", host,
            "--headless", "-u", str(users), "-r", str(users), "-t", f"{duration}s",
            "--csv", str(prefix), "--only-summary",
        ],
        check=False,  # exit code checked below: 1 only means some requests failed
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # locust exits 1 when some requests failed
        raise SystemExit(f"locust failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    hits_after, queries_after = prefix_cache_counters(vllm_url)
    queries = queries_after - queries_before

    with open(f"{prefix}_stats.csv", newline="") as f:
        rows = {row["Name"]: row for row in csv.DictReader(f)}
    ask, ttft, total = rows["/ask"], rows["ttft"], rows["total"]
    return {
        "users": users,
        "requests": int(total["Request Count"]),
        "failures": int(ask["Failure Count"]),
        "rps": float(total["Requests/s"]),
        "ttft_p50": float(ttft["50%"]),
        "ttft_p95": float(ttft["95%"]),
        "total_p50": float(total["50%"]),
        "total_p95": float(total["95%"]),
        "cache_hit": (hits_after - hits_before) / queries if queries else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--host", default="http://127.0.0.1:8080")
    parser.add_argument("--vllm", default="http://localhost:8000")
    parser.add_argument("--users", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    print(f"### {args.tag}")
    print("| Users | Requests | Failures | Req/s | TTFT p50 | TTFT p95 | Total p50 | Total p95 "
          "| Prefix-cache hits |")
    print("|---|---|---|---|---|---|---|---|---|")
    for users in args.users:
        r = run(users, args.duration, args.host, args.tag, args.vllm)
        print(
            f"| {r['users']} | {r['requests']} | {r['failures']} | {r['rps']:.2f} "
            f"| {r['ttft_p50'] / 1000:.2f} s | {r['ttft_p95'] / 1000:.2f} s "
            f"| {r['total_p50'] / 1000:.1f} s | {r['total_p95'] / 1000:.1f} s "
            f"| {r['cache_hit']:.0%} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
