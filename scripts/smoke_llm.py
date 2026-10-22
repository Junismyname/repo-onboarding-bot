"""Saturday-morning check: is the LLM server up, and how fast is it?

Usage:
    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --base-url http://localhost:11434/v1 --model qwen2.5-coder:7b  # Ollama
"""

import argparse
import time

from openai import OpenAI

PROMPT = "Explain what a Python context manager is, with a short example."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen2.5-coder-7b")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    start = time.perf_counter()
    first_token_at = None
    pieces = 0
    usage = None

    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=300,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
    )
    for event in stream:
        if event.usage:
            usage = event.usage
        if event.choices and event.choices[0].delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces += 1
            print(event.choices[0].delta.content, end="", flush=True)
    end = time.perf_counter()

    output_tokens = usage.completion_tokens if usage else pieces  # pieces ~= tokens
    decode_s = end - (first_token_at or end)
    print("\n\n--- results ---")
    print(f"TTFT:          {((first_token_at or end) - start) * 1000:.0f} ms")
    print(f"output tokens: {output_tokens}")
    print(f"decode speed:  {output_tokens / decode_s:.1f} tok/s" if decode_s else "n/a")


if __name__ == "__main__":
    main()
