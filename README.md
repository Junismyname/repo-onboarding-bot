# Repo Onboarding Bot: Chat With Any GitHub Repository

A self-hosted retrieval-augmented generation (RAG) system for understanding unfamiliar codebases. Paste a GitHub URL, ask questions in plain English (including follow-ups like "what about the first one?"), and get answers that cite the exact lines. Every citation is clickable and opens the code in a side panel. The language model, embedding model and vector database all run on a single consumer GPU (RTX 5070, 12 GB).

**Result: every answer cites its sources, 12/12 setup questions answered with the repo's exact commands, 15/15 follow-ups resolved, 3.6 s per answer, and 16 concurrent users served with zero failures, on one 12 GB GPU**

![Web app: an answer with a clickable citation open in the source viewer](docs/ui.png)

This project was built to learn how to deploy a large language model end to end: serving it with vLLM on limited GPU memory, building retrieval around it, and measuring both answer quality and serving performance.

Every design decision was measured rather than assumed, and several assumptions turned out wrong. Hybrid search and a re-ranker made retrieval *worse*. A CPU re-ranker took 15 s per question. An fp8 KV cache bought throughput at a cost in answer quality. And a "slow model" turned out to be Windows spending 2 s per connection on IPv6. Each of these is documented below, along with the fix.

## Project Overview

Generic "chat with your docs" tools treat code like prose. This project uses the structure of code instead:

- **Chunks follow the code.** Functions, classes and methods come from tree-sitter, and markdown is split by heading.
- **Code and docs are searched separately**, so plain-English documentation can't crowd out the implementing code.
- **Structural questions get the file's real outline.** "What's the second class in model.py?" is answered from the actual order of definitions.
- **Setup questions get build files.** "How do I run the tests?" pulls in `Makefile` and `package.json`.

### Evaluation Repositories

All evals run against these repositories at commits pinned in [`eval/repos.json`](eval/repos.json), so expected answers stay valid as the repositories evolve.

| Repository | Language | Chunks | Pinned commit | Used for |
| --- | --- | --- | --- | --- |
| [psf/requests](https://github.com/psf/requests) | Python | 826 | `611c616` | Retrieval, answers, structure, follow-ups, how-to |
| [pallets/click](https://github.com/pallets/click) | Python | 1,669 | `2247b35` | Retrieval, answers, structure, follow-ups, how-to |
| [axios/axios](https://github.com/axios/axios) | JavaScript / TypeScript | 2,551 | `fceeb10` | Retrieval, answers, structure, follow-ups, how-to |
| [pallets/itsdangerous](https://github.com/pallets/itsdangerous) | Python | 157 | `672971d` | Structure, follow-ups |
| [Junismyname/room-presence-yolo](https://github.com/Junismyname/room-presence-yolo) | Python | 33 | `84c9ad1` | Structure, follow-ups, how-to |

### Performance & Analysis

Final test of the current code (2026-10-22). Raw results are written to `eval/results/` and `benchmarks/results/` by the scripts listed under [Getting Started](#5-reproducing-the-evaluation).

| Eval | Questions | Result |
| --- | --- | --- |
| Retrieval: right file ranked first / in top 10 | 46 | **65% / 88%** (MRR 0.75, 31 ms) |
| Answers that cite their sources | 46 | **100%** (2 runs) |
| Answers citing the file that implements the answer | 46 | 65–72% across runs |
| Citations matching a chunk the model was given | 46 | 83–88% |
| Setup questions answered with the repo's exact commands | 12 | **12/12** (2 runs) |
| File-structure questions ("second class in X") | 46 | **85%** (was 33%) |
| Follow-up questions resolved from the conversation | 15 | **15/15** (was 8/15) |
| Single user: first token / full answer (p50) | 46 | 1.1 s / 3.6 s |
| 16 concurrent users | 90 s load test | 0.88 answers/s, 0 failures, first token 3.1 s |

* **Retrieval:** Dense search over code with a code-aware embedding model (gte-modernbert-base) ranks the implementing file first for 65% of newcomer-phrased questions and puts it in the top 10 for 88%. Searching code and docs separately was the biggest single gain (top-1 from 49% to 65%).
* **Grounding:** A closing "Sources:" instruction took answers with citations from 67% to 100%. About 30% of answers still cite nearby code or docs rather than the exact implementing file. That's the main limit of a 7B model with plain search.
* **Structure:** Ordinal questions failed because chunks were listed by relevance and the model counted in prompt order. Presenting code in file order, with a numbered outline placed next to the question, raised accuracy from 33% to 85–87%.
* **Conversation:** Follow-ups are rewritten into standalone questions before retrieval ("Which class does that?" becomes "Which class does the `TimestampSigner.sign` method belong to?"). This costs about 0.7 s on follow-up turns only.
* **Setup questions:** The model used to summarize README instructions instead of quoting the commands. A how-to instruction, applied only to command questions, and always searching the root build files fixed every case.

### Retrieval Ablations

File-level recall on 46 questions. 20 configurations were tested; the main ones are shown. One question ≈ 0.02, so differences under about 0.05 are noise. The "all files" rows were measured before markdown was split by sections. With section chunks, docs compete harder in a mixed search (dense R@1 0.47, R@5 0.73 in the final run), which strengthens the case for searching code separately. The code-only rows are unchanged.

| Chunking | Scope | Search | R@1 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| 60-line windows | all files | dense | 0.28 | 0.78 | 0.85 | 0.51 |
| tree-sitter | all files | dense | 0.49 | 0.82 | 0.84 | 0.62 |
| tree-sitter | all files | hybrid (dense + BM25) | 0.24 | 0.72 | 0.84 | 0.48 |
| tree-sitter | all files | hybrid + re-ranker | 0.21 | 0.68 | 0.76 | 0.42 |
| **tree-sitter** | **code only** | **dense** ✅ | **0.65** | 0.83 | 0.88 | **0.75** |
| tree-sitter | code only | hybrid | 0.55 | 0.85 | 0.86 | 0.72 |
| tree-sitter | code only | dense + bge-reranker-v2-m3 | 0.54 | 0.89 | 0.93 | 0.70 |

* **Prose crowds out code.** Questions are in plain English, and so are docs and changelogs. With all files in scope, the re-ranker put a docs file at rank 1 for 30 of 46 questions.
* **BM25 hurts newcomer questions.** Newcomers don't know the identifiers yet, so keyword matching pulls dense results down.
* **The re-ranker isn't worth it here.** It gains a little at depth 10, which is within noise, but costs 1.5 GiB of VRAM and 316 ms. It's off by default (`REPOBOT_USE_RERANKER=true` turns it on).

### Serving Under Load

Locust load test, 90 s per level. Users ask random eval questions (prompts are about 5–7k tokens) and wait 1–3 s between questions. fp16 KV cache, final code:

| Users | Answers/s | First token p50 | First token p95 | Full answer p50 | Failures |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.16 | 1.1 s | 1.9 s | 3.5 s | 0 |
| 4 | 0.56 | 1.1 s | 2.7 s | 4.5 s | 0 |
| 8 | 0.59 | 2.9 s | 7.8 s | 10 s | 0 |
| 16 | 0.88 | 3.1 s | 12 s | 14 s | 0 |

* **KV cache memory is the ceiling.** The 2.57 GiB cache holds about 48k tokens, which is about 7 RAG prompts in flight. Beyond 4–8 users, vLLM's scheduler shows requests waiting while cache usage is at 85–90%.
* **fp8 KV cache** holds 82k tokens. In an earlier sweep it raised 16-user throughput by 58%, but answers citing the expected file fell from about 52% to 28–35%, consistently across two runs. It stays opt-in (`VLLM_EXTRA_ARGS=--kv-cache-dtype fp8`). That comparison predates a fix to the API's event loop, which by itself took the fp16 numbers from 0.57 to 0.88 answers/s, so fp8's real advantage is probably smaller.

## Evaluation

### Evaluation Protocol

* **Ground truth from the repositories, not from the model:**
  * Retrieval questions are labeled with the files that implement the answer, checked against the source.
  * Structure questions are generated from tree-sitter line ranges.
  * Setup questions list the exact commands found in the README, `Makefile` or `package.json`.
* **Pinned commits.** Ingestion can fetch a specific commit, and every eval and the load test resolve repositories through [`eval/pins.py`](eval/pins.py). An unpinned re-index once silently moved axios to a newer commit.
* **Repeated runs.** Identical configurations varied by up to 8 points between runs (70%, 70%, 78%). Decisions compare ranges over repeated runs, not single numbers.
* **Citations are checked, not trusted.** A citation counts as real only if its `path:lines` matches a chunk the API actually sent. Prompt examples are placeholders: a concrete example once caused the model to copy its fake file paths into answers.

### What the Measurements Changed

| Assumption | Measured | Change |
| --- | --- | --- |
| `gpu-memory-utilization 0.65` leaves room for everything | vLLM refused to start: 0.78 GiB of KV cache, below the 0.88 GiB one 16k-token request needs | 0.80 |
| CPU re-ranking takes a few hundred ms | 15.4 s for 50 candidates (about 14 TFLOPs) | GPU fp16 (160 ms), then off by default |
| Hybrid search plus a re-ranker beats dense search | Both lower top-1 recall; prose crowds out code | Dense search, with code and docs searched separately |
| Rewording the prompt fixes missing citations | 67% cited | Closing `Sources:` line: 100% |
| An fp8 KV cache doubles capacity for free | +58% throughput, but grounding dropped | Opt-in only |
| 5 benchmark questions are enough | 97% prefix-cache hits made first-token latency look 5× better | 46-question workload, with hit rate reported per run |
| End-to-end latency is model time | 2 s of every 5.8 s answer was Windows trying IPv6 for `localhost` | Clients use `127.0.0.1`; per-request stage timing in logs |
| Wrong ordinal answers mean retrieval missed the code | The right chunks were retrieved but presented in relevance order | File order plus an outline: 33% → 85–87% |
| A missing command means retrieval missed it | The whole Getting Started section was in the prompt, and "be concise" made the model summarize it | How-to instruction for command questions only |
| The LLM server is either up or down | After 6 hours, a CUDA error killed vLLM's engine while its HTTP server and container stayed "up" | Docker healthcheck, restart policy, and an error event instead of a broken stream |

## Architecture & Data Flow

```text
                         +---------------------------+
                         |  GitHub URL (+ optional   |
                         |  pinned commit)           |
                         +-------------+-------------+
                                       |
                         +-------------v-------------+
                         | Ingestion (background job)|
                         | shallow clone -> filter   |
                         | -> tree-sitter / markdown |
                         |    sections / build files |
                         +-------------+-------------+
                                       |
              +------------------------+------------------------+
              |                                                 |
+-------------v-------------+                     +-------------v-------------+
| Dense + BM25 embeddings   |                     | Repo map (README excerpt, |
| (gte-modernbert, GPU fp16)|                     |  files, top-level symbols)|
| -> Qdrant, 1 collection   |                     |  written last = "ready"   |
|    per repo@commit        |                     |                           |
+-------------+-------------+                     +-------------+-------------+
              |                                                 |
              |        +---------------------------+            |
              |        | Question (+ last turns)   |            |
              |        +-------------+-------------+            |
              |                      |                          |
              |        +-------------v-------------+            |
              |        | Follow-up? LLM rewrites it|            |
              |        | into a standalone question|            |
              |        +-------------+-------------+            |
              |                      |                          |
+-------------v----------------------v-------------+            |
| Retrieval: code top 10 + docs top 3 (separate),  |            |
| named files -> outline + chunks,                 |            |
| command questions -> root build files            |            |
+-------------------------+------------------------+            |
                          |                                     |
            +-------------v-------------+                       |
            | Prompt: code in file order|<----------------------+
            | + outline + recent turns  |
            | + citation rules, fitted  |
            |   to the token budget     |
            +-------------+-------------+
                          |
            +-------------v-------------+
            | vLLM: Qwen2.5-Coder-7B    |
            | Instruct AWQ (4-bit)      |
            +-------------+-------------+
                          |
            +-------------v-------------+
            | SSE stream -> web UI:     |
            | answer + clickable        |
            | [path:lines] citations    |
            +---------------------------+
```

### 1. Ingestion & Chunking

* **Clone:** shallow (`--depth 1`), or a single pinned commit fetched by SHA. Checkouts are reused per commit, and `.git` is removed after reading the SHA.
* **Code:** tree-sitter (Python, JavaScript, TypeScript) produces one chunk per function or class. Classes over 80 lines are split into a header plus one chunk per method, and comments directly above a definition stay with it. Symbol names like `Session.request` go into the embedded text.
* **Docs:** markdown is split by heading sections, keeping a section whole (with its subsections) up to 80 lines. `#` lines inside code fences are shell comments, not headings.
* **Build files:** `Makefile`, `package.json` and `Dockerfile` are indexed by name. Other `.json` files are skipped, since they're mostly data or lock files.
* **Completion marker:** the repo map is written only after the last chunk is stored. A repo still being indexed, or interrupted midway, is never served, and leftovers are rebuilt.

### 2. Retrieval

* **Separate code and docs searches** (top 10 and top 3) with dense vectors. BM25 vectors are stored too, for hybrid search experiments.
* **Named files:** "model.py" in a question is matched to an indexed path (with fuzzy matching, so `model.py` also finds `models.py`). The file's numbered outline of classes and functions is added, along with its chunks.
* **Command questions** ("how do I install / run / test …") always include the best chunks of root-level build files. Nested ones are skipped: axios has 7 `package.json` files, mostly test fixtures.

### 3. Prompting & Generation

* **File order:** code is grouped by file and sorted by line, and the outline sits right before the question. Placement mattered: with the outline at the top of the prompt, structure accuracy was 76%; next to the question, it was 87%.
* **Citations:** every snippet is labeled `[path:start-end]`, and the answer ends with a `Sources:` line.
* **Multi-turn:** the last 3 question/answer pairs are sent as plain text, trimmed and without their retrieved code, so the prompt doesn't grow with each turn.
* **Token budget:** prompts are counted with Qwen's tokenizer and chat template. If prompt plus output would exceed the context window, the lowest-scoring chunks are dropped. The window size is read from vLLM at startup.

### 4. Serving on 12 GB

| Consumer | VRAM |
| --- | --- |
| vLLM (`gpu-memory-utilization 0.80`) | ~9.6 GiB |
|  ↳ model weights (AWQ 4-bit) | 5.29 GiB |
|  ↳ CUDA graphs + activations | ~1.7 GiB |
|  ↳ KV cache (fp16, ~56 KB/token) | 2.57 GiB = 48k tokens |
| API process (embedder fp16 + CUDA context) | ~0.7 GiB |
| Windows desktop | ~0.5–0.8 GiB |

* **Decode speed:** about 108 tokens/s on a short prompt and about 97 tokens/s with 4–7k tokens of context.
* **Startup warm-up:** models load and run once when the API starts, so the first question returns sources in 0.17 s instead of 23–35 s.
* **Ingestion speed:** 2–18 s per repository. Embedding dominates, and grouping 512 chunks per `encode()` call made it 1.9× faster by cutting padding.

### 5. Web Application

* **Plain HTML, CSS and JavaScript** served by FastAPI. No build step.
* **Streaming answers:** markdown is sanitized with DOMPurify, and each `[path:lines]` becomes a link. Citations that don't match a retrieved snippet are shown in amber.
* **Source viewer:** the whole file with syntax highlighting, the cited lines highlighted and scrolled into view, and a GitHub link. On narrow screens it becomes a full-screen overlay.
* **File endpoint:** serves only indexed paths, and only inside the repo checkout. Traversal attempts like `../../.env` are refused, and tests cover it.

## Repository Structure

```text
repo-onboarding-bot/
├── docker-compose.yml        # vLLM (healthcheck, restart) + Qdrant (+ API with --profile app)
├── docker/api.Dockerfile     # API image (Python 3.12, PyTorch cu128)
├── .env.example              # vLLM and app settings
├── pyproject.toml            # Package and dependencies
├── LICENSE
├── docs/ui.png               # Screenshot
├── src/repobot/
│   ├── config.py             # Settings (env vars prefixed REPOBOT_)
│   ├── models.py             # Chunk, RepoRef, ScoredChunk
│   ├── services.py           # Shared models and clients, loaded once
│   ├── intent.py             # Command-question detection
│   ├── metrics.py            # recall@k, MRR
│   ├── ingestion/            # clone -> filters -> chunker -> repo map -> pipeline
│   ├── indexing/             # Dense and BM25 embedders, Qdrant store
│   ├── retrieval/            # Code/docs retrieval, file outlines, optional re-ranker
│   ├── generation/           # Prompts, follow-up rewriting, token budget, LLM client
│   ├── api/                  # FastAPI app, ingestion jobs, schemas
│   └── web/                  # UI: chat, clickable citations, source viewer
├── eval/
│   ├── repos.json, pins.py   # Pinned eval commits
│   ├── questions/            # Retrieval, follow-up and how-to question sets
│   ├── run_eval.py           # Retrieval ablations
│   ├── run_answer_eval.py    # Citation and grounding
│   ├── run_structure_eval.py # Ordinal questions generated from the index
│   ├── run_followup_eval.py  # Two-turn conversations
│   └── run_howto_eval.py     # Exact commands in setup answers
├── benchmarks/
│   ├── locustfile.py         # Load test (first-token and total latency)
│   └── sweep.py              # Concurrency sweep with prefix-cache hit rate
├── scripts/smoke_llm.py      # Check the LLM server: first-token latency, tokens/s
└── tests/                    # 57 tests: chunkers, retrieval, budget, pipeline, API security
```

## Getting Started

### 1. Prerequisites & Environment Setup

You need an NVIDIA GPU with about 12 GB of VRAM, Docker Desktop with GPU support (WSL2 on Windows), and Python 3.12 or 3.13. Blackwell GPUs (RTX 50xx) need the CUDA 12.8 PyTorch build, so install it first:

```bash
git clone https://github.com/Junismyname/repo-onboarding-bot.git
cd repo-onboarding-bot
cp .env.example .env            # On Windows cmd: copy .env.example .env

conda create -n repobot python=3.12 -y
conda activate repobot
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[bench,dev]"
```

### 2. Starting the Model Server

```bash
docker compose up -d vllm qdrant
docker compose ps                # wait until vllm shows "(healthy)": 3-10 min on first start
python scripts/smoke_llm.py      # prints time to first token and tokens/s
```

The first start downloads the vLLM image and about 5 GB of model weights. If vLLM won't run on your machine, use Ollama as a fallback: set `REPOBOT_LLM_BASE_URL=http://localhost:11434/v1` and `REPOBOT_LLM_MODEL=qwen2.5-coder:7b`.

### 3. Running the Application

```bash
uvicorn repobot.api.main:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080/, paste a GitHub URL into the top bar and click **Index**. Indexing takes a few seconds to a minute, after which you can start asking questions. Use `127.0.0.1` rather than `localhost`: on Windows, `localhost` adds about 2 s to every new connection.

### 4. Using the API Directly

```bash
curl -X POST 127.0.0.1:8080/repos -H "Content-Type: application/json" \
     -d '{"url": "https://github.com/psf/requests"}'            # -> job id
curl 127.0.0.1:8080/jobs/<job_id>                               # -> progress, repo_id
curl -N -X POST 127.0.0.1:8080/repos/<repo_id>/ask -H "Content-Type: application/json" \
     -d '{"question": "How are retries configured?"}'           # -> streamed answer
```

### 5. Reproducing the Evaluation

With the API running and vLLM up:

```bash
python eval/run_eval.py eval/questions/requests.jsonl eval/questions/click.jsonl eval/questions/axios.jsonl
python eval/run_answer_eval.py eval/questions/requests.jsonl eval/questions/click.jsonl eval/questions/axios.jsonl
python eval/run_structure_eval.py
python eval/run_followup_eval.py
python eval/run_howto_eval.py
python benchmarks/sweep.py --tag fp16 --users 1 2 4 8 16 --duration 90
```

`run_eval.py` indexes each repository at its pinned commit the first time it runs. The other evals expect those repositories to be indexed already.

### 6. Tests

```bash
python -m pytest tests
```

## Limitations

* **Grounding isn't perfect.** About 30% of answers cite nearby code or docs rather than the exact file that implements the answer. 12–17% of citations don't match a snippet the model was given; these are shown in amber in the UI.
* **Small eval sets.** The question sets have 10–46 questions, and identical runs vary by up to 8 points. Treat results as ranges.
* **Correctness isn't scored directly.** The evals check retrieval, citations, structure and commands, not whether each explanation is right. An LLM-as-judge or human review would be the next step.
* **Language coverage.** Tree-sitter chunking covers Python, JavaScript and TypeScript. Other languages fall back to 60-line windows.
* **Prompt injection.** Repository content goes into the prompt, so a malicious README could try to steer an answer. Output is sanitized before rendering, so it can't run code in the browser; the risk is a misleading answer.
* **Not yet done:** a re-run of the fp8 sweep on the current API, a build of the API Docker image, a public demo, and tracing.

## Acknowledgments & Citations

Built with [Claude Code](https://claude.com/claude-code) as a pair-programming partner.

Models and serving:
* **Qwen2.5-Coder:** Hui, Binyuan, et al. (2024). Qwen2.5-Coder Technical Report. arXiv:2409.12186.
* **vLLM:** Kwon, Woosuk, et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*. arXiv:2309.06180.
* **ModernBERT:** Warner, Benjamin, et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv:2412.13663.
* **GTE embeddings:** Li, Zehan, et al. (2023). Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281.

Tools: [Qdrant](https://qdrant.tech/), [tree-sitter](https://tree-sitter.github.io/), [FastAPI](https://fastapi.tiangolo.com/), [Locust](https://locust.io/), [highlight.js](https://highlightjs.org/), [marked](https://marked.js.org/), [DOMPurify](https://github.com/cure53/DOMPurify).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
