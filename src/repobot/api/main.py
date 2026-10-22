"""HTTP API and web UI.

GET  /ui/                   web UI (chat + source viewer)
GET  /repos                 indexed repos
POST /repos                 start ingesting a GitHub repo -> job
GET  /jobs/{job_id}         poll ingestion progress
GET  /repos/{repo_id}/file  one indexed file's contents (for the source viewer)
POST /repos/{repo_id}/ask   ask a question (optionally with earlier turns); streams SSE:
                              {"query": standalone question, "files": [...]}, {"sources": [...]},
                              then {"token": "..."} chunks, then [DONE]
"""

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from repobot.api.jobs import JobStore
from repobot.api.schemas import AskRequest, FileOut, IngestRequest, JobOut, RepoOut, Source
from repobot.config import settings
from repobot.generation.budget import count_tokens, fit_to_budget
from repobot.generation.conversation import standalone_question, trim_history
from repobot.generation.llm import served_max_model_len, stream_answer
from repobot.generation.prompts import build_messages
from repobot.ingestion.filters import detect_language
from repobot.ingestion.pipeline import ingest_repo, is_indexed
from repobot.ingestion.repo_map import load_repo_map
from repobot.models import RepoRef
from repobot.retrieval.hybrid import retrieve
from repobot.services import get_components

WEB_DIR = Path(__file__).parent.parent / "web"

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load models and run one query before serving: otherwise the first question waited
    # 20-35 s for model loading and CUDA warm-up.
    started = time.perf_counter()
    components = await run_in_threadpool(get_components)
    await run_in_threadpool(components.dense.embed_query, "warm up")
    await run_in_threadpool(count_tokens, [{"role": "user", "content": "warm up"}])
    if served_len := await served_max_model_len():
        settings.llm_max_model_len = served_len  # budget follows the server's real limit
    logger.info(
        "models ready in %.1f s; prompt budget for max_model_len=%d",
        time.perf_counter() - started,
        settings.llm_max_model_len,
    )
    yield


app = FastAPI(title="Repo Onboarding Bot", version="0.1.0", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
jobs = JobStore()
# One ingestion at a time: concurrent embedding would fight vLLM for the GPU.
_ingest_lock = threading.Lock()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": settings.llm_model}


@app.get("/repos", response_model=list[RepoOut])
def list_repos() -> list[RepoOut]:
    store = get_components().store
    repos = [_repo_from_id(rid) for rid in store.list_repos() if is_indexed(store, rid)]
    return [
        RepoOut(repo_id=r.repo_id, name=f"{r.owner}/{r.name}", sha=r.sha) for r in repos
    ]


@app.get("/repos/{repo_id}/file", response_model=FileOut)
def get_file(repo_id: str, path: str) -> FileOut:
    """Contents of one indexed file. Only paths that are in the index are served, and the
    resolved path must stay inside the repo checkout, so `../` tricks can't read other files."""
    store = get_components().store
    if not is_indexed(store, repo_id) or path not in store.paths(repo_id):
        raise HTTPException(404, "File not indexed")
    root = (settings.data_dir / "repos" / repo_id).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "File not found")
    repo = _repo_from_id(repo_id)
    return FileOut(
        path=path,
        language=detect_language(target) or "text",
        content=target.read_text(encoding="utf-8", errors="replace"),
        github_url=f"https://github.com/{repo.owner}/{repo.name}/blob/{repo.sha}/{path}",
    )


@app.post("/repos", response_model=JobOut, status_code=202)
def start_ingestion(req: IngestRequest, background: BackgroundTasks) -> JobOut:
    job = jobs.create(req.url)
    background.add_task(_run_ingestion, job.id, req.url)
    return JobOut(**asdict(job))


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobOut(**asdict(job))


@app.post("/repos/{repo_id}/ask")
async def ask(repo_id: str, req: AskRequest) -> StreamingResponse:
    started = time.perf_counter()
    components = get_components()
    # Qdrant calls block, so they run in the threadpool; inline they'd stall every request.
    if not await run_in_threadpool(is_indexed, components.store, repo_id):
        raise HTTPException(404, "Repo not indexed (or still indexing); POST /repos first")

    code_k = req.top_k or settings.code_k
    history = trim_history([turn.model_dump() for turn in req.history])
    rewrite_started = time.perf_counter()
    standalone = await standalone_question(history, req.question)
    retrieve_started = time.perf_counter()
    result = await run_in_threadpool(retrieve, repo_id, standalone, components, code_k)
    retrieved = time.perf_counter()
    repo_map = load_repo_map(repo_id)

    def build(chunks):
        return build_messages(
            req.question, repo_map, chunks, result.outlines, history, standalone
        )

    messages, chunks = await run_in_threadpool(fit_to_budget, build, result.chunks)
    repo = _repo_from_id(repo_id)
    logger.info(
        "ask repo=%s exists_ms=%.0f rewrite_ms=%.0f retrieve_ms=%.0f prompt_ms=%.0f "
        "turns=%d chunks=%d/%d",
        repo_id,
        (rewrite_started - started) * 1000,
        (retrieve_started - rewrite_started) * 1000,
        (retrieved - retrieve_started) * 1000,
        (time.perf_counter() - retrieved) * 1000,
        len(history),
        len(chunks),
        len(result.chunks),
    )

    async def events():
        sources = [
            Source(
                label=c.chunk.label,
                path=c.chunk.path,
                start_line=c.chunk.start_line,
                end_line=c.chunk.end_line,
                url=repo.blob_url(c.chunk.path, c.chunk.start_line, c.chunk.end_line),
                score=c.score,
            ).model_dump()
            for c in chunks
        ]
        yield _sse({"query": standalone, "files": result.files})
        yield _sse({"sources": sources})
        try:
            async for token in stream_answer(messages):
                yield _sse({"token": token})
        except Exception as exc:  # tell the client instead of cutting the stream
            logger.exception("LLM stream failed for repo=%s", repo_id)
            yield _sse({"error": f"The language model failed: {type(exc).__name__}. Try again."})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _run_ingestion(job_id: str, url: str) -> None:
    jobs.update(job_id, status="running")
    try:
        with _ingest_lock:
            repo = ingest_repo(
                url,
                get_components(),
                on_progress=lambda stage, progress: jobs.update(
                    job_id, stage=stage, progress=round(progress, 3)
                ),
            )
        jobs.finish(job_id, repo.repo_id)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the client via the job
        jobs.fail(job_id, f"{type(exc).__name__}: {exc}")


def _repo_from_id(repo_id: str) -> RepoRef:
    # GitHub owners can't contain "_", so the first "__" ends the owner; the sha is last.
    owner, rest = repo_id.split("__", 1)
    name, sha = rest.rsplit("__", 1)
    return RepoRef(owner=owner, name=name, sha=sha, local_path=settings.data_dir / "repos" / repo_id)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
