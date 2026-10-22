from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    url: str = Field(examples=["https://github.com/psf/requests"])


class JobOut(BaseModel):
    id: str
    url: str
    status: str
    stage: str
    progress: float
    repo_id: str | None = None
    error: str | None = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=50, description="earlier turns")
    top_k: int | None = Field(default=None, ge=1, le=30, description="code chunks to retrieve")


class RepoOut(BaseModel):
    repo_id: str
    name: str  # owner/name
    sha: str  # short commit sha


class FileOut(BaseModel):
    path: str
    language: str
    content: str
    github_url: str


class Source(BaseModel):
    label: str
    path: str
    start_line: int
    end_line: int
    url: str
    score: float
