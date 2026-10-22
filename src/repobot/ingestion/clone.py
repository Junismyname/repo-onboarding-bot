"""Clone a GitHub repo (shallow) into data/repos/<repo_id>."""

import os
import re
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

from repobot.errors import InvalidRepoUrlError, RepoBotError, RepoTooLargeError
from repobot.models import RepoRef

GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[\w.-]+)/(?P<name>[\w.-]+?)(?:\.git)?/?$"
)
CLONE_TIMEOUT_S = 300


def parse_github_url(url: str) -> tuple[str, str]:
    match = GITHUB_URL_RE.match(url.strip())
    if not match:
        raise InvalidRepoUrlError(f"Not a GitHub repository URL: {url}")
    return match["owner"], match["name"]


def clone_repo(url: str, dest_root: Path, max_repo_mb: int, sha: str | None = None) -> RepoRef:
    """Shallow-clone `url` (latest commit, or `sha` if given) and return its RepoRef.

    The checkout lives at dest_root/repos/<repo_id>; an existing checkout of the same
    commit is reused, offline when `sha` is given. Pinning matters for evals: their
    expected answers are tied to one commit. `.git` is removed after reading the commit
    SHA, since only the working tree is indexed. A fresh clone contains only tracked
    files, so .gitignore'd junk is already absent.
    """
    owner, name = parse_github_url(url)
    repos_dir = dest_root / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)
    if sha and (existing := repos_dir / f"{owner}__{name}__{sha[:12]}").is_dir():
        return RepoRef(owner=owner, name=name, sha=sha, local_path=existing)
    tmp_dir = repos_dir / f".tmp-{uuid.uuid4().hex}"

    try:
        clone_url = f"https://github.com/{owner}/{name}.git"
        if sha:
            # GitHub serves single commits by sha, so a pinned commit is still a shallow fetch.
            _git("init", "--quiet", str(tmp_dir))
            _git("-C", str(tmp_dir), "fetch", "--quiet", "--depth", "1", clone_url, sha)
            _git("-C", str(tmp_dir), "checkout", "--quiet", "FETCH_HEAD")
        else:
            _git("clone", "--depth", "1", "--single-branch", clone_url, str(tmp_dir))
        sha = _git("-C", str(tmp_dir), "rev-parse", "HEAD").strip()
        local_path = repos_dir / f"{owner}__{name}__{sha[:12]}"
        repo = RepoRef(owner=owner, name=name, sha=sha, local_path=local_path)
        if repo.local_path.exists():
            return repo

        _rmtree(tmp_dir / ".git")
        size_mb = _dir_size(tmp_dir) / 1024**2
        if size_mb > max_repo_mb:
            raise RepoTooLargeError(f"{owner}/{name} is {size_mb:.0f} MB (limit {max_repo_mb} MB)")
        tmp_dir.rename(repo.local_path)
        return repo
    finally:
        if tmp_dir.exists():
            _rmtree(tmp_dir)


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_S,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # private/missing repo: fail, don't prompt
        )
    except subprocess.CalledProcessError as exc:
        raise RepoBotError(f"git {args[0]} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RepoBotError(f"git {args[0]} timed out after {CLONE_TIMEOUT_S}s") from exc
    return result.stdout


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _rmtree(path: Path) -> None:
    # Git marks pack files read-only; on Windows rmtree can't delete those without help.
    def clear_readonly_and_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=clear_readonly_and_retry)
