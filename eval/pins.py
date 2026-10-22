"""The commits every eval runs against (eval/repos.json), and their collection names.

Evals must not pick "whatever version of requests is indexed": expected files, commands and
definitions are only valid at the pinned commit.
"""

import json
from pathlib import Path

PINNED: dict[str, str] = {
    url: sha
    for url, sha in json.loads((Path(__file__).parent / "repos.json").read_text("utf-8")).items()
    if not url.startswith("_")
}


def repo_id(url: str) -> str:
    """Collection name of the pinned commit, e.g. psf__requests__611c6162cbc4."""
    owner, name = url.rstrip("/").split("/")[-2:]
    return f"{owner}__{name}__{PINNED[url][:12]}"
