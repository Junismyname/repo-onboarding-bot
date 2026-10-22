class RepoBotError(Exception):
    """Base error for user-facing failures."""


class InvalidRepoUrlError(RepoBotError):
    pass


class RepoTooLargeError(RepoBotError):
    pass
