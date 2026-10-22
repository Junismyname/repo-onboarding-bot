"""Cheap question-intent checks shared by retrieval and prompting."""

import re

# Questions about commands to type ("how do I install / run / test ..."). They get the
# repo's build files in retrieval and a how-to answer format in the prompt.
COMMAND_QUESTION_RE = re.compile(
    r"\b(install|set ?up|setup|run|running|build|train|training|test|tests|launch|start|"
    r"deploy|execute|compile|get started|getting started)\b",
    re.IGNORECASE,
)


def is_command_question(question: str) -> bool:
    return bool(COMMAND_QUESTION_RE.search(question))
