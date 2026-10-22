import pytest

from repobot.generation.prompts import build_messages, is_command_question


@pytest.mark.parametrize(
    "question",
    [
        "How do I run this repository?",
        "How do I set up the environment?",
        "how to install it",
        "How do I train the model?",
        "What's the setup for tests?",
        "How do I get started?",
    ],
)
def test_command_questions(question):
    assert is_command_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "How does the session persist cookies?",
        "What's the second class in model.py?",
        "Where is the retry logic implemented?",
        "Why is the trust_env flag needed?",
    ],
)
def test_explanation_questions(question):
    assert not is_command_question(question)


def test_howto_instruction_only_for_command_questions():
    howto = build_messages("How do I run this repository?", "", [])[-1]["content"]
    explain = build_messages("How do sessions persist cookies?", "", [])[-1]["content"]
    assert "```bash" in howto
    assert "```bash" not in explain


def test_rewritten_follow_up_decides():
    # "And after that?" alone isn't about commands; its standalone rewrite is.
    content = build_messages(
        "And after that?", "", [], standalone="How do I train the model after setup?"
    )[-1]["content"]
    assert "```bash" in content
