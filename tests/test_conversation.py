import asyncio

from repobot.generation import conversation
from repobot.generation.conversation import standalone_question, trim_history
from repobot.generation.prompts import build_messages


def test_trim_history_keeps_recent_turns_and_drops_sources():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    history.append({"role": "assistant", "content": "A is first [m.py:1-5].\n\nSources:\n- [m.py:1-5]"})
    trimmed = trim_history(history)
    assert len(trimmed) == conversation.MAX_TURNS
    assert trimmed[-1]["content"] == "A is first [m.py:1-5]."


def test_no_history_skips_the_llm_call(monkeypatch):
    async def fail(*args, **kwargs):
        raise AssertionError("should not call the LLM")

    monkeypatch.setattr(conversation, "complete", fail)
    assert asyncio.run(standalone_question([], "What is in model.py?")) == "What is in model.py?"


def test_rewrite_falls_back_when_output_is_unusable(monkeypatch):
    async def answer_instead(*args, **kwargs):
        return "x" * 600  # the model answered instead of rewriting

    monkeypatch.setattr(conversation, "complete", answer_instead)
    history = [{"role": "user", "content": "second class of model.py?"}]
    assert asyncio.run(standalone_question(history, "and the first?")) == "and the first?"


def test_prompt_includes_history_and_interpretation():
    history = [
        {"role": "user", "content": "What's the second class of model.py?"},
        {"role": "assistant", "content": "MobileNetYOLO."},
    ]
    messages = build_messages(
        "What about the first one?", "", [], history=history,
        standalone="What is the first class in model.py?",
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "(In context, this means: What is the first class in model.py?)" in messages[-1]["content"]


def test_trim_history_never_starts_with_an_answer_and_caps_questions():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2" * 5000},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
    ]
    trimmed = trim_history(history)  # last 6 would start with "a1"
    assert trimmed[0]["role"] == "user"
    assert len(trimmed[0]["content"]) <= conversation.MAX_QUESTION_CHARS + 2
