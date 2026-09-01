"""A reasoning model's <think> block never reaches a caller.

Qwen3.5 thinks before it answers (``Qwen3_5_2bModel.thinking_default``), and
everything downstream of the LLM wrapper treats the generated text as the
answer — the planner parses it as JSON, the chat endpoint streams it to the
user.  So the monologue is filtered out of both the blocking reply and the
token stream, including when its tags arrive split across stream chunks.
"""

import pytest

from smartloop.llms.thinking import ThinkingFilter, strip_thinking
from smartloop.model.qwen3_5_2b import Qwen3_5_2bModel
from smartloop.model.qwen2_vl_2b import Qwen2VL_2bModel


def _stream(chunks):
    """Run chunks through the streaming filter, as ``LlamaCpp._stream`` does."""
    f = ThinkingFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_reasoning_block_is_removed():
    assert strip_thinking("<think>weigh it up</think>The answer is 4.") == "The answer is 4."


def test_text_without_reasoning_is_untouched():
    assert strip_thinking("The answer is 4.") == "The answer is 4."


def test_pre_filled_open_tag_leaves_only_a_close_tag():
    # Some chat templates put the opening <think> in the generation prompt, so
    # the model only emits the closing tag.
    assert strip_thinking("weigh it up</think>The answer is 4.") == "The answer is 4."


def test_json_reply_survives_reasoning():
    out = strip_thinking('<think>which tool?</think>{"tool": "search"}')
    assert out == '{"tool": "search"}'


def test_tags_split_across_stream_chunks():
    assert _stream(["<thi", "nk>hmm", "m</thin", "k>Fo", "ur."]) == "Four."


def test_partial_tag_that_turns_out_to_be_text_is_released():
    assert _stream(["Compare a <", "b in C."]) == "Compare a <b in C."


def test_reasoning_is_withheld_chunk_by_chunk():
    f = ThinkingFilter()
    assert f.feed("<think>step one") == ""
    assert f.feed(" step two") == ""
    assert f.feed("</think>Done.") == "Done."
    assert f.flush() == ""
    assert not f.unterminated


def test_truncated_thought_falls_back_to_the_reasoning():
    # max_tokens cut generation off mid-thought: there is no reply to show, and
    # a blank answer is worse output than the truncated reasoning.
    f = ThinkingFilter()
    assert f.feed("<think>halfway through") == ""
    assert f.unterminated
    assert f.flush() == "halfway through"


def test_truncated_thought_after_a_reply_adds_nothing():
    f = ThinkingFilter()
    assert f.feed("Four.<think>second thoughts") == "Four."
    assert f.flush() == ""


def test_gemma4_reasoning_channel_is_removed():
    # Gemma 4 does not use <think>: its template emits reasoning on a
    # <|channel>thought…<channel|> channel (and carries a strip_thinking macro
    # that splits on exactly these markers).
    out = strip_thinking("<|channel>thought\nweigh it up\n<channel|>The answer is 4.")
    assert out == "The answer is 4."


def test_gemma4_channel_split_across_stream_chunks():
    assert _stream(["<|chan", "nel>thought\nhmm", "<chan", "nel|>Four."]) == "Four."


def test_reasoning_is_opt_in_for_every_model():
    # Support is declared by ModelCapability.THINKING; the default is off, so
    # reasoning costs nothing until switched on in the model picker.
    assert Qwen3_5_2bModel.thinking_default() is False
    assert Qwen2VL_2bModel.thinking_default() is False
