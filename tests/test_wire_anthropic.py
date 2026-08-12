"""The Anthropic wire, checked against what a real client actually sends.

The client of record is NOON, whose reader (`crates/noon-core/src/stream.rs`)
is stricter than the spec: it reads only `data:` lines, treats a missing
`stop_reason` as truncation, and positions blocks by `index`. These tests
encode those three, plus the two keys NOON always sends that must be tolerated
rather than refused.
"""

from __future__ import annotations

import json

import pytest

from maatml.wire.anthropic import Translator, to_openai, to_openai_messages, to_openai_tools


def frames(raw: list[bytes]) -> list[dict]:
    """Parse the way stream.rs does: `data:` lines only, everything else skipped."""
    out = []
    for chunk in raw:
        for line in chunk.decode().splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


# ------------------------------------------------------------------ request


def test_system_blocks_become_a_leading_message():
    body = {
        "system": [{"type": "text", "text": "you are NOON",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "messages": [{"role": "user", "content": "hello"}],
    }
    out = to_openai_messages(body)
    assert out[0] == {"role": "system", "content": "you are NOON"}
    assert out[1] == {"role": "user", "content": "hello"}


def test_cache_control_and_output_config_are_dropped_not_refused():
    """NOON sends both on every request. A 400 here breaks every ask."""
    body = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
        ]}],
        "max_tokens": 32000,
    }
    out = to_openai(body, "local")
    assert "cache_control" not in json.dumps(out)
    assert "output_config" not in out
    assert out["max_tokens"] == 32000


def test_tool_use_and_tool_result_round_trip_to_openai_shapes():
    body = {"messages": [
        {"role": "assistant", "content": [
            {"type": "text", "text": "reading it"},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file",
             "input": {"path": "docs/shots.json"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "{...}"},
        ]},
    ]}
    out = to_openai_messages(body)
    assert out[0]["tool_calls"][0]["id"] == "toolu_1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "read_file"
    args = json.loads(out[0]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "docs/shots.json"}
    assert out[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "{...}"}


def test_tool_result_content_may_be_a_block_array():
    """NOON stores it as a bare string in 164 cases and an array in 96."""
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "ok"}]},
    ]}]}
    assert to_openai_messages(body)[0]["content"] == "ok"


def test_input_schema_becomes_parameters():
    tools = [{"name": "set_shot_cell", "description": "d",
              "input_schema": {"type": "object", "properties": {"shot": {"type": "string"}}}}]
    out = to_openai_tools(tools)
    assert out[0]["function"]["parameters"]["properties"]["shot"]["type"] == "string"
    assert "input_schema" not in json.dumps(out)


def test_thinking_is_switched_off_at_the_template():
    """The corpus is trained with thinking stripped; a server that reasons
    first is a different model than the one measured."""
    out = to_openai({"messages": []}, "local")
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


# ----------------------------------------------------------------- response


def test_text_streams_as_one_indexed_block_and_always_stops():
    talk = Translator("local")
    raw = [talk.start()]
    raw += list(talk.delta({"choices": [{"delta": {"content": "he"}}]}))
    raw += list(talk.delta({"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}]}))
    raw += list(talk.finish())

    seen = frames(raw)
    kinds = [f["type"] for f in seen]
    assert kinds[0] == "message_start"
    assert "content_block_start" in kinds
    assert kinds[-1] == "message_stop"

    said = "".join(f["delta"]["text"] for f in seen if f["type"] == "content_block_delta")
    assert said == "hello"
    assert all(f["index"] == 0 for f in seen if "index" in f)

    ending = [f for f in seen if f["type"] == "message_delta"]
    assert ending and ending[0]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.parametrize("upstream,expected", [
    ("stop", "end_turn"),
    ("length", "max_tokens"),
    ("tool_calls", "tool_use"),
])
def test_finish_reason_maps_to_anthropic(upstream, expected):
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "x"}, "finish_reason": upstream}]}))
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"]
    assert ending[0]["delta"]["stop_reason"] == expected


def test_a_stream_that_never_said_why_still_carries_a_stop_reason():
    """stream.rs treats a missing stop_reason as truncation and errors. An
    upstream that closes without one must not become a broken reply."""
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "x"}}]}))
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"]
    assert ending[0]["delta"]["stop_reason"] == "end_turn"


def test_tool_calls_get_their_own_block_index_and_json_deltas():
    talk = Translator("local")
    list(talk.delta({"choices": [{"delta": {"content": "calling"}}]}))
    out = list(talk.delta({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "function": {"name": "edit_file", "arguments": '{"pa'}},
    ]}}]}))
    out += list(talk.delta({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": 'th":"a.py"}'}},
    ]}, "finish_reason": "tool_calls"}]}))

    seen = frames(out)
    opened = [f for f in seen if f["type"] == "content_block_start"]
    assert opened[0]["index"] == 1, "text took block 0, so the call takes 1"
    assert opened[0]["content_block"]["name"] == "edit_file"

    args = "".join(f["delta"]["partial_json"] for f in seen
                   if f["type"] == "content_block_delta"
                   and f["delta"]["type"] == "input_json_delta")
    assert json.loads(args) == {"path": "a.py"}


def test_two_tool_calls_do_not_share_an_index():
    talk = Translator("local")
    out = list(talk.delta({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "a", "function": {"name": "read_file", "arguments": "{}"}},
        {"index": 1, "id": "b", "function": {"name": "get_state", "arguments": "{}"}},
    ]}}]}))
    opened = [f for f in frames(out) if f["type"] == "content_block_start"]
    assert [f["index"] for f in opened] == [0, 1]


def test_usage_uses_anthropics_four_names():
    talk = Translator("local")
    list(talk.delta({"usage": {"prompt_tokens": 120, "completion_tokens": 8},
                     "choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}))
    ending = [f for f in frames(list(talk.finish())) if f["type"] == "message_delta"][0]
    assert ending["usage"] == {
        "input_tokens": 120, "output_tokens": 8,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }


def test_no_thinking_block_is_ever_emitted():
    """Anthropic rejects a thinking block handed back without its signature,
    and a local server has none to give."""
    talk = Translator("local")
    raw = [talk.start()]
    raw += list(talk.delta({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}))
    raw += list(talk.finish())
    assert "thinking" not in json.dumps(frames(raw))
