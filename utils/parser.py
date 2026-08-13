import json

_THINK_START = "<think>"
_THINK_END = "</think>"


def iter_stream_text(stream, reasoning_sink=None):
    """Yield visible text chunks from a streamed LLM response.

    Reasoning wrapped in <think>...</think> is stripped from the visible
    stream. If *reasoning_sink* (a list) is given, the reasoning text is
    appended to it chunk-by-chunk instead of being discarded, so callers can
    render it (e.g. in a collapsed "thinking" expander) after streaming ends.
    """
    yield from _process_think_tags(_extract_text(stream), reasoning_sink)


def _extract_text(stream):
    """Yield plain text from streamed chunks, tolerating dict/bytes/SDK chunk shapes."""
    for chunk in stream:
        text = _chunk_to_text(chunk)
        if text is not None:
            yield text


def _chunk_to_text(chunk):
    if isinstance(chunk, (bytes, bytearray)):
        try:
            chunk = chunk.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(chunk, str):
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            return chunk
        return _payload_to_text(payload)

    if isinstance(chunk, dict):
        return _payload_to_text(chunk)

    if hasattr(chunk, "choices") and getattr(chunk, "choices", None):
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if isinstance(content, str):
            return content

    if hasattr(chunk, "delta") and getattr(chunk, "delta", None):
        content = getattr(chunk.delta, "content", None)
        if isinstance(content, str):
            return content

    return None


def _payload_to_text(payload):
    if not isinstance(payload, dict):
        return None

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    choices = payload.get("choices", [])
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta", {})
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

    if isinstance(payload.get("content"), str):
        return payload["content"]

    return None


def _process_think_tags(text_stream, reasoning_sink=None):
    """Split *text_stream* into visible text (yielded) and reasoning (sunk).

    Text outside <think> tags is always yielded. Text inside <think> tags is
    appended to *reasoning_sink* (if given) instead of being yielded. Tags
    split across streamed chunks are handled correctly.
    """
    buffer = ""
    in_think = False

    for text in text_stream:
        buffer += text
        while buffer:
            if not in_think:
                start = buffer.find(_THINK_START)
                if start == -1:
                    # Keep a tail long enough to not split a partial <think> tag.
                    if len(buffer) > len(_THINK_START):
                        yield buffer[: -len(_THINK_START)]
                        buffer = buffer[-len(_THINK_START):]
                    break
                if start > 0:
                    yield buffer[:start]
                buffer = buffer[start + len(_THINK_START):]
                in_think = True
                continue

            end = buffer.find(_THINK_END)
            if end == -1:
                if len(buffer) > len(_THINK_END):
                    if reasoning_sink is not None:
                        reasoning_sink.append(buffer[: -len(_THINK_END)])
                    buffer = buffer[-len(_THINK_END):]
                break

            if reasoning_sink is not None:
                reasoning_sink.append(buffer[:end])
            buffer = buffer[end + len(_THINK_END):]
            in_think = False

    # Flush any remaining plain text once the stream ends.
    if buffer and not in_think:
        yield buffer
    elif buffer and in_think and reasoning_sink is not None:
        reasoning_sink.append(buffer)


def extract_reasoning(content_string):
    """Split a complete (non-streamed) string into (visible_text, reasoning_text)."""
    reasoning_sink = []
    visible = "".join(_process_think_tags(iter([content_string]), reasoning_sink))
    return visible, "".join(reasoning_sink)


def parse_tool_calls(payload_json):
    """Extract the tool_calls array from a chat-completion payload, if present.

    Placeholder for the phase-2 agentic loop; not yet wired into the UI.
    """
    choices = payload_json.get("choices", [])
    if not choices:
        return None
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        return None
    return choice.get("message", {}).get("tool_calls")
