import uuid


class ChatSession:
    """A single chat thread: its own message history, model, and sampling params."""

    def __init__(self, system_prompt, hf_model, temperature=0.7, top_p=0.95, tools=None):
        self.session_id = uuid.uuid4().hex
        self.title = "New Chat"
        self.system_prompt = system_prompt
        self.hf_model = hf_model
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_message(self, role, content, reasoning=None, tool_calls=None, tool_call_id=None, name=None):
        """Append a message to the history.

        Optionally attaches a reasoning trace (assistant), tool_calls
        (assistant messages that invoke tools), or tool_call_id/name (role
        "tool" messages carrying a tool's result) -- mirroring the shapes
        core.payload.PayloadBuilder expects when replaying history.
        """
        message = {"role": role, "content": content}
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if name is not None:
            message["name"] = name
        self.messages.append(message)

    def get_history(self):
        """Return the full message list (including the system prompt) for the LLM payload."""
        return self.messages

    def maybe_set_title_from(self, text):
        """Derive the sidebar title from the first user message, once."""
        if self.title != "New Chat":
            return
        stripped = text.strip()
        self.title = stripped[:40] + ("..." if len(stripped) > 40 else "")
