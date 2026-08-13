import uuid


class ChatSession:
    """A single chat thread: its own message history, model, and sampling params."""

    def __init__(self, system_prompt, hf_model, temperature=0.7, top_p=0.95):
        self.session_id = uuid.uuid4().hex
        self.title = "New Chat"
        self.system_prompt = system_prompt
        self.hf_model = hf_model
        self.temperature = temperature
        self.top_p = top_p
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_message(self, role, content, reasoning=None):
        """Append a message to the history. Optionally attach a reasoning trace."""
        message = {"role": role, "content": content}
        if reasoning:
            message["reasoning"] = reasoning
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
