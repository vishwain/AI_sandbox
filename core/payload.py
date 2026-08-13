class PayloadBuilder:
    """Builder Pattern: assembles the exact JSON payload the HF InferenceClient expects."""

    def __init__(self, model, tools=None, temperature=0.7, top_p=0.95):
        self.model = model
        self.tools = tools or []
        self.temperature = temperature
        self.top_p = top_p
        self.messages = []

    def add_system_prompt(self, instructions):
        """Insert the system prompt at index 0 (it must always lead the message list)."""
        self.messages.insert(0, {"role": "system", "content": instructions})
        return self

    def add_user_message(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        return self

    def add_assistant_message(self, content, tool_calls=None):
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)
        return self

    def add_tool_result(self, tool_call_id, function_name, result_content):
        """Attach a tool's output to the exact tool_call_id the model requested."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": result_content,
            }
        )
        return self

    def truncate_context(self, max_tokens):
        """Drop oldest non-system turns until the estimated token count fits.

        Keeps the system prompt (if present at index 0) and as many of the
        most recent turns as fit within max_tokens (estimated as len(text) // 4).
        """
        def estimate(msg):
            return max(len(str(msg.get("content", ""))) // 4, 1)

        if not self.messages:
            return self

        has_system = self.messages[0].get("role") == "system"
        system_msg = [self.messages[0]] if has_system else []
        rest = self.messages[1:] if has_system else self.messages[:]

        kept = []
        total = sum(estimate(m) for m in system_msg)
        for msg in reversed(rest):
            cost = estimate(msg)
            if total + cost > max_tokens and kept:
                break
            kept.append(msg)
            total += cost
        kept.reverse()

        self.messages = system_msg + kept
        return self

    def build(self):
        """Return the fully structured dict, ready to pass as **kwargs to the InferenceClient."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.tools:
            payload["tools"] = self.tools
        return payload
