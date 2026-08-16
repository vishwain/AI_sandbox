import json

_REGISTRY = {}


def register_tool(name, schema, handler):
    """Register a callable under *name* with its OpenAI-style function *schema*."""
    _REGISTRY[name] = {"schema": schema, "handler": handler}


def get_schemas(names=None):
    """Return the function-calling schemas for *names* (all registered tools if omitted)."""
    keys = names if names is not None else _REGISTRY.keys()
    return [_REGISTRY[name]["schema"] for name in keys]


def dispatch(name, arguments_json):
    """Run the tool named *name* with JSON-encoded *arguments_json*, returning a string result.

    Never raises: unknown tools, bad JSON, and handler exceptions are all
    turned into an "Error: ..." string so the model can react instead of the
    agentic loop crashing.
    """
    if name not in _REGISTRY:
        return f"Error: unknown tool '{name}'"

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: invalid arguments JSON ({exc})"

    try:
        result = _REGISTRY[name]["handler"](**arguments)
    except Exception as exc:  # noqa: BLE001 - surface any handler failure to the model
        return f"Error: {exc}"

    return str(result)
